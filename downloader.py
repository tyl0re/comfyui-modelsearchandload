"""Threaded download manager with progress tracking."""

from __future__ import annotations

import collections
import os
import threading
import time
import uuid
import urllib.request
import urllib.error
from typing import Any

from .config import load_config
from .scanner import get_target_directory


# Window size (seconds) for rolling-average speed calculation.
_SPEED_WINDOW_S = 3.0


class DownloadJob:
    def __init__(
        self,
        url: str,
        dest_dir: str,
        filename: str,
        source: str = "unknown",
        expected_size: int | None = None,
    ):
        self.id = uuid.uuid4().hex
        self.url = url
        self.dest_dir = dest_dir
        self.filename = filename
        self.source = source
        # Expected size (from search candidate). Used by the linker to
        # confirm a same-name local file is actually the same model.
        self.expected_size = expected_size
        # connecting | linking | linked | downloading | done | error | cancelled
        self.status = "connecting"
        self.error: str | None = None
        self.bytes_done: int = 0
        self.bytes_total: int = 0
        # If we resolved this job by linking, these record what we did.
        self.link_method: str | None = None  # "hardlink" | "symlink" | None
        self.link_source: str | None = None  # absolute path of the original
        self.started_at: float | None = None
        self.finished_at: float | None = None
        # Transient flag: set by DownloadManager.enqueue when the caller
        # asked for a job that's already in flight, so the API layer can
        # report it instead of pretending it just queued a fresh one.
        # Not serialised to to_dict().
        self._is_duplicate_of_active: bool = False
        # Rolling window of (timestamp, bytes_done) snapshots, used to derive
        # an instantaneous-ish download speed without big jitter.
        self._samples: collections.deque[tuple[float, int]] = collections.deque(maxlen=64)
        self._cancel = threading.Event()

    @property
    def dest_path(self) -> str:
        return os.path.join(self.dest_dir, self.filename)

    @property
    def temp_path(self) -> str:
        return self.dest_path + ".part"

    def cancel(self):
        self._cancel.set()

    def _record_sample(self, now: float | None = None) -> None:
        if now is None:
            now = time.time()
        self._samples.append((now, self.bytes_done))
        # Drop entries older than the window
        cutoff = now - _SPEED_WINDOW_S
        while self._samples and self._samples[0][0] < cutoff:
            self._samples.popleft()

    def _speed_bps(self) -> float:
        """Bytes per second over the rolling window. Returns 0 if no data."""
        if len(self._samples) < 2:
            return 0.0
        t0, b0 = self._samples[0]
        t1, b1 = self._samples[-1]
        dt = t1 - t0
        if dt <= 0:
            return 0.0
        return max(0.0, (b1 - b0) / dt)

    def to_dict(self) -> dict[str, Any]:
        speed = self._speed_bps()
        eta = None
        if speed > 0 and self.bytes_total and self.status == "downloading":
            remaining = max(0, self.bytes_total - self.bytes_done)
            eta = remaining / speed
        progress = 0.0
        if self.bytes_total:
            progress = self.bytes_done / self.bytes_total
        return {
            "id": self.id,
            "url": self.url,
            "filename": self.filename,
            "dest_dir": self.dest_dir,
            "source": self.source,
            "status": self.status,
            "error": self.error,
            "bytes_done": self.bytes_done,
            "bytes_total": self.bytes_total,
            "progress": progress,
            "speed_bps": speed,
            "eta_seconds": eta,
            "link_method": self.link_method,
            "link_source": self.link_source,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }


class DownloadManager:
    def __init__(self):
        self._jobs: dict[str, DownloadJob] = {}
        self._lock = threading.Lock()

    # ---- Public API ----

    def enqueue(
        self,
        url: str,
        folder: str,
        filename: str,
        source: str = "unknown",
        subfolder: str = "",
        expected_size: int | None = None,
    ) -> DownloadJob:
        # Place the file at:   <models>/<folder>/<subfolder>/<filename>
        # The subfolder is the relative path component the workflow used
        # (e.g. "Wan2_2/lightx2v"). Without it, ComfyUI would fail to find
        # the file because its loaders look for the EXACT relative path
        # the user typed in the workflow.
        # The path manipulation here is platform-neutral: subfolder strings
        # from workflows can use either separator (Windows authors often
        # use backslashes), but on disk we always join via os.path.join.
        base_dir = get_target_directory(folder)
        sub = (subfolder or "").replace("\\", "/").strip("/")
        # Defensive: refuse traversal attempts and absolute paths
        if sub:
            parts = [p for p in sub.split("/") if p not in ("", ".", "..")]
            sub = "/".join(parts)
        dest_dir = os.path.join(base_dir, *sub.split("/")) if sub else base_dir

        # Duplicate-detection: if there is already an ACTIVE job downloading
        # the exact same filename to the exact same destination directory,
        # return the existing job instead of starting a second one. This
        # prevents the user (or a workflow that references the same model
        # from multiple nodes) from blasting the same URL twice.
        # Finished/error/cancelled jobs are NOT considered active so the
        # user can retry after a failure.
        active_states = ("queued", "connecting", "linking", "downloading")
        with self._lock:
            for existing in self._jobs.values():
                if existing.status not in active_states:
                    continue
                if existing.filename != filename:
                    continue
                # Compare normalized destination dirs (case-insensitive on
                # Windows, case-sensitive on Linux) so we don't trip over
                # trailing separators or path-style differences.
                try:
                    same = os.path.normcase(os.path.abspath(existing.dest_dir)) \
                        == os.path.normcase(os.path.abspath(dest_dir))
                except Exception:
                    same = existing.dest_dir == dest_dir
                if same:
                    # Mark the returned job so the caller can tell the user
                    # this wasn't a fresh enqueue.
                    existing._is_duplicate_of_active = True
                    return existing

        os.makedirs(dest_dir, exist_ok=True)
        job = DownloadJob(
            url=url, dest_dir=dest_dir, filename=filename,
            source=source, expected_size=expected_size,
        )
        with self._lock:
            self._jobs[job.id] = job
        t = threading.Thread(target=self._run, args=(job,), daemon=True)
        t.start()
        return job

    def list_jobs(self) -> list[dict]:
        with self._lock:
            return [j.to_dict() for j in self._jobs.values()]

    def get(self, job_id: str) -> DownloadJob | None:
        return self._jobs.get(job_id)

    def cancel(self, job_id: str) -> bool:
        job = self._jobs.get(job_id)
        if not job:
            return False
        job.cancel()
        return True

    def clear_finished(self) -> int:
        with self._lock:
            old = list(self._jobs.items())
            removed = 0
            for jid, j in old:
                if j.status in ("done", "error", "cancelled"):
                    del self._jobs[jid]
                    removed += 1
            return removed

    # ---- Worker ----

    def _build_request(self, job: DownloadJob, resume_from: int = 0) -> urllib.request.Request:
        cfg = load_config()
        headers = {"User-Agent": "ComfyUI-ModelDownloader/1.0"}
        if "huggingface.co" in job.url and cfg.get("huggingface_token"):
            headers["Authorization"] = f"Bearer {cfg['huggingface_token']}"
        elif "civitai.com" in job.url and cfg.get("civitai_token"):
            # CivitAI accepts token in either header or as ?token= query string
            sep = "&" if "?" in job.url else "?"
            job.url = f"{job.url}{sep}token={cfg['civitai_token']}"
        if resume_from > 0:
            headers["Range"] = f"bytes={resume_from}-"
        return urllib.request.Request(job.url, headers=headers)

    def _try_link_existing(self, job: DownloadJob) -> bool:
        """If the user enabled linking, look for a same-name + same-size copy
        already on disk and link it to the destination. Returns True if the
        job was satisfied via linking (no download needed)."""
        cfg = load_config()
        if not cfg.get("enable_linking"):
            return False
        try:
            from .linker import find_existing_copy, link_existing
        except Exception:
            return False
        # If we don't know the expected size yet, try to obtain it cheaply
        # via a HEAD request so the size match is reliable.
        size = job.expected_size
        if size is None or size <= 0:
            size = self._head_content_length(job.url)
            if size:
                job.expected_size = size

        existing = find_existing_copy(job.filename, size)
        if not existing:
            return False
        # Skip if the existing copy IS already the destination - nothing to do.
        try:
            if os.path.samefile(existing, job.dest_path):
                job.status = "done"
                job.link_method = "already-linked"
                job.link_source = existing
                job.bytes_total = job.bytes_done = os.path.getsize(existing)
                job.finished_at = time.time()
                return True
        except OSError:
            pass

        job.status = "linking"
        mode = cfg.get("linking_mode", "auto") or "auto"
        result = link_existing(existing, job.dest_path, mode=mode)
        if result.get("linked"):
            job.status = "linked"
            job.link_method = result["method"]
            job.link_source = existing
            try:
                sz = os.path.getsize(job.dest_path)
                job.bytes_total = sz
                job.bytes_done = sz
            except OSError:
                pass
            job.finished_at = time.time()
            return True
        # Linking failed (e.g. cross-fs hardlink + no symlink permission on
        # Windows). Don't fail the job - just fall back to a normal download.
        return False

    def _head_content_length(self, url: str) -> int | None:
        """Cheap HEAD probe to learn the expected size before downloading.
        Returns None if the server doesn't answer or doesn't supply a length."""
        try:
            req = urllib.request.Request(url, method="HEAD")
            cfg = load_config()
            if "huggingface.co" in url and cfg.get("huggingface_token"):
                req.add_header("Authorization", f"Bearer {cfg['huggingface_token']}")
            with urllib.request.urlopen(req, timeout=10) as resp:
                cl = resp.headers.get("Content-Length")
                return int(cl) if cl else None
        except Exception:
            return None

    def _run(self, job: DownloadJob) -> None:
        # Stays "connecting" until the first byte arrives so the UI can
        # distinguish "queued/handshake in progress" from "actually transferring".
        job.status = "connecting"
        job.started_at = time.time()

        # Try to satisfy the job by linking an already-present copy first.
        # Only when the user opted in via Settings.
        try:
            if self._try_link_existing(job):
                return
        except Exception as e:
            # Linking is best-effort; never let it abort the download path.
            print(f"[ModelDownloader] link probe failed: {e}")

        try:
            # Resume support if .part file exists
            resume_from = 0
            if os.path.exists(job.temp_path):
                resume_from = os.path.getsize(job.temp_path)

            req = self._build_request(job, resume_from=resume_from)
            try:
                resp = urllib.request.urlopen(req, timeout=30)
            except urllib.error.HTTPError as e:
                # Fallback: if 416 (range not satisfiable), restart
                if e.code == 416 and resume_from > 0:
                    os.remove(job.temp_path)
                    resume_from = 0
                    req = self._build_request(job, resume_from=0)
                    resp = urllib.request.urlopen(req, timeout=30)
                else:
                    raise

            total_header = resp.headers.get("Content-Length")
            if total_header:
                job.bytes_total = int(total_header) + resume_from
            job.bytes_done = resume_from
            job._record_sample()

            mode = "ab" if resume_from > 0 else "wb"
            # Smaller chunk size = more frequent progress updates for the UI
            # at the cost of slightly more Python overhead. 64 KB is a good
            # balance: ~150 updates/sec at 10 MB/s.
            chunk = 1024 * 64
            first_chunk = True
            last_sample_t = time.time()

            with open(job.temp_path, mode) as f:
                while True:
                    if job._cancel.is_set():
                        job.status = "cancelled"
                        return
                    buf = resp.read(chunk)
                    if not buf:
                        break
                    f.write(buf)
                    job.bytes_done += len(buf)
                    if first_chunk:
                        # First payload byte received -> we are actually transferring
                        job.status = "downloading"
                        first_chunk = False
                        job._record_sample()
                    else:
                        # Sample at most ~5 times per second to avoid lock contention
                        now = time.time()
                        if now - last_sample_t >= 0.2:
                            job._record_sample(now)
                            last_sample_t = now
                # Final sample so the UI shows the last bytes immediately
                job._record_sample()

            # Sanity: file should be > 0 bytes and not look like an HTML error page
            if os.path.getsize(job.temp_path) < 1024:
                with open(job.temp_path, "rb") as f:
                    head = f.read(512).lower()
                if b"<html" in head or b"<!doctype html" in head:
                    raise RuntimeError("Server returned HTML (likely auth error or gated model).")

            os.replace(job.temp_path, job.dest_path)
            job.status = "done"
            job.finished_at = time.time()
        except Exception as e:
            job.status = "error"
            job.error = str(e)
            job.finished_at = time.time()


# Singleton
manager = DownloadManager()
