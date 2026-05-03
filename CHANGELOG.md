# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-05-03

Initial public release.

### Added

- Sidebar **Models** tab with one-click workflow scanning.
- Multi-strategy HuggingFace search (repo-name + full-text README +
  curated fallback list of well-known umbrella repos).
- CivitAI search with token support.
- Curated `known_models.json` covering popular models (FLUX.1, SDXL,
  ControlNet v1.1, Stable Diffusion 1.5, common upscalers, ...).
- Background `DownloadManager` with adaptive 500ms / 2s polling, smooth
  inter-poll progress animation via `requestAnimationFrame`, rolling
  3-second speed window, ETA calculation, resume support via `.part`
  files, HTTP error / HTML detection, cancel.
- **Download all** bulk endpoint that auto-resolves every missing file.
- **Find sources** per-row search modal with HuggingFace / CivitAI /
  curated DB candidates.
- **Move existing** action - relocates locally-present files with the
  wrong path into the location ComfyUI expects.
- **Disk space**: optional file linking. Hardlink (default), symlink,
  or auto. Matches by filename + size; never reads file content.
- Settings dialog with masked-preview HuggingFace and CivitAI token
  storage. Tokens never leave the server in plaintext.
- Filename-based folder heuristics covering ONNX detection / pose /
  rembg / insightface / sam, FILM / RIFE frame interpolation,
  `clip_vision_*`, `t5xxl*`, `clip_l*`, etc.
- UI workflow node-slot map for the common ComfyUI loaders plus
  Kijai's `OnnxDetectionModelLoader`,
  `FrameInterpolationModelLoader` (core),
  `DwposeDetector`, `UltralyticsDetectorProvider`, ...
- Subfolder preservation for workflow references like
  `Wan2_2/lightx2v/foo.safetensors`.
- Cross-platform support (tested on Windows + Linux).
- Duplicate-download protection: enqueueing the same filename twice
  returns the existing job instead of starting a parallel transfer.
- Toast notifications for "already downloading" feedback.

### Security

- `config.json` is in `.gitignore` and never committed.
- `GET /model_downloader/config` returns tokens only as masked
  preview (`hf_xx••••wxyz`) plus a length count.
- Saving a value that contains the bullet character `•` is refused on
  both client and server, so an accidentally-pasted masked preview
  cannot overwrite a real token.
- Path traversal (`..`) in subfolder strings is stripped.
