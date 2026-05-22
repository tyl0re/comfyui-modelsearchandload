"""Inspect every LoRA file under one or more roots and report what we can
learn about its trigger words and training captions.

The tool reads ``.safetensors`` file metadata (mostly written by kohya_ss
/ sd-scripts trainers) and produces a per-file summary. It never touches
the tensor data, so it stays cheap even on very large model trees.

Usage:

    python tools/lora_triggers.py
    python tools/lora_triggers.py C:/Ai/ComfyUI/models/loras
    python tools/lora_triggers.py --json --top 20 > report.json
    python tools/lora_triggers.py --query "gopro lens" --query "low lighting"

Per LoRA we extract:
  * ``modelspec.title`` / ``ss_output_name`` - the model's own name
  * ``modelspec.architecture`` / ``ss_base_model_version`` - what base model
    this LoRA was trained on
  * ``ss_dataset_dirs`` - the dataset folder names. Kohya-style trainers
    encode the "class token" / trigger phrase in the folder name, e.g.
    ``1_ultra_real_photo ultra-realistic photograph``. The text after the
    "<repeats>_<id>" prefix is the conventional trigger.
  * ``ss_tag_frequency`` - per-caption tag occurrences. We report the
    most common tags. This is what the model actually saw during
    training, so it's a better hint than what a model card claims.
  * Optional ``--query`` strings: report exact + substring matches in
    the tag frequencies. Use this to verify whether a "suggested word"
    from a model card was actually in the training set.

This is a read-only diagnostic; nothing is written unless you redirect
stdout to a file.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from typing import Any


def _iter_lora_files(roots: list[str]):
    for root in roots:
        if not os.path.isdir(root):
            continue
        for d, _dirs, files in os.walk(root, followlinks=True):
            for f in files:
                if f.lower().endswith(".safetensors"):
                    yield os.path.join(d, f)


def _read_metadata(path: str) -> dict[str, str]:
    try:
        from safetensors import safe_open
    except ImportError:
        print("safetensors not installed; run: pip install safetensors", file=sys.stderr)
        raise
    with safe_open(path, framework="pt") as f:
        return dict(f.metadata() or {})


def _parse_lora(meta: dict[str, str], top: int = 10, queries: list[str] | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {
        "title": meta.get("modelspec.title") or meta.get("ss_output_name"),
        "architecture": meta.get("modelspec.architecture") or meta.get("ss_base_model_version"),
        "resolution": meta.get("modelspec.resolution"),
        "date": meta.get("modelspec.date"),
        "network_dim": meta.get("ss_network_dim"),
        "network_alpha": meta.get("ss_network_alpha"),
        "num_train_images": meta.get("ss_num_train_images"),
        "num_epochs": meta.get("ss_num_epochs"),
        "steps": meta.get("ss_steps"),
        "dataset_dirs": [],
        "triggers": [],
        "top_tags": [],
        "total_unique_tags": 0,
        "has_metadata": bool(meta),
        "query_hits": {},
    }
    raw_dirs = meta.get("ss_dataset_dirs")
    if raw_dirs:
        try:
            data = json.loads(raw_dirs)
            for key in data.keys():
                out["dataset_dirs"].append(key)
                # Kohya convention: "<repeats>_<id> [trigger phrase]".
                parts = key.split(" ", 1)
                if len(parts) == 2 and parts[1].strip():
                    out["triggers"].append(parts[1].strip())
        except Exception:
            pass
    raw_tags = meta.get("ss_tag_frequency")
    bucket: Counter = Counter()
    if raw_tags:
        try:
            data = json.loads(raw_tags)
            for _dataset, freqs in data.items():
                for tag, n in freqs.items():
                    bucket[tag] += int(n)
        except Exception:
            pass
    out["total_unique_tags"] = len(bucket)
    out["top_tags"] = [
        {"tag": t, "count": n} for t, n in bucket.most_common(top)
    ]
    for q in queries or []:
        ql = q.lower()
        hits = sorted(
            ((t, n) for t, n in bucket.items() if ql in t.lower()),
            key=lambda x: -x[1],
        )[:10]
        out["query_hits"][q] = [{"tag": t, "count": n} for t, n in hits]
    return out


def _print_human(rows: list[dict[str, Any]], queries: list[str]) -> None:
    with_t = [r for r in rows if r["info"].get("triggers")]
    no_t = [r for r in rows if not r["info"].get("triggers")]

    print(f"Total LoRAs: {len(rows)}")
    print(f"With explicit dataset trigger: {len(with_t)}")
    print(f"Without explicit dataset trigger: {len(no_t)}")
    print()

    print("WITH EXPLICIT TRIGGER")
    print("---------------------")
    for r in with_t:
        info = r["info"]
        rel = r["rel"]
        print(f"- {rel}")
        if info.get("title"):
            print(f"    title: {info['title']}")
        if info.get("architecture"):
            print(f"    arch:  {info['architecture']}")
        for t in info["triggers"]:
            print(f"    trigger: {t}")
        if info["top_tags"]:
            preview = ", ".join(f"{x['tag']}({x['count']})" for x in info["top_tags"][:5])
            print(f"    top tags: {preview}")
        _print_queries(info)

    print()
    print("NO EXPLICIT TRIGGER")
    print("-------------------")
    for r in no_t:
        info = r["info"]
        rel = r["rel"]
        extras: list[str] = []
        if info.get("title"):
            extras.append(f"title={info['title']}")
        if info.get("architecture"):
            extras.append(f"arch={info['architecture']}")
        if not info["has_metadata"]:
            extras.append("no metadata")
        elif info["total_unique_tags"] == 0:
            extras.append("no tag frequency")
        suffix = " (" + ", ".join(extras) + ")" if extras else ""
        print(f"- {rel}{suffix}")
        if info["top_tags"]:
            preview = ", ".join(f"{x['tag']}({x['count']})" for x in info["top_tags"][:5])
            print(f"    top tags: {preview}")
        _print_queries(info)


def _print_queries(info: dict[str, Any]) -> None:
    qh = info.get("query_hits") or {}
    for q, hits in qh.items():
        if not hits:
            print(f"    query {q!r}: no match")
            continue
        joined = ", ".join(f"{h['tag']}({h['count']})" for h in hits[:5])
        print(f"    query {q!r}: {joined}")


def _default_roots() -> list[str]:
    try:
        import folder_paths  # type: ignore
        return list(folder_paths.get_folder_paths("loras") or [])
    except Exception:
        # ComfyUI not on sys.path; fall back to the common install layout.
        guesses = [
            os.path.expanduser("~/ComfyUI/models/loras"),
            r"C:\Ai\ComfyUI\models\loras",
        ]
        return [g for g in guesses if os.path.isdir(g)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "roots", nargs="*",
        help="Directories to scan. Defaults to ComfyUI's loras folder(s).",
    )
    parser.add_argument("--top", type=int, default=10, help="Top-N tags per LoRA (default 10).")
    parser.add_argument(
        "--query", action="append", default=[],
        help="Optional substring to look up in each LoRA's tag frequency. May repeat.",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of human-readable text.")
    args = parser.parse_args(argv)

    roots = args.roots or _default_roots()
    if not roots:
        print("No LoRA roots found. Pass a directory as argument.", file=sys.stderr)
        return 2

    rows: list[dict[str, Any]] = []
    for path in _iter_lora_files(roots):
        try:
            meta = _read_metadata(path)
        except Exception as e:
            rows.append({
                "path": path,
                "rel": _relpath(path, roots),
                "info": {
                    "title": None, "architecture": None, "resolution": None,
                    "date": None, "network_dim": None, "network_alpha": None,
                    "num_train_images": None, "num_epochs": None, "steps": None,
                    "dataset_dirs": [], "triggers": [], "top_tags": [],
                    "total_unique_tags": 0, "has_metadata": False,
                    "query_hits": {q: [] for q in args.query},
                    "error": str(e),
                },
            })
            continue
        info = _parse_lora(meta, top=args.top, queries=args.query)
        rows.append({
            "path": path,
            "rel": _relpath(path, roots),
            "info": info,
        })

    if args.json:
        json.dump(rows, sys.stdout, indent=2, ensure_ascii=False)
        sys.stdout.write("\n")
    else:
        _print_human(rows, args.query)
    return 0


def _relpath(p: str, roots: list[str]) -> str:
    best = p
    for r in roots:
        try:
            cand = os.path.relpath(p, r)
        except ValueError:
            continue
        if not cand.startswith(".."):
            if len(cand) < len(best):
                best = cand
    return best


if __name__ == "__main__":
    raise SystemExit(main())
