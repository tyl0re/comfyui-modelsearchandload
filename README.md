# ComfyUI Model Search and Load

A ComfyUI extension that scans your loaded workflow for missing model
files, finds them on HuggingFace and CivitAI, and downloads them straight
into the correct folder. Saves disk space by hardlinking copies you
already have.

> [Deutsche Version: README.de.md](README.de.md)

---

> ## ⚠️ Disclaimer
>
> **This project was vibe-coded.** It was built iteratively in a
> conversation with an AI coding assistant - shipped because it works
> for the author, not because every code path was reviewed line by line.
>
> **Use at your own risk.** The author provides this software AS IS,
> without warranty of any kind. The author accepts **no liability** for
> anything that happens as a result of running this code, including but
> not limited to: deleted or corrupted files, downloads to the wrong
> location, exhausted disk space, leaked API tokens, broken ComfyUI
> installations, unexpected bandwidth or storage costs, or any other
> direct or indirect damage.
>
> The plugin moves files, deletes the `.part` of incomplete downloads,
> creates hardlinks/symlinks across your `models/` tree, and stores API
> tokens in a local `config.json`. **Back up anything you cannot afford
> to lose before you run it for the first time.** Try it on a test
> ComfyUI install first if you have one.
>
> See the [LICENSE](LICENSE) (MIT) for the formal terms.

---

## Features

- **One-click scan** of the loaded workflow detects every referenced
  model file - across all custom-node packs - and tells you which ones
  are missing.
- **Smart folder routing** uses ComfyUI's own `folder_paths`, your
  `extra_model_paths.yaml`, custom-node-registered folders (e.g. Kijai's
  `detection/` for ViTPose / YOLO), and filename heuristics to pick the
  right destination. Subfolder paths from the workflow
  (`Wan2_2/lightx2v/...`) are preserved.
- **Multi-strategy search**:
  1. Curated database of well-known models (FLUX, SDXL, ControlNet v1.1, ...)
  2. HuggingFace repo-name search
  3. HuggingFace **full-text README search** (catches files referenced
     in READMEs even when the repo name doesn't match the filename)
  4. Fallback list of well-known umbrella repos (`Kijai/WanVideo_comfy`,
     `Comfy-Org/frame_interpolation`, `lightx2v/Wan2.2-Distill-Loras`, ...)
  5. CivitAI search
- **Background downloader** with progress bar, smooth animation,
  download speed and ETA, resume support, cancel, and HTTP-error
  detection (catches the "server returned an HTML login page instead of
  the model" trap).
- **Disk-space saver**: when a model with the same name + size already
  lives anywhere in your `models/` tree, the plugin can hardlink (or
  symlink) it instead of downloading again. Optional, opt-in.
- **Move existing**: shifts files that landed in the wrong folder into
  the right one, no re-download needed.
- **Duplicate protection**: clicking download a second time on a model
  that's already in flight produces a clear toast notification instead
  of starting a parallel transfer.
- **API tokens** for HuggingFace gated repos (FLUX.1-dev) and CivitAI
  models. Tokens are stored in a local `config.json` and **never** sent
  back to the browser - only a masked preview (`hf_xx••••wxyz`) is
  shown.

---

## Installation

Clone (or copy) the repository into your ComfyUI custom nodes folder:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/<your-username>/comfyui-modelsearchandload.git
```

Restart ComfyUI. A new **Models** tab appears in the sidebar
(download icon). On older ComfyUI builds without the sidebar API, a
floating "Models" button shows up in the bottom-right corner.

No additional Python packages are required - the plugin uses only the
standard library (`urllib`, `threading`, `json`, ...) and ComfyUI's
existing `aiohttp` server.

### Cross-platform

Tested on Windows and Linux. All paths use `os.path.join`, all string
comparisons use `os.path.normcase`, and link operations fall back
gracefully when the OS doesn't support them (e.g. Windows symlinks
without Developer Mode → falls back to hardlink → falls back to copy).

---

## Usage

1. **Load a workflow** in ComfyUI.
2. Open the **Models** sidebar tab.
3. Click **Scan workflow** - missing models appear in a list with their
   target folder path.
4. Either:
   - Click **Download all** to auto-resolve every missing file at once,
     or
   - Click **Find sources** on a single entry, pick a candidate, and
     hit **Download** for fine-grained control.
5. Watch progress in the **Downloads** panel below. Each row shows the
   live status, percentage, speed, and ETA. Finished files animate
   away from the missing list.

Other useful actions:

- **Move existing** scans your `models/` tree for files that match a
  missing entry by name, then moves them to the location ComfyUI
  expects (correct folder + subfolder).
- **Settings** opens a modal for HuggingFace / CivitAI API tokens and
  the disk-space (linking) options.

---

## Disk space: link instead of copy

When the same model is referenced from two different paths
(e.g. `Wan2_2/foo.safetensors` and `foo.safetensors`), or when the
plugin needs to satisfy a download for a file that's already on disk
elsewhere, it can create a filesystem link instead of downloading the
data again.

Open **Settings → Disk space**, tick *Reuse existing files via filesystem
links*, and pick a mode:

| Mode | Behavior |
|---|---|
| **Auto** *(recommended)* | Try hardlink first (instant, no extra space). If that fails (cross-filesystem, no permission), try symlink. If symlinks also fail (Windows without Developer Mode), download normally. |
| **Hardlink only** | Use only hardlinks. Fails if the file is on a different filesystem. |
| **Symlink only** | Use only symlinks. May fail on Windows without Developer Mode or admin rights. |

The plugin matches by **filename + file size**. SHA256 verification is
not done because reading multi-GB models would be too slow; the size
check is good enough in practice for distinct model versions.

When linking succeeds, the job status reads `✓ Hardlinked` or
`✓ Symlinked` and no bytes are downloaded.

---

## API tokens

Some models require authentication:

- **HuggingFace token** for gated repos like `black-forest-labs/FLUX.1-dev`.
  Get one at https://huggingface.co/settings/tokens (a read-only token
  is enough). You also need to accept the model's license on the
  HuggingFace web UI before the download will work.
- **CivitAI API key** for many CivitAI downloads.
  Get one at https://civitai.com/user/account.

Open **Settings**, paste the token, click **Save**. After saving, the
input clears and the status line shows `✓ Stored: hf_xx••••wxyz`. The
real value never leaves the server. Click **Clear** to remove a stored
token.

---

## HTTP API

The plugin registers these endpoints on the running ComfyUI server, so
external scripts can drive it too:

| Method | Path | Body |
|---|---|---|
| `POST` | `/model_downloader/scan` | `{"workflow": <api-format-prompt>}` |
| `POST` | `/model_downloader/search` | `{"filename": "...", "folder": "..."}` |
| `POST` | `/model_downloader/download` | `{"url": "...", "folder": "...", "filename": "...", "subfolder": "...", "size": <bytes>}` |
| `POST` | `/model_downloader/download_all` | `{"items": [{"name": "...", "folder": "...", "subfolder": "..."}, ...]}` |
| `GET`  | `/model_downloader/jobs` | - |
| `POST` | `/model_downloader/cancel` | `{"id": "<job-id>"}` |
| `POST` | `/model_downloader/clear` | `{}` |
| `POST` | `/model_downloader/relocate` | `{"items": [...]}` (same shape as `download_all`) |
| `GET`  | `/model_downloader/config` | - (tokens are masked) |
| `POST` | `/model_downloader/config` | `{"huggingface_token": "...", "civitai_token": "...", "enable_linking": true, "linking_mode": "auto"}` |

The download endpoint returns `{"job": {...}, "duplicate": true}` when
a job for the same filename + destination is already running, so
clients can avoid spawning parallel transfers.

---

## Adding your own known models

Edit `known_models.json` to teach the plugin about a custom model:

```json
{
  "my_favourite_lora.safetensors": {
    "folder": "loras",
    "url": "https://huggingface.co/user/repo/resolve/main/file.safetensors",
    "source": "huggingface",
    "size": 134217728
  }
}
```

Folder keys follow the ComfyUI convention: `checkpoints`, `loras`,
`vae`, `controlnet`, `clip` (= `text_encoders`), `clip_vision`,
`diffusion_models`, `upscale_models`, `embeddings`, `style_models`,
`ipadapter`, `gligen`, `hypernetworks`, `frame_interpolation`,
`detection`, ...

Pull requests with additions to the curated database are welcome.

---

## Known limitations

- **HuggingFace search blind spots**: very obscure filenames that
  appear neither in any repo name nor in any README cannot be found
  automatically. For these, paste the URL directly into the candidate's
  search modal or add the model to `known_models.json`.
- **Gated models** require both a token *and* an accepted license on
  the HuggingFace web UI.
- **CivitAI rate limits** - bulk-downloading lots of CivitAI models
  in quick succession can hit their API rate limit.

---

## Contributing

PRs welcome. Please:

- Run `python -c "import ast, pathlib; [ast.parse(p.read_text(encoding='utf-8')) for p in pathlib.Path('.').glob('*.py')]"`
  before pushing to make sure all Python files parse.
- Validate the JS file with `node --check web/model_downloader.js` (the
  file is written as a plain ES module).
- Don't commit `config.json` (it's in `.gitignore`) - that file holds
  per-installation tokens.

---

## License

[MIT](LICENSE)
