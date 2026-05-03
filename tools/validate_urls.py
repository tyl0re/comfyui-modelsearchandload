"""Validate every URL in patterns.py + known_models.json with a HEAD probe.

Run from the repo root:  python tools/validate_urls.py

Exits non-zero on any failure so you can wire it into a CI job. A 401
counts as 'reachable but gated' and is not treated as a failure (FLUX,
some Llama models). Anything else (404, 5xx, network error) fails the
build because it means the user would silently get a broken download.

This catches mistakes like 'RealESRGAN_x4plus_anime_6B.pth at
ai-forever/Real-ESRGAN' which doesn't exist there but the wrong
gemasai/Real-ESRGAN does. Without this check, the download manager
just writes a 404 HTML page to disk and reports success.
"""
import sys, importlib.util, urllib.request, urllib.parse, json, os
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Allow running from either the repo root or the tools/ folder
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
os.chdir(ROOT)

# Load patterns module
spec = importlib.util.spec_from_file_location("patterns", "patterns.py")
mod = importlib.util.module_from_spec(spec)
sys.modules["patterns"] = mod
spec.loader.exec_module(mod)

def probe(url, timeout=15):
    """Returns (ok, status, content_length, error)."""
    try:
        req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "Mozilla/5.0"})
        resp = urllib.request.urlopen(req, timeout=timeout)
        return True, resp.status, resp.headers.get("Content-Length"), None
    except urllib.error.HTTPError as e:
        # 401 = gated, that's not a broken URL just an auth requirement
        if e.code == 401:
            return True, 401, None, "gated (token required)"
        return False, e.code, None, str(e)
    except Exception as e:
        return False, None, None, str(e)

failures = []
total = 0

# === Test all pattern URLs ===
print("=== Pattern URLs ===")
# Build a sample filename for each rule by trying its regex against a few
# canonical filenames; if no canonical found, just emit the URL as-is.
samples = {
    # canonical filenames per rule
    "lcm-lora-sdv1-5":      "lcm_lora_weights_sd15.safetensors",
    "lcm-lora-sdxl":        "lcm_lora_weights_sdxl.safetensors",
    "lcm-lora-ssd-1b":      "lcm_lora_ssd_1b.safetensors",
    "image_encoder.*sd1":   "clip_vision_sd15.safetensors",
    "sdxl_models/image":    "clip_vision_sdxl.safetensors",
    "sigclip":              "clip_vision_h.safetensors",
    "clip_vision_g":        "clip_vision_g.safetensors",
    "depth_anything":       "depth_anything_vitl14.pth",
    "Depth-Anything-V2":    "depth_anything_v2_vitl.pth",
    "ControlNet-v1-1":      "control_v11p_sd15_canny.pth",
    "AnimateLCM/resolve":   "animatelcm_sd15_t2v.ckpt",
    "AnimateLCM-I2V":       "animatelcm_i2v.ckpt",
    "animatediff/resolve.*mm_sd_v": "mm_sd_v15_v2.ckpt",
    "animatediff/resolve.*mm_sdxl": "mm_sdxl_v10_beta.ckpt",
    "animatediff/resolve.*v3_sd15": "v3_sd15_mm.ckpt",
    "animatediff/resolve.*v2_lora": "v2_lora_zoomin.ckpt",
    "Real-ESRGAN.*RealESRGAN_x4plus_anime": "realesrgan_x4plus_anime_6b.pth",
    "Real-ESRGAN.*RealESRGAN_x2": "realesrgan_x2plus.pth",
    "Real-ESRGAN.*RealESRGAN_x4": "realesrgan_x4plus.pth",
    "Real-ESRGAN.*RealESRGAN_x8": "realesrgan_x8plus.pth",
}

# Walk every rule, render its URL with a sample filename
for rule in mod._RULES:
    pat = rule["pattern"]
    # Find a sample filename that matches the rule's pattern
    sample = None
    for s in samples.values():
        if pat.match(mod.normalise_filename(s)):
            sample = s
            break
    # Fallback: emit the raw URL with no backref expansion if no sample
    if sample is None:
        url = rule["url"].replace("\\g<1>", "X").replace("{SIZE}", "Large")
    else:
        match = pat.match(mod.normalise_filename(sample))
        url = mod._render(rule["url"], match, rule)
    title = rule.get("title", "?")
    if sample:
        title = mod._render(title, match, rule)

    total += 1
    ok, status, cl, err = probe(url)
    cl_s = ""
    if cl:
        cl_mb = int(cl) / 1024 / 1024
        cl_s = f" ({cl_mb:.0f} MB)"
    if ok:
        print(f"  ✓ [{status}{cl_s}]  {title[:40]:40s}  {url[:100]}")
    else:
        print(f"  ✗ [{status}]   {title[:40]:40s}  {url[:100]}")
        print(f"      {err}")
        failures.append((title, url, err))

# === Test known_models.json URLs ===
print("\n=== known_models.json URLs ===")
with open("known_models.json", encoding="utf-8") as f:
    db = json.load(f)
for fn, entry in db.items():
    if fn.startswith("_"):
        continue
    url = entry.get("url")
    if not url:
        continue
    total += 1
    # CivitAI URLs need a token to actually download but the URL itself
    # is reachable - we use a HEAD that follows redirects.
    ok, status, cl, err = probe(url)
    cl_s = ""
    if cl:
        cl_mb = int(cl) / 1024 / 1024
        cl_s = f" ({cl_mb:.0f} MB)"
    if ok:
        print(f"  ✓ [{status}{cl_s}]  {fn[:50]:50s}  {url[:80]}")
    else:
        print(f"  ✗ [{status}]   {fn[:50]:50s}  {url[:80]}")
        print(f"      {err}")
        failures.append((fn, url, err))

print()
print(f"=== Summary: {total - len(failures)}/{total} URLs reachable ===")
if failures:
    print("\nFailures:")
    for f in failures:
        print(f"  {f[0]}")
        print(f"    URL: {f[1]}")
        print(f"    err: {f[2]}")
    sys.exit(1)
sys.exit(0)
