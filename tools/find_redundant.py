"""Audit script: which curated DB entries are still needed?

Run from the repo root:  python tools/find_redundant.py

For each entry in known_models.json, disables the curated lookup and
verifies that pure search (HF API + trusted-author probe + fulltext
scan) still returns the canonical repo as Top-1 hit. Entries where
search succeeds are flagged as redundant - they can be removed from
the DB without losing functionality.

Use this whenever:
  - HF improves their search index
  - We add a new pattern to patterns.py
  - We expand the trusted-author whitelist
to find DB entries that are now dead weight.
"""
import sys, os, importlib.util, time, json
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
os.chdir(ROOT)

# Bootstrap modules without ComfyUI present
spec_p = importlib.util.spec_from_file_location("patterns", "patterns.py")
pat_mod = importlib.util.module_from_spec(spec_p)
sys.modules["patterns"] = pat_mod
spec_p.loader.exec_module(pat_mod)

spec_cfg = importlib.util.spec_from_file_location("config", "config.py")
cfg_mod = importlib.util.module_from_spec(spec_cfg)
sys.modules["config"] = cfg_mod
spec_cfg.loader.exec_module(cfg_mod)

src = open("sources.py", encoding="utf-8").read()
src = src.replace("from .config import", "from config import")
src = src.replace("from .patterns import", "from patterns import")
ns = {"__name__": "sources_test"}
exec(compile(src, "sources.py", "exec"), ns)

# Disable DB + patterns to simulate "what if this entry didn't exist?"
ns["lookup_known"] = lambda fn: None
ns["lookup_known_or_pattern"] = lambda fn, fh=None: None

db = json.load(open("known_models.json", encoding="utf-8"))


def _expected_substrings(entry: dict) -> list[str]:
    """Extract owner/repo from the entry's URL so we can verify the
    search returns a hit pointing at the canonical source."""
    url = entry.get("url", "")
    if "huggingface.co" in url:
        try:
            tail = url.split("huggingface.co/", 1)[1]
            return [tail.split("/resolve/", 1)[0].lower()]
        except Exception:
            return []
    if "civitai.com" in url:
        return ["civitai.com"]
    return []


redundant: list[str] = []
needed: list[tuple[str, str, str]] = []

for fn, entry in db.items():
    if fn.startswith("_") or not isinstance(entry, dict):
        continue
    expected = _expected_substrings(entry)
    test_keys = [fn] + (entry.get("aliases") or [])
    all_pass = True
    fail_detail = ("", "")

    for key in test_keys:
        ns["_search_cache"].clear()
        if "_author_repo_cache" in ns:
            ns["_author_repo_cache"].clear()
        results = ns["find_candidates"](key)
        if not results:
            all_pass = False
            fail_detail = (key, "(no result)")
            break
        url = (results[0].get("url") or "").lower()
        if not any(s.lower() in url for s in expected):
            all_pass = False
            fail_detail = (key, results[0].get("url", "")[:120])
            break

    if all_pass:
        print(f"  REDUNDANT: {fn}  (search finds it for primary key + {len(test_keys)-1} aliases)")
        redundant.append(fn)
    else:
        print(f"  NEEDED   : {fn}")
        print(f"             -> {fail_detail[0]!r} returned {fail_detail[1]}")
        needed.append((fn, fail_detail[0], fail_detail[1]))

print()
print(f"=== Summary: {len(redundant)} redundant, {len(needed)} needed ===")

if redundant:
    print("\nRemove these from known_models.json:")
    for fn in redundant:
        print(f"  - {fn}")
    sys.exit(1)
else:
    print("All DB entries are still required - nothing to remove.")
    sys.exit(0)
