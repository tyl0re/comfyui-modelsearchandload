// ComfyUI-ModelDownloader frontend (sidebar tab)
// Registers a sidebar panel and a settings section.

import { app } from "../../scripts/app.js";

const API = {
  scan:         "/model_downloader/scan",
  search:       "/model_downloader/search",
  web_search:   "/model_downloader/web_search",
  download:     "/model_downloader/download",
  relocate:     "/model_downloader/relocate",
  dedupe_scan:  "/model_downloader/dedupe_scan",
  dedupe_apply: "/model_downloader/dedupe_apply",
  check_path:   "/model_downloader/check_path",
  clear_dedupe_cache: "/model_downloader/clear_dedupe_cache",
  dedupe_cache_stats: "/model_downloader/dedupe_cache_stats",
  jobs:         "/model_downloader/jobs",
  cancel:       "/model_downloader/cancel",
  clear:        "/model_downloader/clear",
  retry:        "/model_downloader/retry",
  lora_list:    "/model_downloader/lora_list",
  lora_meta:    "/model_downloader/lora_meta",
  config:       "/model_downloader/config",
};

async function jsonFetch(url, opts = {}) {
  const res = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...opts,
  });
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

function fmtBytes(n) {
  if (!n) return "?";
  const u = ["B", "KB", "MB", "GB", "TB"];
  let i = 0;
  while (n >= 1024 && i < u.length - 1) { n /= 1024; i++; }
  return `${n.toFixed(1)} ${u[i]}`;
}

function fmtCount(n) {
  n = Number(n || 0);
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(n >= 10_000_000 ? 0 : 1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(n >= 10_000 ? 0 : 1)}K`;
  return String(n);
}

function el(tag, props = {}, children = []) {
  const node = document.createElement(tag);
  Object.assign(node, props);
  if (props.style) Object.assign(node.style, props.style);
  for (const c of [].concat(children)) {
    if (c == null) continue;
    node.appendChild(typeof c === "string" ? document.createTextNode(c) : c);
  }
  return node;
}

// ---------- Sidebar UI ----------

function buildPanel(container) {
  container.innerHTML = "";
  Object.assign(container.style, {
    padding: "10px",
    display: "flex",
    flexDirection: "column",
    gap: "10px",
    height: "100%",
    overflow: "auto",
    color: "var(--input-text, #ddd)",
    fontFamily: "sans-serif",
    fontSize: "13px",
  });

  // Two-row header: row 1 = plugin name + Settings button (right-aligned),
  // row 2 = action buttons (Scan / Move existing / Free Space Via Link).
  const header = el("div", {
    style: { display: "flex", flexDirection: "column", gap: "6px" },
  });
  const headerRow1 = el("div", {
    style: { display: "flex", gap: "6px", alignItems: "center" },
  });
  const headerRow2 = el("div", {
    style: { display: "flex", gap: "6px", alignItems: "center", flexWrap: "wrap" },
  });

  const title = el("div", {
    textContent: "Model Downloader",
    style: { fontWeight: "bold", fontSize: "14px", flex: "1" },
  });

  const btnScan = el("button", { textContent: "Scan workflow" });
  const btnRelocate = el("button", { textContent: "Move existing" });
  const btnDedupe = el("button", { textContent: "Free Space Via Link" });
  const btnLoraTags = el("button", { textContent: "Read LoRA tags" });
  const btnSettings = el("button", { textContent: "Settings" });

  btnDedupe.title = "Find duplicate model files anywhere in your models tree and replace duplicates with hardlinks pointing at one master copy. Frees disk space without breaking any workflow. Visibility depends on whether linking is enabled in Settings.";
  btnLoraTags.title = "Pick a LoRA, read its trigger words + training tags from the safetensors metadata, and write the selection into a 'LoRA Tag Selector' node in the canvas.";

  for (const b of [btnScan, btnRelocate, btnDedupe, btnLoraTags, btnSettings]) {
    Object.assign(b.style, {
      padding: "5px 10px",
      cursor: "pointer",
      background: "var(--comfy-input-bg, #333)",
      color: "var(--input-text, #ddd)",
      border: "1px solid var(--border-color, #555)",
      borderRadius: "4px",
    });
  }
  btnRelocate.disabled = true;
  btnRelocate.style.opacity = "0.5";
  btnRelocate.title = "Move files that are already on disk into the folder ComfyUI actually expects.";

  // Dedupe button is hidden by default; enabled by the config-fetch
  // result below if `enable_linking` is true.
  btnDedupe.style.display = "none";

  headerRow1.append(title, btnSettings);
  headerRow2.append(btnScan, btnRelocate, btnDedupe, btnLoraTags);
  header.append(headerRow1, headerRow2);

  // Fetch config once on panel open, toggle the dedupe button visibility.
  // Also re-check whenever the settings modal closes (in case the user
  // just enabled or disabled linking).
  async function refreshDedupeButtonVisibility() {
    try {
      const cfg = await jsonFetch(API.config);
      const linkingOn = !!cfg.enable_linking;
      const method = (cfg.dedupe_method || "hash");
      btnDedupe.style.display = (linkingOn && method !== "disabled") ? "" : "none";
    } catch (e) {
      // If config fetch fails, default to hidden.
      btnDedupe.style.display = "none";
    }
  }
  refreshDedupeButtonVisibility();
  // Expose so the settings modal can call it after Save.
  container._refreshDedupeButtonVisibility = refreshDedupeButtonVisibility;

  const status = el("div", {
    style: { fontSize: "12px", opacity: "0.8", minHeight: "16px" },
  });

  const missingSection = el("div", { style: { display: "flex", flexDirection: "column", gap: "6px" } });
  const jobsSection = el("div", { style: { display: "flex", flexDirection: "column", gap: "6px" } });

  const missingHeader = el("div", {
    textContent: "Missing models",
    style: { fontWeight: "bold", marginTop: "4px" },
  });
  const jobsHeader = el("div", {
    textContent: "Downloads",
    style: { fontWeight: "bold", marginTop: "10px", display: "flex", alignItems: "center", gap: "6px" },
  });
  const btnClear = el("button", {
    textContent: "Clear finished",
    style: { fontSize: "11px", padding: "2px 6px", marginLeft: "auto", cursor: "pointer" },
  });
  jobsHeader.append(btnClear);

  container.append(header, status, missingHeader, missingSection, jobsHeader, jobsSection);

  // When the settings modal closes, refresh the dedupe button visibility.
  // We hook into Setting modal's close by checking after a small delay
  // every time it's opened. Simpler: poll for the modal's open state in
  // settings handler below.

  // State shared by scan + per-candidate downloads
  let lastMissing = [];

  // Settings modal (lazy)
  let settingsModal = null;

  btnSettings.onclick = () => {
    if (!settingsModal) {
      settingsModal = buildSettingsModal();
    }
    // Pass our refresh callback so the modal can re-trigger the dedupe
    // button visibility check when the user clicks Save / Close.
    settingsModal.open(refreshDedupeButtonVisibility);
  };

  btnDedupe.onclick = () => {
    runDedupeFlow(status);
  };

  btnLoraTags.onclick = () => {
    runLoraTagFlow(status).catch((e) => {
      status.textContent = "LoRA tag reader failed: " + (e?.message || e);
    });
  };

  function setActionsEnabled(enabled) {
    btnRelocate.disabled = !enabled;
    btnRelocate.style.opacity = enabled ? "1" : "0.5";
    btnRelocate.style.cursor = enabled ? "pointer" : "not-allowed";
  }

  btnScan.onclick = async () => {
    status.textContent = "Scanning workflow...";
    missingSection.innerHTML = "";
    setActionsEnabled(false);
    try {
      // Send the full UI-format workflow (app.graph.serialize()) instead of
      // the API-format prompt (graphToPrompt().output). The API format
      // strips bypassed/muted nodes (mode != 0) and may not expose model
      // filenames as `inputs` on custom nodes — so model references in
      // bypassed nodes or in widget-only slots would be invisible.
      // The backend scanner handles both formats but UI format is strictly
      // more complete.
      const workflow = app.graph.serialize();
      const data = await jsonFetch(API.scan, {
        method: "POST",
        body: JSON.stringify({ workflow }),
      });
      lastMissing = data.missing || [];
      renderMissing(missingSection, lastMissing, status);
      const n = lastMissing.length;
      status.textContent = n === 0
        ? "All referenced models are installed."
        : `${n} missing model(s) found.`;
      setActionsEnabled(n > 0);
    } catch (e) {
      status.textContent = "Scan failed: " + e.message;
    }
  };

  // Map: missing-model-name -> job_id (set when we queue a download for it
  // via the per-candidate "Download" button).
  // The poll loop uses this to keep each missing row's status badge in sync
  // with the actual download job state.
  const missingNameToJobId = new Map();
  // Container is shared with the per-row Download buttons via this object,
  // so the candidate-row code can register a job_id without re-passing refs.
  container._mdLink = { missingNameToJobId, missingSection };

  btnRelocate.onclick = async () => {
    if (!lastMissing.length) return;
    const ok = confirm(
      `Try to move ${lastMissing.length} file(s) into the location ComfyUI expects?\n\n` +
      `This searches your models tree for files matching by name and moves the ` +
      `first match into the correct folder + subfolder. Useful when a download ` +
      `landed in the wrong place. Files that already sit at the right path are ` +
      `left alone; targets that already exist are not overwritten.`,
    );
    if (!ok) return;

    btnRelocate.disabled = true;
    btnRelocate.textContent = "Moving...";
    status.textContent = "Searching for misplaced files...";

    try {
      const items = lastMissing.map(m => ({
        name: m.name,
        folder: m.folder,
        subfolder: m.subfolder || "",
      }));
      const data = await jsonFetch(API.relocate, {
        method: "POST",
        body: JSON.stringify({ items }),
      });
      const results = data.results || [];

      // Annotate each missing row with the outcome
      for (const r of results) {
        const row = missingSection.querySelector(`[data-missing-name="${cssEscape(r.name)}"]`);
        if (!row) continue;
        const badge = _ensureBadge(row);
        if (r.status === "moved") {
          _setBadge(badge, `✓ Moved to correct folder`, "#66bb6a");
          // Auto-remove the row after a short pause
          setTimeout(() => {
            row.style.transition = "opacity 0.4s, max-height 0.4s, padding 0.4s, margin 0.4s";
            row.style.overflow = "hidden";
            row.style.maxHeight = row.offsetHeight + "px";
            requestAnimationFrame(() => {
              row.style.opacity = "0";
              row.style.maxHeight = "0";
              row.style.padding = "0";
              row.style.margin = "0";
              row.style.border = "0";
            });
            setTimeout(() => {
              row.remove();
              const idx = lastMissing.findIndex(m => m.name === r.name);
              if (idx >= 0) lastMissing.splice(idx, 1);
              if (missingSection.children.length === 0) {
                const ph = document.createElement("div");
                ph.textContent = "Nothing missing.";
                Object.assign(ph.style, { opacity: "0.7", fontStyle: "italic", padding: "4px 0" });
                missingSection.appendChild(ph);
              }
            }, 450);
          }, 1000);
        } else if (r.status === "already_correct") {
          _setBadge(badge, "✓ Already at correct path (rescan to refresh)", "#66bb6a");
        } else if (r.status === "not_found") {
          _setBadge(badge, "Not found locally - inspect sources and download manually", "#ffb74d");
        } else if (r.status === "target_exists") {
          _setBadge(badge, "⚠ Target file already exists - skipped", "#ffb74d");
        } else if (r.status === "error") {
          _setBadge(badge, "✗ " + (r.reason || "error"), "#ef9a9a");
        }
      }

      const moved = results.filter(r => r.status === "moved").length;
      const ok = results.filter(r => r.status === "already_correct").length;
      const notFound = results.filter(r => r.status === "not_found").length;
      const parts = [];
      if (moved) parts.push(`✓ ${moved} moved`);
      if (ok) parts.push(`${ok} already correct`);
      if (notFound) parts.push(`${notFound} not found`);
      status.textContent = parts.join(" · ") || "Nothing to do.";
    } catch (e) {
      status.textContent = "Move failed: " + e.message;
    } finally {
      btnRelocate.textContent = "Move existing";
      btnRelocate.disabled = false;
      btnRelocate.style.opacity = "1";
    }
  };

  btnClear.onclick = async () => {
    try {
      await jsonFetch(API.clear, { method: "POST", body: "{}" });
      refreshJobs(jobsSection);
    } catch (e) { /* ignore */ }
  };

  // Poll jobs.
  // We use an adaptive interval: while at least one job is active we poll
  // every 500 ms for snappy progress, otherwise back off to 2 s to keep
  // load down. Between polls, the UI animates the "smoothed" byte counter
  // so the bar moves continuously instead of jumping.
  let pollInProgress = false;
  let pollTimer = null;
  const ACTIVE_INTERVAL = 500;
  const IDLE_INTERVAL = 2000;

  // Set of filenames that have an active job right now. Shared with
  // candidate rows via container._mdLink so the per-candidate "Download"
  // buttons can grey themselves out.
  const activeFilenames = new Set();
  container._mdLink.activeFilenames = activeFilenames;

  async function poll() {
    if (pollInProgress) return;
    pollInProgress = true;
    let nextDelay = IDLE_INTERVAL;
    try {
      const data = await jsonFetch(API.jobs);
      const jobs = data.jobs || [];
      renderJobs(jobsSection, jobs);
      // Sync the badges + auto-remove rows whose download finished.
      syncMissingBadges(missingSection, missingNameToJobId, jobs, lastMissing);
      // Refresh the activeFilenames set.
      activeFilenames.clear();
      for (const j of jobs) {
        if (["queued", "connecting", "linking", "downloading"].includes(j.status)) {
          activeFilenames.add(j.filename);
        }
      }
      // Tell every candidate-row in the panel to re-evaluate its button.
      refreshCandidateButtons(container, activeFilenames);
      const hasActive = activeFilenames.size > 0;
      nextDelay = hasActive ? ACTIVE_INTERVAL : IDLE_INTERVAL;
    } catch (e) {
      /* keep idle interval on errors */
    } finally {
      pollInProgress = false;
      pollTimer = setTimeout(poll, nextDelay);
    }
  }
  poll();
}

// Re-evaluate every "Download" button in candidate rows so the user sees
// at a glance which models are already being fetched. The button itself
// stays clickable - clicking it shows a clear "already running" message
// instead of silently doing nothing.
function refreshCandidateButtons(root, activeFilenames) {
  root.querySelectorAll("[data-md-candidate-filename]").forEach(row => {
    const fn = row.getAttribute("data-md-candidate-filename");
    const btn = row.querySelector("[data-md-download-btn]");
    if (!btn) return;
    const isActive = activeFilenames.has(fn);
    // Use a data-flag so the click handler knows whether to short-circuit
    // with a message vs. actually starting the download.
    btn.dataset.alreadyActive = isActive ? "1" : "";
    if (isActive) {
      // Visually demote the button but keep it clickable so the message
      // can fire when a user tries again.
      btn.style.opacity = "0.6";
      btn.style.cursor = "help";
      if (!btn.dataset.activeLabel) {
        btn.dataset.originalLabel = btn.textContent;
        btn.textContent = "In progress...";
        btn.dataset.activeLabel = "1";
      }
      btn.title = "A download for this file is already running. Click for details.";
    } else {
      if (btn.dataset.activeLabel) {
        btn.textContent = btn.dataset.originalLabel || "Download";
        delete btn.dataset.activeLabel;
        delete btn.dataset.originalLabel;
      }
      btn.style.opacity = "1";
      btn.style.cursor = "pointer";
      btn.title = "";
    }
  });
}

// One-line non-blocking notice that floats above the bottom of the panel.
// Used when the user tries to download a model that's already in progress.
function showToast(panel, msg, color = "#ffb74d", durationMs = 4000) {
  if (!panel) return;
  let toast = panel.querySelector(".md-toast");
  if (!toast) {
    toast = document.createElement("div");
    toast.className = "md-toast";
    Object.assign(toast.style, {
      position: "fixed",
      left: "50%",
      bottom: "20px",
      transform: "translateX(-50%)",
      padding: "8px 14px",
      background: "var(--comfy-menu-bg, #222)",
      border: "1px solid var(--border-color, #555)",
      borderRadius: "6px",
      boxShadow: "0 4px 14px rgba(0,0,0,0.5)",
      fontSize: "13px",
      fontWeight: "bold",
      zIndex: "9999",
      transition: "opacity 0.2s",
      opacity: "0",
      pointerEvents: "none",
      maxWidth: "80vw",
      textAlign: "center",
    });
    document.body.appendChild(toast);
  }
  toast.textContent = msg;
  toast.style.color = color;
  toast.style.borderColor = color;
  // Force a reflow so the next opacity change actually transitions.
  void toast.offsetWidth;
  toast.style.opacity = "1";
  if (toast._hideTimer) clearTimeout(toast._hideTimer);
  toast._hideTimer = setTimeout(() => {
    toast.style.opacity = "0";
  }, durationMs);
}

function _ensureBadge(row) {
  let badge = row.querySelector(".md-bulk-badge");
  if (!badge) {
    badge = document.createElement("div");
    badge.className = "md-bulk-badge";
    Object.assign(badge.style, {
      marginTop: "4px",
      fontSize: "11px",
      fontWeight: "bold",
      display: "flex",
      alignItems: "center",
      gap: "6px",
    });
    row.appendChild(badge);
  }
  return badge;
}

function _setBadge(badge, text, color, withSpinner = false) {
  badge.innerHTML = "";
  if (withSpinner) {
    const sp = document.createElement("span");
    sp.className = "md-spinner";
    badge.appendChild(sp);
  }
  const t = document.createElement("span");
  t.textContent = text;
  badge.appendChild(t);
  badge.style.color = color;
}

// Show a small icon next to the filename indicating that the row was
// satisfied by linking instead of a real download. Hover-tooltip shows
// the original file the link points at.
//
// `mode` one of:
//   "hardlink"        -> 🔗  green
//   "symlink"         -> ⇲  green
//   "already-linked"  -> 🔗  blue (was already at the right path)
//   "linking"         -> 🔗  orange (in progress)
//   "remove" / null   -> removes the icon
//
// Looks up the filename header in two ways:
//   1. an explicit element marked with [data-md-name-el="1"] (preferred)
//   2. fallback: the first direct child <div> of the row
function _setLinkIcon(row, mode, sourcePath) {
  if (!row) return;
  let header = row.querySelector("[data-md-name-el]");
  if (!header) header = row.querySelector(":scope > div");
  if (!header) return;

  let icon = header.querySelector(".md-link-icon");
  if (!mode || mode === "remove") {
    if (icon) icon.remove();
    return;
  }
  if (!icon) {
    icon = document.createElement("span");
    icon.className = "md-link-icon";
    Object.assign(icon.style, {
      display: "inline-block",
      marginLeft: "6px",
      fontSize: "13px",
      verticalAlign: "middle",
      cursor: "help",
      // Reset bold from the parent so the icon reads as a glyph.
      fontWeight: "normal",
    });
    header.appendChild(icon);
  }
  let glyph = "🔗";
  let color = "#66bb6a";
  let label = "linked";
  switch (mode) {
    case "hardlink":
      glyph = "🔗"; color = "#66bb6a"; label = "hardlinked";
      break;
    case "symlink":
      glyph = "⇲"; color = "#66bb6a"; label = "symlinked";
      break;
    case "already-linked":
      glyph = "🔗"; color = "#64b5f6"; label = "already linked at correct path";
      break;
    case "linking":
      glyph = "🔗"; color = "#ffb74d"; label = "linking...";
      break;
  }
  icon.textContent = glyph;
  icon.style.color = color;
  icon.title = sourcePath
    ? `${label}\nfrom: ${sourcePath}`
    : label;
}

// Set of (name, jobId) pairs that have already been scheduled for
// auto-removal. Prevents the same row from queueing multiple removal
// timers if 'done' is observed across several poll ticks.
const _missingRowRemovalScheduled = new Set();

function _scheduleMissingRowRemoval(row, parent, name, lastMissing, badgeLabel, badgeColor) {
  const key = name;
  if (_missingRowRemovalScheduled.has(key)) return;
  _missingRowRemovalScheduled.add(key);

  const badge = _ensureBadge(row);
  _setBadge(badge, badgeLabel, badgeColor);

  setTimeout(() => {
    row.style.transition = "opacity 0.4s, max-height 0.4s, padding 0.4s, margin 0.4s";
    row.style.overflow = "hidden";
    row.style.maxHeight = row.offsetHeight + "px";
    requestAnimationFrame(() => {
      row.style.opacity = "0";
      row.style.maxHeight = "0";
      row.style.padding = "0";
      row.style.margin = "0";
      row.style.border = "0";
    });
    setTimeout(() => {
      row.remove();
      if (Array.isArray(lastMissing)) {
        const idx = lastMissing.findIndex(m => m.name === name);
        if (idx >= 0) lastMissing.splice(idx, 1);
      }
      _missingRowRemovalScheduled.delete(key);
      if (parent.children.length === 0) {
        const ph = document.createElement("div");
        ph.textContent = "All downloads complete.";
        Object.assign(ph.style, {
          opacity: "0.7", fontStyle: "italic", padding: "4px 0",
        });
        parent.appendChild(ph);
      }
    }, 450);
  }, 1200);
}

// Called every poll-tick. Looks at the current jobs list and updates the
// status badge of each missing-row that has an associated download job.
// Rows whose download completed are removed entirely after a short delay.
function syncMissingBadges(parent, nameToJobId, jobs, lastMissing) {
  if (nameToJobId.size === 0) return;
  const jobById = new Map(jobs.map(j => [j.id, j]));

  for (const [name, jobId] of [...nameToJobId.entries()]) {
    const row = parent.querySelector(`[data-missing-name="${cssEscape(name)}"]`);
    if (!row) {
      // Row already gone (cleared by user)
      nameToJobId.delete(name);
      continue;
    }
    const job = jobById.get(jobId);
    if (!job) {
      // Job vanished from the manager - this happens when:
      //   a) the user clicked "Clear finished" while a download was in
      //      progress, or
      //   b) the job finished + was cleaned up before our poll caught it
      //      in the 'done' state (very fast / linked downloads).
      // Either way: if the file is now ACTUALLY on disk, treat as done
      // and remove the row. We approximate "on disk" by trusting that
      // the registered job ran to completion. The file-system index is
      // refreshed only on the next /scan, so we don't try to verify it
      // here client-side; we just remove the row optimistically.
      console.log(`[ModelDownloader] job ${jobId} for '${name}' vanished - assuming done, removing row`);
      _scheduleMissingRowRemoval(
        row, parent, name, lastMissing,
        "✓ Done - removing from list...", "#66bb6a",
      );
      nameToJobId.delete(name);
      continue;
    }
    const badge = _ensureBadge(row);
    switch (job.status) {
      case "queued":
      case "connecting":
        _setBadge(badge, "Connecting...", "#ffb74d", true);
        break;
      case "linking":
        _setBadge(badge, "Linking existing copy...", "#ffb74d", true);
        _setLinkIcon(row, "linking", null);
        break;
      case "linked": {
        const m = job.link_method || "linked";
        const label = m === "hardlink" ? "Hardlinked"
                    : m === "symlink"  ? "Symlinked"
                    : m === "already-linked" ? "Already linked"
                    : "Linked";
        _setLinkIcon(row, m, job.link_source);
        _scheduleMissingRowRemoval(
          row, parent, name, lastMissing,
          `✓ ${label} (no download) - removing from list...`, "#66bb6a",
        );
        nameToJobId.delete(name);
        break;
      }
      case "downloading": {
        const pct = job.bytes_total
          ? Math.min(100, (job.bytes_done / job.bytes_total) * 100)
          : 0;
        const speed = job.speed_bps ? fmtSpeed(job.speed_bps) : "";
        const eta = job.eta_seconds ? "ETA " + fmtEta(job.eta_seconds) : "";
        const txt = job.bytes_total
          ? `Downloading ${pct.toFixed(0)}%  ${speed}  ${eta}`.replace(/\s+/g, " ").trim()
          : `Downloading  ${speed}`.trim();
        _setBadge(badge, txt, "#64b5f6", true);
        break;
      }
      case "done":
        _scheduleMissingRowRemoval(
          row, parent, name, lastMissing,
          "✓ Downloaded - removing from list...", "#66bb6a",
        );
        nameToJobId.delete(name);
        break;
      case "error":
        _setBadge(badge, "✗ " + (job.error || "error"), "#ef9a9a");
        nameToJobId.delete(name);
        break;
      case "cancelled":
        _setBadge(badge, "Cancelled", "#bdbdbd");
        nameToJobId.delete(name);
        break;
    }
  }
}

// Escape a string so it can be safely used as the value-side of an attribute
// selector. Falls back to a manual escape if CSS.escape is unavailable.
function cssEscape(s) {
  if (typeof CSS !== "undefined" && CSS.escape) return CSS.escape(s);
  return String(s).replace(/[^\w-]/g, c => "\\" + c.charCodeAt(0).toString(16) + " ");
}

function renderMissing(parent, missing, status) {
  parent.innerHTML = "";
  if (missing.length === 0) {
    parent.append(el("div", {
      textContent: "Nothing missing.",
      style: { opacity: "0.7", fontStyle: "italic" },
    }));
    return;
  }
  for (const m of missing) {
    const row = el("div", {
      style: {
        border: "1px solid var(--border-color, #555)",
        borderRadius: "4px",
        padding: "6px 8px",
        background: "var(--comfy-menu-bg, #222)",
      },
    });
    row.setAttribute("data-missing-name", m.name);
    const name = el("div", {
      textContent: m.name,
      style: { fontWeight: "bold", wordBreak: "break-all" },
    });
    name.setAttribute("data-md-name-el", "1");
    const meta = el("div", {
      textContent: `${m.folder}${m.subfolder ? "/" + m.subfolder : ""}  ·  used by ${m.node_type || "?"}`,
      style: { fontSize: "11px", opacity: "0.8" },
    });
    // Full destination path so the user can spot wrong-folder routing
    // before downloading. Uses the workflow's filename as the final
    // component (the downloader preserves it even when the upstream
    // file has a different name like 'pytorch_lora_weights.safetensors').
    const targetEl = el("div", {
      textContent: m.target_path
        ? "→ " + m.target_path
        : "→ (no target path resolved - see folder above)",
      style: {
        fontSize: "10px",
        opacity: "0.7",
        marginTop: "2px",
        fontFamily: "ui-monospace, Consolas, Menlo, monospace",
        wordBreak: "break-all",
        color: m.target_path ? "var(--input-text, #aaa)" : "#ef9a9a",
      },
      title: "Where the file will be saved on disk. If this looks wrong, the download will land in the wrong place — please report it.",
    });
    // If the curated DB or pattern engine already resolved a source
    // URL during scan, show it inline so the user can see at a glance
    // WHERE the file would be downloaded from before clicking
    // 'Find sources'. Suppressed when the source is unknown - in that
    // case 'Find sources' will fetch candidates on demand.
    let sourceEl = null;
    if (m.source_url) {
      sourceEl = el("div", {
        style: {
          fontSize: "10px",
          opacity: "0.85",
          marginTop: "2px",
          fontFamily: "ui-monospace, Consolas, Menlo, monospace",
          wordBreak: "break-all",
        },
      });
      const kind = m.source_kind || "?";
      const tag = el("span", {
        textContent: kind === "known" ? "DB"
                   : kind === "pattern" ? "PATTERN"
                   : kind.toUpperCase(),
        style: {
          display: "inline-block",
          marginRight: "5px",
          padding: "0 4px",
          borderRadius: "2px",
          background: "#4caf50",
          color: "#000",
          fontSize: "9px",
          fontWeight: "bold",
        },
      });
      const src = el("a", {
        textContent: "↓ " + m.source_url,
        href: m.source_url,
        target: "_blank",
        rel: "noopener noreferrer",
        style: {
          color: "var(--input-text, #aaa)",
          textDecoration: "none",
        },
        title: "Click to verify this URL in a new browser tab before downloading.",
      });
      sourceEl.append(tag, src);
    }
    const btnFind = el("button", { textContent: "Find sources" });
    const btnWeb = el("button", { textContent: "Search web" });
    const actions = el("div", {
      style: {
        marginTop: "4px",
        display: "flex",
        gap: "6px",
        flexWrap: "wrap",
      },
    });
    for (const b of [btnFind, btnWeb]) {
      Object.assign(b.style, {
        padding: "3px 8px",
        cursor: "pointer",
        fontSize: "12px",
      });
    }
    Object.assign(btnWeb.style, {
      background: "var(--comfy-input-bg, #333)",
      color: "var(--input-text, #ddd)",
      border: "1px solid var(--border-color, #555)",
      borderRadius: "4px",
    });
    btnWeb.title = "Search the web for exact filename mentions, verify HuggingFace hits, and show them as WEB candidates.";
    actions.append(btnFind, btnWeb);

    btnWeb.onclick = async () => {
      btnWeb.disabled = true;
      btnWeb.textContent = "Checking web...";
      try {
        const data = await jsonFetch(API.web_search, {
          method: "POST",
          body: JSON.stringify({ filename: m.name, folder: m.folder }),
        });
        renderCandidates(candidatesBox, data.candidates || [], m.folder, m.name, status, m.subfolder || "");
        candidatesBox.style.display = "block";
        const n = (data.candidates || []).length;
        status.textContent = n
          ? `Search found ${n} verified HuggingFace candidate(s).`
          : `Search found no verified HuggingFace source for ${m.name}.`;
      } catch (e) {
        status.textContent = "Web search failed: " + e.message;
      } finally {
        btnWeb.disabled = false;
        btnWeb.textContent = "Search web";
      }
    };

    Object.assign(btnFind.style, {
      marginTop: "4px",
    });
    const candidatesBox = el("div", { style: { marginTop: "6px", display: "none" } });

    btnFind.onclick = async () => {
      btnFind.disabled = true;
      btnFind.textContent = "Searching...";
      try {
        const data = await jsonFetch(API.search, {
          method: "POST",
          body: JSON.stringify({ filename: m.name, folder: m.folder, source_hint: m.raw || "" }),
        });
        renderCandidates(candidatesBox, data.candidates || [], m.folder, m.name, status, m.subfolder || "");
        candidatesBox.style.display = "block";
      } catch (e) {
        status.textContent = "Search failed: " + e.message;
      } finally {
        btnFind.disabled = false;
        btnFind.textContent = "Find sources";
      }
    };

    if (sourceEl) {
      row.append(name, meta, targetEl, sourceEl, actions, candidatesBox);
    } else {
      row.append(name, meta, targetEl, actions, candidatesBox);
    }
    parent.append(row);
  }
}

function renderCandidates(parent, candidates, folder, filename, status, subfolder = "") {
  parent.innerHTML = "";
  if (candidates.length === 0) {
    parent.append(el("div", {
      textContent: "No candidates found. Try adjusting the filename or add a manual URL in Settings.",
      style: { fontSize: "11px", opacity: "0.8" },
    }));
    return;
  }
  // Resolve which filename this candidate row stands for. We use the
  // workflow filename (the one ComfyUI complains about) as the key, not
  // the candidate's own filename, because that's what the user sees in
  // the missing-list and what the download manager keys on.
  const targetFilename = filename;
  if (candidates.some(c => c.ambiguous || c.confidence_label === "ambiguous" || c.confidence_label === "low")) {
    const q = encodeURIComponent(`"${targetFilename}"`);
    const webLine = el("div", {
      style: {
        fontSize: "11px",
        margin: "4px 0 6px",
        opacity: "0.85",
        display: "flex",
        gap: "8px",
        flexWrap: "wrap",
      },
    });
    webLine.append(
      el("span", { textContent: "Ambiguous? Verify on web:" }),
      el("a", {
        textContent: "Google",
        href: `https://www.google.com/search?q=${q}`,
        target: "_blank",
        rel: "noopener noreferrer",
        style: { color: "#64b5f6" },
      }),
      el("a", {
        textContent: "DuckDuckGo",
        href: `https://duckduckgo.com/?q=${q}`,
        target: "_blank",
        rel: "noopener noreferrer",
        style: { color: "#64b5f6" },
      }),
    );
    parent.append(webLine);
  }
  for (const c of candidates) {
    const row = el("div", {
      style: {
        marginTop: "4px",
        padding: "5px 7px",
        background: "var(--comfy-input-bg, #2a2a2a)",
        borderRadius: "3px",
        fontSize: "12px",
      },
    });
    // Tag the row so the poll loop can find the button and update it
    // when an unrelated UI part starts a download for the same file.
    row.setAttribute("data-md-candidate-filename", targetFilename);
    const sourceLabel = c.web_found ? "WEB"
                      : c.hf_fallback_found ? "HF"
                      : c.search_fallback_found ? "SRC"
                      : c.source.toUpperCase();
    const tag = el("span", {
      textContent: sourceLabel,
      style: {
        display: "inline-block",
        padding: "1px 5px",
        marginRight: "5px",
        borderRadius: "3px",
        background: c.web_found ? "#9c27b0"
                  : c.source === "huggingface" ? "#ffc107"
                  : c.source === "civitai" ? "#1976d2"
                  : "#4caf50",
        color: c.web_found ? "#fff" : "#000",
        fontSize: "10px",
        fontWeight: "bold",
      },
    });
    const title = el("span", { textContent: c.title || c.filename || c.url });
    const confidenceLabel = c.confidence_label || "unknown";
    const confidenceColor = confidenceLabel === "exact" || confidenceLabel === "high"
      ? "#66bb6a"
      : confidenceLabel === "likely"
        ? "#ffb74d"
        : confidenceLabel === "ambiguous"
          ? "#ff9800"
          : "#ef9a9a";
    const confidenceTag = el("span", {
      textContent: `${confidenceLabel.toUpperCase()}${c.confidence != null ? " " + c.confidence : ""}`,
      title: (c.confidence_reasons || []).join("; "),
      style: {
        display: "inline-block",
        padding: "1px 5px",
        marginLeft: "5px",
        borderRadius: "3px",
        background: confidenceColor,
        color: "#000",
        fontSize: "10px",
        fontWeight: "bold",
      },
    });
    const downloads = Number(c.downloads || 0);
    const downloadsTag = el("span", {
      textContent: `${fmtCount(downloads)} DL`,
      title: `${downloads} downloads`,
      style: {
        display: "inline-block",
        minWidth: "82px",
        textAlign: "center",
        whiteSpace: "nowrap",
        padding: "1px 5px",
        marginLeft: "5px",
        borderRadius: "3px",
        background: downloads >= 100000 ? "#42a5f5"
                  : downloads >= 10000 ? "#64b5f6"
                  : downloads > 0 ? "#90a4ae"
                  : "#616161",
        color: "#fff",
        fontSize: "10px",
        fontWeight: "bold",
      },
    });
    downloadsTag.onmouseenter = () => {
      downloadsTag.textContent = `${fmtCount(downloads)} downloads`;
    };
    downloadsTag.onmouseleave = () => {
      downloadsTag.textContent = `${fmtCount(downloads)} DL`;
    };
    const meta = el("div", {
      style: { fontSize: "10px", opacity: "0.7", marginTop: "2px" },
      textContent: [
        c.size ? fmtBytes(c.size) : null,
        c.web_found ? "verified via web search" : null,
        c.hf_fallback_found ? "found via HuggingFace fallback" : null,
        c.search_fallback_found ? "found via normal source fallback" : null,
        c.match_type ? `match: ${c.match_type}` : null,
        c.gated ? "gated (HF token required)" : null,
        c.needs_token ? "may require CivitAI token" : null,
        c.auto_safe ? "high confidence" : "manual review",
      ].filter(Boolean).join(" · "),
    });
    // Always show the full source URL as a small monospace line so the
    // user can verify exactly which repo the file is downloaded from
    // before clicking Download. Click the link to open it in a new tab
    // for manual verification.
    const sourceLine = el("div", {
      style: {
        fontSize: "10px",
        marginTop: "2px",
        opacity: "0.85",
        fontFamily: "ui-monospace, Consolas, Menlo, monospace",
        wordBreak: "break-all",
      },
    });
    const sourceLink = el("a", {
      textContent: c.url || "(no URL)",
      href: c.url || "#",
      target: "_blank",
      rel: "noopener noreferrer",
      style: {
        color: "var(--input-text, #aaa)",
        textDecoration: "none",
      },
      title: "Click to open the source URL in a new browser tab.",
    });
    sourceLine.appendChild(sourceLink);
    const btn = el("button", {
      textContent: "Download",
      style: {
        marginTop: "3px",
        padding: "2px 8px",
        cursor: "pointer",
        fontSize: "11px",
      },
    });
    btn.setAttribute("data-md-download-btn", "1");
    btn.onclick = async () => {
      // Walk up once to find the panel that holds shared state.
      let panel = parent;
      while (panel && !panel._mdLink) panel = panel.parentElement;

      // Pre-flight check: is a job for this file already running?
      // This catches the case where the user clicks a second candidate
      // for the same model. The poll loop also sets data-already-active
      // for visual feedback, but the canonical check is the activeFilenames
      // set updated on every poll.
      const activeSet = panel?._mdLink?.activeFilenames;
      if (activeSet && activeSet.has(targetFilename)) {
        showToast(
          panel,
          `Already downloading: ${targetFilename}`,
          "#ffb74d",
        );
        status.textContent = `Already in progress: ${targetFilename}`;
        return;
      }

      btn.disabled = true;
      btn.textContent = "Queued...";
      try {
        // Always save the file under the name ComfyUI expects (targetFilename).
        // The upstream file on HF/CivitAI may have a different name
        // (e.g. diffusion_pytorch_model.safetensors vs z_image_base-bf16.safetensors).
        // Using c.filename here would land the file under the wrong name
        // and ComfyUI would never find it.
        const downloadFilename = targetFilename;
        const resp = await jsonFetch(API.download, {
          method: "POST",
          body: JSON.stringify({
            url: c.url,
            folder: c.folder || folder,
            filename: downloadFilename,
            requested_filename: targetFilename,
            subfolder: subfolder,
            source: c.source,
            title: c.title,
            size: c.size,
          }),
        });

        // Backend tells us if it just returned an already-running job
        // instead of starting a new one (race between client checks).
        if (resp?.duplicate) {
          showToast(
            panel,
            `A download for ${targetFilename} is already running.`,
            "#ffb74d",
          );
          status.textContent = `Already in progress: ${targetFilename}`;
          // Don't reset the button - the poll loop will switch it to
          // "In progress..." on the next tick because the file is now
          // in activeFilenames.
        } else {
          status.textContent = `Queued: ${c.filename || filename}`;
        }

        // Register the job-id with the parent panel so the missing-row's
        // status badge gets updated by the poll loop. This works for both
        // fresh and duplicate responses.
        const jobId = resp?.job?.id;
        if (jobId && panel?._mdLink) {
          panel._mdLink.missingNameToJobId.set(filename, jobId);
          // Set an immediate "Queued" / "In progress" badge on the matching
          // missing row so the user gets feedback right away.
          const row = panel._mdLink.missingSection.querySelector(
            `[data-missing-name="${cssEscape(filename)}"]`,
          );
          if (row) {
            const badge = _ensureBadge(row);
            const label = resp?.duplicate
              ? "Already in progress"
              : `Queued (${c.source})`;
            _setBadge(badge, label, "#64b5f6", true);
          }
          // Eagerly add to activeFilenames so very fast double-clicks on a
          // sibling candidate get caught even before the next poll tick.
          if (panel._mdLink.activeFilenames) {
            panel._mdLink.activeFilenames.add(targetFilename);
            refreshCandidateButtons(panel, panel._mdLink.activeFilenames);
          }
        }
      } catch (e) {
        status.textContent = "Download failed: " + e.message;
        btn.disabled = false;
        btn.textContent = "Download";
      }
    };
    row.append(tag, title, confidenceTag, downloadsTag, meta, sourceLine, btn);
    parent.append(row);
  }
}

// ---------- Job rendering with live smoothing ----------

// Inject the indeterminate-stripes animation once, globally.
function ensureJobStyles() {
  if (document.getElementById("md-job-styles")) return;
  const style = document.createElement("style");
  style.id = "md-job-styles";
  style.textContent = `
    @keyframes md-stripes {
      from { background-position: 0 0; }
      to   { background-position: 28px 0; }
    }
    .md-bar-fill-indeterminate {
      background-image: linear-gradient(
        45deg,
        rgba(255,255,255,0.18) 25%, transparent 25%,
        transparent 50%, rgba(255,255,255,0.18) 50%,
        rgba(255,255,255,0.18) 75%, transparent 75%, transparent);
      background-size: 28px 28px;
      animation: md-stripes 1s linear infinite;
    }
    .md-spinner {
      display: inline-block;
      width: 10px; height: 10px;
      border: 2px solid rgba(255,255,255,0.2);
      border-top-color: #1976d2;
      border-radius: 50%;
      animation: md-spin 0.8s linear infinite;
      vertical-align: -1px;
      margin-right: 5px;
    }
    @keyframes md-spin {
      to { transform: rotate(360deg); }
    }
  `;
  document.head.appendChild(style);
}

function fmtSpeed(bps) {
  if (!bps || bps <= 0) return "";
  return fmtBytes(bps) + "/s";
}

function fmtEta(seconds) {
  if (seconds == null || !isFinite(seconds) || seconds <= 0) return "";
  if (seconds < 60) return `${Math.ceil(seconds)}s`;
  const m = Math.floor(seconds / 60);
  const s = Math.round(seconds % 60);
  if (m < 60) return `${m}m ${s}s`;
  const h = Math.floor(m / 60);
  return `${h}h ${m % 60}m`;
}

// Per-row state: caches DOM nodes + smoothing state so we can update in
// place rather than rebuilding the row every poll.
const jobRowState = new Map(); // id -> { row, els, lastBytes, lastBytesTime, smoothBytes, ... }

function renderJobs(parent, jobs) {
  ensureJobStyles();

  if (jobs.length === 0) {
    // Wipe any leftover state and show placeholder.
    if (!parent.firstChild || parent.firstChild.dataset?.placeholder !== "1") {
      parent.innerHTML = "";
      const ph = el("div", {
        textContent: "No active downloads.",
        style: { opacity: "0.7", fontStyle: "italic", fontSize: "12px" },
      });
      ph.dataset.placeholder = "1";
      parent.append(ph);
    }
    jobRowState.clear();
    stopAnimationLoop();
    return;
  }

  // Remove placeholder if present
  if (parent.firstChild?.dataset?.placeholder === "1") {
    parent.innerHTML = "";
  }

  const seenIds = new Set();
  for (const j of jobs) {
    seenIds.add(j.id);
    let entry = jobRowState.get(j.id);
    if (!entry) {
      entry = createJobRow(j);
      jobRowState.set(j.id, entry);
      parent.append(entry.row);
    }
    updateJobRow(entry, j);
  }

  // Remove rows for jobs that are gone (e.g. after Clear finished)
  for (const [id, entry] of [...jobRowState.entries()]) {
    if (!seenIds.has(id)) {
      entry.row.remove();
      jobRowState.delete(id);
    }
  }

  // Make sure the smoothing animation loop is running while there is
  // anything in-flight.
  const anyActive = jobs.some(j => j.status === "downloading" || j.status === "connecting");
  if (anyActive) startAnimationLoop();
  else stopAnimationLoop();
}

function createJobRow(j) {
  const row = el("div", {
    style: {
      border: "1px solid var(--border-color, #555)",
      borderRadius: "4px",
      padding: "6px 8px",
      background: "var(--comfy-menu-bg, #222)",
      fontSize: "12px",
      display: "flex",
      flexDirection: "column",
      gap: "4px",
    },
  });

  const name = el("div", {
    textContent: j.filename,
    style: { fontWeight: "bold", wordBreak: "break-all" },
  });
  name.setAttribute("data-md-name-el", "1");

  const bar = el("div", {
    style: {
      height: "8px",
      background: "#1a1a1a",
      borderRadius: "4px",
      overflow: "hidden",
      position: "relative",
    },
  });
  const barFill = el("div", {
    style: {
      height: "100%",
      width: "0%",
      background: "#1976d2",
      transition: "width 0.25s linear",
    },
  });
  bar.append(barFill);

  const line1 = el("div", {
    style: { display: "flex", gap: "8px", fontSize: "11px", alignItems: "center" },
  });
  const statusEl = el("span", { style: { fontWeight: "bold" } });
  const pctEl = el("span", { style: { opacity: "0.85" } });
  const bytesEl = el("span", { style: { opacity: "0.7", marginLeft: "auto" } });
  line1.append(statusEl, pctEl, bytesEl);

  const line2 = el("div", {
    style: { display: "flex", gap: "8px", fontSize: "11px", opacity: "0.7", alignItems: "center" },
  });
  const speedEl = el("span", {});
  const etaEl = el("span", {});
  const btnCancel = el("button", {
    textContent: "Cancel",
    style: {
      marginLeft: "auto",
      padding: "1px 8px",
      fontSize: "10px",
      cursor: "pointer",
      background: "transparent",
      color: "#bbb",
      border: "1px solid #555",
      borderRadius: "3px",
    },
  });
  btnCancel.onclick = () => {
    btnCancel.disabled = true;
    btnCancel.textContent = "Cancelling...";
    jsonFetch(API.cancel, { method: "POST", body: JSON.stringify({ id: j.id }) })
      .catch(() => { btnCancel.disabled = false; btnCancel.textContent = "Cancel"; });
  };
  line2.append(speedEl, etaEl, btnCancel);

  const errEl = el("div", {
    style: { color: "#ef9a9a", fontSize: "10px", display: "none" },
  });

  // Action row for errored jobs: "Open repo page" + optional token input +
  // "Retry". Hidden by default; shown when the backend reports an auth-style
  // failure (HTTP 401/403) so the user can accept the model's license and
  // resume without re-scanning. The token input is only displayed when no HF
  // token is configured yet, and gets persisted to plugin Settings on retry.
  const actionEl = el("div", {
    style: { display: "none", flexDirection: "column", gap: "4px", marginTop: "2px" },
  });
  const actionRow = el("div", {
    style: { display: "flex", gap: "6px", flexWrap: "wrap", alignItems: "center" },
  });
  const btnOpenRepo = el("button", {
    textContent: "Open repo page",
    style: {
      padding: "2px 8px",
      fontSize: "10px",
      cursor: "pointer",
      background: "#1976d2",
      color: "#fff",
      border: "0",
      borderRadius: "3px",
    },
  });
  btnOpenRepo.title =
    "Open the HuggingFace model card so you can accept the license. " +
    "Accept it with the SAME account whose token is configured here, then click Retry.";
  const btnRetry = el("button", {
    textContent: "Retry",
    style: {
      padding: "2px 8px",
      fontSize: "10px",
      cursor: "pointer",
      background: "transparent",
      color: "#bbb",
      border: "1px solid #555",
      borderRadius: "3px",
    },
  });
  btnRetry.title = "Re-run this download after accepting the license / setting a token.";
  actionRow.append(btnOpenRepo, btnRetry);

  const tokenWrap = el("div", {
    style: { display: "none", gap: "6px", alignItems: "center", fontSize: "10px" },
  });
  const tokenInput = el("input", {
    type: "password",
    placeholder: "hf_… token (saved to plugin Settings)",
    style: {
      flex: "1 1 auto",
      minWidth: "120px",
      padding: "2px 6px",
      fontSize: "10px",
      background: "#111",
      color: "#ddd",
      border: "1px solid #555",
      borderRadius: "3px",
    },
  });
  tokenInput.title =
    "Paste a HuggingFace access token (read access). It's saved to the plugin's " +
    "Settings and reused for future downloads. The retry will run authenticated.";
  tokenWrap.append(tokenInput);
  actionEl.append(actionRow, tokenWrap);

  row.append(name, bar, line1, line2, errEl, actionEl);

  return {
    row,
    els: {
      name, bar, barFill, statusEl, pctEl, bytesEl, speedEl, etaEl, btnCancel, errEl,
      actionEl, actionRow, btnOpenRepo, btnRetry, tokenWrap, tokenInput,
    },
    // Smoothing state: extrapolate bytes_done between polls based on the
    // server-reported speed so the bar moves continuously.
    smoothBytes: 0,
    lastSyncTime: performance.now(),
    lastSyncBytes: 0,
    speedBps: 0,
    bytesTotal: 0,
    status: "",
  };
}

function statusLabel(status, job) {
  switch (status) {
    case "connecting":  return "Connecting...";
    case "downloading": return "Downloading";
    case "linking":     return "Linking existing copy...";
    case "linked": {
      const m = job?.link_method;
      if (m === "hardlink") return "✓ Hardlinked (no download)";
      if (m === "symlink")  return "✓ Symlinked (no download)";
      return "✓ Linked (no download)";
    }
    case "done":        return "Done";
    case "error":       return "Error";
    case "cancelled":   return "Cancelled";
    case "queued":      return "Queued";
    default:            return status;
  }
}

function statusColor(status) {
  switch (status) {
    case "error":     return "#ef5350";
    case "done":      return "#66bb6a";
    case "linked":    return "#66bb6a";
    case "linking":   return "#ffb74d";
    case "cancelled": return "#bdbdbd";
    case "connecting":return "#ffb74d";
    default:          return "#64b5f6";
  }
}

function updateJobRow(entry, j) {
  const { els } = entry;
  entry.status = j.status;
  entry.bytesTotal = j.bytes_total || 0;
  entry.speedBps = j.speed_bps || 0;
  entry.lastSyncBytes = j.bytes_done || 0;
  entry.smoothBytes = entry.lastSyncBytes;
  entry.lastSyncTime = performance.now();
  entry.etaSeconds = j.eta_seconds;

  els.statusEl.textContent = statusLabel(j.status, j);
  els.statusEl.style.color = statusColor(j.status);

  // Show / hide the link-icon next to the filename in the job row,
  // mirroring the behaviour in the missing-list above. Linking jobs
  // get a 🔗 (or ⇲ for symlink) so the user can tell at a glance which
  // entries used disk-space-saving and which did a real download.
  _setLinkIcon(entry.row, j.status === "linking" ? "linking"
                          : j.status === "linked"  ? (j.link_method || "linked")
                          : "remove",
               j.link_source || null);

  // Bar look: striped indeterminate while connecting (no total yet),
  // solid blue during transfer, green on done/linked, red on error.
  if (j.status === "connecting" || j.status === "linking"
      || (j.status === "downloading" && !j.bytes_total)) {
    els.barFill.style.width = "100%";
    els.barFill.style.background = j.status === "linking" ? "#fb8c00" : "#1976d2";
    els.barFill.classList.add("md-bar-fill-indeterminate");
  } else {
    els.barFill.classList.remove("md-bar-fill-indeterminate");
    if (j.status === "done" || j.status === "linked") {
      els.barFill.style.background = "#2e7d32";
      els.barFill.style.width = "100%";
    } else if (j.status === "error") {
      els.barFill.style.background = "#c62828";
    } else if (j.status === "cancelled") {
      els.barFill.style.background = "#616161";
    } else {
      els.barFill.style.background = "#1976d2";
    }
  }

  // The width and text are updated continuously by the animation loop,
  // but set them once now too in case the loop isn't running yet.
  applySmoothFrame(entry);

  els.btnCancel.style.display =
    (j.status === "downloading" || j.status === "connecting" || j.status === "queued")
      ? "" : "none";

  if (j.status === "error" && j.error) {
    els.errEl.style.display = "";
    els.errEl.textContent = j.error;
  } else {
    els.errEl.style.display = "none";
  }

  // Auth-style failures: surface "Open repo page" + optional token input +
  // "Retry" so the user can accept a gated HF license (or paste a token) and
  // resume without losing the scanned-missing context. The token field is
  // only shown when the backend reports that no HF token is configured yet.
  const isAuthError = j.status === "error" && (j.error_code === 401 || j.error_code === 403);
  if (isAuthError) {
    els.actionEl.style.display = "flex";
    els.btnOpenRepo.style.display = j.repo_page_url ? "" : "none";
    els.btnOpenRepo.onclick = () => {
      if (j.repo_page_url) window.open(j.repo_page_url, "_blank", "noopener");
    };
    // Token input only when no token is configured server-side. If the user
    // already has a token but still got 401/403, the issue is license/account
    // mismatch, not a missing token — don't ask for another one.
    els.tokenWrap.style.display = j.has_hf_token ? "none" : "flex";
    // Avoid double-firing: keep button disabled while a retry is in flight,
    // and re-enable only on outright failure. The poll loop will move the row
    // out of the "error" state on success, which hides the action row.
    if (!entry._retryInFlight) {
      els.btnRetry.disabled = false;
      els.btnRetry.textContent = "Retry";
    }
    els.btnRetry.onclick = () => {
      if (entry._retryInFlight) return;
      entry._retryInFlight = true;
      els.btnRetry.disabled = true;
      els.btnRetry.textContent = "Retrying...";
      const body = { id: j.id };
      const tok = (els.tokenInput.value || "").trim();
      if (tok) body.huggingface_token = tok;
      jsonFetch(API.retry, {
        method: "POST",
        body: JSON.stringify(body),
      }).then(() => {
        els.tokenInput.value = "";
      }).catch(() => {
        entry._retryInFlight = false;
        els.btnRetry.disabled = false;
        els.btnRetry.textContent = "Retry";
      });
    };
  } else {
    els.actionEl.style.display = "none";
    // Job left the error state - clear the retry latch so a future failure
    // is once again actionable.
    entry._retryInFlight = false;
  }
}

function applySmoothFrame(entry) {
  const { els, status, bytesTotal, speedBps } = entry;

  // Extrapolate progress between server polls so the bar appears to move
  // continuously instead of jumping every 500 ms.
  let displayBytes = entry.lastSyncBytes;
  if (status === "downloading" && speedBps > 0 && bytesTotal > 0) {
    const elapsed = (performance.now() - entry.lastSyncTime) / 1000;
    // Cap extrapolation at ~750 ms so we don't run wildly ahead of reality.
    displayBytes = entry.lastSyncBytes + speedBps * Math.min(elapsed, 0.75);
    if (displayBytes > bytesTotal) displayBytes = bytesTotal;
  } else if (status === "done") {
    displayBytes = bytesTotal || entry.lastSyncBytes;
  }
  entry.smoothBytes = displayBytes;

  if (bytesTotal > 0 && status !== "connecting") {
    const pct = Math.min(100, (displayBytes / bytesTotal) * 100);
    els.barFill.style.width = pct.toFixed(1) + "%";
    els.pctEl.textContent = pct.toFixed(1) + "%";
  } else if (status === "connecting") {
    els.pctEl.textContent = "";
  } else {
    els.pctEl.textContent = "";
  }

  if (bytesTotal > 0) {
    els.bytesEl.textContent = `${fmtBytes(displayBytes)} / ${fmtBytes(bytesTotal)}`;
  } else if (displayBytes > 0) {
    els.bytesEl.textContent = fmtBytes(displayBytes);
  } else {
    els.bytesEl.textContent = "";
  }

  if (status === "downloading") {
    els.speedEl.textContent = fmtSpeed(speedBps);
    // Recompute ETA locally based on the smoothed bytes for nicer UX
    if (speedBps > 0 && bytesTotal > 0) {
      const remain = bytesTotal - displayBytes;
      els.etaEl.textContent = remain > 0 ? "ETA " + fmtEta(remain / speedBps) : "";
    } else {
      els.etaEl.textContent = "";
    }
  } else if (status === "connecting") {
    els.speedEl.innerHTML = '<span class="md-spinner"></span>waiting for server';
    els.etaEl.textContent = "";
  } else {
    els.speedEl.textContent = "";
    els.etaEl.textContent = "";
  }
}

let animationFrameId = null;
function startAnimationLoop() {
  if (animationFrameId != null) return;
  const tick = () => {
    let any = false;
    for (const entry of jobRowState.values()) {
      if (entry.status === "downloading" || entry.status === "connecting") {
        applySmoothFrame(entry);
        any = true;
      }
    }
    if (any) {
      animationFrameId = requestAnimationFrame(tick);
    } else {
      animationFrameId = null;
    }
  };
  animationFrameId = requestAnimationFrame(tick);
}
function stopAnimationLoop() {
  if (animationFrameId != null) {
    cancelAnimationFrame(animationFrameId);
    animationFrameId = null;
  }
}

// ---------- Settings Modal ----------

function buildSettingsModal() {
  const overlay = el("div", {
    style: {
      position: "fixed",
      inset: "0",
      background: "rgba(0,0,0,0.6)",
      display: "none",
      alignItems: "center",
      justifyContent: "center",
      zIndex: "10000",
    },
  });
  const box = el("div", {
    style: {
      background: "var(--comfy-menu-bg, #2a2a2a)",
      color: "var(--input-text, #ddd)",
      padding: "20px",
      borderRadius: "8px",
      minWidth: "380px",
      width: "min(560px, 92vw)",
      maxHeight: "88vh",
      overflowY: "auto",
      boxShadow: "0 4px 20px rgba(0,0,0,0.5)",
      fontFamily: "sans-serif",
    },
  });
  const title = el("h3", { textContent: "Model Downloader Settings", style: { marginTop: "0" } });

  const hfLabel = el("label", { textContent: "HuggingFace Token", style: { display: "block", marginTop: "10px" } });
  const hfInput = el("input", { type: "text", placeholder: "hf_...", style: { width: "100%", padding: "5px", boxSizing: "border-box", fontFamily: "monospace" } });
  const hfStatus = el("div", { style: { fontSize: "11px", marginTop: "3px", fontFamily: "monospace" } });

  const civLabel = el("label", { textContent: "CivitAI API Key", style: { display: "block", marginTop: "10px" } });
  const civInput = el("input", { type: "text", placeholder: "civitai api key", style: { width: "100%", padding: "5px", boxSizing: "border-box", fontFamily: "monospace" } });
  const civStatus = el("div", { style: { fontSize: "11px", marginTop: "3px", fontFamily: "monospace" } });

  const note = el("div", {
    textContent: "Tokens are saved locally in this plugin's config.json. The masked preview (e.g. hf_xx••••••wxyz) confirms a token is stored — the real value is never sent back to the browser.",
    style: { fontSize: "11px", opacity: "0.7", marginTop: "10px" },
  });

  // Track which token is currently "stored" so we can decide whether to keep
  // the masked preview as placeholder or show "Not set".
  const stored = { hf: false, civ: false };

  // When user types into a field, hide the placeholder mask suggestion (the
  // input itself replaces it visually, but we also clear the status so it's
  // obvious the user is overwriting).
  function setupTypingHints(input, statusNode, key) {
    input.addEventListener("focus", () => {
      if (stored[key] && !input.value) {
        statusNode.style.opacity = "0.5";
      }
    });
    input.addEventListener("blur", () => {
      statusNode.style.opacity = "1";
    });
  }
  setupTypingHints(hfInput, hfStatus, "hf");
  setupTypingHints(civInput, civStatus, "civ");

  // Storage-saving section. The toggle itself is always visible; the
  // sub-options below it only appear when linking is enabled. This
  // keeps the modal short for users who don't use linking at all.
  const sep = el("hr", {
    style: { margin: "16px 0 10px", border: "none", borderTop: "1px solid #444" },
  });
  const linkHeader = el("div", {
    textContent: "Disk space",
    style: { fontWeight: "bold", marginBottom: "6px" },
  });

  const linkToggleRow = el("label", {
    style: { display: "flex", alignItems: "center", gap: "8px", cursor: "pointer", margin: "4px 0" },
  });
  const linkToggle = el("input", { type: "checkbox" });
  const linkToggleText = el("span", { textContent: "Reuse existing files via filesystem links" });
  linkToggleRow.append(linkToggle, linkToggleText);

  // All linking-related sub-controls live inside this container so
  // syncLinkUI() can hide them in one go.
  const linkSubsection = el("div", {
    style: { display: "none", flexDirection: "column" },
  });

  const linkModeRow = el("div", {
    style: { display: "flex", alignItems: "center", gap: "8px", margin: "4px 0", marginLeft: "22px" },
  });
  const linkModeLabel = el("span", { textContent: "Mode:", style: { fontSize: "12px", opacity: "0.85" } });
  const linkMode = el("select", {
    style: { padding: "3px 5px", background: "var(--comfy-input-bg, #333)", color: "var(--input-text, #ddd)", border: "1px solid var(--border-color, #555)", borderRadius: "3px" },
  });
  for (const [v, label] of [
    ["auto",     "Auto (hardlink, fall back to symlink)"],
    ["hardlink", "Hardlink only"],
    ["symlink",  "Symlink only"],
  ]) {
    const opt = document.createElement("option");
    opt.value = v;
    opt.textContent = label;
    linkMode.appendChild(opt);
  }
  linkModeRow.append(linkModeLabel, linkMode);

  const linkNote = el("div", {
    textContent: "When enabled, before each download we look for an existing copy of the same file (matched by name + size) anywhere in your models folder. If found, we link it instead of downloading again. Hardlinks are instant and invisible; symlinks may need Developer Mode on Windows.",
    style: { fontSize: "11px", opacity: "0.7", marginTop: "4px", marginLeft: "22px" },
  });

  // Dedupe method (used by both the "Free Space Via Link" button and
  // the post-download dedupe pass).
  const dedupeRow = el("div", {
    style: { display: "flex", alignItems: "center", gap: "8px", margin: "8px 0 4px", marginLeft: "22px" },
  });
  const dedupeLabel = el("span", {
    textContent: "Duplicate detection:",
    style: { fontSize: "12px", opacity: "0.85" },
  });
  const dedupeMode = el("select", {
    style: { padding: "3px 5px", background: "var(--comfy-input-bg, #333)", color: "var(--input-text, #ddd)", border: "1px solid var(--border-color, #555)", borderRadius: "3px" },
  });
  for (const [v, label] of [
    ["hash",      "SHA-256 hash (slow, 100% safe)"],
    ["size_name", "Name + size only (fast, may false-positive)"],
    ["disabled",  "Disabled (hide dedupe button)"],
  ]) {
    const opt = document.createElement("option");
    opt.value = v;
    opt.textContent = label;
    dedupeMode.appendChild(opt);
  }
  dedupeRow.append(dedupeLabel, dedupeMode);

  const autoDedupeRow = el("label", {
    style: { display: "flex", alignItems: "center", gap: "8px", cursor: "pointer", margin: "4px 0", marginLeft: "22px" },
  });
  const autoDedupeToggle = el("input", { type: "checkbox" });
  const autoDedupeText = el("span", {
    textContent: "After download, check for existing duplicates and replace with hardlink",
    style: { fontSize: "12px" },
  });
  autoDedupeRow.append(autoDedupeToggle, autoDedupeText);

  const dedupeNote = el("div", {
    textContent: "Hash mode reads every byte of every candidate file - slow but never wrong. Name+size is instant but two unrelated files could in theory collide. The 'Free Space Via Link' button (in the panel header) and the post-download dedupe pass both use this setting.",
    style: { fontSize: "11px", opacity: "0.7", marginTop: "4px", marginLeft: "22px" },
  });

  // Hash cache section
  const cacheRow = el("label", {
    style: { display: "flex", alignItems: "center", gap: "8px", cursor: "pointer", margin: "8px 0 4px", marginLeft: "22px" },
  });
  const cacheToggle = el("input", { type: "checkbox" });
  const cacheText = el("span", {
    textContent: "Cache SHA-256 hashes (re-scan is fast)",
    style: { fontSize: "12px" },
  });
  cacheRow.append(cacheToggle, cacheText);
  const cacheStatsRow = el("div", {
    style: { display: "flex", gap: "8px", alignItems: "center",
             marginTop: "2px", marginLeft: "22px" },
  });
  const cacheStatsText = el("span", {
    style: { fontSize: "11px", opacity: "0.7", flex: "1" },
    textContent: "(loading cache stats...)",
  });
  const cacheClearBtn = el("button", {
    textContent: "Clear cache",
    style: { padding: "2px 8px", fontSize: "11px", cursor: "pointer",
             background: "transparent", border: "1px solid #555",
             color: "#bbb", borderRadius: "3px" },
  });
  const cachePruneBtn = el("button", {
    textContent: "Prune missing",
    style: { padding: "2px 8px", fontSize: "11px", cursor: "pointer",
             background: "transparent", border: "1px solid #555",
             color: "#bbb", borderRadius: "3px" },
  });
  cachePruneBtn.title = "Remove cache entries whose underlying file no longer exists. Faster than clearing the whole cache.";
  cacheClearBtn.title = "Wipe all cached hashes. The next dedupe scan will re-hash every file from scratch.";
  cacheStatsRow.append(cacheStatsText, cachePruneBtn, cacheClearBtn);

  const cacheNote = el("div", {
    textContent: "When enabled, computed hashes are saved to dedupe_cache.json and reused on the next scan if the file's size and mtime are unchanged. Massively speeds up repeated 'Free Space Via Link' runs.",
    style: { fontSize: "11px", opacity: "0.7", marginTop: "4px", marginLeft: "22px" },
  });

  async function refreshCacheStats() {
    try {
      const s = await jsonFetch(API.dedupe_cache_stats);
      cacheStatsText.textContent =
        `${s.entries} cached hash${s.entries === 1 ? "" : "es"} ` +
        `· ${fmtBytes(s.file_bytes || 0)} on disk`;
    } catch (e) {
      cacheStatsText.textContent = "(stats unavailable)";
    }
  }
  cacheClearBtn.onclick = async () => {
    if (!confirm("Wipe all cached file hashes? Next scan will be slow.")) return;
    cacheClearBtn.disabled = true;
    try {
      const r = await jsonFetch(API.clear_dedupe_cache,
        { method: "POST", body: "{}" });
      showFlash(`✓ Cleared ${r.removed || 0} cache entries.`, "#66bb6a");
      await refreshCacheStats();
    } catch (e) {
      showFlash("Clear failed: " + e.message, "#ef9a9a");
    } finally {
      cacheClearBtn.disabled = false;
    }
  };
  cachePruneBtn.onclick = async () => {
    cachePruneBtn.disabled = true;
    try {
      const r = await jsonFetch(API.clear_dedupe_cache,
        { method: "POST", body: JSON.stringify({ prune_only: true }) });
      showFlash(`✓ Pruned ${r.pruned || 0} stale entries.`, "#66bb6a");
      await refreshCacheStats();
    } catch (e) {
      showFlash("Prune failed: " + e.message, "#ef9a9a");
    } finally {
      cachePruneBtn.disabled = false;
    }
  };

  // Extra model paths section
  const sep2 = el("hr", {
    style: { margin: "16px 0 10px", border: "none", borderTop: "1px solid #444" },
  });
  const extraHeader = el("div", {
    textContent: "Extra model search paths",
    style: { fontWeight: "bold", marginBottom: "6px" },
  });
  const extraNote = el("div", {
    textContent: "Additional folders to scan for existing model files. Useful when you have a second ComfyUI installation, an external models drive, or a shared models tree. Paths are walked recursively. Files found here will be considered for linking and dedupe just like the main models/ folder.",
    style: { fontSize: "11px", opacity: "0.7", marginTop: "4px", marginBottom: "8px" },
  });

  // Container that holds one row per existing path + an empty add row.
  const extraList = el("div", {
    style: { display: "flex", flexDirection: "column", gap: "4px" },
  });
  const extraAddRow = el("div", {
    style: { display: "flex", gap: "6px", alignItems: "center", marginTop: "6px" },
  });
  // Pick a placeholder that matches the OS the user is on. Use
  // navigator.platform / userAgent because we cannot reach the server
  // to ask. Defaults to the Linux/Mac form when we cannot tell.
  const _isWindows = (() => {
    try {
      const p = (navigator.platform || "").toLowerCase();
      const ua = (navigator.userAgent || "").toLowerCase();
      return p.includes("win") || ua.includes("windows");
    } catch (e) {
      return false;
    }
  })();
  const _extraPlaceholder = _isWindows
    ? "C:\\Other\\ComfyUI\\models"
    : "/path/to/other/ComfyUI/models";

  const extraAddInput = el("input", {
    type: "text",
    placeholder: _extraPlaceholder,
    style: { flex: "1", padding: "5px", boxSizing: "border-box",
             fontFamily: "monospace", fontSize: "12px",
             background: "var(--comfy-input-bg, #333)",
             color: "var(--input-text, #ddd)",
             border: "1px solid var(--border-color, #555)",
             borderRadius: "3px" },
  });
  const extraAddBtn = el("button", {
    textContent: "Add",
    style: { padding: "5px 12px", cursor: "pointer", fontSize: "12px" },
  });
  const extraValidationMsg = el("div", {
    style: { fontSize: "11px", marginTop: "2px", minHeight: "14px" },
  });

  // In-memory state for extra paths. Synced from /config on open and
  // saved on Save click. Each entry is { path: string, status?: 'ok'|'warn'|'err',
  // info?: string }.
  let extraPaths = [];

  function renderExtraPaths() {
    extraList.innerHTML = "";
    if (extraPaths.length === 0) {
      const empty = el("div", {
        textContent: "(none configured)",
        style: { fontSize: "11px", opacity: "0.6", fontStyle: "italic", padding: "2px 0" },
      });
      extraList.appendChild(empty);
      return;
    }
    extraPaths.forEach((entry, i) => {
      const row = el("div", {
        style: { display: "flex", gap: "6px", alignItems: "center",
                 padding: "3px 0", fontFamily: "monospace", fontSize: "12px" },
      });
      // Status badge
      let badgeColor = "#90a4ae", badgeText = "?";
      if (entry.status === "ok") { badgeColor = "#66bb6a"; badgeText = "OK"; }
      else if (entry.status === "warn") { badgeColor = "#ffb74d"; badgeText = "!"; }
      else if (entry.status === "err") { badgeColor = "#ef5350"; badgeText = "X"; }
      const badge = el("span", {
        textContent: badgeText,
        title: entry.info || "",
        style: { display: "inline-block", minWidth: "22px", textAlign: "center",
                 padding: "1px 4px", borderRadius: "3px",
                 background: badgeColor, color: "#000", fontSize: "10px",
                 fontWeight: "bold", flexShrink: "0" },
      });
      const text = el("span", {
        textContent: entry.path,
        style: { flex: "1", wordBreak: "break-all" },
      });
      const removeBtn = el("button", {
        textContent: "Remove",
        style: { padding: "2px 8px", fontSize: "10px", cursor: "pointer",
                 background: "transparent", border: "1px solid #555",
                 color: "#bbb", borderRadius: "3px" },
      });
      removeBtn.onclick = () => {
        extraPaths.splice(i, 1);
        renderExtraPaths();
      };
      row.append(badge, text, removeBtn);
      extraList.appendChild(row);
    });
  }

  async function validatePath(p) {
    try {
      const data = await jsonFetch(API.check_path, {
        method: "POST",
        body: JSON.stringify({ path: p }),
      });
      if (!data.exists) {
        return { status: "err", info: "Path does not exist" };
      }
      if (!data.is_dir) {
        return { status: "err", info: "Not a directory" };
      }
      return { status: "ok",
               info: `${data.file_count} top-level entr${data.file_count === 1 ? "y" : "ies"}` };
    } catch (e) {
      return { status: "warn", info: "Validation failed: " + e.message };
    }
  }

  extraAddBtn.onclick = async () => {
    const raw = (extraAddInput.value || "").trim();
    if (!raw) return;
    // Reject duplicates
    if (extraPaths.some(e => e.path.toLowerCase() === raw.toLowerCase())) {
      extraValidationMsg.textContent = "Already in the list.";
      extraValidationMsg.style.color = "#ffb74d";
      return;
    }
    extraValidationMsg.textContent = "Checking...";
    extraValidationMsg.style.color = "#bbb";
    const v = await validatePath(raw);
    extraPaths.push({ path: raw, status: v.status, info: v.info });
    extraAddInput.value = "";
    extraValidationMsg.textContent = v.status === "ok"
      ? `Added: ${v.info}`
      : `Added (warning): ${v.info}`;
    extraValidationMsg.style.color = v.status === "ok" ? "#66bb6a" : "#ffb74d";
    renderExtraPaths();
  };
  // Pressing Enter in the input also triggers Add.
  extraAddInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      extraAddBtn.click();
    }
  });

  extraAddRow.append(extraAddInput, extraAddBtn);

  const flash = el("div", {
    style: {
      fontSize: "12px",
      marginTop: "10px",
      minHeight: "16px",
      fontWeight: "bold",
      transition: "opacity 0.3s",
      opacity: "0",
    },
  });

  const btnRow = el("div", { style: { marginTop: "10px", display: "flex", gap: "8px", justifyContent: "flex-end" } });
  const btnSave = el("button", { textContent: "Save", style: { padding: "5px 12px", cursor: "pointer" } });
  const btnClose = el("button", { textContent: "Close", style: { padding: "5px 12px", cursor: "pointer" } });
  const btnClearHF = el("button", { textContent: "Clear HF", style: { padding: "3px 8px", fontSize: "11px", cursor: "pointer" } });
  const btnClearCiv = el("button", { textContent: "Clear CivitAI", style: { padding: "3px 8px", fontSize: "11px", cursor: "pointer" } });
  btnRow.append(btnClearHF, btnClearCiv, btnSave, btnClose);

  // Hide / show the entire link-dependent subsection based on the
  // master toggle. Keeps the modal short for users who don't care
  // about linking at all.
  function syncLinkUI() {
    const on = linkToggle.checked;
    linkSubsection.style.display = on ? "flex" : "none";
  }
  linkToggle.addEventListener("change", syncLinkUI);
  syncLinkUI();

  // Build the link-dependent subsection. Every control inside is
  // shown/hidden as a unit by syncLinkUI().
  linkSubsection.append(
    linkModeRow, linkNote,
    dedupeRow, autoDedupeRow, dedupeNote,
    cacheRow, cacheStatsRow, cacheNote,
    sep2, extraHeader, extraNote, extraList, extraAddRow, extraValidationMsg,
  );

  box.append(
    title,
    hfLabel, hfInput, hfStatus,
    civLabel, civInput, civStatus,
    note,
    sep, linkHeader, linkToggleRow,
    linkSubsection,
    flash, btnRow,
  );
  overlay.append(box);
  document.body.append(overlay);

  function setStatus(node, input, isSet, masked, length, key) {
    stored[key] = !!isSet;
    if (isSet) {
      node.textContent = `✓ Stored: ${masked || "•".repeat(8)}  (${length} chars)`;
      node.style.color = "#66bb6a";
      // Show the masked preview as the input's placeholder so the user
      // sees at a glance that something is set without exposing it.
      input.placeholder = masked || "••••••••";
    } else {
      node.textContent = "✗ Not set";
      node.style.color = "#ef9a9a";
      input.placeholder = key === "hf" ? "hf_..." : "civitai api key";
    }
  }

  let flashTimer = null;
  function showFlash(msg, color) {
    flash.textContent = msg;
    flash.style.color = color;
    flash.style.opacity = "1";
    if (flashTimer) clearTimeout(flashTimer);
    flashTimer = setTimeout(() => { flash.style.opacity = "0"; }, 3000);
  }

  async function refreshStatus() {
    try {
      const cfg = await jsonFetch(API.config);
      setStatus(
        hfStatus, hfInput,
        cfg.huggingface_token_set,
        cfg.huggingface_token_masked,
        cfg.huggingface_token_length,
        "hf",
      );
      setStatus(
        civStatus, civInput,
        cfg.civitai_token_set,
        cfg.civitai_token_masked,
        cfg.civitai_token_length,
        "civ",
      );
      // Reflect the linking settings in the toggle/select.
      linkToggle.checked = !!cfg.enable_linking;
      const mode = cfg.linking_mode || "auto";
      // Make sure the option exists, then select it.
      const opts = Array.from(linkMode.options).map(o => o.value);
      linkMode.value = opts.includes(mode) ? mode : "auto";
      // Dedupe settings
      const dedupe = cfg.dedupe_method || "hash";
      const dopts = Array.from(dedupeMode.options).map(o => o.value);
      dedupeMode.value = dopts.includes(dedupe) ? dedupe : "hash";
      autoDedupeToggle.checked = cfg.auto_dedupe_after_download !== false; // default true
      cacheToggle.checked = cfg.use_hash_cache !== false; // default true
      // Stats are async, fire and forget
      refreshCacheStats();
      // Extra paths: revalidate each one in the background so the user
      // sees broken paths flagged. We render synchronously first with
      // an "?" status, then update as results come in.
      const incoming = Array.isArray(cfg.extra_model_paths) ? cfg.extra_model_paths : [];
      extraPaths = incoming.filter(p => typeof p === "string" && p.trim())
                            .map(p => ({ path: p, status: undefined, info: "" }));
      renderExtraPaths();
      // Validate in parallel; refresh the row each time one completes.
      extraPaths.forEach((entry, i) => {
        validatePath(entry.path).then(v => {
          // Make sure entry index is still valid (user might have removed)
          if (extraPaths[i] && extraPaths[i].path === entry.path) {
            extraPaths[i].status = v.status;
            extraPaths[i].info = v.info;
            renderExtraPaths();
          }
        });
      });
      syncLinkUI();
      return cfg;
    } catch (e) {
      hfStatus.textContent = civStatus.textContent = "Error: " + e.message;
      hfStatus.style.color = civStatus.style.color = "#ef9a9a";
      return null;
    }
  }

  btnSave.onclick = async () => {
    const hfVal = hfInput.value;
    const civVal = civInput.value;
    // Client-side guard: refuse to even send a value that contains the
    // bullet/dot character used in the masked preview.
    if (hfVal.includes("•") || civVal.includes("•")) {
      showFlash(
        "That looks like the masked preview, not a real token. Clear the field and paste the actual token.",
        "#ef9a9a",
      );
      return;
    }
    btnSave.disabled = true;
    btnSave.textContent = "Saving...";
    try {
      const body = {
        huggingface_token: hfVal,
        civitai_token: civVal,
        enable_linking: linkToggle.checked,
        linking_mode: linkMode.value,
        dedupe_method: dedupeMode.value,
        auto_dedupe_after_download: autoDedupeToggle.checked,
        extra_model_paths: extraPaths.map(e => e.path),
        use_hash_cache: cacheToggle.checked,
      };
      const resp = await jsonFetch(API.config, {
        method: "POST",
        body: JSON.stringify(body),
      });
      hfInput.value = "";
      civInput.value = "";
      const cfg = await refreshStatus();
      const saved = [];
      if (hfVal && !(resp.ignored || []).includes("huggingface_token") && cfg?.huggingface_token_set) {
        saved.push("HuggingFace");
      }
      if (civVal && !(resp.ignored || []).includes("civitai_token") && cfg?.civitai_token_set) {
        saved.push("CivitAI");
      }
      // Always report the linking status so the user knows it took effect.
      const linkParts = [];
      linkParts.push("linking " + (cfg?.enable_linking ? `ON (${cfg.linking_mode})` : "OFF"));
      if ((resp.ignored || []).length) {
        showFlash(
          `Ignored masked input for: ${resp.ignored.join(", ")}. Other settings saved.`,
          "#ffb74d",
        );
      } else {
        const tokenPart = saved.length ? `tokens: ${saved.join(" + ")}` : "";
        const parts = [tokenPart, ...linkParts].filter(Boolean);
        showFlash("✓ Saved · " + parts.join(" · "), "#66bb6a");
      }
    } catch (e) {
      showFlash("Save failed: " + e.message, "#ef9a9a");
    } finally {
      btnSave.disabled = false;
      btnSave.textContent = "Save";
    }
  };

  btnClearHF.onclick = async () => {
    try {
      await jsonFetch(API.config, {
        method: "POST",
        body: JSON.stringify({ huggingface_token: "", clear_huggingface_token: true }),
      });
      await refreshStatus();
      showFlash("HuggingFace token cleared.", "#66bb6a");
    } catch (e) {
      showFlash("Clear failed: " + e.message, "#ef9a9a");
    }
  };
  btnClearCiv.onclick = async () => {
    try {
      await jsonFetch(API.config, {
        method: "POST",
        body: JSON.stringify({ civitai_token: "", clear_civitai_token: true }),
      });
      await refreshStatus();
      showFlash("CivitAI token cleared.", "#66bb6a");
    } catch (e) {
      showFlash("Clear failed: " + e.message, "#ef9a9a");
    }
  };
  // closeCallback is set by .open() so the panel can refresh its button
  // visibility when the modal closes (e.g. user just toggled linking).
  let onCloseCallback = null;
  function close() {
    overlay.style.display = "none";
    if (typeof onCloseCallback === "function") {
      try { onCloseCallback(); } catch (e) { /* ignore */ }
    }
  }
  btnClose.onclick = close;
  overlay.onclick = (e) => { if (e.target === overlay) close(); };

  // Also re-fire the close callback right after Save so the panel updates
  // even if the user keeps the modal open.
  const _origSave = btnSave.onclick;
  btnSave.onclick = async () => {
    await _origSave.call(btnSave);
    if (typeof onCloseCallback === "function") {
      try { onCloseCallback(); } catch (e) { /* ignore */ }
    }
  };

  return {
    open: (cb) => {
      onCloseCallback = cb || null;
      overlay.style.display = "flex";
      refreshStatus();
    },
  };
}

// ---------- LoRA tag reader ----------
//
// Sidebar workflow: user clicks "Read LoRA tags", picks a LoRA from a modal,
// the backend returns trigger phrases + top training tags, the user ticks
// which ones to use (with weights), and we write the resulting text into a
// `LoraTagSelector` node on the canvas. Both the node's `selected_tags`
// widget value and its visible text are updated immediately, no restart.

function _findLoraTagSelectorNodes() {
  const graph = app?.graph;
  if (!graph || !Array.isArray(graph._nodes)) return [];
  return graph._nodes.filter((n) => n?.type === "LoraTagSelector");
}

function _setLoraSelectorWidget(node, text) {
  if (!node) return false;
  const widget = (node.widgets || []).find((w) => w?.name === "selected_tags");
  if (!widget) return false;
  widget.value = text;
  if (typeof widget.callback === "function") {
    try { widget.callback(text); } catch (_) { /* ignore */ }
  }
  if (node.onWidgetChanged) {
    try { node.onWidgetChanged(widget.name, text, widget.value, widget); } catch (_) {}
  }
  // Refresh canvas so the new value renders without restart.
  if (app?.graph?.setDirtyCanvas) app.graph.setDirtyCanvas(true, true);
  if (app?.canvas?.setDirty) app.canvas.setDirty(true, true);
  return true;
}

async function runLoraTagFlow(statusEl) {
  const data = await jsonFetch(API.lora_list);
  const loras = (data && data.loras) || [];
  if (!loras.length) {
    statusEl.textContent = "No LoRAs found on disk.";
    return;
  }

  // Build modal: LoRA picker + (after pick) tag selection with weights.
  const overlay = el("div", {
    style: {
      position: "fixed", inset: "0",
      background: "rgba(0,0,0,0.55)",
      display: "flex", alignItems: "center", justifyContent: "center",
      zIndex: 10000,
    },
  });
  const modal = el("div", {
    style: {
      background: "var(--comfy-menu-bg, #2a2a2a)",
      color: "var(--input-text, #ddd)",
      border: "1px solid var(--border-color, #555)",
      borderRadius: "6px",
      padding: "14px",
      minWidth: "480px", maxWidth: "720px",
      maxHeight: "80vh", overflow: "auto",
      display: "flex", flexDirection: "column", gap: "10px",
      fontFamily: "sans-serif", fontSize: "13px",
    },
  });
  const title = el("div", {
    textContent: "Read LoRA tags",
    style: { fontWeight: "bold", fontSize: "14px" },
  });

  const search = el("input", {
    type: "text",
    placeholder: "Filter LoRA…",
    style: {
      padding: "6px 8px", borderRadius: "4px",
      border: "1px solid #555", background: "#111", color: "#ddd",
    },
  });
  const list = el("select", {
    size: 12,
    style: {
      width: "100%", background: "#111", color: "#ddd",
      border: "1px solid #555", borderRadius: "4px", padding: "4px",
    },
  });
  function rebuildList(filterText) {
    list.innerHTML = "";
    const lc = (filterText || "").toLowerCase().trim();
    for (const name of loras) {
      if (lc && !name.toLowerCase().includes(lc)) continue;
      const opt = document.createElement("option");
      opt.value = name;
      opt.textContent = name;
      list.appendChild(opt);
    }
    if (list.options.length > 0) list.selectedIndex = 0;
  }
  rebuildList("");
  search.oninput = () => rebuildList(search.value);

  const tagsArea = el("div", { style: { display: "none", flexDirection: "column", gap: "8px" } });
  const tagsTitle = el("div", { style: { fontWeight: "bold" } });
  const tagsList = el("div", {
    style: {
      display: "flex", flexDirection: "column", gap: "4px",
      maxHeight: "260px", overflow: "auto",
      padding: "6px", background: "#111", borderRadius: "4px",
      border: "1px solid #444",
    },
  });
  const tagsPreviewLabel = el("div", {
    textContent: "Preview (this will be written into the node):",
    style: { fontSize: "11px", opacity: "0.7" },
  });
  const tagsPreview = el("textarea", {
    rows: 4,
    style: {
      background: "#111", color: "#ddd",
      border: "1px solid #555", borderRadius: "4px", padding: "6px",
      fontFamily: "monospace", fontSize: "12px",
    },
  });
  tagsArea.append(tagsTitle, tagsList, tagsPreviewLabel, tagsPreview);

  const buttonRow = el("div", {
    style: { display: "flex", gap: "8px", justifyContent: "flex-end" },
  });
  const btnClose = el("button", { textContent: "Close" });
  const btnNext = el("button", { textContent: "Read tags →" });
  const btnApply = el("button", { textContent: "Write into node" });
  btnApply.style.display = "none";
  for (const b of [btnClose, btnNext, btnApply]) {
    Object.assign(b.style, {
      padding: "5px 12px", cursor: "pointer",
      background: "#333", color: "#ddd",
      border: "1px solid #555", borderRadius: "4px",
    });
  }
  btnApply.style.background = "#1976d2";
  btnApply.style.color = "#fff";
  btnApply.style.borderColor = "#1976d2";

  buttonRow.append(btnClose, btnNext, btnApply);
  modal.append(title, search, list, tagsArea, buttonRow);
  overlay.append(modal);
  document.body.appendChild(overlay);

  function close() {
    document.body.removeChild(overlay);
  }
  btnClose.onclick = close;
  overlay.onclick = (e) => { if (e.target === overlay) close(); };

  // State for the tag-selection step.
  const tagRows = [];   // [{ tag, weight, checkbox, weightInput }]
  let triggerPhrases = [];

  function rebuildPreview() {
    const parts = [];
    for (const r of tagRows) {
      if (!r.checkbox.checked) continue;
      const tag = r.tag.trim();
      if (!tag) continue;
      const w = Number(r.weightInput.value);
      if (!Number.isFinite(w) || Math.abs(w - 1.0) < 1e-3) {
        parts.push(tag);
      } else {
        parts.push(`${tag} :: ${w.toFixed(2)}`);
      }
    }
    tagsPreview.value = parts.join("\n");
  }

  function tagRow(tag, count, weight, checked) {
    const row = el("label", {
      style: {
        display: "grid",
        gridTemplateColumns: "auto 1fr auto 70px",
        gap: "6px", alignItems: "center",
        fontSize: "12px",
      },
    });
    const cb = el("input", { type: "checkbox" });
    cb.checked = !!checked;
    const lbl = el("span", { textContent: tag });
    const cnt = el("span", {
      textContent: count != null ? String(count) : "",
      style: { opacity: "0.6", fontSize: "11px", textAlign: "right" },
    });
    const w = el("input", {
      type: "number",
      value: String(weight),
      step: "0.05", min: "0", max: "3",
      style: {
        background: "#1a1a1a", color: "#ddd",
        border: "1px solid #555", borderRadius: "3px",
        padding: "2px 4px", width: "60px",
      },
    });
    cb.onchange = rebuildPreview;
    w.oninput = rebuildPreview;
    row.append(cb, lbl, cnt, w);
    tagRows.push({ tag, weight, checkbox: cb, weightInput: w });
    return row;
  }

  btnNext.onclick = async () => {
    const opt = list.options[list.selectedIndex];
    if (!opt) return;
    btnNext.disabled = true;
    btnNext.textContent = "Loading…";
    try {
      const url = `${API.lora_meta}?name=${encodeURIComponent(opt.value)}&top=30`;
      const meta = await jsonFetch(url);
      tagsList.innerHTML = "";
      tagRows.length = 0;
      triggerPhrases = Array.isArray(meta.triggers) ? meta.triggers.slice() : [];

      const summaryBits = [];
      if (meta.title) summaryBits.push(meta.title);
      if (meta.architecture) summaryBits.push(meta.architecture);
      tagsTitle.textContent = summaryBits.length
        ? `${opt.value}  —  ${summaryBits.join("  ·  ")}`
        : opt.value;

      if (triggerPhrases.length) {
        const triggersHeader = el("div", {
          textContent: "Dataset trigger phrases (checked by default):",
          style: { fontSize: "11px", opacity: "0.7" },
        });
        tagsList.appendChild(triggersHeader);
        for (const t of triggerPhrases) {
          tagsList.appendChild(tagRow(t, null, 1.0, true));
        }
      }
      if (Array.isArray(meta.top_tags) && meta.top_tags.length) {
        const tagsHeader = el("div", {
          textContent: "Top training tags (pick the ones to keep):",
          style: { fontSize: "11px", opacity: "0.7", marginTop: "4px" },
        });
        tagsList.appendChild(tagsHeader);
        for (const t of meta.top_tags) {
          tagsList.appendChild(tagRow(t.tag, t.count, 1.0, false));
        }
      }
      if (!triggerPhrases.length && (!meta.top_tags || !meta.top_tags.length)) {
        const empty = el("div", {
          textContent: meta.error
            ? `No metadata: ${meta.error}`
            : "No trigger words or training tags found in this LoRA.",
          style: { opacity: "0.7", fontStyle: "italic" },
        });
        tagsList.appendChild(empty);
      }
      rebuildPreview();
      tagsArea.style.display = "flex";
      btnApply.style.display = "";
    } catch (e) {
      statusEl.textContent = "Failed to read LoRA meta: " + (e?.message || e);
    } finally {
      btnNext.disabled = false;
      btnNext.textContent = "Read tags →";
    }
  };

  btnApply.onclick = () => {
    const text = (tagsPreview.value || "").trim();
    if (!text) {
      statusEl.textContent = "Nothing selected.";
      return;
    }
    const nodes = _findLoraTagSelectorNodes();
    if (nodes.length === 0) {
      statusEl.textContent =
        "Add a 'LoRA Tag Selector' node to the canvas first, then click 'Read LoRA tags' again.";
      return;
    }
    let target = nodes[0];
    if (nodes.length > 1) {
      // Prefer the currently selected node if it is a LoraTagSelector.
      const selected = (app.canvas?.selected_nodes || {});
      const selList = Object.values(selected).filter((n) => n?.type === "LoraTagSelector");
      if (selList.length === 1) target = selList[0];
    }
    const ok = _setLoraSelectorWidget(target, text);
    if (ok) {
      statusEl.textContent = `Wrote ${text.split("\n").filter(Boolean).length} tag(s) into LoRA Tag Selector node #${target.id}.`;
      close();
    } else {
      statusEl.textContent = "Could not write into the node; widget missing.";
    }
  };
}

// ---------- Register sidebar tab ----------

app.registerExtension({
  name: "comfyui.model_downloader",
  async setup() {
    if (app.extensionManager?.registerSidebarTab) {
      app.extensionManager.registerSidebarTab({
        id: "modelDownloader",
        icon: "pi pi-download",
        title: "Models",
        tooltip: "Find and download missing models",
        type: "custom",
        render: (containerEl) => buildPanel(containerEl),
      });
      console.log("[ModelDownloader] Sidebar tab registered.");
    } else {
      // Fallback: floating button (older ComfyUI without sidebar API)
      const fab = el("button", {
        textContent: "Models",
        style: {
          position: "fixed",
          right: "12px",
          bottom: "12px",
          zIndex: "9999",
          padding: "8px 14px",
          borderRadius: "20px",
          background: "#1976d2",
          color: "#fff",
          border: "none",
          cursor: "pointer",
          boxShadow: "0 2px 8px rgba(0,0,0,0.4)",
        },
      });
      const panel = el("div", {
        style: {
          position: "fixed",
          right: "12px",
          bottom: "60px",
          zIndex: "9999",
          width: "360px",
          height: "500px",
          background: "var(--comfy-menu-bg, #222)",
          border: "1px solid #555",
          borderRadius: "8px",
          display: "none",
        },
      });
      document.body.append(fab, panel);
      let built = false;
      fab.onclick = () => {
        const showing = panel.style.display === "block";
        panel.style.display = showing ? "none" : "block";
        if (!built && !showing) { buildPanel(panel); built = true; }
      };
      console.log("[ModelDownloader] Sidebar API unavailable, using floating button.");
    }
  },
});

// ---------- Dedupe (Free Space Via Link) flow ----------
//
// 1. User clicks the toolbar button.
// 2. We show a confirmation dialog warning about the (possibly long)
//    full-disk hash scan.
// 3. On confirm, hit /dedupe_scan and stream-display progress.
// 4. Render a results modal listing every duplicate group with:
//      - filename, size, count
//      - per-group radio (which copy to KEEP) and per-row checkboxes (which to remove/link)
//      - total disk space that will be freed
// 5. On Apply, hit /dedupe_apply with the user's choices and show results.

async function runDedupeFlow(statusEl) {
  // The slow-scan warning lives in the Settings tooltip / docs; no
  // need to nag the user every time they click the toolbar button.
  if (statusEl) statusEl.textContent = "Dedupe: scanning + hashing files (this can take a long time)...";

  // Block UI with a simple progress overlay so the user knows something is happening.
  const overlay = el("div", {
    style: {
      position: "fixed", inset: "0",
      background: "rgba(0,0,0,0.7)",
      display: "flex", alignItems: "center", justifyContent: "center",
      zIndex: "10000",
      color: "#ddd",
      fontFamily: "sans-serif",
      flexDirection: "column",
      gap: "12px",
    },
  });
  const spin = el("div", { className: "md-spinner",
    style: { width: "32px", height: "32px", borderWidth: "4px" } });
  ensureJobStyles();
  const msg = el("div", {
    textContent: "Hashing model files... (do not close this tab)",
    style: { fontSize: "16px", fontWeight: "bold" } });
  const sub = el("div", {
    textContent: "May take minutes to hours.",
    style: { fontSize: "13px", opacity: "0.85" } });
  overlay.append(spin, msg, sub);
  document.body.appendChild(overlay);

  let data;
  try {
    data = await jsonFetch(API.dedupe_scan, { method: "POST", body: "{}" });
  } catch (e) {
    overlay.remove();
    if (statusEl) statusEl.textContent = "Dedupe scan failed: " + e.message;
    alert("Dedupe scan failed: " + e.message);
    return;
  }
  overlay.remove();

  const groups = data.groups || [];
  const totalScanned = data.total_files_scanned || 0;
  const totalSavings = data.potential_savings_bytes || 0;

  if (statusEl) {
    statusEl.textContent = `Dedupe: ${groups.length} duplicate group(s) found, ` +
      `${fmtBytes(totalSavings)} reclaimable from ${totalScanned} files scanned.`;
  }

  if (groups.length === 0) {
    alert(`No duplicates found.\n\nScanned ${totalScanned} model files; every one is unique.`);
    return;
  }

  showDedupeResultsModal(groups, totalSavings, totalScanned, statusEl);
}

function showDedupeResultsModal(groups, totalSavings, totalScanned, statusEl) {
  // Build a modal listing every group. Each group has:
  //   - title: filename (size, count copies)
  //   - radio buttons "keep this one" - one per path, default = first
  //   - the unselected ones are auto-displayed as "will be linked"
  const overlay = el("div", {
    style: {
      position: "fixed", inset: "0",
      background: "rgba(0,0,0,0.6)",
      display: "flex", alignItems: "center", justifyContent: "center",
      zIndex: "10000",
    },
  });
  const box = el("div", {
    style: {
      background: "var(--comfy-menu-bg, #2a2a2a)",
      color: "var(--input-text, #ddd)",
      padding: "16px",
      borderRadius: "8px",
      width: "min(900px, 92vw)",
      maxHeight: "85vh",
      display: "flex", flexDirection: "column", gap: "10px",
      fontFamily: "sans-serif",
      fontSize: "13px",
      boxShadow: "0 4px 24px rgba(0,0,0,0.6)",
    },
  });

  const title = el("h3", {
    textContent: `Free Space Via Link - ${groups.length} duplicate group(s)`,
    style: { margin: "0 0 4px 0" },
  });
  const summary = el("div", {
    textContent: `Up to ${fmtBytes(totalSavings)} can be reclaimed by replacing duplicates with hardlinks. ` +
                 `${totalScanned} files scanned in total.`,
    style: { fontSize: "12px", opacity: "0.85" },
  });
  const help = el("div", {
    style: { fontSize: "11px", opacity: "0.7", lineHeight: "1.4" },
  });
  help.innerHTML =
    "For each file decide its action: " +
    "<b style='color:#66bb6a'>KEEP</b> = master copy (one per group), " +
    "<b style='color:#64b5f6'>LINK</b> = delete and replace with a hardlink to the keeper, " +
    "<b style='color:#bdbdbd'>LEAVE</b> = ignore this file, don't touch it. " +
    "The data on disk after linking is bit-identical, so no workflow will break.";

  // Scrollable list
  const list = el("div", {
    style: {
      flex: "1 1 auto",
      overflow: "auto",
      border: "1px solid var(--border-color, #444)",
      borderRadius: "4px",
      padding: "6px",
      background: "var(--comfy-input-bg, #1f1f1f)",
    },
  });

  // Per-group state. groupState[i] = {
  //   paths: string[], actions: ("keep"|"link"|"leave")[], hash, size, name
  // }
  // Default: first path = keep, all others = link. Matches the old
  // behaviour. The user can flip any path to "leave" at will.
  const groupState = groups.map((g) => ({
    paths: g.paths,
    actions: g.paths.map((_, i) => i === 0 ? "keep" : "link"),
    hash: g.hash,
    size: g.size,
    name: g.name,
    saving: g.saving_bytes,
  }));

  // Action metadata used to render the per-row badge.
  const ACTION_META = {
    keep:  { label: "KEEP",  bg: "#2e7d32", color: "#fff" },
    link:  { label: "LINK",  bg: "#1976d2", color: "#fff" },
    leave: { label: "LEAVE", bg: "#616161", color: "#fff" },
  };

  // Forward-declare the apply button so updateTotal() can toggle its
  // enabled state. We attach the actual onclick later.
  const btnApply = el("button", {
    textContent: "Apply (replace LINK rows with hardlinks)",
    style: {
      padding: "6px 14px", cursor: "pointer",
      background: "#1976d2", border: "none",
      color: "#fff", borderRadius: "4px", fontWeight: "bold",
    },
  });

  // Build the rendered groups. Each path gets a 3-state cycle button.
  const groupRenderers = [];
  for (let gi = 0; gi < groupState.length; gi++) {
    const g = groupState[gi];
    const groupBox = el("div", {
      style: {
        padding: "8px",
        marginBottom: "6px",
        border: "1px solid var(--border-color, #444)",
        borderRadius: "4px",
        background: "var(--comfy-menu-bg, #2c2c2c)",
      },
    });
    const groupHeader = el("div", {
      style: { display: "flex", gap: "8px", alignItems: "baseline",
               marginBottom: "4px", flexWrap: "wrap" },
    });
    groupHeader.append(
      el("span", { textContent: g.name,
        style: { fontWeight: "bold", fontFamily: "ui-monospace, monospace" } }),
      el("span", { textContent: `${fmtBytes(g.size)} × ${g.paths.length}`,
        style: { fontSize: "11px", opacity: "0.85" } }),
    );
    if (g.hash) {
      groupHeader.appendChild(
        el("span", { textContent: `sha256: ${g.hash.slice(0, 12)}...`,
          style: { fontSize: "10px", opacity: "0.55",
                   fontFamily: "ui-monospace, monospace" } }),
      );
    }
    // Per-group warning line for invalid configurations (shown by validate()).
    const warnLine = el("div", {
      style: { fontSize: "10px", color: "#ef9a9a", marginTop: "2px",
               minHeight: "12px" },
    });
    groupBox.append(groupHeader, warnLine);

    const rowEls = [];
    g.paths.forEach((p, pi) => {
      const row = el("div", {
        style: {
          display: "flex", gap: "6px", alignItems: "center",
          padding: "3px 6px",
          fontFamily: "ui-monospace, monospace",
          fontSize: "11px",
          borderRadius: "3px",
        },
      });
      const actionBtn = el("button", {
        style: {
          minWidth: "60px",
          padding: "2px 6px",
          fontSize: "10px",
          fontWeight: "bold",
          borderRadius: "3px",
          border: "none",
          cursor: "pointer",
          color: "#fff",
        },
      });
      actionBtn.title = "Click to cycle: KEEP -> LINK -> LEAVE";
      // Cycle on click. Order: keep -> link -> leave -> keep ...
      actionBtn.onclick = () => {
        const cur = g.actions[pi];
        let next;
        if (cur === "keep") next = "link";
        else if (cur === "link") next = "leave";
        else next = "keep";
        // Enforce: only one keep per group. If the user picks keep here,
        // demote any other keep to link.
        if (next === "keep") {
          for (let i = 0; i < g.actions.length; i++) {
            if (i !== pi && g.actions[i] === "keep") {
              g.actions[i] = "link";
            }
          }
        }
        g.actions[pi] = next;
        renderRows();
        updateTotal();
      };
      const pathSpan = el("span", {
        textContent: p,
        style: { wordBreak: "break-all", flex: "1" },
      });
      row.append(actionBtn, pathSpan);
      rowEls.push({ row, actionBtn, pathSpan });
      groupBox.appendChild(row);
    });

    function renderRows() {
      // Update each row's badge + bg from the current action state and
      // recompute the warning line.
      rowEls.forEach((r, pi) => {
        const a = g.actions[pi];
        const meta = ACTION_META[a] || ACTION_META.leave;
        r.actionBtn.textContent = meta.label;
        r.actionBtn.style.background = meta.bg;
        r.actionBtn.style.color = meta.color;
        // Greyed text for LEAVE rows so the user can spot active vs ignored.
        r.pathSpan.style.opacity = (a === "leave") ? "0.5" : "1";
        r.pathSpan.style.textDecoration = (a === "leave") ? "line-through" : "none";
      });
      // Validation: check group state and surface a warning if needed.
      const keeps = g.actions.filter(a => a === "keep").length;
      const links = g.actions.filter(a => a === "link").length;
      let warn = "";
      if (links > 0 && keeps === 0) {
        warn = "⚠ At least one file must be marked KEEP if any are set to LINK.";
      } else if (keeps > 1) {
        warn = "⚠ Only one file per group can be KEEP.";
      }
      warnLine.textContent = warn;
    }
    renderRows();
    groupRenderers.push({ render: renderRows });

    list.appendChild(groupBox);
  }

  // Footer with total + apply button
  const footer = el("div", {
    style: { display: "flex", gap: "8px", alignItems: "center", marginTop: "4px" },
  });
  const totalEl = el("div", {
    style: { flex: "1", fontWeight: "bold", color: "#66bb6a" },
  });
  function updateTotal() {
    let saving = 0;
    let hasInvalid = false;
    for (const g of groupState) {
      const keeps = g.actions.filter(a => a === "keep").length;
      const links = g.actions.filter(a => a === "link").length;
      if ((links > 0 && keeps === 0) || keeps > 1) {
        hasInvalid = true;
        continue;
      }
      // Each LINK in a valid group reclaims `size` bytes.
      saving += g.size * links;
    }
    if (hasInvalid) {
      totalEl.textContent = "⚠ Some groups have invalid action combinations - fix to enable Apply.";
      totalEl.style.color = "#ef9a9a";
      btnApply.disabled = true;
      btnApply.style.opacity = "0.5";
      btnApply.style.cursor = "not-allowed";
    } else if (saving === 0) {
      totalEl.textContent = "Nothing to do (no LINK rows selected).";
      totalEl.style.color = "#bdbdbd";
      btnApply.disabled = true;
      btnApply.style.opacity = "0.5";
      btnApply.style.cursor = "not-allowed";
    } else {
      totalEl.textContent = `Will free ~${fmtBytes(saving)}`;
      totalEl.style.color = "#66bb6a";
      btnApply.disabled = false;
      btnApply.style.opacity = "1";
      btnApply.style.cursor = "pointer";
    }
  }

  const btnCancel = el("button", {
    textContent: "Cancel",
    style: {
      padding: "6px 14px", cursor: "pointer",
      background: "transparent", border: "1px solid #555",
      color: "#bbb", borderRadius: "4px",
    },
  });
  btnCancel.onclick = () => overlay.remove();

  btnApply.onclick = async () => {
    // Build payload: only groups that have at least one KEEP and one LINK
    // produce a server request. LEAVE rows are simply omitted from the
    // remove list. Groups where everything is LEAVE skip entirely.
    const reqGroups = [];
    let totalLinks = 0;
    for (const g of groupState) {
      const keepIdx = g.actions.indexOf("keep");
      if (keepIdx === -1) continue;
      const removePaths = [];
      g.actions.forEach((a, i) => {
        if (a === "link") removePaths.push(g.paths[i]);
      });
      if (removePaths.length === 0) continue;
      reqGroups.push({ keep: g.paths[keepIdx], remove: removePaths });
      totalLinks += removePaths.length;
    }
    if (reqGroups.length === 0) {
      // Apply button is disabled by updateTotal() in this case, but
      // guard anyway in case of a race.
      return;
    }
    btnApply.disabled = true;
    btnApply.textContent = "Working...";

    try {
      const res = await jsonFetch(API.dedupe_apply, {
        method: "POST",
        body: JSON.stringify({ groups: reqGroups }),
      });
      const linked = res.linked_count || 0;
      const freed = res.freed_bytes || 0;
      const errors = (res.results || []).filter(r => r.status === "error");
      let msg = `Done. Linked ${linked} file(s). Freed ${fmtBytes(freed)}.`;
      if (errors.length) {
        msg += `\n\n${errors.length} error(s):\n` +
               errors.slice(0, 5).map(e => `  ${e.remove}: ${e.reason}`).join("\n");
        if (errors.length > 5) msg += `\n  ... and ${errors.length - 5} more`;
      }
      alert(msg);
      if (statusEl) {
        statusEl.textContent = `Dedupe: linked ${linked} file(s), freed ${fmtBytes(freed)}.`;
      }
      overlay.remove();
    } catch (e) {
      alert("Dedupe apply failed: " + e.message);
      btnApply.disabled = false;
      btnApply.textContent = "Apply (replace LINK rows with hardlinks)";
    }
  };

  footer.append(totalEl, btnCancel, btnApply);

  // Initial total update so the button starts in the correct enabled state.
  updateTotal();

  box.append(title, summary, help, list, footer);
  overlay.appendChild(box);
  document.body.appendChild(overlay);
}
