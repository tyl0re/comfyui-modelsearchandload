// ComfyUI-ModelDownloader frontend (sidebar tab)
// Registers a sidebar panel and a settings section.

import { app } from "../../scripts/app.js";

const API = {
  scan:         "/model_downloader/scan",
  search:       "/model_downloader/search",
  web_search:   "/model_downloader/web_search",
  download:     "/model_downloader/download",
  relocate:     "/model_downloader/relocate",
  jobs:         "/model_downloader/jobs",
  cancel:       "/model_downloader/cancel",
  clear:        "/model_downloader/clear",
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

  const header = el("div", {
    style: { display: "flex", gap: "6px", alignItems: "center", flexWrap: "wrap" },
  });

  const title = el("div", {
    textContent: "Model Downloader",
    style: { fontWeight: "bold", fontSize: "14px", flex: "1" },
  });

  const btnScan = el("button", { textContent: "Scan workflow" });
  const btnRelocate = el("button", { textContent: "Move existing" });
  const btnSettings = el("button", { textContent: "Settings" });

  for (const b of [btnScan, btnRelocate, btnSettings]) {
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

  header.append(title, btnScan, btnRelocate, btnSettings);

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

  // State shared by scan + per-candidate downloads
  let lastMissing = [];

  // Settings modal (lazy)
  let settingsModal = null;

  btnSettings.onclick = () => {
    if (!settingsModal) settingsModal = buildSettingsModal();
    settingsModal.open();
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

  row.append(name, bar, line1, line2, errEl);

  return {
    row,
    els: { name, bar, barFill, statusEl, pctEl, bytesEl, speedEl, etaEl, btnCancel, errEl },
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

  // Storage-saving section
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

  // Disable mode-select when toggle is off, for clarity.
  function syncLinkUI() {
    linkMode.disabled = !linkToggle.checked;
    linkMode.style.opacity = linkToggle.checked ? "1" : "0.5";
  }
  linkToggle.addEventListener("change", syncLinkUI);
  syncLinkUI();

  box.append(
    title,
    hfLabel, hfInput, hfStatus,
    civLabel, civInput, civStatus,
    note,
    sep, linkHeader, linkToggleRow, linkModeRow, linkNote,
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
  btnClose.onclick = () => { overlay.style.display = "none"; };
  overlay.onclick = (e) => { if (e.target === overlay) overlay.style.display = "none"; };

  return {
    open: () => {
      overlay.style.display = "flex";
      refreshStatus();
    },
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
