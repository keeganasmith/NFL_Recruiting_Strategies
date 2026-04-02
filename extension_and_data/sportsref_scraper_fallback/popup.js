function setStatus(msg) {
  document.getElementById("status").textContent = msg;
}

function countQueueStatuses(players) {
  return (Array.isArray(players) ? players : []).reduce(
    (acc, player) => {
      const status = String(player.status || "").toLowerCase();
      if (status === "pending") acc.pending += 1;
      else if (status === "processing") acc.processing += 1;
      else if (status === "matched") acc.matched += 1;
      else if (status === "unmatched") acc.unmatched += 1;
      else if (status === "error") acc.errors += 1;
      return acc;
    },
    { pending: 0, processing: 0, matched: 0, unmatched: 0, errors: 0 }
  );
}

function reasonMessage(reason, fallbackPrefix) {
  const reasonMap = {
    empty_queue: "No queued players are pending or processing.",
    not_running: "Run is not currently marked as running.",
    player_not_processing: "No processing player matched the reported result."
  };
  if (!reason) return `${fallbackPrefix}: unknown`;
  return reasonMap[reason] || `${fallbackPrefix}: ${reason}`;
}

function formatControllerStatus(response, successMessage, failurePrefix) {
  if (response?.ok) return successMessage;
  return reasonMessage(response?.reason, failurePrefix);
}

const NAV_THROTTLE_MS = 5000;
let throttleTicker = null;

function renderThrottleStatus(lastNavigationAt) {
  const last = Number(lastNavigationAt) || 0;
  const remainingMs = Math.max(0, NAV_THROTTLE_MS - (Date.now() - last));
  const remainingSeconds = Math.ceil(remainingMs / 1000);
  document.getElementById("throttleStatus").textContent = `Next request in ${remainingSeconds}s`;
}

function startThrottleTicker(lastNavigationAt) {
  if (throttleTicker) clearInterval(throttleTicker);
  renderThrottleStatus(lastNavigationAt);
  throttleTicker = setInterval(() => renderThrottleStatus(lastNavigationAt), 500);
}

function normalizeName(name) {
  return String(name || "")
    .toLowerCase()
    .normalize("NFKD")
    .replace(/[^\w\s-]/g, "")
    .replace(/\s+/g, " ")
    .trim();
}

function normalizeSportsRefUrl(url) {
  if (!url) return "";
  try {
    const u = new URL(String(url).trim());
    if (!u.hostname.includes("sports-reference.com")) return "";
    return `${u.origin.toLowerCase()}${u.pathname.toLowerCase()}`;
  } catch {
    return "";
  }
}

function playerKey(row) {
  const nflId = String(row.NFL_id || row.nfl_id || "").trim();
  if (nflId) return `nfl:${nflId}`;

  const name = String(row.Player || row.player || "").trim().toLowerCase();
  const draftYear = String(row.draft_year || row.combine_year || "").trim();
  const pos = String(row.Pos || row.position || "").trim().toLowerCase();
  return `name:${name}|year:${draftYear}|pos:${pos}`;
}

function extractDraftYear(row) {
  const direct = Number(String(row.draft_year || row.combine_year || "").trim());
  if (Number.isFinite(direct)) return String(direct);

  const drafted = String(row["Drafted (tm/rnd/yr)"] || "").trim();
  const match = drafted.match(/\b(19|20)\d{2}\b/);
  return match ? match[0] : "";
}

function isDraftedRow(row) {
  const draftYear = extractDraftYear(row);
  return Boolean(draftYear);
}

function buildPlayerRecord(row) {
  const draftYear = extractDraftYear(row);
  const key = playerKey({ ...row, draft_year: draftYear });
  const playerName = String(row.Player || row.player || "").trim();
  const pos = String(row.Pos || row.position || "").trim();
  const slugSource = selectSportsRefSlugSource(row, playerName);
  const slug = normalizeSlugBase(slugSource);

  return {
    playerKey: key,
    playerName,
    pos,
    draftYear,
    slug,
    attemptIndex: 1,
    status: "pending",
    lastTriedUrl: "",
    matchedUrl: ""
  };
}

function firstUnprocessedPendingIndex(players) {
  const idx = players.findIndex(p => p.status === "pending");
  return idx >= 0 ? idx : players.length;
}

function mergeQueueState(existingState, importedRecords) {
  const queueState = existingState || {};
  const existingPlayers = Array.isArray(queueState.players) ? queueState.players : [];
  const byKey = new Map(existingPlayers.map(p => [p.playerKey, p]));

  for (const imported of importedRecords) {
    const prev = byKey.get(imported.playerKey);
    if (!prev) {
      byKey.set(imported.playerKey, imported);
      continue;
    }

    byKey.set(imported.playerKey, {
      ...prev,
      playerName: imported.playerName || prev.playerName,
      pos: imported.pos || prev.pos,
      draftYear: imported.draftYear || prev.draftYear,
      slug: imported.slug || prev.slug
    });
  }

  const players = Array.from(byKey.values());
  return {
    ...queueState,
    players,
    nextIndex: firstUnprocessedPendingIndex(players)
  };
}

function parseCsvLine(line) {
  const out = [];
  let value = "";
  let inQuotes = false;

  for (let i = 0; i < line.length; i++) {
    const ch = line[i];
    if (ch === '"') {
      if (inQuotes && line[i + 1] === '"') {
        value += '"';
        i++;
      } else {
        inQuotes = !inQuotes;
      }
      continue;
    }
    if (ch === "," && !inQuotes) {
      out.push(value);
      value = "";
      continue;
    }
    value += ch;
  }
  out.push(value);
  return out.map(v => v.trim());
}

function parseCsv(text) {
  const lines = String(text || "")
    .replace(/\r\n/g, "\n")
    .replace(/\r/g, "\n")
    .split("\n")
    .filter(line => line.trim().length > 0);

  if (!lines.length) return [];
  const headers = parseCsvLine(lines[0]).map(h => h.trim());
  const rows = [];

  for (let i = 1; i < lines.length; i++) {
    const values = parseCsvLine(lines[i]);
    const row = {};
    headers.forEach((h, idx) => {
      row[h] = values[idx] ?? "";
    });
    rows.push(row);
  }
  return rows;
}

function findColumn(columns, wanted) {
  const wantedSet = new Set(wanted.map(w => w.toLowerCase()));
  return columns.find(c => wantedSet.has(c.toLowerCase())) || "";
}

function buildDraftYearLookup(csvRows) {
  const lookup = {};
  if (!csvRows.length) return lookup;

  const cols = Object.keys(csvRows[0] || {});
  const playerCol = findColumn(cols, ["Player", "player", "name", "Name"]);
  const yearCol = findColumn(cols, ["year", "draft_year", "Draft Year"]);
  const urlCol = findColumn(cols, [
    "sportsref_predicted_url",
    "sportsref_url",
    "page_url",
    "url"
  ]);

  if (!yearCol || (!playerCol && !urlCol)) return lookup;

  for (const row of csvRows) {
    const year = Number(String(row[yearCol] || "").trim());
    if (!Number.isFinite(year)) continue;

    const player = normalizeName(row[playerCol] || "");
    const url = normalizeSportsRefUrl(row[urlCol] || "");

    if (player) lookup[`name:${player}`] = year;
    if (url) lookup[`url:${url}`] = year;
  }
  return lookup;
}

function escapeCsv(value) {
  if (value === null || value === undefined) return "";
  const s = String(value);
  if (/[",\n]/.test(s)) return `"${s.replace(/"/g, '""')}"`;
  return s;
}

function toCsv(rows) {
  if (!rows.length) return "";
  const headers = Array.from(rows.reduce((set, row) => {
    Object.keys(row).forEach(k => set.add(k));
    return set;
  }, new Set()));

  const lines = [headers.join(",")];
  for (const row of rows) {
    lines.push(headers.map(h => escapeCsv(row[h])).join(","));
  }
  return lines.join("\n");
}

function buildUnmatchedExportRows(unmatchedRows) {
  return (Array.isArray(unmatchedRows) ? unmatchedRows : []).map(row => ({
    playerKey: row.playerKey || "",
    Player: row.Player || row.playerName || "",
    Pos: row.Pos || row.pos || "",
    draftYear: row.draftYear || "",
    slug: row.slug || "",
    attemptsTried: row.attemptsTried || row.attemptedUrlsCount || "",
    lastTriedUrl: row.lastTriedUrl || "",
    reason: row.reason || "",
    timestamp: row.timestamp || row.attemptedAt || ""
  }));
}

function downloadBlob(content, type, filename) {
  const blob = new Blob([content], { type });
  const url = URL.createObjectURL(blob);
  chrome.downloads.download({
    url,
    filename,
    saveAs: true
  });
}

async function ensureLocalStateShape() {
  const current = await chrome.storage.local.get([
    "queueState",
    "processedKeys",
    "unmatchedRows",
    "runConfig",
    "lastNavigationAt"
  ]);
  const queueState = current.queueState || {};
  const normalizedQueueState = {
    players: Array.isArray(queueState.players) ? queueState.players : [],
    nextIndex:
      Number.isInteger(queueState.nextIndex) && queueState.nextIndex >= 0
        ? queueState.nextIndex
        : 0
  };

  await chrome.storage.local.set({
    queueState: normalizedQueueState,
    processedKeys: Array.isArray(current.processedKeys) ? current.processedKeys : [],
    unmatchedRows: Array.isArray(current.unmatchedRows) ? current.unmatchedRows : [],
    runConfig: current.runConfig || {},
    lastNavigationAt: Number(current.lastNavigationAt) || 0
  });
}

async function refresh() {
  const result = await chrome.storage.local.get([
    "rows",
    "draftYearLookup",
    "queueState",
    "processedKeys",
    "unmatchedRows",
    "runConfig",
    "lastNavigationAt",
    "draftYearLookupMeta"
  ]);
  const rows = result.rows || [];
  const lookup = result.draftYearLookup || {};
  const queueState = result.queueState || {};
  const queuePlayers = Array.isArray(queueState.players) ? queueState.players : [];
  const processedCount = Array.isArray(result.processedKeys)
    ? result.processedKeys.length
    : Object.keys(result.processedKeys || {}).length;
  const unmatchedCount = Array.isArray(result.unmatchedRows) ? result.unmatchedRows.length : 0;
  const statusCounts = countQueueStatuses(queuePlayers);
  const importMeta = result.draftYearLookupMeta || {};
  const importSummary = importMeta.filename
    ? `${importMeta.filename} (${Number(importMeta.source_rows) || 0} rows, drafted ${Number(importMeta.drafted_rows) || 0}, queued ${Number(importMeta.queue_players) || 0})`
    : "none";
  const runStatus = result.runConfig?.status || "idle";
  document.getElementById("count").textContent = `Saved records: ${rows.length}`;
  document.getElementById("lookupCount").textContent =
    `Lookup: ${Object.keys(lookup).length} | Queue: ${queuePlayers.length} | Processed: ${processedCount} | Unmatched: ${unmatchedCount} | Run: ${result.runConfig?.status || "idle"}`;
  document.getElementById("preview").value = queuePlayers.length
    ? JSON.stringify(queuePlayers[Math.max(0, queueState.nextIndex || 0)] || queuePlayers[0], null, 2)
    : "";
  document.getElementById("queueCounts").textContent =
    `Pending: ${statusCounts.pending} | Processing: ${statusCounts.processing} | Matched: ${statusCounts.matched} | Unmatched: ${statusCounts.unmatched} | Errors: ${statusCounts.errors}`;
  document.getElementById("diagQueueSize").textContent = `Queue size: ${queuePlayers.length}`;
  document.getElementById("diagPending").textContent = `Pending: ${statusCounts.pending}`;
  document.getElementById("diagImportMeta").textContent = `Last import: ${importSummary}`;
  document.getElementById("diagRunStatus").textContent = `Run status: ${runStatus}`;
  startThrottleTicker(result.lastNavigationAt);
}

async function sendControllerMessage(type, payload = {}) {
  return chrome.runtime.sendMessage({ type, ...payload });
}

console.log("[popup] initial visibilityState:", document.visibilityState);
console.log("[popup] current window handlers:", {
  onblur: window.onblur,
  onunload: window.onunload
});
document.addEventListener("visibilitychange", () => {
  console.log("[popup] visibilitychange:", document.visibilityState);
});
window.addEventListener("blur", event => {
  console.log("[popup] blur event fired", {
    visibilityState: document.visibilityState,
    eventType: event?.type
  });
});
window.addEventListener("unload", event => {
  console.log("[popup] unload event fired", {
    visibilityState: document.visibilityState,
    eventType: event?.type
  });
});

document.getElementById("importCombineCsvBtn").addEventListener("click", () => {
  document.getElementById("combineCsvInput").click();
});

document.getElementById("combineCsvInput").addEventListener("change", async event => {
  const file = event.target.files && event.target.files[0];
  console.log("[combineCsvInput] file selected:", file ? {
    name: file.name,
    size: file.size,
    type: file.type,
    visibilityState: document.visibilityState
  } : null);
  if (!file) return;

  try {
    const text = await file.text();
    console.log("[combineCsvInput] file.text() complete:", {
      length: text.length,
      visibilityState: document.visibilityState
    });
    const rows = parseCsv(text);
    console.log("[combineCsvInput] parseCsv(text) complete:", {
      rowCount: rows.length,
      visibilityState: document.visibilityState
    });
    const lookup = buildDraftYearLookup(rows);
    const draftedRows = rows.filter(isDraftedRow);
    const importedRecordsByKey = new Map();
    for (const row of draftedRows) {
      const record = buildPlayerRecord(row);
      if (!record.playerKey || importedRecordsByKey.has(record.playerKey)) continue;
      importedRecordsByKey.set(record.playerKey, record);
    }
    const importedRecords = Array.from(importedRecordsByKey.values());

    const current = await chrome.storage.local.get([
      "queueState",
      "processedKeys",
      "unmatchedRows",
      "runConfig"
    ]);

    const mergedQueueState = mergeQueueState(current.queueState, importedRecords);
    const processedSet = new Set(
      Array.isArray(current.processedKeys)
        ? current.processedKeys
        : Object.keys(current.processedKeys || {})
    );

    for (const p of mergedQueueState.players) {
      if (p.status === "matched" || p.status === "unmatched") {
        processedSet.add(p.playerKey);
      }
    }

    console.log("[combineCsvInput] before chrome.storage.local.set", {
      visibilityState: document.visibilityState,
      sourceRows: rows.length,
      draftedRows: draftedRows.length,
      queuePlayers: mergedQueueState.players.length,
      lookupEntries: Object.keys(lookup).length
    });
    await chrome.storage.local.set({
      draftYearLookup: lookup,
      draftYearLookupMeta: {
        imported_at: new Date().toISOString(),
        filename: file.name,
        source_rows: rows.length,
        drafted_rows: draftedRows.length,
        queue_players: importedRecords.length,
        lookup_entries: Object.keys(lookup).length
      },
      queueState: mergedQueueState,
      processedKeys: Array.from(processedSet),
      unmatchedRows: Array.isArray(current.unmatchedRows) ? current.unmatchedRows : [],
      runConfig: current.runConfig || {}
    });
    console.log("[combineCsvInput] after chrome.storage.local.set", {
      visibilityState: document.visibilityState
    });
    setStatus(
      `Imported ${rows.length} rows (${draftedRows.length} drafted). Queue now has ${mergedQueueState.players.length} players; ${Object.keys(lookup).length} lookup entries`
    );
    await refresh();
  } catch (err) {
    console.log("[combineCsvInput] import failed with error object:", err);
    setStatus(`Import failed: ${err && err.message ? err.message : String(err)}`);
  } finally {
    event.target.value = "";
  }
});

document.getElementById("exportCsvBtn").addEventListener("click", async () => {
  const result = await chrome.storage.local.get(["rows"]);
  const rows = result.rows || [];
  downloadBlob(toCsv(rows), "text/csv", "sportsref_final_season_data.csv");
  setStatus(`Exported ${rows.length} rows to CSV`);
});

document.getElementById("startRunBtn").addEventListener("click", async () => {
  const localState = await chrome.storage.local.get(["queueState", "runConfig"]);
  const queueState = localState.queueState || {};
  const players = Array.isArray(queueState.players) ? queueState.players : [];
  const counts = countQueueStatuses(players);
  if (counts.pending + counts.processing === 0) {
    setStatus(
      `Start blocked: empty_queue (pending=${counts.pending}, processing=${counts.processing}, run=${localState.runConfig?.status || "idle"})`
    );
    await refresh();
    return;
  }
  const response = await sendControllerMessage("START_RUN");
  setStatus(formatControllerStatus(response, "Run started", "Start failed"));
  await refresh();
});

document.getElementById("pauseRunBtn").addEventListener("click", async () => {
  const response = await sendControllerMessage("PAUSE_RUN");
  setStatus(formatControllerStatus(response, "Run paused", "Pause failed"));
  await refresh();
});

document.getElementById("resumeRunBtn").addEventListener("click", async () => {
  const response = await sendControllerMessage("RESUME_RUN");
  setStatus(formatControllerStatus(response, "Run resumed", "Resume failed"));
  await refresh();
});

document.getElementById("processNextBtn").addEventListener("click", async () => {
  const response = await sendControllerMessage("PROCESS_NEXT");
  setStatus(formatControllerStatus(response, "Queued next player", "Process failed"));
  await refresh();
});

document.getElementById("exportJsonBtn").addEventListener("click", async () => {
  const result = await chrome.storage.local.get(["rows"]);
  const rows = result.rows || [];
  downloadBlob(JSON.stringify(rows, null, 2), "application/json", "sportsref_final_season_data.json");
  setStatus(`Exported ${rows.length} rows to JSON`);
});

document.getElementById("exportUnmatchedCsvBtn").addEventListener("click", async () => {
  const result = await chrome.storage.local.get(["unmatchedRows"]);
  const unmatchedRows = buildUnmatchedExportRows(result.unmatchedRows || []);
  downloadBlob(toCsv(unmatchedRows), "text/csv", "unmatched_players.csv");
  setStatus(`Exported ${unmatchedRows.length} unmatched rows to CSV`);
});

document.getElementById("exportRunStateJsonBtn").addEventListener("click", async () => {
  const result = await chrome.storage.local.get([
    "queueState",
    "runConfig",
    "processedKeys",
    "lastNavigationAt",
    "currentPlayer"
  ]);
  const queueState = result.queueState || {};
  const players = Array.isArray(queueState.players) ? queueState.players : [];
  const processedCount = Array.isArray(result.processedKeys)
    ? result.processedKeys.length
    : Object.keys(result.processedKeys || {}).length;
  const runState = {
    exportedAt: new Date().toISOString(),
    runConfig: result.runConfig || {},
    nextIndex: Number.isInteger(queueState.nextIndex) ? queueState.nextIndex : 0,
    processedCount,
    lastNavigationAt: Number(result.lastNavigationAt) || 0,
    currentPlayer: result.currentPlayer || null,
    queue: players.map(player => ({
      playerKey: player.playerKey || "",
      playerName: player.playerName || "",
      pos: player.pos || "",
      draftYear: player.draftYear || "",
      slug: player.slug || "",
      status: player.status || "pending",
      attempts: Number(player.attemptIndex) || 0,
      lastTriedUrl: player.lastTriedUrl || "",
      matchedUrl: player.matchedUrl || "",
      processingStartedAt: player.processingStartedAt || "",
      updatedAt: player.updatedAt || "",
      completedAt: player.completedAt || ""
    }))
  };
  downloadBlob(JSON.stringify(runState, null, 2), "application/json", "run_state.json");
  setStatus(`Exported run state for ${players.length} queued players`);
});

document.getElementById("clearBtn").addEventListener("click", async () => {
  await chrome.storage.local.set({
    rows: [],
    queueState: { players: [], nextIndex: 0 },
    processedKeys: [],
    unmatchedRows: [],
    runConfig: {},
    runTabId: null,
    currentPlayer: null,
    lastNavigationAt: 0
  });
  setStatus("Cleared saved data");
  await refresh();
});

ensureLocalStateShape().then(refresh);
