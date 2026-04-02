function setStatus(msg) {
  document.getElementById("status").textContent = msg;
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
  const slug = normalizeSlugBase(row["Player-additional"] || row.slug || playerName);

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

async function ensureLocalStateShape() {
  const current = await chrome.storage.local.get([
    "queueState",
    "processedKeys",
    "unmatchedRows",
    "runConfig"
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
    runConfig: current.runConfig || {}
  });
}

async function refresh() {
  const result = await chrome.storage.local.get([
    "rows",
    "draftYearLookup",
    "queueState",
    "processedKeys",
    "unmatchedRows",
    "runConfig"
  ]);
  const rows = result.rows || [];
  const lookup = result.draftYearLookup || {};
  const queueState = result.queueState || {};
  const queuePlayers = Array.isArray(queueState.players) ? queueState.players : [];
  const processedCount = Array.isArray(result.processedKeys)
    ? result.processedKeys.length
    : Object.keys(result.processedKeys || {}).length;
  const unmatchedCount = Array.isArray(result.unmatchedRows) ? result.unmatchedRows.length : 0;
  document.getElementById("count").textContent = `Saved records: ${rows.length}`;
  document.getElementById("lookupCount").textContent =
    `Lookup: ${Object.keys(lookup).length} | Queue: ${queuePlayers.length} | Processed: ${processedCount} | Unmatched: ${unmatchedCount} | Run: ${result.runConfig?.status || "idle"}`;
  document.getElementById("preview").value = queuePlayers.length
    ? JSON.stringify(queuePlayers[Math.max(0, queueState.nextIndex || 0)] || queuePlayers[0], null, 2)
    : "";
}

async function sendControllerMessage(type, payload = {}) {
  return chrome.runtime.sendMessage({ type, ...payload });
}

document.getElementById("importCombineCsvBtn").addEventListener("click", () => {
  document.getElementById("combineCsvInput").click();
});

document.getElementById("combineCsvInput").addEventListener("change", async event => {
  const file = event.target.files && event.target.files[0];
  if (!file) return;

  try {
    const text = await file.text();
    const rows = parseCsv(text);
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
    setStatus(
      `Imported ${rows.length} rows (${draftedRows.length} drafted). Queue now has ${mergedQueueState.players.length} players; ${Object.keys(lookup).length} lookup entries`
    );
    await refresh();
  } catch (err) {
    setStatus(`Import failed: ${err && err.message ? err.message : String(err)}`);
  } finally {
    event.target.value = "";
  }
});

document.getElementById("exportCsvBtn").addEventListener("click", async () => {
  const result = await chrome.storage.local.get(["rows"]);
  const rows = result.rows || [];
  const blob = new Blob([toCsv(rows)], { type: "text/csv" });
  const url = URL.createObjectURL(blob);
  chrome.downloads.download({
    url,
    filename: "sportsref_final_season_data.csv",
    saveAs: true
  });
  setStatus(`Exported ${rows.length} rows to CSV`);
});

document.getElementById("startRunBtn").addEventListener("click", async () => {
  const response = await sendControllerMessage("START_RUN");
  setStatus(response?.ok ? "Run started" : `Start failed: ${response?.reason || "unknown"}`);
  await refresh();
});

document.getElementById("pauseRunBtn").addEventListener("click", async () => {
  const response = await sendControllerMessage("PAUSE_RUN");
  setStatus(response?.ok ? "Run paused" : `Pause failed: ${response?.reason || "unknown"}`);
  await refresh();
});

document.getElementById("resumeRunBtn").addEventListener("click", async () => {
  const response = await sendControllerMessage("RESUME_RUN");
  setStatus(response?.ok ? "Run resumed" : `Resume failed: ${response?.reason || "unknown"}`);
  await refresh();
});

document.getElementById("processNextBtn").addEventListener("click", async () => {
  const response = await sendControllerMessage("PROCESS_NEXT");
  setStatus(response?.ok ? "Queued next player" : `Process failed: ${response?.reason || "unknown"}`);
  await refresh();
});

document.getElementById("exportJsonBtn").addEventListener("click", async () => {
  const result = await chrome.storage.local.get(["rows"]);
  const rows = result.rows || [];
  const blob = new Blob([JSON.stringify(rows, null, 2)], {
    type: "application/json"
  });
  const url = URL.createObjectURL(blob);
  chrome.downloads.download({
    url,
    filename: "sportsref_final_season_data.json",
    saveAs: true
  });
  setStatus(`Exported ${rows.length} rows to JSON`);
});

document.getElementById("clearBtn").addEventListener("click", async () => {
  await chrome.storage.local.set({
    rows: [],
    queueState: { players: [], nextIndex: 0 },
    processedKeys: [],
    unmatchedRows: [],
    runConfig: {},
    runTabId: null,
    currentPlayer: null
  });
  setStatus("Cleared saved data");
  await refresh();
});

ensureLocalStateShape().then(refresh);
