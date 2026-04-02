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

async function refresh() {
  const result = await chrome.storage.local.get(["rows", "draftYearLookup"]);
  const rows = result.rows || [];
  const lookup = result.draftYearLookup || {};
  document.getElementById("count").textContent = `Saved records: ${rows.length}`;
  document.getElementById("lookupCount").textContent =
    `Draft-year lookup entries: ${Object.keys(lookup).length}`;
  document.getElementById("preview").value = rows.length
    ? JSON.stringify(rows[rows.length - 1], null, 2)
    : "";
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
    await chrome.storage.local.set({
      draftYearLookup: lookup,
      draftYearLookupMeta: {
        imported_at: new Date().toISOString(),
        filename: file.name,
        source_rows: rows.length,
        lookup_entries: Object.keys(lookup).length
      }
    });
    setStatus(
      `Imported ${rows.length} CSV rows; built ${Object.keys(lookup).length} draft-year lookup entries`
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
  await chrome.storage.local.set({ rows: [] });
  setStatus("Cleared saved data");
  await refresh();
});

refresh();
