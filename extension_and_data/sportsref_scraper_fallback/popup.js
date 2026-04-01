function setStatus(msg) {
  document.getElementById("status").textContent = msg;
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
  const result = await chrome.storage.local.get(["rows"]);
  const rows = result.rows || [];
  document.getElementById("count").textContent = `Saved records: ${rows.length}`;
  document.getElementById("preview").value = rows.length
    ? JSON.stringify(rows[rows.length - 1], null, 2)
    : "";
}

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