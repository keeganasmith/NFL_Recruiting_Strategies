function txt(el) {
  return el ? el.textContent.trim() : "";
}

function cleanKey(s) {
  return (s || "")
    .toLowerCase()
    .replace(/[%/]/g, "_pct_")
    .replace(/[^\w]+/g, "_")
    .replace(/^_+|_+$/g, "");
}

function extractTablesFromComments() {
  const tables = [];
  const walker = document.createTreeWalker(document.documentElement, NodeFilter.SHOW_COMMENT);
  let node;
  while ((node = walker.nextNode())) {
    const text = node.nodeValue || "";
    if (text.includes("<table")) {
      const div = document.createElement("div");
      div.innerHTML = text;
      tables.push(...div.querySelectorAll("table"));
    }
  }
  return tables;
}

function getAllTables() {
  return [...document.querySelectorAll("table"), ...extractTablesFromComments()];
}

function getMetaPlayerName() {
  return txt(document.querySelector("#meta h1")) || txt(document.querySelector("h1"));
}

function getMetaPosition() {
  const meta = document.querySelector("#meta");
  const raw = meta ? (meta.innerText || meta.textContent || "") : "";
  const match = raw.match(/Position:\s*([^\n]+)/i);
  return match ? match[1].trim() : "";
}

function getHeaderRow(table) {
  return table.querySelector("thead tr:last-child") || table.querySelector("tr");
}

function getHeaders(table) {
  const row = getHeaderRow(table);
  if (!row) return [];
  return Array.from(row.querySelectorAll("th, td")).map(el => txt(el));
}

function rowCells(row) {
  return Array.from(row.querySelectorAll("th, td")).map(el => txt(el));
}

function findRows(table) {
  const tbodyRows = Array.from(table.querySelectorAll("tbody tr"));
  return tbodyRows.length ? tbodyRows : Array.from(table.querySelectorAll("tr")).slice(1);
}

function seasonScore(table) {
  const headers = getHeaders(table).map(h => h.toLowerCase());
  const hs = new Set(headers);
  if (!hs.has("season") || !hs.has("g")) return -1;
  const keys = ["solo", "ast", "comb", "tfl", "sk", "int", "pd", "ff", "fr"];
  let score = 0;
  for (const k of keys) if (hs.has(k)) score += 1;
  return score >= 3 ? score : -1;
}

function gameLogScore(table) {
  const headers = getHeaders(table).map(h => h.toLowerCase());
  const hs = new Set(headers);
  const hasDateLike = hs.has("date") || hs.has("opp") || hs.has("opponent");
  const keys = ["solo", "ast", "comb", "tfl", "sk", "int", "pd", "ff", "fr"];
  let score = 0;
  for (const k of keys) if (hs.has(k)) score += 1;
  return hasDateLike && score >= 3 ? score : -1;
}

function extractFinalSeasonRow(table) {
  const headers = getHeaders(table);
  const rows = findRows(table).filter(row => {
    const first = txt(row.querySelector("th, td"));
    return /^\d{4}\*?$/.test(first);
  });
  if (!rows.length) return null;

  const finalRow = rows[rows.length - 1];
  const cells = rowCells(finalRow);
  const raw = {};

  headers.forEach((h, i) => {
    let key = cleanKey(h);
    if (raw[key] !== undefined) {
      let n = 2;
      while (raw[`${key}_${n}`] !== undefined) n++;
      key = `${key}_${n}`;
    }
    raw[key] = cells[i] ?? "";
  });

  return raw;
}

function parseNum(v) {
  if (v === null || v === undefined) return 0;
  const s = String(v).trim();
  if (!s || s === "-" || s === "Did not play") return 0;
  const n = parseFloat(s.replace(/,/g, ""));
  return Number.isFinite(n) ? n : 0;
}

function normalizeLookupName(name) {
  return String(name || "")
    .toLowerCase()
    .normalize("NFKD")
    .replace(/[^\w\s-]/g, "")
    .replace(/\s+/g, " ")
    .trim();
}

function normalizeCurrentSportsRefUrl(url) {
  try {
    const u = new URL(String(url || window.location.href).trim());
    return `${u.origin.toLowerCase()}${u.pathname.toLowerCase()}`;
  } catch {
    return "";
  }
}

function inferYear() {
  const url = window.location.href;
  const m1 = url.match(/\/(\d{4})\/gamelog/);
  if (m1) return m1[1];
  const titleText = txt(document.querySelector("h1")) + " " + (document.body.innerText || "");
  const m2 = titleText.match(/\b(19|20)\d{2}\b/);
  return m2 ? m2[0] : "";
}

function aggregateGameLog(table) {
  const headers = getHeaders(table);
  const rows = findRows(table).filter(row => {
    const cells = rowCells(row);
    if (!cells.length) return false;
    const joined = cells.join(" ").toLowerCase();
    if (joined.includes("date") || joined.includes("opponent")) return false;
    const first = cells[0] || "";
    return !!first && !/^(totals?|career)$/i.test(first);
  });

  const rawRows = [];
  for (const row of rows) {
    const cells = rowCells(row);
    const raw = {};
    headers.forEach((h, i) => {
      let key = cleanKey(h);
      if (raw[key] !== undefined) {
        let n = 2;
        while (raw[`${key}_${n}`] !== undefined) n++;
        key = `${key}_${n}`;
      }
      raw[key] = cells[i] ?? "";
    });
    rawRows.push(raw);
  }

  if (!rawRows.length) return null;

  const total = {
    season: inferYear(),
    g: String(rawRows.length),
    team: "",
    conf: "",
    class: ""
  };

  const sumFields = ["solo", "ast", "comb", "tfl", "sk", "int", "pd", "ff", "fr", "yds", "inttd", "frtd"];
  for (const field of sumFields) {
    total[field] = String(rawRows.reduce((acc, r) => acc + parseNum(r[field]), 0));
  }

  for (const k of Object.keys(total)) {
    if (/^\d+\.0$/.test(total[k])) total[k] = String(parseInt(total[k], 10));
  }

  return total;
}

/**
 * Read the draft/combine year that came from your CSV.
 * This assumes you stored it in chrome.storage.local before opening the page.
 */
async function getExpectedDraftYear() {
  const result = await chrome.storage.local.get(["currentPlayer", "draftYearLookup"]);
  const direct = result.currentPlayer?.year ? Number(result.currentPlayer.year) : null;
  if (Number.isFinite(direct)) return direct;

  const lookup = result.draftYearLookup || {};
  const byUrl = lookup[`url:${normalizeCurrentSportsRefUrl(window.location.href)}`];
  if (Number.isFinite(Number(byUrl))) return Number(byUrl);

  const byName = lookup[`name:${normalizeLookupName(getMetaPlayerName())}`];
  if (Number.isFinite(Number(byName))) return Number(byName);

  return null;
}

function isYearMatch(finalSeasonYear, draftYear) {
  const fs = Number(String(finalSeasonYear).replace(/\*/g, ""));
  const dy = Number(draftYear);
  if (!Number.isFinite(fs) || !Number.isFinite(dy)) return false;
  return fs + 1 === dy;
}

async function scrapePage() {
  try {
    const player = getMetaPlayerName();
    const pos = getMetaPosition();
    const page_url = window.location.href;
    const expected_draft_year = await getExpectedDraftYear();
    const allTables = getAllTables();

    let bestSeasonTable = null, bestSeasonScore = -1;
    let bestGameLogTable = null, bestGameLogScoreVal = -1;

    for (const table of allTables) {
      const ss = seasonScore(table);
      if (ss > bestSeasonScore) {
        bestSeasonScore = ss;
        bestSeasonTable = table;
      }
      const gs = gameLogScore(table);
      if (gs > bestGameLogScoreVal) {
        bestGameLogScoreVal = gs;
        bestGameLogTable = table;
      }
    }

    let row = null;
    let source_type = "";

    if (bestSeasonTable) {
      row = extractFinalSeasonRow(bestSeasonTable);
      source_type = "season_totals";
    } else if (bestGameLogTable) {
      row = aggregateGameLog(bestGameLogTable);
      source_type = "gamelog_aggregated";
    }

    if (!row) {
      return {
        status: "no_table",
        final_season_year: "",
        expected_draft_year,
        page_url
      };
    }

    const final_season_year = (row.season || "").replace(/\*/g, "");

    // Reject same-name wrong-player pages
    if (expected_draft_year !== null && !isYearMatch(final_season_year, expected_draft_year)) {
      console.log(
        "[SportsRef scraper] rejected year mismatch:",
        player,
        "final season =", final_season_year,
        "expected draft year =", expected_draft_year
      );
      return {
        status: "mismatch",
        final_season_year,
        expected_draft_year,
        page_url
      };
    }

    return {
      status: "matched",
      final_season_year,
      expected_draft_year,
      page_url,
      row: {
        player,
        pos,
        expected_draft_year,
        final_season_year,
        source_type,
        page_url,
        school: row.team || "",
        conference: row.conf || "",
        class: row.class || "",
        games: row.g || "",
        solo_tackles: row.solo || "",
        assisted_tackles: row.ast || "",
        combined_tackles: row.comb || "",
        tfl: row.tfl || "",
        sacks: row.sk || "",
        interceptions: row.int || "",
        interception_yards: row.yds || "",
        interception_tds: row.inttd || "",
        passes_defended: row.pd || "",
        forced_fumbles: row.ff || "",
        fumble_recoveries: row.fr || "",
        fumble_recovery_yards: row.yds_2 || row.yds_3 || "",
        fumble_recovery_tds: row.frtd || "",
        awards: row.awards || "",
        scraped_at: new Date().toISOString()
      }
    };
  } catch (error) {
    return {
      status: "error",
      reason: String(error?.message || error || "unknown_scrape_error"),
      final_season_year: "",
      expected_draft_year: await getExpectedDraftYear(),
      page_url: window.location.href
    };
  }
}

function isLikely404Page() {
  const title = String(document.title || "").toLowerCase();
  const bodyText = String(document.body?.innerText || "").toLowerCase();
  return (
    title.includes("404") ||
    title.includes("not found") ||
    bodyText.includes("404 error") ||
    bodyText.includes("page not found")
  );
}

async function sendResult(resultType, row = null) {
  const { currentPlayer } = await chrome.storage.local.get(["currentPlayer"]);
  if (!currentPlayer?.key) return;
  try {
    await chrome.runtime.sendMessage({
      type: "MARK_RESULT",
      playerKey: currentPlayer.key,
      resultType,
      row,
      pageUrl: window.location.href
    });
  } catch (err) {
    console.warn("[SportsRef scraper] failed to send MARK_RESULT", err);
  }
}

// retry a few times in case the table loads late
async function autoScrapeWithRetry(maxAttempts = 10, delayMs = 800) {
  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    const outcome = await scrapePage();

    if (outcome.status === "matched") {
      const row = outcome.row;
      console.log(
        "[SportsRef scraper] matched",
        row.player,
        "draft year:", outcome.expected_draft_year,
        "final season:", outcome.final_season_year,
        row.source_type
      );
      await sendResult("matched", row);
      return;
    }

    if (outcome.status === "mismatch") {
      console.log("[SportsRef scraper] year mismatch on", window.location.href);
      await sendResult("mismatch", outcome);
      return;
    }

    if (outcome.status === "error") {
      console.warn("[SportsRef scraper] scrape error on", window.location.href, outcome.reason || "");
      await sendResult("error", outcome);
      return;
    }

    if (outcome.status === "no_table" && attempt < maxAttempts) {
      await new Promise(resolve => setTimeout(resolve, delayMs));
      continue;
    }

    if (outcome.status === "no_table") {
      if (isLikely404Page()) {
        console.log("[SportsRef scraper] 404 on", window.location.href);
        await sendResult("not_found_404", outcome);
        return;
      }
      console.log("[SportsRef scraper] no valid matching table found on", window.location.href);
      await sendResult("no_table", outcome);
      return;
    }

    await new Promise(resolve => setTimeout(resolve, delayMs));
  }
  await sendResult("error", {
    status: "error",
    reason: "retry_limit_exhausted",
    final_season_year: "",
    expected_draft_year: await getExpectedDraftYear(),
    page_url: window.location.href
  });
}

autoScrapeWithRetry();
