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

const HEADER_ALIASES = {
  year_id: "season",
  games: "g",
  team_name_abbr: "team",
  conf_abbr: "conf",
  def_int: "int",
  def_int_yds: "int_yds",
  def_int_td: "inttd",
  def_int_yds_per_int: "int_yds_per_int",
  fumbles_rec_yds: "fr_yds",
  fumbles_rec_td: "frtd",
  int_td: "inttd",
  inttd: "inttd",
  int_yds: "int_yds",
  int_yds_: "int_yds",
  interception_yds: "int_yds",
  interception_yards: "int_yds",
  yards: "yds",
  pass_defended: "pd",
  passes_defended: "pd",
  pass_breakups: "pd",
  pbus: "pd",
  pbu: "pd"
};

function normalizeHeaderKey(label) {
  const cleaned = cleanKey(label);
  return HEADER_ALIASES[cleaned] || cleaned;
}

function getNormalizedHeaders(table) {
  const row = getHeaderRow(table);
  if (!row) return [];
  return Array.from(row.querySelectorAll("th, td")).map(el => {
    const dataStat = normalizeHeaderKey(el.getAttribute("data-stat") || "");
    if (dataStat && !/^header_empty/.test(dataStat)) return dataStat;
    return normalizeHeaderKey(txt(el));
  });
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
  const details = seasonScoreDetails(table);
  return details.score;
}

function seasonScoreDetails(table, options = {}) {
  const minDenseDefensiveKeys = Number.isFinite(options.minDenseDefensiveKeys)
    ? options.minDenseDefensiveKeys
    : 3;
  const minSparseSubsetKeys = Number.isFinite(options.minSparseSubsetKeys)
    ? options.minSparseSubsetKeys
    : 2;

  const headers = getNormalizedHeaders(table);
  const hs = new Set(headers);
  if (!hs.has("season") || !hs.has("g")) {
    return {
      score: -1,
      reason: "rejected_missing_base_headers",
      headers,
      statKeyScore: 0,
      sparseSubsetHits: 0
    };
  }

  const keys = ["solo", "ast", "comb", "tfl", "sk", "int", "pd", "ff", "fr"];
  let score = 0;
  for (const k of keys) if (hs.has(k)) score += 1;

  if (score >= minDenseDefensiveKeys) {
    return {
      score,
      reason: "accepted_dense_defensive_keys",
      headers,
      statKeyScore: score,
      sparseSubsetHits: 0
    };
  }

  const sparseSubset = ["int", "inttd", "int_yds", "pd"];
  let sparseSubsetHits = 0;
  for (const k of sparseSubset) if (hs.has(k)) sparseSubsetHits += 1;
  if (sparseSubsetHits >= minSparseSubsetKeys) {
    return {
      score: sparseSubsetHits,
      reason: "accepted_sparse_defensive_summary",
      headers,
      statKeyScore: score,
      sparseSubsetHits
    };
  }

  const optionalStatKeys = ["inttd", "int_yds", "fr_yds", "frtd", "yds", "awards"];
  let optionalStatHits = 0;
  for (const k of optionalStatKeys) if (hs.has(k)) optionalStatHits += 1;
  if (score + optionalStatHits > 0) {
    return {
      score: Math.max(score, optionalStatHits),
      reason: "accepted_minimal_defensive_stats",
      headers,
      statKeyScore: score,
      sparseSubsetHits
    };
  }

  return {
    score: -1,
    reason: "rejected_insufficient_defensive_keys",
    headers,
    statKeyScore: score,
    sparseSubsetHits
  };
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
  const headerRow = getHeaderRow(table);
  const headerCells = headerRow ? Array.from(headerRow.querySelectorAll("th, td")) : [];
  const headers = headerCells.map(el => normalizeHeaderKey(el.getAttribute("data-stat") || txt(el)));
  const rows = findRows(table).filter(row => {
    const first = txt(row.querySelector("th, td"));
    return /^\d{4}\*?$/.test(first);
  });
  if (!rows.length) return null;

  const finalRow = rows[rows.length - 1];
  const cells = rowCells(finalRow);
  const raw = {};

  headers.forEach((h, i) => {
    let key = normalizeHeaderKey(h);
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

function normalizeYearString(value) {
  const candidate = String(value ?? "").trim();
  return /^\d{4}$/.test(candidate) ? candidate : "";
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

  const sumFields = [
    "solo",
    "ast",
    "comb",
    "tfl",
    "sk",
    "int",
    "pd",
    "ff",
    "fr",
    "int_yds",
    "inttd",
    "fr_yds",
    "frtd"
  ];
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
  const direct = normalizeYearString(result.currentPlayer?.year || "");
  if (direct) return { year: direct, source: "currentPlayer" };

  const lookup = result.draftYearLookup || {};
  const byUrl = lookup[`url:${normalizeCurrentSportsRefUrl(window.location.href)}`];
  const normalizedByUrl = normalizeYearString(byUrl || "");
  if (normalizedByUrl) return { year: normalizedByUrl, source: "lookup:url" };

  const byName = lookup[`name:${normalizeLookupName(getMetaPlayerName())}`];
  const normalizedByName = normalizeYearString(byName || "");
  if (normalizedByName) return { year: normalizedByName, source: "lookup:name" };

  return { year: null, source: "none" };
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
    const expectedDraftYear = await getExpectedDraftYear();
    const expected_draft_year = expectedDraftYear.year;
    const expected_draft_year_source = expectedDraftYear.source;
    const allTables = getAllTables();

    let bestSeasonTable = null, bestSeasonScore = -1;
    let bestSeasonDiagnosticRank = -1;
    let seasonTableMatchReason = "no_season_table_candidate";
    let bestGameLogTable = null, bestGameLogScoreVal = -1;

    for (const table of allTables) {
      const seasonDetails = seasonScoreDetails(table);
      const ss = seasonDetails.score;
      if (ss > bestSeasonScore) {
        bestSeasonScore = ss;
        bestSeasonTable = table;
        seasonTableMatchReason = seasonDetails.reason;
      }

      const candidateRank = ss >= 0
        ? 100 + ss
        : seasonDetails.statKeyScore + (seasonDetails.sparseSubsetHits * 0.1);
      if (candidateRank > bestSeasonDiagnosticRank) {
        bestSeasonDiagnosticRank = candidateRank;
        if (ss < 0 || !bestSeasonTable) {
          seasonTableMatchReason = seasonDetails.reason;
        }
      }

      const gs = gameLogScore(table);
      if (gs > bestGameLogScoreVal) {
        bestGameLogScoreVal = gs;
        bestGameLogTable = table;
      }
    }

    let row = null;
    let source_type = "";
    const makeDiagnostic = (status, extra = {}) => ({
      status,
      player,
      page_url,
      expected_draft_year,
      expected_draft_year_source,
      final_season_year: extra.final_season_year ?? "",
      source_type,
      bestSeasonScore,
      bestGameLogScoreVal,
      seasonTableMatchReason,
      ...extra
    });

    if (bestSeasonTable) {
      row = extractFinalSeasonRow(bestSeasonTable);
      source_type = "season_totals";
    } else if (bestGameLogTable) {
      row = aggregateGameLog(bestGameLogTable);
      source_type = "gamelog_aggregated";
    }

    if (!row) {
      const diagnostic = makeDiagnostic("no_table");
      return {
        status: "no_table",
        player,
        final_season_year: "",
        expected_draft_year,
        expected_draft_year_source,
        page_url,
        source_type,
        bestSeasonScore,
        bestGameLogScoreVal,
        seasonTableMatchReason,
        diagnostic
      };
    }

    const final_season_year = (row.season || "").replace(/\*/g, "");

    // Reject same-name wrong-player pages
    if (expected_draft_year !== null && !isYearMatch(final_season_year, expected_draft_year)) {
      const fs = Number(String(final_season_year).replace(/\*/g, ""));
      const dy = Number(expected_draft_year);
      const eitherParseNonFinite = !Number.isFinite(fs) || !Number.isFinite(dy);
      const diagnostic = makeDiagnostic("mismatch", {
        final_season_year,
        fs,
        dy,
        eitherParseNonFinite
      });
      console.log(
        "[SportsRef scraper] rejected year mismatch:",
        player,
        "final season =", final_season_year,
        "expected draft year =", expected_draft_year
      );
      return {
        status: "mismatch",
        player,
        final_season_year,
        expected_draft_year,
        expected_draft_year_source,
        page_url,
        source_type,
        bestSeasonScore,
        bestGameLogScoreVal,
        seasonTableMatchReason,
        diagnostic
      };
    }

    const diagnostic = makeDiagnostic("matched", { final_season_year });
    return {
      status: "matched",
      player,
      final_season_year,
      expected_draft_year,
      expected_draft_year_source,
      page_url,
      source_type,
      bestSeasonScore,
      bestGameLogScoreVal,
      seasonTableMatchReason,
      diagnostic,
      row: {
        player,
        pos,
        expected_draft_year,
        expected_draft_year_source,
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
        interception_yards: row.int_yds || row.yds || "",
        interception_tds: row.inttd || "",
        passes_defended: row.pd || "",
        forced_fumbles: row.ff || "",
        fumble_recoveries: row.fr || "",
        fumble_recovery_yards: row.fr_yds || row.yds_2 || row.yds_3 || "",
        fumble_recovery_tds: row.frtd || "",
        awards: row.awards || "",
        scraped_at: new Date().toISOString()
      }
    };
  } catch (error) {
    const expectedDraftYear = await getExpectedDraftYear();
    const expected_draft_year = expectedDraftYear.year;
    const expected_draft_year_source = expectedDraftYear.source;
    const page_url = window.location.href;
    const player = getMetaPlayerName();
    const diagnostic = {
      status: "error",
      player,
      page_url,
      expected_draft_year,
      expected_draft_year_source,
      final_season_year: "",
      source_type: "",
      bestSeasonScore: -1,
      bestGameLogScoreVal: -1,
      seasonTableMatchReason: "error_before_table_scoring"
    };
    return {
      status: "error",
      reason: String(error?.message || error || "unknown_scrape_error"),
      player,
      final_season_year: "",
      expected_draft_year,
      expected_draft_year_source,
      page_url,
      source_type: "",
      bestSeasonScore: -1,
      bestGameLogScoreVal: -1,
      seasonTableMatchReason: "error_before_table_scoring",
      diagnostic
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

async function sendResult(resultType, row = null, diagnostic = null) {
  const { currentPlayer } = await chrome.storage.local.get(["currentPlayer"]);
  if (!currentPlayer?.key) return;
  try {
    await chrome.runtime.sendMessage({
      type: "MARK_RESULT",
      playerKey: currentPlayer.key,
      resultType,
      row,
      diagnostic,
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
      await sendResult("matched", row, outcome.diagnostic || null);
      return;
    }

    if (outcome.status === "mismatch") {
      console.log("[SportsRef scraper] year mismatch on", window.location.href);
      await sendResult("mismatch", outcome, outcome.diagnostic || null);
      return;
    }

    if (outcome.status === "error") {
      console.warn("[SportsRef scraper] scrape error on", window.location.href, outcome.reason || "");
      await sendResult("error", outcome, outcome.diagnostic || null);
      return;
    }

    if (outcome.status === "no_table" && attempt < maxAttempts) {
      await new Promise(resolve => setTimeout(resolve, delayMs));
      continue;
    }

    if (outcome.status === "no_table") {
      if (isLikely404Page()) {
        console.log("[SportsRef scraper] 404 on", window.location.href);
        await sendResult("not_found_404", outcome, outcome.diagnostic || null);
        return;
      }
      console.log("[SportsRef scraper] no valid matching table found on", window.location.href);
      await sendResult("no_table", outcome, outcome.diagnostic || null);
      return;
    }

    await new Promise(resolve => setTimeout(resolve, delayMs));
  }
  const retryExpectedDraft = await getExpectedDraftYear();
  await sendResult("error", {
    status: "error",
    reason: "retry_limit_exhausted",
    player: getMetaPlayerName(),
    final_season_year: "",
    expected_draft_year: retryExpectedDraft.year,
    expected_draft_year_source: retryExpectedDraft.source,
    page_url: window.location.href,
    source_type: "",
    bestSeasonScore: -1,
    bestGameLogScoreVal: -1,
    seasonTableMatchReason: "retry_limit_exhausted"
  }, {
    status: "error",
    player: getMetaPlayerName(),
    page_url: window.location.href,
    expected_draft_year: retryExpectedDraft.year,
    expected_draft_year_source: retryExpectedDraft.source,
    final_season_year: "",
    source_type: "",
    bestSeasonScore: -1,
    bestGameLogScoreVal: -1,
    seasonTableMatchReason: "retry_limit_exhausted"
  });
}

autoScrapeWithRetry();
