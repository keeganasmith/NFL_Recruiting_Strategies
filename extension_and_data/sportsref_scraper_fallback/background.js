importScripts("shared.js");

const RUN_TAB_KEY = "runTabId";
const MAX_SUFFIX_ATTEMPTS = 25;
const NAV_THROTTLE_MS = 5000;

function dedupeKey(row) {
  return `${row.player}|${row.expected_draft_year}|${row.page_url}`;
}

async function saveMatchedRow(row) {
  if (!row || !row.player) return;
  const result = await chrome.storage.local.get(["rows"]);
  const rows = result.rows || [];
  const key = dedupeKey(row);
  const deduped = rows.filter(r => dedupeKey(r) !== key);
  deduped.push(row);
  await chrome.storage.local.set({ rows: deduped });
}

function nextPlayerIndex(players) {
  if (!Array.isArray(players) || !players.length) return -1;
  const processing = players.findIndex(p => p.status === "processing");
  if (processing >= 0) return processing;
  return players.findIndex(p => p.status === "pending");
}

async function getRunState() {
  const result = await chrome.storage.local.get([
    "queueState",
    "runConfig",
    RUN_TAB_KEY,
    "unmatchedRows",
    "diagnosticRows"
  ]);
  const queueState = result.queueState || {};
  return {
    queueState: {
      players: Array.isArray(queueState.players) ? queueState.players : [],
      nextIndex: Number.isInteger(queueState.nextIndex) ? queueState.nextIndex : 0
    },
    runConfig: {
      status: result.runConfig?.status || "idle",
      startedAt: result.runConfig?.startedAt || "",
      pausedAt: result.runConfig?.pausedAt || "",
      completedAt: result.runConfig?.completedAt || ""
    },
    runTabId: Number.isInteger(result[RUN_TAB_KEY]) ? result[RUN_TAB_KEY] : null,
    unmatchedRows: Array.isArray(result.unmatchedRows) ? result.unmatchedRows : [],
    diagnosticRows: Array.isArray(result.diagnosticRows) ? result.diagnosticRows : []
  };
}

function buildDiagnosticRow(player, resultType, pageUrl, diagnostic) {
  return {
    playerKey: player?.playerKey || "",
    playerName: player?.playerName || "",
    draftYear: player?.draftYear || "",
    slug: player?.slug || "",
    resultType: resultType || "",
    pageUrl: pageUrl || player?.lastTriedUrl || "",
    status: diagnostic?.status || resultType || "",
    player: diagnostic?.player || player?.playerName || "",
    page_url: diagnostic?.page_url || pageUrl || player?.lastTriedUrl || "",
    expected_draft_year: diagnostic?.expected_draft_year ?? player?.draftYear ?? "",
    final_season_year: diagnostic?.final_season_year ?? "",
    source_type: diagnostic?.source_type || "",
    bestSeasonScore: diagnostic?.bestSeasonScore ?? "",
    bestGameLogScoreVal: diagnostic?.bestGameLogScoreVal ?? "",
    fs: diagnostic?.fs ?? "",
    dy: diagnostic?.dy ?? "",
    eitherParseNonFinite: diagnostic?.eitherParseNonFinite ?? "",
    timestamp: new Date().toISOString()
  };
}

async function setRunStatus(status, extra = {}) {
  const now = new Date().toISOString();
  const result = await chrome.storage.local.get(["runConfig"]);
  const runConfig = result.runConfig || {};
  await chrome.storage.local.set({
    runConfig: {
      ...runConfig,
      ...extra,
      status,
      ...(status === "running" ? { startedAt: runConfig.startedAt || now, pausedAt: "" } : {}),
      ...(status === "paused" ? { pausedAt: now } : {}),
      ...(status === "complete" ? { completedAt: now } : {})
    }
  });
}

async function waitForNavigationWindow() {
  const result = await chrome.storage.local.get(["lastNavigationAt"]);
  const lastNavigationAt = Number(result.lastNavigationAt) || 0;
  const waitMs = Math.max(0, NAV_THROTTLE_MS - (Date.now() - lastNavigationAt));
  if (waitMs > 0) {
    await new Promise(resolve => setTimeout(resolve, waitMs));
  }
}

async function getOrCreateRunTab(targetUrl, existingTabId) {
  await waitForNavigationWindow();
  if (existingTabId !== null) {
    try {
      const updatedTab = await chrome.tabs.update(existingTabId, { url: targetUrl, active: false });
      await chrome.storage.local.set({ lastNavigationAt: Date.now() });
      return updatedTab;
    } catch {
      // fall through to create
    }
  }
  const createdTab = await chrome.tabs.create({ url: targetUrl, active: false });
  await chrome.storage.local.set({ lastNavigationAt: Date.now() });
  return createdTab;
}

async function processNext() {
  const state = await getRunState();
  if (state.runConfig.status !== "running") {
    return { ok: false, reason: "not_running", runStatus: state.runConfig.status };
  }

  const players = state.queueState.players;
  const index = nextPlayerIndex(players);

  if (index < 0) {
    await chrome.storage.local.set({
      queueState: { ...state.queueState, nextIndex: players.length }
    });
    await setRunStatus("complete");
    return { ok: false, reason: "empty_queue", queueSize: players.length };
  }

  const player = players[index];
  const attemptIndex = Number.isInteger(player.attemptIndex) && player.attemptIndex > 0 ? player.attemptIndex : 1;

  if (attemptIndex > MAX_SUFFIX_ATTEMPTS) {
    players[index] = {
      ...player,
      status: "error",
      completedAt: new Date().toISOString()
    };
    state.unmatchedRows.push({
      playerKey: player.playerKey,
      playerName: player.playerName,
      draftYear: player.draftYear,
      slug: player.slug,
      lastTriedUrl: player.lastTriedUrl || "",
      attemptedUrlsCount: attemptIndex - 1,
      reason: "suffix_scan_cap_reached_non_404",
      attemptedAt: new Date().toISOString()
    });
    await chrome.storage.local.set({
      queueState: { ...state.queueState, players, nextIndex: nextPlayerIndex(players) },
      unmatchedRows: state.unmatchedRows
    });
    return processNext();
  }

  const targetUrl = computePlayerUrl(player.slug || player.playerName, attemptIndex);
  if (!targetUrl) {
    players[index] = {
      ...player,
      status: "unmatched",
      lastTriedUrl: "",
      updatedAt: new Date().toISOString()
    };
    await chrome.storage.local.set({
      queueState: { ...state.queueState, players, nextIndex: nextPlayerIndex(players) },
      unmatchedRows: [
        ...state.unmatchedRows,
        {
          playerKey: player.playerKey,
          playerName: player.playerName,
          draftYear: player.draftYear,
          reason: "missing_slug",
          attemptedAt: new Date().toISOString()
        }
      ]
    });
    return processNext();
  }

  const tab = await getOrCreateRunTab(targetUrl, state.runTabId);
  players[index] = {
    ...player,
    status: "processing",
    attemptIndex,
    lastTriedUrl: targetUrl,
    processingStartedAt: new Date().toISOString()
  };

  await chrome.storage.local.set({
    [RUN_TAB_KEY]: tab.id,
    currentPlayer: {
      key: player.playerKey,
      year: player.draftYear,
      name: player.playerName,
      slug: player.slug,
      attemptIndex,
      url: targetUrl
    },
    queueState: { ...state.queueState, players, nextIndex: index }
  });

  return { ok: true, targetUrl, playerKey: player.playerKey };
}

async function handleMarkResult(message) {
  const { playerKey, resultType, row, pageUrl, diagnostic } = message;
  const state = await getRunState();
  const processedState = await chrome.storage.local.get(["processedKeys"]);
  const processedKeys = new Set(
    Array.isArray(processedState.processedKeys)
      ? processedState.processedKeys
      : Object.keys(processedState.processedKeys || {})
  );
  const players = state.queueState.players;
  const idx = players.findIndex(p => p.playerKey === playerKey && p.status === "processing");
  if (idx < 0) {
    return {
      ok: false,
      reason: "player_not_processing",
      playerKey: playerKey || "",
      processingCount: players.filter(p => p.status === "processing").length
    };
  }

  const player = players[idx];
  const now = new Date().toISOString();
  state.diagnosticRows.push(buildDiagnosticRow(player, resultType, pageUrl, diagnostic));

  if (resultType === "matched") {
    await saveMatchedRow(row);
    players[idx] = {
      ...player,
      status: "matched",
      matchedUrl: pageUrl || player.lastTriedUrl || "",
      completedAt: now
    };
    processedKeys.add(player.playerKey);
  } else if (resultType === "not_found_404") {
    players[idx] = {
      ...player,
      status: "unmatched",
      completedAt: now
    };
    const attemptedUrlsCount = Number(player.attemptIndex) || 1;
    state.unmatchedRows.push({
      playerKey: player.playerKey,
      playerName: player.playerName,
      draftYear: player.draftYear,
      slug: player.slug,
      lastTriedUrl: pageUrl || player.lastTriedUrl || "",
      attemptedUrlsCount,
      reason: "not_found_after_suffix_scan",
      attemptedAt: now,
      diagnostic: diagnostic || null
    });
    processedKeys.add(player.playerKey);
  } else if (resultType === "mismatch" || resultType === "no_table") {
    players[idx] = {
      ...player,
      status: "pending",
      attemptIndex: (Number(player.attemptIndex) || 1) + 1,
      completedAt: "",
      updatedAt: now
    };
  } else if (resultType === "error") {
    const nextAttempt = (Number(player.attemptIndex) || 1) + 1;
    if (nextAttempt > MAX_SUFFIX_ATTEMPTS) {
      players[idx] = {
        ...player,
        status: "error",
        completedAt: now,
        updatedAt: now
      };
      state.unmatchedRows.push({
        playerKey: player.playerKey,
        playerName: player.playerName,
        draftYear: player.draftYear,
        slug: player.slug,
        lastTriedUrl: pageUrl || player.lastTriedUrl || "",
        attemptedUrlsCount: Number(player.attemptIndex) || 1,
        reason: row?.reason || "scrape_error",
        attemptedAt: now,
        diagnostic: diagnostic || null
      });
      processedKeys.add(player.playerKey);
    } else {
      players[idx] = {
        ...player,
        status: "pending",
        attemptIndex: nextAttempt,
        completedAt: "",
        updatedAt: now
      };
    }
  } else {
    players[idx] = {
      ...player,
      status: "pending",
      attemptIndex: (Number(player.attemptIndex) || 1) + 1,
      completedAt: "",
      updatedAt: now
    };
  }

  await chrome.storage.local.set({
    queueState: { ...state.queueState, players, nextIndex: nextPlayerIndex(players) },
    unmatchedRows: state.unmatchedRows,
    diagnosticRows: state.diagnosticRows,
    processedKeys: Array.from(processedKeys)
  });

  return processNext();
}

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (!message || !message.type) return;

  if (message.type === "START_RUN") {
    (async () => {
      await setRunStatus("running", { startedAt: new Date().toISOString(), completedAt: "" });
      sendResponse(await processNext());
    })();
    return true;
  }

  if (message.type === "PAUSE_RUN") {
    (async () => {
      await setRunStatus("paused");
      sendResponse({ ok: true });
    })();
    return true;
  }

  if (message.type === "RESUME_RUN") {
    (async () => {
      await setRunStatus("running");
      sendResponse(await processNext());
    })();
    return true;
  }

  if (message.type === "PROCESS_NEXT") {
    (async () => {
      sendResponse(await processNext());
    })();
    return true;
  }

  if (message.type === "MARK_RESULT") {
    (async () => {
      sendResponse(await handleMarkResult(message));
    })();
    return true;
  }
});
