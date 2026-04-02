# SportsRef Scraper Fallback — Operator Guide

This guide is the day-to-day runbook for operating the unpacked Chrome extension in
`extension_and_data/sportsref_scraper_fallback`.

## End-to-end usage (exact run sequence)

1. **Load unpacked extension from `sportsref_scraper_fallback`.**
   - Open `chrome://extensions`.
   - Enable **Developer mode**.
   - Click **Load unpacked** and select:
     `extension_and_data/sportsref_scraper_fallback/`

2. **Click `Import Combine CSV` and select `NFL_data/combine_with_stats.csv`.**
   - Open the extension popup.
   - Click **Import Combine CSV**.
   - Choose `NFL_data/combine_with_stats.csv` from this repo.

3. **Start run (new button).**
   - Click **Start Run**.

4. **Extension auto-opens player URLs with 5s minimum interval.**
   - The extension enforces a 5-second throttle between navigations.
   - You can confirm in popup status text: **Next request in Xs**.

5. **Monitor live counts (pending/matched/unmatched/errors) in popup.**
   - In popup queue counters, monitor:
     - Pending
     - Matched
     - Unmatched
     - Errors

6. **Pause/resume anytime without losing state.**
   - Use **Pause Run** to stop processing.
   - Use **Resume Run** to continue.
   - Queue state and saved rows remain in extension storage.

7. **On completion, export:**
   - **matched CSV** via **Export Matched CSV**
   - **unmatched CSV** via **Export Unmatched CSV**
   - **run-state JSON** via **Export Run State JSON**

8. **If rerunning, reset options:**

   ### Option A — Continue from saved state
   Use when you want to pick up where prior processing left off.

   - Do **not** clear data.
   - Re-open popup and click **Resume Run** (if paused) or **Start Run** (if idle).

   ### Option B — Reset only unmatched
   Use when you only want failed/unmatched players retried and want to keep matched data.

   - Open popup and click **Export Run State JSON** first (backup).
   - In `chrome://extensions`, open **Service Worker** for this extension (Inspect views).
   - In the console, run:

   ```js
   (async () => {
     const { queueState = {}, unmatchedRows = [], processedKeys = [] } =
       await chrome.storage.local.get(["queueState", "unmatchedRows", "processedKeys"]);

     const players = Array.isArray(queueState.players) ? queueState.players : [];
     const unmatchedKeys = new Set(
       players
         .filter(p => ["unmatched", "error"].includes(String(p.status || "").toLowerCase()))
         .map(p => p.playerKey)
     );

     const resetPlayers = players.map(p => {
       const status = String(p.status || "").toLowerCase();
       if (status !== "unmatched" && status !== "error") return p;
       return {
         ...p,
         status: "pending",
         attemptIndex: 1,
         completedAt: "",
         updatedAt: new Date().toISOString()
       };
     });

     const nextIndex = resetPlayers.findIndex(p => p.status === "pending");
     const processedSet = new Set(Array.isArray(processedKeys) ? processedKeys : Object.keys(processedKeys || {}));
     unmatchedKeys.forEach(k => processedSet.delete(k));

     await chrome.storage.local.set({
       queueState: {
         players: resetPlayers,
         nextIndex: nextIndex >= 0 ? nextIndex : resetPlayers.length
       },
       unmatchedRows: [],
       processedKeys: Array.from(processedSet)
     });

     console.log("Reset unmatched/error players to pending complete.");
   })();
   ```

   - Re-open popup, then click **Start Run**.

   ### Option C — Full reset
   Use when starting a fully fresh run.

   - Click **Clear Saved Data** in popup.
   - Re-import `NFL_data/combine_with_stats.csv`.
   - Click **Start Run**.

---

## Operator tips

- Export outputs at the end of every run before any reset.
- Prefer **Pause Run** over closing tabs mid-run; pause preserves clean state transitions.
- Keep `run_state.json` snapshots if you need auditability across multiple run attempts.
