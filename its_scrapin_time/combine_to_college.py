#!/usr/bin/env python3
"""
combine_to_college.py

Reads a combine CSV that already includes an ID column (default: NFL_id), fetches
college-football ESPN stats per unique numeric ID, and writes a long-form merged CSV.

Behavior is intentionally parallel to combine_to_nfl.py:
- SQLite response cache
- Rate limiting between requests
- Retry/timeout controls
- Progress prints while fetching
- Unmatched/failure CSV with reason
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import time
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests


ESPN_URL_TMPL = (
    "https://site.web.api.espn.com/apis/common/v3/sports/football/college-football/athletes/{athlete_id}/stats"
)
DEFAULT_PARAMS = {"region": "us", "lang": "en", "contentorigin": "espn"}


def init_cache(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS athlete_stats_cache (
            athlete_id INTEGER PRIMARY KEY,
            fetched_at_utc INTEGER NOT NULL,
            payload_json TEXT NOT NULL
        )
        """
    )
    conn.commit()
    print(f"[INFO] Cache DB ready: {db_path}")
    return conn


def cache_get(conn: sqlite3.Connection, athlete_id: int) -> Optional[dict]:
    cur = conn.execute(
        "SELECT payload_json FROM athlete_stats_cache WHERE athlete_id = ?",
        (athlete_id,),
    )
    row = cur.fetchone()
    if not row:
        return None
    try:
        return json.loads(row[0])
    except Exception:
        return None


def cache_put(conn: sqlite3.Connection, athlete_id: int, payload: dict) -> None:
    conn.execute(
        """
        INSERT INTO athlete_stats_cache (athlete_id, fetched_at_utc, payload_json)
        VALUES (?, ?, ?)
        ON CONFLICT(athlete_id) DO UPDATE SET
          fetched_at_utc=excluded.fetched_at_utc,
          payload_json=excluded.payload_json
        """,
        (athlete_id, int(time.time()), json.dumps(payload)),
    )
    conn.commit()


class RateLimiter:
    def __init__(self, min_interval_sec: float):
        self.min_interval = float(min_interval_sec)
        self._last = None  # Optional[float]

    def wait(self):
        now = time.monotonic()
        if self._last is None:
            self._last = now
            return
        elapsed = now - self._last
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last = time.monotonic()


def fetch_espn_stats(
    session: requests.Session,
    limiter: RateLimiter,
    athlete_id: int,
    timeout_sec: float,
    max_retries: int,
) -> dict:
    url = ESPN_URL_TMPL.format(athlete_id=athlete_id)
    last_err: Optional[Exception] = None

    for attempt in range(max_retries + 1):
        try:
            limiter.wait()
            resp = session.get(url, params=DEFAULT_PARAMS, timeout=timeout_sec)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:  # noqa: BLE001 - CLI retry path
            last_err = e
            if attempt >= max_retries:
                break
            sleep_s = min(2.0, 0.25 * (2**attempt))
            time.sleep(sleep_s)

    raise RuntimeError(f"fetch_failed after {max_retries + 1} attempts: {last_err}")


def flatten_espn_payload_to_rows(payload: dict) -> List[dict]:
    rows: Dict[Tuple[int, str], dict] = {}

    categories = payload.get("categories", [])
    for cat in categories:
        cat_name = cat.get("name", "unknown")
        names = cat.get("names", [])
        stats_list = cat.get("statistics", [])
        for entry in stats_list:
            season = entry.get("season", {}) or {}
            year = season.get("year")
            if year is None:
                continue
            team_id = str(entry.get("teamId", ""))

            key = (int(year), team_id)
            if key not in rows:
                rows[key] = {
                    "season_year": int(year),
                    "teamId": team_id,
                    "teamSlug": entry.get("teamSlug", ""),
                    "position": entry.get("position", ""),
                }

            stat_values = entry.get("stats", [])
            for i, nm in enumerate(names):
                if i >= len(stat_values):
                    break
                rows[key][f"{cat_name}_{nm}"] = stat_values[i]

    totals_row = {"season_year": -1, "teamId": "", "teamSlug": "", "position": ""}
    any_totals = False
    for cat in categories:
        cat_name = cat.get("name", "unknown")
        names = cat.get("names", [])
        totals = cat.get("totals")
        if not totals or not isinstance(totals, list):
            continue
        any_totals = True
        for i, nm in enumerate(names):
            if i >= len(totals):
                break
            totals_row[f"{cat_name}_{nm}"] = totals[i]

    out = list(rows.values())
    if any_totals:
        out.append(totals_row)
    return out


def parse_numeric_id(val) -> Tuple[Optional[int], str]:
    if pd.isna(val):
        return None, "missing_id"
    s = str(val).strip()
    if not s or s.upper() == "N/A":
        return None, "missing_id"
    if s.endswith(".0") and s[:-2].isdigit():
        s = s[:-2]
    if not s.isdigit():
        return None, "non_numeric_id"
    return int(s), ""


def fmt_duration(seconds: float) -> str:
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h}h{m:02d}m{s:02d}s"
    if m > 0:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", default="NFL_data/combine_with_stats.csv")
    ap.add_argument("--id_column", default="NFL_id")
    ap.add_argument("--out_csv", default="NFL_data/combine_with_college_stats.csv")
    ap.add_argument("--unmatched_csv", default="NFL_data/college_unmatched_ids.csv")
    ap.add_argument("--cache_db", default="espn_college_cache.sqlite")
    ap.add_argument("--min_delay_ms", type=int, default=200, help="Minimum delay between network requests")
    ap.add_argument("--timeout_sec", type=float, default=30.0)
    ap.add_argument("--max_retries", type=int, default=2)
    ap.add_argument("--print_every_fetch", type=int, default=50)
    args = ap.parse_args()

    t0 = time.time()

    df = pd.read_csv(args.input_csv)
    if args.id_column not in df.columns:
        raise SystemExit(f"Missing id column '{args.id_column}' in {args.input_csv}")

    conn = init_cache(args.cache_db)
    limiter = RateLimiter(args.min_delay_ms / 1000.0)
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) combine_to_college/1.0",
            "Accept": "*/*",
            "Origin": "https://www.espn.com",
            "Referer": "https://www.espn.com/",
        }
    )

    unmatched_rows: List[dict] = []
    valid_ids: List[int] = []
    seen_ids = set()

    for idx, row in df.iterrows():
        athlete_id, reason = parse_numeric_id(row.get(args.id_column))
        if athlete_id is None:
            unmatched_rows.append(
                {
                    "row_index": idx,
                    "id_column": args.id_column,
                    "id_value": row.get(args.id_column, ""),
                    "reason": reason,
                }
            )
            continue
        if athlete_id not in seen_ids:
            seen_ids.add(athlete_id)
            valid_ids.append(athlete_id)

    print(
        f"[INFO] Loaded {len(df):,} input rows | unique numeric IDs to resolve: {len(valid_ids):,} | "
        f"pre-unmatched rows: {len(unmatched_rows):,}"
    )

    payload_by_id: Dict[int, dict] = {}
    total_cache_hits = 0
    total_network_fetches = 0
    total_fetch_failures = 0

    t_fetch0 = time.time()
    for i, athlete_id in enumerate(valid_ids, start=1):
        if args.print_every_fetch > 0 and i % args.print_every_fetch == 0:
            print(
                f"[FETCH] {i:,}/{len(valid_ids):,} IDs processed | "
                f"cache_hits={total_cache_hits:,} fetched={total_network_fetches:,} failures={total_fetch_failures:,} | "
                f"elapsed={fmt_duration(time.time() - t_fetch0)}"
            )

        payload = cache_get(conn, athlete_id)
        if payload is not None:
            total_cache_hits += 1
            payload_by_id[athlete_id] = payload
            continue

        try:
            payload = fetch_espn_stats(
                session=session,
                limiter=limiter,
                athlete_id=athlete_id,
                timeout_sec=args.timeout_sec,
                max_retries=args.max_retries,
            )
            cache_put(conn, athlete_id, payload)
            payload_by_id[athlete_id] = payload
            total_network_fetches += 1
        except Exception as e:  # noqa: BLE001 - CLI failure collection
            total_fetch_failures += 1
            unmatched_rows.append(
                {
                    "row_index": "",
                    "id_column": args.id_column,
                    "id_value": athlete_id,
                    "reason": f"fetch_failed: {type(e).__name__}: {e}",
                }
            )

    print(f"[FETCH DONE] in {fmt_duration(time.time() - t_fetch0)}")

    all_output_rows: List[dict] = []
    for _, row in df.iterrows():
        athlete_id, _ = parse_numeric_id(row.get(args.id_column))
        if athlete_id is None:
            continue

        payload = payload_by_id.get(athlete_id)
        if payload is None:
            continue

        stat_rows = flatten_espn_payload_to_rows(payload)
        base = row.to_dict()
        for sr in stat_rows:
            out = dict(base)
            out.update(sr)
            all_output_rows.append(out)

    if unmatched_rows:
        pd.DataFrame(unmatched_rows).to_csv(args.unmatched_csv, index=False)
        print(f"[WRITE] Unmatched / failed IDs -> {args.unmatched_csv} ({len(unmatched_rows):,} rows)")

    if all_output_rows:
        out_df = pd.DataFrame(all_output_rows)
        front_cols = [
            args.id_column,
            "Player",
            "Pos",
            "School",
            "College",
            "Drafted (tm/rnd/yr)",
            "season_year",
            "teamId",
            "teamSlug",
            "position",
        ]
        cols = [c for c in front_cols if c in out_df.columns] + [c for c in out_df.columns if c not in front_cols]
        out_df = out_df[cols]
        out_df.to_csv(args.out_csv, index=False)
        print(f"[WRITE] Stats CSV -> {args.out_csv} ({len(out_df):,} rows)")
    else:
        print("⚠️  No stats output rows generated.")

    print("\n========== SUMMARY ==========")
    print(f"Total time: {fmt_duration(time.time() - t0)}")
    print(f"Input rows: {len(df):,}")
    print(f"Unique numeric IDs: {len(valid_ids):,}")
    print(f"Cache hits: {total_cache_hits:,}")
    print(f"Network fetches: {total_network_fetches:,}")
    print(f"Fetch failures: {total_fetch_failures:,}")
    print(f"Cache DB: {args.cache_db}")
    if unmatched_rows:
        print(f"Unmatched/failures CSV: {args.unmatched_csv}")
    print("=============================")


if __name__ == "__main__":
    main()
