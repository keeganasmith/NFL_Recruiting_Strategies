#!/usr/bin/env python3
"""
Collect unique NFL athletes (athlete.id + athlete.displayName) into ONE shared CSV.

Requirements from user:
- Iterate all NFL teams
- Iterate all seasons from 2000–2025 (inclusive)
- >= 200 ms delay between *every* network request, but run as fast as possible
- Maintain a single shared CSV of unique athletes (global de-dupe by athlete_id)
- Resumable: if CSV exists, load it and keep adding only new athlete_ids

Default season type:
- seasontype=2 (regular season). You can change SEASON_TYPES if you want postseason too.

Usage:
  pip install requests
  python collect_unique_athletes.py --out all_nfl_athletes.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import requests


BASE = "https://site.web.api.espn.com/apis/site/v2/sports/football/nfl"
TEAM_LIST_URL = f"{BASE}/teams?region=us&lang=en&contentorigin=espn"

# Regular season only by default (2). If you want postseason too, set to (2, 3).
SEASON_TYPES: Tuple[int, ...] = (2,)
YEAR_START = 2000
YEAR_END = 2025

MIN_DELAY_S = 0.200  # 200 ms between each network request


DEFAULT_HEADERS = {
    "accept": "*/*",
    "accept-language": "en-US,en;q=0.9",
    "origin": "https://www.espn.com",
    "referer": "https://www.espn.com/",
    "user-agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/145.0.0.0 Safari/537.36"
    ),
}


def build_stats_url(team_abbr: str, season: int, seasontype: int) -> str:
    return (
        f"{BASE}/teams/{team_abbr}/athletes/statistics"
        f"?region=us&lang=en&contentorigin=espn&season={season}&seasontype={seasontype}"
    )


class RateLimiter:
    """Ensures at least `min_interval_s` between request starts."""

    def __init__(self, min_interval_s: float):
        self.min_interval_s = float(min_interval_s)
        self._next_allowed = time.monotonic()

    def wait(self) -> None:
        now = time.monotonic()
        if now < self._next_allowed:
            time.sleep(self._next_allowed - now)
        # schedule next slot starting *now* (request start) + interval
        self._next_allowed = time.monotonic() + self.min_interval_s


def fetch_json(
    session: requests.Session,
    limiter: RateLimiter,
    url: str,
    timeout_s: float = 20.0,
    retries: int = 3,
) -> Dict[str, Any]:
    last_err: Optional[Exception] = None
    for attempt in range(retries):
        try:
            limiter.wait()
            resp = session.get(url, timeout=timeout_s)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            last_err = e
            # small backoff; still respects limiter on next attempt
            time.sleep(0.2 * (attempt + 1))
    raise RuntimeError(f"Failed after {retries} attempts: {url} :: {last_err}") from last_err


def iter_dicts(obj: Any) -> Iterable[Dict[str, Any]]:
    """Yield all dicts recursively from nested JSON."""
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from iter_dicts(v)
    elif isinstance(obj, list):
        for it in obj:
            yield from iter_dicts(it)


def extract_team_abbrs(team_payload: Dict[str, Any]) -> List[str]:
    """
    Robustly extract team abbreviations from the teams endpoint.

    ESPN typically provides something like:
      sports[0].leagues[0].teams[*].team.abbreviation

    This function also includes a fallback recursive search for dicts like:
      {"team": {"abbreviation": "lar", ...}}
    """
    abbrs: Set[str] = set()

    # Preferred path
    try:
        sports = team_payload.get("sports", [])
        if isinstance(sports, list) and sports:
            leagues = sports[0].get("leagues", [])
            if isinstance(leagues, list) and leagues:
                teams = leagues[0].get("teams", [])
                if isinstance(teams, list):
                    for t in teams:
                        if not isinstance(t, dict):
                            continue
                        team = t.get("team")
                        if isinstance(team, dict):
                            a = team.get("abbreviation")
                            if isinstance(a, str) and a.strip():
                                abbrs.add(a.strip().lower())
    except Exception:
        pass

    # Fallback recursive search
    if not abbrs:
        for d in iter_dicts(team_payload):
            team = d.get("team")
            if isinstance(team, dict):
                a = team.get("abbreviation")
                if isinstance(a, str) and a.strip():
                    abbrs.add(a.strip().lower())

    return sorted(abbrs)


def _iter_dicts(obj: Any):
    """Yield all dicts recursively from nested JSON."""
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from _iter_dicts(v)
    elif isinstance(obj, list):
        for it in obj:
            yield from _iter_dicts(it)

def extract_athletes_from_stats(stats_payload: Dict[str, Any]) -> List[Tuple[str, str]]:
    """
    Robustly extract (athlete_id, athlete_display_name) from ESPN stats payload.

    Primary path:
      categories[*].leaders[*].athlete.{id,displayName}

    Fallback:
      recursively find any dict that contains an 'athlete' dict with id+displayName
      OR any dict itself that looks like an athlete object (id+displayName + optional uid/guid).
    """
    found: Set[Tuple[str, str]] = set()

    # ---- Primary path (fast) ----
    categories = stats_payload.get("categories")
    if isinstance(categories, list):
        for category in categories:
            if not isinstance(category, dict):
                continue
            leaders = category.get("leaders")
            if not isinstance(leaders, list):
                continue
            for leader in leaders:
                if not isinstance(leader, dict):
                    continue
                athlete = leader.get("athlete")
                if not isinstance(athlete, dict):
                    continue
                athlete_id = athlete.get("id")
                name = athlete.get("displayName")
                if athlete_id is not None and name is not None:
                    athlete_id_s = str(athlete_id).strip()
                    name_s = str(name).strip()
                    if athlete_id_s and name_s:
                        found.add((athlete_id_s, name_s))

    if found:
        return sorted(found)

    # ---- Fallback path (schema drift / older seasons) ----
    for d in _iter_dicts(stats_payload):
        # Case A: dict has an 'athlete' sub-dict
        athlete = d.get("athlete")
        if isinstance(athlete, dict):
            athlete_id = athlete.get("id")
            name = athlete.get("displayName")
            if athlete_id is not None and name is not None:
                athlete_id_s = str(athlete_id).strip()
                name_s = str(name).strip()
                if athlete_id_s and name_s:
                    found.add((athlete_id_s, name_s))

        # Case B: dict itself looks like an athlete object
        # (ESPN athlete dicts often include id + displayName + (uid or guid))
        if "id" in d and "displayName" in d and ("uid" in d or "guid" in d):
            athlete_id = d.get("id")
            name = d.get("displayName")
            if athlete_id is not None and name is not None:
                athlete_id_s = str(athlete_id).strip()
                name_s = str(name).strip()
                if athlete_id_s and name_s:
                    found.add((athlete_id_s, name_s))

    return sorted(found)


def load_existing_ids(csv_path: str) -> Set[str]:
    """Load existing athlete_id values so the script can resume."""
    ids: Set[str] = set()
    if not os.path.exists(csv_path):
        return ids

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        # tolerate older/variant headers by checking a few possibilities
        for row in reader:
            if not row:
                continue
            athlete_id = row.get("athlete_id") or row.get("id") or row.get("athleteId")
            if athlete_id:
                ids.add(str(athlete_id))
    return ids


def count_duplicate_display_names(csv_path: str) -> int:
    """Count how many display names map to multiple athlete IDs in the CSV."""
    if not os.path.exists(csv_path):
        return 0

    name_to_ids: Dict[str, Set[str]] = {}
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            athlete_id = str(row.get("athlete_id") or "").strip()
            display_name = str(row.get("athlete_display_name") or "").strip()
            if not athlete_id or not display_name:
                continue
            name_to_ids.setdefault(display_name, set()).add(athlete_id)

    return sum(1 for ids in name_to_ids.values() if len(ids) > 1)


def ensure_csv_header(csv_path: str) -> None:
    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
        return
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["athlete_id", "athlete_display_name"])


def append_rows(csv_path: str, rows: List[Tuple[str, str]]) -> None:
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for athlete_id, name in rows:
            w.writerow([athlete_id, name])
        f.flush()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="unique_athletes.csv", help="Shared output CSV path.")
    ap.add_argument("--timeout", type=float, default=20.0, help="Per-request timeout seconds.")
    ap.add_argument("--retries", type=int, default=3, help="Retries per request.")
    ap.add_argument("--min-delay-ms", type=int, default=200, help="Minimum delay between requests in ms (>=200).")
    args = ap.parse_args()

    min_delay_s = max(0.200, args.min_delay_ms / 1000.0)

    ensure_csv_header(args.out)
    seen_ids = load_existing_ids(args.out)
    existing_duplicate_names = count_duplicate_display_names(args.out)
    print(f"[info] Loaded {len(seen_ids)} existing athlete_ids from {args.out}", file=sys.stderr)
    print(f"[info] Existing duplicate display names in CSV: {existing_duplicate_names}", file=sys.stderr)

    limiter = RateLimiter(min_delay_s)

    session = requests.Session()
    session.headers.update(DEFAULT_HEADERS)

    # 1) Fetch all NFL teams (one request)
    try:
        teams_payload = fetch_json(session, limiter, TEAM_LIST_URL, timeout_s=args.timeout, retries=args.retries)
    except Exception as e:
        print(f"[error] Could not fetch team list: {e}", file=sys.stderr)
        return 2

    team_abbrs = extract_team_abbrs(teams_payload)
    if not team_abbrs:
        print("[error] No teams found in team list response (schema may have changed).", file=sys.stderr)
        return 3

    print(f"[info] Found {len(team_abbrs)} teams", file=sys.stderr)

    # 2) Iterate seasons/teams, collect uniques, append incrementally
    newly_added = 0
    total_requests = 0
    start = time.time()

    # Write in small batches to reduce file open/close overhead but keep safety.
    batch: List[Tuple[str, str]] = []
    BATCH_FLUSH = 200  # flush every N new athletes

    for year in range(YEAR_START, YEAR_END + 1):
        for seasontype in SEASON_TYPES:
            for abbr in team_abbrs:
                url = build_stats_url(abbr, year, seasontype)
                try:
                    payload = fetch_json(session, limiter, url, timeout_s=args.timeout, retries=args.retries)
                    total_requests += 1
                except Exception as e:
                    total_requests += 1
                    print(f"[warn] request failed year={year} type={seasontype} team={abbr}: {e}", file=sys.stderr)
                    continue

                athletes = extract_athletes_from_stats(payload)
                if not athletes:
                    print("no athletes found, url: ", url)
                    continue

                for athlete_id, name in athletes:
                    if athlete_id in seen_ids:
                        continue
                    seen_ids.add(athlete_id)
                    batch.append((athlete_id, name))
                    newly_added += 1

                    if len(batch) >= BATCH_FLUSH:
                        append_rows(args.out, batch)
                        batch.clear()

        # lightweight progress
        elapsed = time.time() - start
        print(
            f"[info] year {year} done | total_unique={len(seen_ids)} | added={newly_added} | "
            f"requests={total_requests} | elapsed={elapsed:.1f}s",
            file=sys.stderr,
        )

    if batch:
        append_rows(args.out, batch)
        batch.clear()

    elapsed = time.time() - start
    print(
        f"[done] unique athletes={len(seen_ids)} | newly added={newly_added} | requests={total_requests} | elapsed={elapsed:.1f}s",
        file=sys.stderr,
    )
    final_duplicate_names = count_duplicate_display_names(args.out)
    print(f"[done] duplicate display names in CSV: {final_duplicate_names}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
