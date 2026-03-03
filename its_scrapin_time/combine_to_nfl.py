#!/usr/bin/env python3
"""
combine_to_espn_stats.py

Adds:
- Updates each combine CSV in-place (or to --combine_out_dir) with a new column: NFL_id
  - NFL_id = matched athlete_id
  - NFL_id = "N/A" if no confident match
- Still produces the big merged output CSV with ESPN stats for drafted players
- Uses SQLite cache to avoid redundant ESPN fetches
- Enforces >= 200ms delay between network requests

Now includes progress/debug prints:
- Overall file progress
- Per-file matching progress (every N rows)
- Drafted stats progress (cached vs fetched) + overall ETA-ish counters
- Periodic “still alive” summaries

Example:
  python combine_to_espn_stats.py \
    --combine_glob "combine_data/*.csv" \
    --athletes_csv "athletes.csv" \
    --out_csv "combine_with_stats.csv"

Optional:
  --combine_out_dir "combine_updated"   # write updated combine CSVs there instead of in-place
  --print_every 500                      # progress print frequency for matching
  --print_every_fetch 50                 # progress print frequency for drafted stats
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sqlite3
import time
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
import requests


ESPN_URL_TMPL = "https://site.web.api.espn.com/apis/common/v3/sports/football/nfl/athletes/{athlete_id}/stats"
DEFAULT_PARAMS = {"region": "us", "lang": "en", "contentorigin": "espn"}

_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}
_PUNCT_RE = re.compile(r"[^a-z0-9\s]")


def normalize_name(name: str) -> str:
    if name is None:
        return ""
    s = str(name).strip().lower()
    s = _PUNCT_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    parts = s.split()
    if parts and parts[-1] in _SUFFIXES:
        parts = parts[:-1]
    return " ".join(parts)


def last_name(norm_name: str) -> str:
    parts = norm_name.split()
    return parts[-1] if parts else ""


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


@dataclass
class Athlete:
    athlete_id: int
    display_name: str
    norm_name: str


def load_athletes(athletes_csv: str) -> Tuple[Dict[str, List[Athlete]], Dict[str, List[Athlete]]]:
    exact: Dict[str, List[Athlete]] = {}
    by_last: Dict[str, List[Athlete]] = {}

    df = pd.read_csv(athletes_csv)
    for _, row in df.iterrows():
        aid = int(row["athlete_id"])
        dn = str(row["athlete_display_name"])
        nn = normalize_name(dn)
        if not nn:
            continue
        ath = Athlete(athlete_id=aid, display_name=dn, norm_name=nn)
        exact.setdefault(nn, []).append(ath)
        ln = last_name(nn)
        by_last.setdefault(ln, []).append(ath)

    duplicate_name_count = sum(1 for athletes in exact.values() if len(athletes) > 1)
    print(
        f"[INFO] Loaded {sum(len(v) for v in exact.values()):,} athletes and "
        f"{len(exact):,} normalized names from {athletes_csv} "
        f"({duplicate_name_count:,} duplicate-name keys)"
    )
    return exact, by_last


def normalize_position(pos: str) -> str:
    if pos is None:
        return ""
    p = str(pos).strip().upper()
    if not p:
        return ""

    groups = {
        "OL": {"OL", "OT", "OG", "C", "G", "T"},
        "DL": {"DL", "DE", "DT", "NT"},
        "DB": {"DB", "CB", "S", "FS", "SS"},
        "LB": {"LB", "ILB", "OLB", "MLB"},
        "RB": {"RB", "HB", "FB"},
        "WR": {"WR"},
        "TE": {"TE"},
        "QB": {"QB"},
        "K": {"K", "PK"},
        "P": {"P"},
        "LS": {"LS"},
    }
    for grp, vals in groups.items():
        if p in vals:
            return grp
    return p


def extract_athlete_meta(payload: dict) -> dict:
    seasons: Set[int] = set()
    positions: Set[str] = set()
    for cat in payload.get("categories", []):
        for entry in cat.get("statistics", []):
            season = entry.get("season", {}) or {}
            year = season.get("year")
            if isinstance(year, int) and year > 0:
                seasons.add(year)
            norm_pos = normalize_position(entry.get("position", ""))
            if norm_pos:
                positions.add(norm_pos)

    debut_year = min(seasons) if seasons else None
    return {
        "debut_year": debut_year,
        "positions": positions,
    }


def score_candidate(
    combine_year: Optional[int],
    combine_pos: str,
    candidate_meta: dict,
) -> float:
    score = 0.0

    debut_year = candidate_meta.get("debut_year")
    if combine_year is not None and debut_year is not None:
        if combine_year <= debut_year <= combine_year + 3:
            score += 4.0
        else:
            score -= min(abs(debut_year - combine_year) * 0.25, 3.0)

    cpos = normalize_position(combine_pos)
    cmeta_positions = candidate_meta.get("positions", set())
    if cpos and cmeta_positions:
        if cpos in cmeta_positions:
            score += 4.0
        else:
            score -= 4.0

    return score


def is_temporally_plausible(combine_year: Optional[int], candidate_meta: dict) -> bool:
    debut_year = candidate_meta.get("debut_year")
    if combine_year is None or debut_year is None:
        return True
    # A combine entrant should usually debut around the combine year.
    # Allow small drift for redshirt/roster timing noise, but block obvious mislinks.
    return (combine_year - 1) <= debut_year <= (combine_year + 4)


def pick_best_match(
    combine_player_name: str,
    combine_year: Optional[int],
    combine_pos: str,
    exact_index: Dict[str, List[Athlete]],
    last_index: Dict[str, List[Athlete]],
    athlete_meta: Dict[int, dict],
    min_ratio: float = 0.92,
) -> Tuple[Optional[Athlete], float, str]:
    cn = normalize_name(combine_player_name)
    if not cn:
        return None, 0.0, "empty"

    if cn in exact_index:
        exact_matches = exact_index[cn]
        if len(exact_matches) == 1:
            only = exact_matches[0]
            meta = athlete_meta.get(only.athlete_id, {})
            if not is_temporally_plausible(combine_year, meta):
                return None, 1.0, "exact_implausible"
            return only, 1.0, "exact"

        cpos = normalize_position(combine_pos)

        def has_position_match(ath: Athlete) -> bool:
            if not cpos:
                return False
            meta_positions = athlete_meta.get(ath.athlete_id, {}).get("positions", set())
            return bool(meta_positions) and cpos in meta_positions

        has_any_pos_match = any(has_position_match(a) for a in exact_matches)
        scored: List[Tuple[float, Athlete]] = []
        for a in exact_matches:
            if has_any_pos_match and cpos:
                meta_positions = athlete_meta.get(a.athlete_id, {}).get("positions", set())
                if meta_positions and cpos not in meta_positions:
                    # If we have at least one exact-name candidate whose tracked NFL
                    # position matches the combine position, ignore exact-name
                    # candidates that confidently disagree on position.
                    continue
            scored.append((score_candidate(combine_year, combine_pos, athlete_meta.get(a.athlete_id, {})), a))

        ranked = sorted(scored, key=lambda x: x[0], reverse=True)
        if not ranked:
            return None, 1.0, "exact_ambiguous"

        top_score, top_athlete = ranked[0]
        second_score = ranked[1][0] if len(ranked) > 1 else float("-inf")
        if top_score - second_score >= 1.0 and top_score >= 0.0:
            return top_athlete, 1.0, "exact_disambiguated"
        return None, 1.0, "exact_ambiguous"

    ln = last_name(cn)
    candidates = last_index.get(ln, [])

    best = None
    best_r = 0.0
    for ath in candidates:
        r = similarity(cn, ath.norm_name)
        r += 0.02 * score_candidate(combine_year, combine_pos, athlete_meta.get(ath.athlete_id, {}))
        if r > best_r:
            best_r = r
            best = ath

    if best and best_r >= min_ratio:
        return best, best_r, "fuzzy_lastname"

    if ln:
        first = ln[0]
        expanded: List[Athlete] = []
        for k, v in last_index.items():
            if k and k[0] == first:
                expanded.extend(v)

        for ath in expanded:
            r = similarity(cn, ath.norm_name)
            r += 0.02 * score_candidate(combine_year, combine_pos, athlete_meta.get(ath.athlete_id, {}))
            if r > best_r:
                best_r = r
                best = ath

        if best and best_r >= (min_ratio + 0.02):
            return best, best_r, "fuzzy_broad"

    return None, best_r, "no_match"


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
    timeout_sec: float = 30.0,
) -> dict:
    limiter.wait()
    url = ESPN_URL_TMPL.format(athlete_id=athlete_id)
    resp = session.get(url, params=DEFAULT_PARAMS, timeout=timeout_sec)
    resp.raise_for_status()
    return resp.json()


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


def infer_year_from_filename(path: str) -> Optional[int]:
    base = os.path.basename(path)
    m = re.search(r"(19|20)\d{2}", base)
    return int(m.group(0)) if m else None


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


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
    ap.add_argument("--combine_glob", required=True, help='Glob for yearly combine CSVs, e.g. "combine/*.csv"')
    ap.add_argument("--athletes_csv", required=True, help="CSV with columns athlete_id, athlete_display_name")
    ap.add_argument("--out_csv", default="combine_with_stats.csv")
    ap.add_argument("--unmatched_csv", default="unmatched_players.csv")
    ap.add_argument("--cache_db", default="espn_cache.sqlite")
    ap.add_argument("--min_delay_ms", type=int, default=200, help="Minimum delay between network requests")
    ap.add_argument("--min_match_ratio", type=float, default=0.92)
    ap.add_argument(
        "--combine_out_dir",
        default="",
        help="If set, write updated combine CSVs here instead of editing in place.",
    )
    ap.add_argument("--print_every", type=int, default=500, help="Progress print frequency while matching NFL_id")
    ap.add_argument("--print_every_fetch", type=int, default=50, help="Progress print frequency while fetching stats")
    args = ap.parse_args()

    t0 = time.time()

    combine_files = sorted(glob.glob(args.combine_glob))
    if not combine_files:
        raise SystemExit(f"No files matched: {args.combine_glob}")

    print(f"[INFO] Found {len(combine_files)} combine CSV files")

    exact_index, last_index = load_athletes(args.athletes_csv)
    conn = init_cache(args.cache_db)

    limiter = RateLimiter(args.min_delay_ms / 1000.0)
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) combine_to_espn_stats/1.0",
            "Accept": "*/*",
            "Origin": "https://www.espn.com",
            "Referer": "https://www.espn.com/",
        }
    )

    all_output_rows: List[dict] = []
    unmatched_rows: List[dict] = []
    athlete_meta: Dict[int, dict] = {}

    # Global counters for stats section
    total_drafted_rows_seen = 0
    total_with_nfl_id = 0
    total_cache_hits = 0
    total_network_fetches = 0
    total_fetch_failures = 0

    if args.combine_out_dir:
        ensure_dir(args.combine_out_dir)
        print(f"[INFO] Writing updated combine files to: {args.combine_out_dir}")
    else:
        print("[INFO] Updating combine files in-place")

    for file_i, fp in enumerate(combine_files, start=1):
        year = infer_year_from_filename(fp)
        file_name = os.path.basename(fp)

        print(f"\n[FILE {file_i}/{len(combine_files)}] Processing {file_name} (year={year if year else 'unknown'})")

        df = pd.read_csv(fp)

        # Ensure NFL_id column exists and is STRING dtype (so it can hold "N/A" and numeric ids)
        if "NFL_id" not in df.columns:
            df["NFL_id"] = "N/A"

        df["NFL_id"] = df["NFL_id"].astype("string")  # <--- IMPORTANT
        n_rows = len(df)
        print(f"[INFO] Rows: {n_rows:,}")

        # Ensure NFL_id column exists
        if "NFL_id" not in df.columns:
            df["NFL_id"] = "N/A"
            print("[INFO] Added missing column: NFL_id")

        drafted_col = "Drafted (tm/rnd/yr)"
        if drafted_col not in df.columns:
            cand = [c for c in df.columns if c.lower().startswith("drafted")]
            if not cand:
                print(f"[WARN] No drafted column in {file_name}; will still update NFL_id, but skip stats output.")
                drafted_col = None
            else:
                drafted_col = cand[0]
                print(f"[INFO] Using drafted column: {drafted_col}")

        # --- Update NFL_id for ALL rows that have a Player name (drafted or not) ---
        t_match0 = time.time()
        matched_count = 0
        na_count = 0
        empty_name_count = 0

        for idx, row in df.iterrows():
            if args.print_every > 0 and (idx + 1) % args.print_every == 0:
                elapsed = time.time() - t_match0
                print(
                    f"[MATCH] {file_name}: {idx+1:,}/{n_rows:,} processed | "
                    f"matched={matched_count:,} N/A={na_count:,} empty={empty_name_count:,} | "
                    f"elapsed={fmt_duration(elapsed)}"
                )

            player_name = str(row.get("Player", "")).strip()
            if not player_name:
                empty_name_count += 1
                continue

            combine_pos = str(row.get("Pos", "")).strip()

            exact_pool = exact_index.get(normalize_name(player_name), [])
            for cand in exact_pool:
                if cand.athlete_id in athlete_meta:
                    continue
                cached_payload = cache_get(conn, cand.athlete_id)
                if cached_payload is None:
                    try:
                        cached_payload = fetch_espn_stats(session, limiter, cand.athlete_id)
                        cache_put(conn, cand.athlete_id, cached_payload)
                    except Exception:
                        athlete_meta[cand.athlete_id] = {}
                        continue
                athlete_meta[cand.athlete_id] = extract_athlete_meta(cached_payload)

            ath, ratio, method = pick_best_match(
                player_name,
                year,
                combine_pos,
                exact_index,
                last_index,
                athlete_meta,
                min_ratio=args.min_match_ratio,
            )
            if ath:
                df.at[idx, "NFL_id"] = str(ath.athlete_id)
                matched_count += 1
            else:
                df.at[idx, "NFL_id"] = "N/A"
                na_count += 1

        print(
            f"[MATCH DONE] {file_name}: matched={matched_count:,} N/A={na_count:,} empty={empty_name_count:,} "
            f"in {fmt_duration(time.time() - t_match0)}"
        )

        # Write updated combine file (in-place or to out dir)
        out_path = fp
        if args.combine_out_dir:
            out_path = os.path.join(args.combine_out_dir, file_name)
        df.to_csv(out_path, index=False)
        print(f"[WRITE] Updated combine CSV -> {out_path}")

        # --- Stats output ONLY for drafted players ---
        if drafted_col is None:
            continue

        df_drafted = df[df[drafted_col].notna() & (df[drafted_col].astype(str).str.strip() != "")]
        drafted_n = len(df_drafted)
        print(f"[DRAFTED] {file_name}: drafted rows={drafted_n:,}")

        if drafted_n == 0:
            continue

        t_stats0 = time.time()

        for j, (_, r) in enumerate(df_drafted.iterrows(), start=1):
            total_drafted_rows_seen += 1

            if args.print_every_fetch > 0 and j % args.print_every_fetch == 0:
                elapsed = time.time() - t_stats0
                print(
                    f"[STATS] {file_name}: {j:,}/{drafted_n:,} drafted processed | "
                    f"cache_hits={total_cache_hits:,} fetched={total_network_fetches:,} failures={total_fetch_failures:,} | "
                    f"elapsed={fmt_duration(elapsed)} (file) total_elapsed={fmt_duration(time.time() - t0)}"
                )

            player_name = str(r.get("Player", "")).strip()
            if not player_name:
                continue

            nfl_id_val = str(r.get("NFL_id", "N/A")).strip()
            if not nfl_id_val.isdigit():
                unmatched_rows.append(
                    {
                        "combine_file": file_name,
                        "combine_year": year if year is not None else "",
                        "combine_player": player_name,
                        "best_ratio_seen": "",
                        "note": "no confident match (NFL_id=N/A)",
                    }
                )
                continue

            total_with_nfl_id += 1
            athlete_id = int(nfl_id_val)

            payload = cache_get(conn, athlete_id)
            if payload is None:
                try:
                    payload = fetch_espn_stats(session, limiter, athlete_id)
                    total_network_fetches += 1
                    if total_network_fetches <= 5:
                        print(f"[FETCH] athlete_id={athlete_id} (network) example_fetch#{total_network_fetches}")
                except Exception as e:
                    total_fetch_failures += 1
                    unmatched_rows.append(
                        {
                            "combine_file": file_name,
                            "combine_year": year if year is not None else "",
                            "combine_player": player_name,
                            "matched_athlete_id": athlete_id,
                            "note": f"fetch_failed: {type(e).__name__}: {e}",
                        }
                    )
                    continue
                cache_put(conn, athlete_id, payload)
            else:
                total_cache_hits += 1

            stat_rows = flatten_espn_payload_to_rows(payload)

            combine_info = r.to_dict()
            combine_info["combine_file"] = file_name
            combine_info["combine_year"] = year if year is not None else ""

            for sr in stat_rows:
                out = dict(combine_info)
                out.update(sr)
                all_output_rows.append(out)

        print(f"[STATS DONE] {file_name}: processed drafted rows in {fmt_duration(time.time() - t_stats0)}")

    # Final writes
    if unmatched_rows:
        pd.DataFrame(unmatched_rows).to_csv(args.unmatched_csv, index=False)
        print(f"[WRITE] Unmatched / failed fetch list -> {args.unmatched_csv} ({len(unmatched_rows):,} rows)")

    if not all_output_rows:
        print("⚠️  No stats output rows generated (maybe no drafted players, no matches, or missing drafted column).")
    else:
        out_df = pd.DataFrame(all_output_rows)

        front_cols = [
            "combine_year",
            "combine_file",
            "Player",
            "NFL_id",
            "Pos",
            "School",
            "College",
            "Ht",
            "Wt",
            "40yd",
            "Vertical",
            "Bench",
            "Broad Jump",
            "3Cone",
            "Shuttle",
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

    # Summary
    print("\n========== SUMMARY ==========")
    print(f"Total time: {fmt_duration(time.time() - t0)}")
    print(f"Drafted rows processed: {total_drafted_rows_seen:,}")
    print(f"Drafted rows with NFL_id: {total_with_nfl_id:,}")
    print(f"Cache hits: {total_cache_hits:,}")
    print(f"Network fetches: {total_network_fetches:,}")
    print(f"Fetch failures: {total_fetch_failures:,}")
    print(f"Cache DB: {args.cache_db}")
    if args.combine_out_dir:
        print(f"Updated combine CSVs: {args.combine_out_dir}")
    else:
        print("Updated combine CSVs: in-place")
    if unmatched_rows:
        print(f"Unmatched/failures CSV: {args.unmatched_csv}")
    print("=============================")


if __name__ == "__main__":
    main()
