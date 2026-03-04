"""Build an interactive overperformer quadrant visualization from combine and NFL stats."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.position_assignment import filter_rows_to_primary_nfl_position

DATA_PATH = Path("NFL_data/combine_with_stats.csv")
OUTPUT_HTML = Path("outputs/visualizations/overperformer_quadrant.html")

COMBINE_METRICS = ["40yd", "Vertical", "Bench", "Broad Jump", "3Cone", "Shuttle", "Ht", "Wt"]
TIMING_METRICS = {"40yd", "3Cone", "Shuttle"}

PASSING_STATS = [
    "passing_passingYards",
    "passing_passingTouchdowns",
    "passing_completionPct",
    "passing_QBRating",
    "passing_adjQBR",
    "passing_yardsPerPassAttempt",
]
RUSH_REC_STATS = [
    "rushing_rushingYards",
    "rushing_rushingTouchdowns",
    "rushing_yardsPerRushAttempt",
    "receiving_receivingYards",
    "receiving_receivingTouchdowns",
    "receiving_receptions",
    "receiving_yardsPerReception",
]
RETURNING_STATS = ["returning_kickReturnYards", "returning_puntReturnYards"]
DEFENSE_STATS = [
    "defensive_totalTackles",
    "defensive_soloTackles",
    "defensive_sacks",
    "defensive_interceptions",
    "defensive_passesDefended",
    "defensive_fumblesForced",
]
SPECIAL_TEAMS_STATS = [
    "kicking_fieldGoalPct",
    "kicking_totalKickingPoints",
    "punting_netAvgPuntYards",
    "punting_puntsInside20",
    "returning_kickReturnYards",
    "returning_puntReturnYards",
]
OFFENSIVE_LINE_STATS = ["rushing_gamesPlayed", "scoring_gamesPlayed"]
FALLBACK_STATS = ["rushing_gamesPlayed", "receiving_gamesPlayed", "defensive_gamesPlayed"]

QB_POSITIONS = {"QB"}
SKILL_POSITIONS = {"RB", "FB", "WR", "TE", "CB/WR"}
DEFENSE_POSITIONS = {"CB", "DB", "DE", "DL", "DT", "EDGE", "ILB", "LB", "OLB", "S", "SAF"}
SPECIAL_POSITIONS = {"K", "P", "LS"}
OFFENSIVE_LINE_POSITIONS = {"C", "G", "OG", "OL", "OT", "T"}


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>NFL Combine Overperformer Quadrant</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body { font-family: Arial, sans-serif; margin: 24px; }
    .controls { display: grid; grid-template-columns: repeat(4, minmax(200px, 1fr)); gap: 12px; margin-bottom: 16px; }
    label { font-size: 13px; font-weight: 600; display: flex; flex-direction: column; gap: 4px; }
    #chart { width: 100%; height: 760px; }
  </style>
</head>
<body>
  <h2>NFL Overperformer Quadrant</h2>
  <div class="controls">
    <label>Position
      <select id="posFilter">
        <option value="ALL">All</option>
      </select>
    </label>
    <label>Draft status
      <select id="draftFilter">
        <option value="ALL">All</option>
        <option value="DRAFTED">Drafted</option>
        <option value="UNDRAFTED">Undrafted</option>
      </select>
    </label>
    <label>Combine year minimum: <span id="yearMinLabel"></span>
      <input id="yearMin" type="range" />
    </label>
    <label>Combine year maximum: <span id="yearMaxLabel"></span>
      <input id="yearMax" type="range" />
    </label>
  </div>
  <div id="chart"></div>

  <script>
    const payload = __PAYLOAD__;
    const posFilter = document.getElementById('posFilter');
    const draftFilter = document.getElementById('draftFilter');
    const yearMin = document.getElementById('yearMin');
    const yearMax = document.getElementById('yearMax');
    const yearMinLabel = document.getElementById('yearMinLabel');
    const yearMaxLabel = document.getElementById('yearMaxLabel');

    payload.positions.forEach(pos => {
      const option = document.createElement('option');
      option.value = pos;
      option.innerText = pos;
      posFilter.appendChild(option);
    });

    [yearMin, yearMax].forEach(el => {
      el.min = payload.year_min;
      el.max = payload.year_max;
      el.step = 1;
    });
    yearMin.value = payload.year_min;
    yearMax.value = payload.year_max;

    function filteredData() {
      let yMin = parseInt(yearMin.value, 10);
      let yMax = parseInt(yearMax.value, 10);
      if (yMin > yMax) {
        [yMin, yMax] = [yMax, yMin];
      }
      yearMinLabel.innerText = yMin;
      yearMaxLabel.innerText = yMax;

      return payload.records.filter(r => {
        const posOk = posFilter.value === 'ALL' || r.Pos === posFilter.value;
        const draftOk = draftFilter.value === 'ALL'
          || (draftFilter.value === 'DRAFTED' && r.is_drafted)
          || (draftFilter.value === 'UNDRAFTED' && !r.is_drafted);
        const yearOk = r.combine_year >= yMin && r.combine_year <= yMax;
        return posOk && draftOk && yearOk;
      });
    }

    function render() {
      const data = filteredData();
      const byPos = {};
      data.forEach(r => {
        if (!byPos[r.Pos]) byPos[r.Pos] = [];
        byPos[r.Pos].push(r);
      });

      const traces = Object.entries(byPos).map(([pos, rows]) => ({
        type: 'scattergl',
        mode: 'markers',
        name: pos,
        x: rows.map(r => r.combine_score),
        y: rows.map(r => r.production_score),
        text: rows.map(r => (
          r.Player + '<br>School: ' + (r.School || 'N/A') +
          '<br>Combine Year: ' + r.combine_year +
          '<br>Draft: ' + (r['Drafted (tm/rnd/yr)'] || 'Undrafted')
        )),
        hovertemplate: '%{text}<extra></extra>',
        marker: { size: 9, opacity: 0.78, color: payload.colors[pos] || '#1f77b4' }
      }));

      const xVals = data.map(d => d.combine_score);
      const yVals = data.map(d => d.production_score);
      const xMin = xVals.length ? Math.min(...xVals, -2.5) : -2.5;
      const xMax = xVals.length ? Math.max(...xVals, 2.5) : 2.5;
      const yMin = yVals.length ? Math.min(...yVals, -2.5) : -2.5;
      const yMax = yVals.length ? Math.max(...yVals, 2.5) : 2.5;

      Plotly.react('chart', traces, {
        title: 'Combine Athleticism vs NFL Production',
        xaxis: { title: 'Normalized Combine Score (z within position)' },
        yaxis: { title: 'Normalized Production Score (z within position)' },
        hovermode: 'closest',
        shapes: [
          { type: 'line', x0: 0, x1: 0, y0: yMin - 0.3, y1: yMax + 0.3, line: { color: 'gray', width: 1, dash: 'dash' } },
          { type: 'line', x0: xMin - 0.3, x1: xMax + 0.3, y0: 0, y1: 0, line: { color: 'gray', width: 1, dash: 'dash' } },
        ],
      }, { responsive: true });
    }

    [posFilter, draftFilter, yearMin, yearMax].forEach(el => el.addEventListener('input', render));
    render();
  </script>
</body>
</html>
"""


def _first_non_empty(series: pd.Series):
    values = series.dropna().astype(str)
    values = values[values.str.strip() != ""]
    return values.iloc[0] if not values.empty else np.nan


def _height_to_inches(value) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    if not text:
        return np.nan
    if "-" in text:
        feet, inches = text.split("-", maxsplit=1)
        if feet.isdigit() and inches.isdigit():
            return int(feet) * 12 + int(inches)
    return pd.to_numeric(text, errors="coerce")


def _zscore(series: pd.Series) -> pd.Series:
    std = series.std(ddof=0)
    if pd.isna(std) or std == 0:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - series.mean()) / std



def _position_stat_columns(pos: str, available_cols: set[str]) -> list[str]:
    if pos in QB_POSITIONS:
        candidates = PASSING_STATS + ["rushing_rushingYards", "rushing_rushingTouchdowns"]
    elif pos == "WR":
        # Keep receiver production focused on offense to avoid special-teams return volume
        # inflating true receiving output for players who handled return duties.
        candidates = RUSH_REC_STATS
    elif pos in SKILL_POSITIONS:
        candidates = RUSH_REC_STATS + RETURNING_STATS
    elif pos in DEFENSE_POSITIONS:
        candidates = DEFENSE_STATS
    elif pos in SPECIAL_POSITIONS:
        candidates = SPECIAL_TEAMS_STATS
    elif pos in OFFENSIVE_LINE_POSITIONS:
        candidates = OFFENSIVE_LINE_STATS
    else:
        candidates = FALLBACK_STATS
    return [c for c in candidates if c in available_cols]


def build_player_level_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["player_key"] = np.where(
        df["NFL_id"].notna() & (df["NFL_id"].astype(str).str.strip() != ""),
        "id_" + df["NFL_id"].astype(str),
        "name_" + df["Player"].astype(str).str.lower().str.replace(r"\s+", " ", regex=True).str.strip(),
    )

    df["Ht"] = df["Ht"].apply(_height_to_inches)
    for col in COMBINE_METRICS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    numeric_cols = [
        c
        for c in df.columns
        if c.startswith(("defensive_", "scoring_", "rushing_", "receiving_", "returning_", "passing_", "punting_", "kicking_"))
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    games_cols = [col for col in numeric_cols if col.endswith("_gamesPlayed")]
    df = filter_rows_to_primary_nfl_position(
        df,
        player_key_col="player_key",
        nfl_position_col="position",
        fallback_position_col="Pos",
        games_played_cols=games_cols,
    )

    agg_spec: dict[str, str] = {col: "sum" for col in numeric_cols}
    agg_spec.update({col: "first" for col in COMBINE_METRICS if col in df.columns})
    agg_spec.update(
        {
            "Player": _first_non_empty,
            "School": _first_non_empty,
            "Pos": _first_non_empty,
            "combine_year": "min",
            "Drafted (tm/rnd/yr)": _first_non_empty,
            "NFL_id": _first_non_empty,
        }
    )

    players = df.groupby("player_key", dropna=False).agg(agg_spec).reset_index(drop=True)
    players["combine_year"] = pd.to_numeric(players["combine_year"], errors="coerce")
    players["is_drafted"] = players["Drafted (tm/rnd/yr)"].notna() & (
        players["Drafted (tm/rnd/yr)"].astype(str).str.strip() != ""
    )

    for metric in TIMING_METRICS:
        if metric in players.columns:
            players[metric] = -players[metric]

    combine_z_columns: list[str] = []
    for col in COMBINE_METRICS:
        if col not in players.columns:
            continue
        z_col = f"{col}_z"
        players[z_col] = players.groupby("Pos")[col].transform(_zscore)
        combine_z_columns.append(z_col)

    players["combine_score"] = players[combine_z_columns].mean(axis=1, skipna=True)

    available_cols = set(players.columns)
    for pos, group_idx in players.groupby("Pos").groups.items():
        pos_cols = _position_stat_columns(str(pos), available_cols)
        if not pos_cols:
            players.loc[group_idx, "production_score"] = np.nan
            continue

        z_cols = []
        for col in pos_cols:
            z_col = f"{col}_z"
            players.loc[group_idx, z_col] = _zscore(players.loc[group_idx, col].fillna(0.0))
            z_cols.append(z_col)
        players.loc[group_idx, "production_score"] = players.loc[group_idx, z_cols].mean(axis=1, skipna=True)

    players["combine_score"] = players.groupby("Pos")["combine_score"].transform(_zscore)
    players["production_score"] = players.groupby("Pos")["production_score"].transform(_zscore)
    return players


def build_interactive_html(players: pd.DataFrame, output_path: Path) -> None:
    viz_df = players[
        [
            "Player",
            "School",
            "Pos",
            "combine_year",
            "Drafted (tm/rnd/yr)",
            "combine_score",
            "production_score",
            "is_drafted",
        ]
    ].copy()
    viz_df = viz_df.dropna(subset=["combine_score", "production_score", "combine_year", "Pos"])

    color_map = {
        pos: color
        for pos, color in zip(sorted(viz_df["Pos"].dropna().unique()), px.colors.qualitative.Alphabet * 5, strict=False)
    }

    payload = {
        "records": viz_df.to_dict(orient="records"),
        "positions": sorted(viz_df["Pos"].dropna().unique().tolist()),
        "year_min": int(viz_df["combine_year"].min()),
        "year_max": int(viz_df["combine_year"].max()),
        "colors": color_map,
    }

    html = HTML_TEMPLATE.replace("__PAYLOAD__", json.dumps(payload))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")


def main() -> None:
    df = pd.read_csv(DATA_PATH, low_memory=False)
    players = build_player_level_dataset(df)
    build_interactive_html(players, OUTPUT_HTML)
    print(f"Wrote visualization to {OUTPUT_HTML}")


if __name__ == "__main__":
    main()
