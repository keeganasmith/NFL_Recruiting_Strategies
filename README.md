# NFL Recruiting Strategies

## Standardized effect-size dotplot workflow

`src/visuals/effect_size_dotplot.py` builds a small-multiples effect-size chart that summarizes how each standardized combine metric relates to modeled NFL production value **within each position group**.

### What this visualization is doing

- Splits the figure into one panel per `position_group`.
- Plots one dot per combine `metric`, where the x-position is the `estimate` (standardized marginal effect size).
- Draws horizontal uncertainty bars from `ci_low` to `ci_high` for each metric.
- Colors dots by sign: blue for positive effects and red for negative effects.
- Sorts metrics inside each panel by descending `abs(estimate)` so the strongest effects are shown first.
- Draws a vertical reference line at zero, making it easy to see positive vs. negative associations.

### How to read it (what it means)

- **Further from zero** means a larger estimated relationship (in standardized units) between that combine metric and production value for that position.
- **Crossing zero in the interval** suggests higher uncertainty about direction for that metric.
- **Panel-specific ranking** highlights which drills appear most influential for each position group, without implying the same ranking applies across all positions.
- Because estimates are standardized and position-specific, the plot is best interpreted as a **relative signal-strength view** rather than a direct causal statement.

### Required input format

`effect_size_dotplot.py` expects a tidy five-column effect table with:

- `position_group`: modeled position bucket used for paneling
- `metric`: combine metric label shown on the y-axis
- `estimate`: standardized marginal effect estimate
- `ci_low`: lower confidence interval bound
- `ci_high`: upper confidence interval bound

This format is intentionally narrow so chart code stays simple and reusable regardless of model internals.

### 1) Run modeling to produce feature effects

```bash
python - <<'PY'
import pandas as pd
from src.modeling.position_models import run_position_modeling_workflow

df = pd.read_csv("NFL_data/combine_with_college_stats.csv", low_memory=False)
run_position_modeling_workflow(df, output_dir="outputs/modeling")
PY
```

### 2) Convert model outputs to the tidy five-column CSV

Use the runner file `src/visuals/prepare_effects_tidy.py`:

```bash
python src/visuals/prepare_effects_tidy.py outputs/modeling/feature_effects.csv \
  --output-csv outputs/model_effects/standardized_effects.csv
```

You can also point it directly at raw player data and it will automatically run modeling first, then build the tidy five-column file:

```bash
python src/visuals/prepare_effects_tidy.py NFL_data/combine_with_college_stats.csv \
  --model-output-dir outputs/modeling \
  --output-csv outputs/model_effects/standardized_effects.csv
```

By default this keeps standardized combine features (`*_z`) and excludes pooled/intercept/missing-indicator rows.

> Note on missing college stats: the position modeling workflow does **not** require college-stat columns for target construction. It uses the production-value components in `src/scoring/config/production_value_config.json` (e.g., starts, approximate_value, snap_share, seasons_active) with configured imputation for missing values. High missingness in college-stat fields therefore does not directly break this analysis pipeline.

### 3) Build the chart

```bash
python src/visuals/effect_size_dotplot.py outputs/model_effects/standardized_effects.csv \
  --heuristic-version 1 --model-version 1
```

Outputs:

- `outputs/visualizations/effect_size_dotplot.svg`
- `outputs/visualizations/effect_size_dotplot.pdf`

## Heuristic experiment pipeline

A modular experiment package now lives under `pipeline/` with clear stages:

- `pipeline/io/`: input loading and schema validation
- `pipeline/features/`: shared preprocessing / feature preparation
- `pipeline/heuristics/`: heuristic interface and pluggable implementations
- `pipeline/evaluation/`: metrics, calibration, and ranking analysis
- `pipeline/reporting/`: persisted tables/artifacts for each run
- `configs/`: heuristic configuration files

### Run an experiment

```bash
python -m pipeline.run_experiment \
  --input-data all_data.csv \
  --heuristic-config configs/default_heuristic.json \
  --output-dir outputs/experiments/default_run
```

Optional split/time context arguments (for reproducibility metadata):

```bash
python -m pipeline.run_experiment \
  --input-data all_data.csv \
  --heuristic-config configs/default_heuristic.json \
  --output-dir outputs/experiments/time_split_run \
  --seed 7 \
  --time-column combine_year \
  --train-end-year 2018 \
  --target-column scoring_totalPoints \
  --calibration-bins 10 \
  --top-k 25
```

Each run writes:

- `scored_players.csv`
- `metrics.json`
- `calibration.csv`
- `ranking_topk.csv`
- `manifest.json` with timestamp, git hash (if available), heuristic+params, data slice details, and evaluation settings.
