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

## NFL production-value heuristic pipeline

The `pipeline/` package is a configurable experiment runner focused on generating an `NFL_production_value` feature and evaluating how informative that heuristic is as a production proxy (not combine talent grading).

### Package layout

- `pipeline/io/`: load + validate `all_data.csv`
- `pipeline/features/`: shared preprocessing and derived features used by heuristics
- `pipeline/heuristics/`: heuristic interface, registry/factory, and implementations
- `pipeline/evaluation/`: score diagnostics, calibration, and rank analysis
- `pipeline/reporting/`: plots + summary table writers
- `configs/`: experiment/heuristic config files

### Run a baseline experiment

```bash
python -m pipeline.run_experiment \
  --input-data all_data.csv \
  --heuristic-config configs/heuristics/nfl_production_value_baseline.json \
  --output-dir outputs/pipeline/default_run \
  --split-seed 42
```

Optional time split (for forward-looking rank stability):

```bash
python -m pipeline.run_experiment \
  --input-data all_data.csv \
  --heuristic-config configs/heuristics/nfl_production_value_baseline.json \
  --output-dir outputs/pipeline/time_split_2018 \
  --time-split-year 2018
```

### Heuristic sweep objective (why this function)

When sweeping heuristic configs, we rank each candidate with a composite objective:

`objective_score = 0.55*proxy_spearman + 0.25*rank_corr + 0.20*top_overlap - 0.15*calibration_rmse`

This was chosen to balance four practical needs:

- **Primary signal quality (proxy_spearman, 55%)**: the score should be strongly monotonic with an independent production proxy rather than only produce large raw values.
- **Temporal ranking robustness (rank_corr, 25%)**: we want top prospects to remain similarly ordered across train/test eras, reducing overfit to a single slice.
- **Top-board consistency (top_overlap, 20%)**: for recruiting use-cases, overlap in the top of the board matters more than tiny reorderings deep in the list.
- **Calibration penalty (calibration_rmse, -15%)**: heuristics that separate players but distort relative bin quality are penalized.

Weights are intentionally interpretable and sum to 1.0 on the positive terms so the objective can be tuned transparently in future experiments.

### First-contract window (first 4 NFL seasons)

All `pipeline/` heuristic experiments now compute `NFL_production_value` only on each player's **first four NFL seasons** (`career_year = season_year - combine_year + 1 <= 4`). This aligns with typical rookie-contract decision windows and makes comparisons more relevant to recruiting and early-career projection.

### Outputs produced per run

- `scored_players.csv`: full dataset with derived features + `NFL_production_value`
- `score_summary.csv`: overall scoring distribution metrics
- `calibration_table.csv`: quantile bins of score vs. proxy NFL outcome
- `ranking_stability.csv`: top-N overlap / Spearman rank stability across train/test split
- `top_players.csv`: top ranked players under the configured heuristic
- `plots/score_distribution.png`
- `plots/calibration_curve.png`
- `experiment_manifest.json`: reproducibility manifest with timestamp, git commit, heuristic metadata, data slice details, and evaluation settings

### Swapping heuristics without changing pipeline code

Heuristics are created by key through `pipeline/heuristics/registry.py`. To swap heuristics, provide a different config file with a different `heuristic_key` and parameter block.

## Explainable model-training pipeline for `db_model_preprocessed.csv`

A dedicated training script is now available at `src/modeling/train_explainable_models.py` for supervised prediction of `NFL_production_value`.

### Why these modeling choices

To keep outputs explainable while still tuning performance, the pipeline trains and fine-tunes three interpretable model families:

- **Ridge regression**: stable linear baseline with coefficient-based interpretation.
- **Elastic Net regression**: sparse linear model for feature selection + coefficient interpretation.
- **Decision tree regressor (depth-constrained)**: non-linear model with transparent split logic and global feature-importance output.

### Data split policy

The script respects the existing `dataset_split` column:

- `train` + `val` rows are used for tuning.
- `test` rows are held out for final evaluation only.

If `val` rows are present, tuning uses a `PredefinedSplit` so hyperparameter selection is anchored to the provided validation partition. If `val` rows are absent, it falls back to 5-fold CV on train rows.

### Leakage controls

- The target (`NFL_production_value`) and split label (`dataset_split`) are excluded from model features.
- Numeric/categorical preprocessing (imputation, scaling/encoding) is encapsulated inside sklearn `Pipeline`/`ColumnTransformer` objects so transforms are fit only on train/validation data during CV.

### How to run

```bash
python src/modeling/train_explainable_models.py \
  --input-csv db_model_preprocessed.csv \
  --output-dir outputs/model_training
```

### Artifacts produced

- `model_metrics.csv`: best CV score + held-out test RMSE/MAE/R² + best hyperparameters.
- `<model>_global_explanations.csv`: global explainability export (coefficients or importances).
- `<model>_test_predictions.csv`: test-set predictions for each model.

