# Model Training Results Summary (`outputs/model_training`)

## Scope
This summary reviews the training outputs in `outputs/model_training`:

- `model_metrics.csv`
- `*_test_predictions.csv`
- `*_global_explanations.csv`
- `training_manifest.json`

## Executive Summary
- Three models were successfully trained and evaluated on the test split (`decision_tree`, `ridge`, `elastic_net`), each with very similar test performance.
- `decision_tree` achieved the best test RMSE (`392.44`), while `ridge` had slightly better bias characteristics and `decision_tree` had slightly better MAE.
- Overall explanatory power is low across all evaluated models (`test_r2` ~ `0.074` to `0.077`), suggesting substantial unexplained variance in `NFL_production_value`.
- The `autogluon_extreme` run was skipped because `autogluon.tabular` was not installed.

## Data + Feature Configuration (from manifest)
- Target column: `NFL_production_value`
- Split column: `dataset_split`
- Identity columns excluded from modeling: `NFL_id`, `Player`
- Model feature set includes positional/categorical fields plus combine and college production features, with accompanying missingness indicators and z-score normalized variants.

Interpretation note: because both raw and z-score versions of many variables are included, linear model importance tables can surface duplicate signals (for example, both `weight_lb` and `weight_lb_z`).

## Model Performance Comparison

| Model | Best CV RMSE | Test RMSE | Test MAE | Test R² | Status |
|---|---:|---:|---:|---:|---|
| decision_tree | 322.86 | **392.44** | **308.02** | **0.0767** | trained |
| ridge | 324.88 | 392.56 | 309.87 | 0.0761 | trained |
| elastic_net | **316.59** | 393.09 | 311.37 | 0.0736 | trained |
| autogluon_extreme | — | — | — | — | skipped (`autogluon.tabular` not installed) |

### Takeaways
- Test results are tightly clustered: RMSE spread is under 1 point across the three trained models.
- The model selected by CV (`elastic_net`, best CV RMSE) did not translate into best holdout performance.
- Low positive R² indicates models do better than a constant baseline, but only modestly.

## Prediction Error Profile (Test Split)
Using the model-specific `*_test_predictions.csv` files:

| Model | Mean Error (Pred - Actual) | Median Abs Error | P90 Abs Error | Correlation(pred, actual) |
|---|---:|---:|---:|---:|
| decision_tree | -24.09 | 284.62 | 587.27 | **0.293** |
| ridge | -15.87 | **260.74** | 606.81 | 0.279 |
| elastic_net | -18.58 | 261.70 | 610.97 | 0.279 |

Notes:
- All models show a negative mean error (underprediction on average).
- Tail errors are large for all three models (90th percentile absolute errors ~587–611), consistent with low overall R².

## Global Explanation Highlights

### Decision Tree
- Importance is extremely concentrated:
  - `num__combine_year`: `0.629`
  - `num__college_interceptions_z`: `0.371`
- Most other features have zero tree importance in the fitted shallow tree.

### Ridge / Elastic Net
Top absolute coefficients are directionally consistent across both linear models:
- Negative: `combine_year`, `forty_yard_sec`, `three_cone_sec_missing`
- Positive: `college_interceptions` (+ z-score variant), `weight_lb` (+ z-score variant), `college_fumble_recovery_tds` (+ z-score variant)

Interpretation note: coefficient magnitude comparisons are affected by duplicated raw/z-score features and should be treated as directional rather than fully causal/orthogonal effects.

## Operational Observations
- `autogluon_extreme_global_explanations.csv` is present but empty, matching the skip in `model_metrics.csv`.
- AutoGluon predictor directories exist (`autogluon_medium_predictor`, `autogluon_extreme_predictor`), but the reported `autogluon_extreme` training result in this run indicates dependency/environment inconsistency worth cleaning up for reproducibility.

## Recommendations
1. **Resolve AutoGluon environment parity** (install and lock `autogluon.tabular` if that model family is required).
2. **Simplify feature representation** by removing raw/z-score duplicates (or use regularization with grouped selection) to improve interpretability and stability.
3. **Add robust evaluation cuts** (e.g., by position group and cohort year) to understand where underprediction is concentrated.
4. **Inspect target distribution/outliers** and consider transformations or robust losses if long-tail error is structurally present.
5. **Consider richer non-linear models** (GBMs/CatBoost/XGBoost or tuned AutoGluon stack) once environment issues are fixed.
