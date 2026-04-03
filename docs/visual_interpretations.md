# Visual Interpretation: Model Training Summary + Draft Round Production

## Context
These visuals support the same high-level conclusion from the project: we aimed to build a defensive-back recruiting strategy using predictive models, but the current models only capture a small amount of signal. The figures still reveal useful patterns we can leverage in a hybrid scouting process.

## 1) `outputs/visualizations/model_training_summary/model_performance.png`

### What the chart shows
- Test RMSE values are tightly grouped (roughly **398 to 411**), and MAE values are also tightly grouped (roughly **309 to 312**) across the four shown models.
- `gbm_huber` is the best of this set on RMSE and R² (R² ≈ **0.050**), while `gbm_quantile` has slightly negative R² (≈ **-0.014**).
- Even the best R² is close to zero, meaning these models explain only a small fraction of variation in NFL production.

### Interpretation
- Model family choice is not currently the bottleneck. Since all models cluster together, changing algorithms alone is unlikely to produce a major gain.
- This is consistent with the project premise that the data available today does not contain enough predictive signal for a stand-alone recruiting model.

## 2) `outputs/visualizations/model_training_summary/prediction_error_profile.png`

### What the chart shows
- Absolute error boxplots are broad for every model, with medians around **218 to 252** and very long tails.
- The bias bars are negative for all models (mean error `Pred - Actual` is about **-54 to -83**), indicating systematic underprediction.
- 90th percentile absolute errors are very large (about **663 to 705** depending on model), indicating weak reliability for high-end outcomes.

### Interpretation
- The models are especially poor at capturing upside players (the exact players recruiting strategy cares about most).
- Current outputs are better used for coarse risk-tiering than for direct rank-order decisioning.

## 3) `outputs/visualizations/model_training_summary/global_explanation_map.png`

### What the chart shows
- Despite low predictive power, some features repeatedly appear across models:
  - **Shuttle time (`num__shuttle_sec_z`)**
  - **Forty time (`num__forty_yard_sec_z`)**
  - **Weight (`num__weight_lb_z`)**
  - **College interceptions (`num__college_interceptions_z`)**
  - **Combined/assisted tackle production**
- Signs and strength vary by model (expected), but these features are consistently part of the highest-importance band.

### Interpretation
- This map gives actionable screening signals even when overall accuracy is weak:
  - Athletic markers (short-area movement and speed) matter.
  - Ball production and tackle involvement matter.
- However, the feature map should be treated as directional guidance, not a complete decision rule.

## 4) `outputs/visualizations/nfl_production_vs_draft_round.png`

### What the chart shows
- The median NFL production value declines as draft round gets later:
  - Round 1 median ≈ **79.8**
  - Round 2 median ≈ **74.3**
  - Round 3 median ≈ **49.6**
  - Round 4 median ≈ **45.0**
  - Round 5 median ≈ **32.4**
  - Round 6 median ≈ **27.2**
  - Round 7 median ≈ **36.4**
- There is large overlap and heavy dispersion in every round, including high outliers in late rounds.

### Interpretation
- Draft round carries real signal (especially from rounds 1–2 vs later rounds), but it is far from deterministic.
- The wide overlap reinforces that identifying exceptions is difficult with the current feature set—and exactly where better college-context data could add value.

## Overall Takeaways for Recruiting Strategy

1. **Models are not accurate enough to be used alone.**
   - Low R² and large/highly skewed errors make them unreliable as a pure ranking engine.

2. **There are still useful scouting signals.**
   - A small set of features repeatedly appears as important: speed/agility metrics and specific college production stats.

3. **Use a hybrid workflow now.**
   - Use model outputs as one input into scouting (triage, flagging, or disagreement checks), not as final authority.

4. **Best path to improvement: richer college information.**
   - To match the project premise, likely gains will come from adding:
     - game-by-game college production,
     - opponent strength / team strength context,
     - role/usage context (coverage vs box responsibilities),
     - and possibly trajectory features (year-over-year growth).

In short: the visuals confirm that the current system has *some* learnable signal but not enough to support an autonomous recruiting strategy yet. The right next step is better context-rich college data, not only more model complexity.
