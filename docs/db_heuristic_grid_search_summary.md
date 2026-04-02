# DB Heuristic Grid Search Summary

Experiment config: `configs/experiments/db_heuristic_grid_search.yaml`.

Scoring window: first 4 NFL seasons (`career_year <= 4`).

Total variants evaluated: 32.

## Top 10 variants by objective score

| variant_id | objective_score | proxy_spearman | mean_score | p90_score |
|---|---|---|---|---|
| db_grid_008 | 0.5294 | 0.9625 | 101.2591 | 242.3700 |
| db_grid_016 | 0.5293 | 0.9623 | 101.6955 | 243.0000 |
| db_grid_004 | 0.5291 | 0.9620 | 99.0363 | 237.3700 |
| db_grid_012 | 0.5290 | 0.9618 | 99.4727 | 237.5800 |
| db_grid_006 | 0.5285 | 0.9609 | 95.3429 | 228.2400 |
| db_grid_014 | 0.5283 | 0.9606 | 95.7794 | 228.6500 |
| db_grid_002 | 0.5280 | 0.9600 | 93.1201 | 224.9200 |
| db_grid_010 | 0.5279 | 0.9598 | 93.5566 | 225.4100 |
| db_grid_024 | 0.5260 | 0.9564 | 125.8936 | 303.2550 |
| db_grid_032 | 0.5259 | 0.9563 | 126.3301 | 303.9300 |

Best variant: **db_grid_008** with objective score **0.5294** and proxy Spearman **0.9625**.

Objective function used:
`0.55*proxy_spearman + 0.25*rank_corr + 0.20*top_overlap - 0.15*calibration_rmse`.

Note: in this time-split run (`time_split_year=2018`), `rank_corr` and `top_overlap` were 0.0 for all variants, so ranking was effectively driven by proxy alignment.
