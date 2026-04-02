# Modeling Target Data Contract

## 1) Target definition

- **Target column name:** `NFL_production_value`.
- **Source file validated:** `outputs/pipeline/sweeps/db_heuristic_grid_search/best_heuristic_scored_players.csv`.
- **Observed type:** continuous numeric (`float`), not categorical tiers.
- **Units:** weighted composite NFL production points produced by the project heuristic (higher = more early-career NFL production value).

### Classification vs regression decision

Because `NFL_production_value` is continuous (many unique real-valued outputs) and no native tier label column is present in this file, the default supervised task is:

- **Primary task:** regression.

If downstream consumers later require tiers, they should be created as a **derived view** from the continuous target (for example quantile bins), with threshold definitions versioned separately.

## 2) Prediction-time assumption (must hold for training and inference)

- **Prediction time = pre-draft only.**
- At prediction time, features are restricted to information available before an NFL player is drafted:
  - combine measurements/anthropometrics
  - college performance/context fields
- Any field that is only known after entering the NFL is disallowed.

## 3) Eligible feature families

The model input contract allows only these feature groups:

1. **Combine / anthropometric fields** (examples):
   - `40yd`, `Vertical`, `Bench`, `Broad Jump`, `3Cone`, `Shuttle`, `Ht`, `Wt`
2. **College fields** (examples):
   - `college_games`, `college_combined_tackles`, `college_solo_tackles`,
     `college_assisted_tackles`, `college_tfl`, `college_sacks`,
     `college_interceptions`, `college_passes_defended`, `college_conference`,
     `college_class`, `college_final_season_year`
3. **Optional identity/context fields known pre-draft** (if needed for joins/grouping only):
   - `Player`, `Pos`, `School`, `College`, `combine_year`

## 4) Explicitly excluded leakage fields

The following must be excluded from predictors because they encode post-draft/NFL outcomes or otherwise leak future information relative to pre-draft prediction time:

1. **All NFL box-score/stat families**, including but not limited to prefixes:
   - `defensive_*`, `scoring_*`, `rushing_*`, `receiving_*`, `returning_*`,
     `passing_*`, `punting_*`, `kicking_*`
2. **Direct or proxy NFL outcome targets:**
   - `NFL_production_value` (target only, never a feature)
   - `season_year`, `career_year`, `teamId`, `teamSlug`, `position`, `Player-additional`
3. **Draft outcome fields (not pre-draft signals):**
   - `Drafted (tm/rnd/yr)`
4. **Operational/scrape metadata not intended as football signal:**
   - `college_scraped_at`, `college_page_url`, `college_expected_draft_year_source`, `college_source_type`

## 5) Enforcement rule

Before fitting any model, enforce:

- feature allowlist = combine + college families only
- denylist = all fields in Section 4
- target = `NFL_production_value`

Any run that violates this contract should fail validation before training starts.
