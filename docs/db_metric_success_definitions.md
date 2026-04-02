# DB metric success definitions

This document defines canonical DB sub-position mapping and Year-3/Year-5 success targets for modeling.

## 1) Canonical DB taxonomy

Allowed mapped labels:

- `CB_outside`
- `CB_slot` (if source data explicitly indicates slot/nickel usage)
- `FS`
- `SS`
- `DB_hybrid` (fallback for ambiguous/multi-role labels)

## 2) Sub-position mapping rules (exact source strings)

Mapping is applied in this source-priority order per row:

1. `pos`
2. `position`
3. `Pos`

First exact match wins. If no exact match is found, map to `DB_hybrid` fallback.

### 2.1 Exact strings mapped to `CB_outside`

- `CB`
- `LCB`
- `RCB`
- `Outside CB`

### 2.2 Exact strings mapped to `CB_slot` (supported when available)

- `CB_slot`
- `SCB`
- `Slot CB`
- `Nickel`
- `NB`

> Note: no explicit slot strings were observed in the current `all_data.csv`, but rules are pre-defined for future data refreshes.

### 2.3 Exact strings mapped to `FS`

- `FS`
- `Free Safety`

### 2.4 Exact strings mapped to `SS`

- `SS`
- `Strong Safety`

### 2.5 Exact strings mapped to `DB_hybrid` fallback

Ambiguous or multi-role labels:

- `S`
- `SAF`
- `DB`
- `CB/DB`
- `DB/CB`
- `S/DB`
- `DB/S`
- `CB/S`
- `LB/S`
- `DB/S/LB`
- `S/CB`
- `CB/DB/S`
- `CB/S/DB`
- `DB/S/CB`
- `S/DB/CB`
- `DB/LB`
- `WR/DB`
- `DB/WR`
- `WR/CB/DB`
- `CB/WR/DB`
- `WR/CB/S`
- `WR/CB`
- `WR/S`
- `WR/LB/S`

Non-DB/other labels retained in mixed pipelines are also forced into fallback for robustness:

- `LB`
- `RB`
- `WR`
- `DL`
- `DT`
- `DL/DT`
- `DL/DE`
- `TE`
- `QB/LB`

## 3) Success targets

Define at least one binary and one continuous target.

### 3.1 Binary target (recommended baseline)

`y_contributor_y3` = 1 if player is a meaningful NFL contributor by end of Year 3, else 0.

Recommended operational definition (choose one and keep fixed within an experiment):

- **Games/starts based:**
  - `games_through_y3 >= 24` **OR**
  - `starts_through_y3 >= 12`
- **Production based (defense):**
  - cumulative defensive snaps, tackles + passes defended + interceptions exceed a threshold tuned on training set percentiles.

Use draft-year + first three NFL seasons as evaluation window.

### 3.2 Continuous targets

Use one or more:

- `starts_through_y3` (count)
- `games_through_y3` (count)
- `starts_through_y5` (count)
- `games_through_y5` (count)
- `av_like_through_y3` or `av_like_through_y5` (Approximate Value-like cumulative production score)

For skewed distributions, apply `log1p` transform before modeling and report raw-scale back-transforms for interpretability.

## 4) Inclusion / exclusion criteria

### 4.1 Include

- Players with valid DB mapping to one of the five canonical labels.
- Players with at least one measurable pre-draft profile row (combine or verified pro-day equivalent when available).

### 4.2 Exclude or flag

- Missing critical identifiers (`Player`, year, or unresolved NFL linkage).
- Implausible anthropometric values (height/weight outliers beyond data-quality cutoffs).
- Players with missing combine participation above threshold:
  - **Hard exclude option:** missing any two of core athletic tests (`40yd`, `Vertical`, `Broad Jump`, `3Cone`, `Shuttle`).
  - **Soft include option:** retain with imputation + missingness indicators; track sensitivity.

### 4.3 Cohort windowing

- Ensure full follow-up horizon exists for outcome windows.
  - Example: Year-3 outcomes require players drafted at least three seasons before data cutoff.
  - Year-5 outcomes require at least five seasons of follow-up.

## 5) Ambiguous labels and position-change handling

- If any source field contains mixed roles (e.g., `CB/S`, `WR/DB`, `DB/S/LB`), map to `DB_hybrid`.
- If a player changes position across seasons and labels disagree, keep deterministic priority (`pos` → `position` → `Pos`) and retain `canonical_db_position_source` for auditability.
- Do not infer `FS` vs `SS` from generic `S` unless an explicit source string indicates free/strong safety.
- When downstream modeling requires strict FS/SS split, treat generic `S` players as `DB_hybrid` or build a separate disambiguation model with snap-alignment context.

## 6) Implemented dataset fields

`all_data.csv` now includes:

- `canonical_db_position`: mapped taxonomy label.
- `canonical_db_position_source`: exact column:value source used for mapping (or `fallback_unmapped`).

These fields should be treated as the canonical join keys for downstream DB success modeling.
