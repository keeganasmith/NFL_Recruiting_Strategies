# Plan: Proposal for Evidence-Backed NFL Combine Recruiting Strategies (Defensive Backs)

## Objective
Build a proposal NFL teams can use to recruit defensive backs (DBs) by combining NFL Combine measurements with downstream football outcomes in `all_data.csv`.

## Core Question
Which measurable traits and context signals available around the NFL Combine are most predictive of high-value DB outcomes, and how should teams translate those findings into recruiting strategy?

## Scope and Definitions
- **Population:** Players in `all_data.csv` with DB positions (CB, S, FS, SS, DB variants).
- **Decision point:** Pre-draft recruiting/scouting discussions.
- **Outcomes to optimize (tiered):**
  1. Draft capital / drafted indicator.
  2. Early career playing time (games played).
  3. Defensive production (interceptions, passes defended, tackles, sacks by role).
  4. Composite production value score (if needed for ranking candidates).
- **Primary predictor groups:**
  - Combine: height, weight, 40-yard dash, vertical, broad jump, shuttle, 3-cone, bench.
  - Context: school/conference, class/final season timing.
  - College on-field signals: interceptions, passes defended, tackles, etc.

## Work Plan

### 1) Data audit and DB cohort construction
- Filter to DB records with a documented mapping table for accepted DB labels.
- Quantify missingness by feature and year; produce a missingness heatmap/table.
- Resolve duplicate player keys (`Player`, `player`, `NFL_id`, `Player-additional`) with deterministic rules.
- Standardize units/formats (e.g., `Ht`, time/event fields, split strings).

**Deliverable:** Reproducible DB-ready analysis dataset + data quality note.

### 2) Feature engineering for scouting relevance
- Create percentile-based athletic scores by combine year to control era effects.
- Build role-specific features:
  - Coverage profile (e.g., agility + ball production).
  - Range profile (speed + explosion).
  - Physicality profile (size + tackling/box production).
- Add interaction terms that mimic scout logic (e.g., speed × weight; vertical × broad jump).

**Deliverable:** Feature dictionary translating raw columns into scouting interpretable metrics.

### 3) Outcome design and validation framing
- Define binary and continuous targets:
  - Drafted vs undrafted.
  - Top-N round threshold.
  - Early-career production indicators.
- Use time-aware train/validation splits by combine year (avoid leakage).
- Set clear success metrics:
  - AUC / PR-AUC for classification.
  - MAE/RMSE + rank correlation for continuous outcomes.

**Deliverable:** Modeling protocol and evaluation rubric.

### 4) Modeling and evidence extraction
- Fit baseline interpretable models first (logistic/linear with regularization).
- Compare with non-linear models (tree ensembles) for signal discovery.
- Estimate effect sizes with uncertainty (confidence intervals/bootstrap).
- Perform subgroup analyses by role archetype (corner vs safety) and conference tier.

**Deliverable:** Ranked drivers of DB success with stability checks.

### 5) Strategy synthesis for NFL recruiting decisions
Convert evidence into practical strategy packages:
- **Threshold strategy:** Minimum athletic thresholds associated with materially better odds.
- **Archetype strategy:** Separate recruiting templates for boundary CB, slot CB, and safety.
- **Value strategy:** Identify undervalued profiles (strong production but less-hyped combine profile, or vice versa).
- **Risk controls:** Red-flag combinations (e.g., poor agility + low ball production) tied to lower hit rates.

**Deliverable:** Decision rules formatted as front-office-ready scouting guidance.

### 6) Proposal artifact production
- Produce a concise proposal deck/one-pager including:
  - Objective and data coverage.
  - Top 5 evidence-backed insights.
  - Archetype-specific recruiting recommendations.
  - Implementation checklist for scouting workflow integration.
- Include transparent caveats (sample size, missing data, position label noise, era drift).

**Deliverable:** Final proposal document ready for team strategy discussion.

## Suggested Analysis Timeline
- **Day 1:** Data audit, DB filtering, missingness report.
- **Day 2:** Feature engineering + target definitions.
- **Day 3:** Baseline and advanced modeling.
- **Day 4:** Insight translation + proposal draft.
- **Day 5:** Review, stress tests, final presentation package.

## Acceptance Criteria for the Proposal
- At least 3 recruiting recommendations are directly supported by quantified effect sizes.
- Recommendations are role-specific (not one-size-fits-all DB guidance).
- Methods are reproducible and leakage-resistant.
- Each recommendation includes expected upside and key risk.

## Immediate Next Steps
1. Confirm final DB label mapping and outcome definitions with stakeholders.
2. Generate DB-only analysis table from `all_data.csv`.
3. Run first-pass models and produce an insight shortlist for proposal narrative.
