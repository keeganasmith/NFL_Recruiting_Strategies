# NFL Recruiting Strategies

## Overperformer quadrant visualization

The script `analysis/visualizations/quadrant_overperformers.py` creates an interactive Plotly HTML chart at `outputs/visualizations/overperformer_quadrant.html`.

### Scoring logic and assumptions

- **Player-level deduplication**: source data is season-grain, so rows are grouped to one record per player using `NFL_id` when available and a normalized `Player` name fallback.
- **Combine score**:
  - Uses `40yd`, `Vertical`, `Bench`, `Broad Jump`, `3Cone`, `Shuttle`, `Ht`, and `Wt`.
  - Height is converted from `feet-inches` to total inches.
  - Timing drills (`40yd`, `3Cone`, `Shuttle`) are sign-inverted so higher is always better.
  - Each combine metric is z-scored **within `Pos`** and averaged to a composite combine score.
- **Production score**:
  - Career production is aggregated as player totals (or included rate stats where available) across NFL seasons.
  - Position-relevant stat families are selected per position group:
    - QB: passing-centered metrics (+ rushing contribution)
    - Skill positions (RB/FB/WR/TE): rushing + receiving (+ return yards)
    - Defensive positions: tackles/sacks/INT/pass defenses/forced fumbles
    - Specialists (K/P/LS): kicking/punting/return outputs
  - Chosen production components are z-scored within `Pos` and averaged to a composite production score.
- **Normalization**: final combine and production composites are both re-z-scored within `Pos` to keep cross-position comparisons centered around 0.
- **Filters in HTML**: position, drafted vs undrafted, and combine year min/max sliders are applied client-side.

### Run

```bash
python analysis/visualizations/quadrant_overperformers.py
```
