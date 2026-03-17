# March Madness Bracket Model

Predicts game totals (over/under) for NCAA March Madness **without** using sportsbook
odds — because brackets must be submitted weeks before games, when no live odds exist.

All scripts in this folder are **self-contained and do not modify any existing files.**

---

## How to run (in order)

All scripts must be run from the **`ncaamb/`** parent directory.

```bash
# 1. Build 2026 parquet (completed games only)
python marchmadness/build_sample2026.py

# 2. Generate features2026.csv
python marchmadness/generate_features2026.py

# 3. Train models (Bayesian opt + final training, ~20–40 min)
python marchmadness/train_bracket_model.py
```

Steps 1 and 2 only need to be rerun if new 2026 games complete.
Step 3 retrains from scratch on features2021–2026.

---

## Models trained

| Model | File | Notes |
|---|---|---|
| LightGBM | `saved/lightgbm_model.txt` | Bayesian opt, 150 iterations |
| XGBoost | `saved/xgboost_model.json` | Bayesian opt, 150 iterations |
| Ridge Regression | `saved/ridge_model.pkl` | RidgeCV alpha search, includes fitted scaler |

Feature column list: `saved/feature_columns.txt`
Hyperparameters + metrics: `saved/hyperparameters.txt`

---

## Dropped features (odds / betting lines)

The following columns are dropped from **all** years (2021–2026) before training,
since these are not available at bracket submission time.

### Per-bookmaker columns (7 books × 9 cols each = 63 cols)
`{book}_ou_line`, `{book}_over_odds`, `{book}_under_odds`,
`{book}_ml_team_1`, `{book}_ml_team_2`,
`{book}_spread_pts_team_1`, `{book}_spread_odds_team_1`,
`{book}_spread_pts_team_2`, `{book}_spread_odds_team_2`

Books: `betmgm`, `betonline`, `bovada`, `draftkings`, `fanduel`, `lowvig`, `mybookie`

### Aggregate odds columns
```
avg_ou_line              ou_line_variance
avg_over_odds            avg_under_odds
num_books_with_ou
avg_spread_pts_team_1    avg_spread_pts_team_2
spread_variance
avg_spread_odds_team_1   avg_spread_odds_team_2
avg_ml_team_1            avg_ml_team_2
num_books_with_spread    num_books_with_ml
```

### Derived odds features
```
implied_team_1_score              implied_team_2_score
spread_ou_agreement               hours_until_game_from_odds
combined_expected_total_closest3rank
combined_expected_total_closest5rank
combined_expected_total_closest7rank
```

---

## Notes

- **No changes** are made to any existing models or scripts outside this folder.
- The 2026 parquet and CSV are written to `ncaamb/` alongside the 2021–2025 files.
- Training uses sample weights: most recent year = 4×, year before = 2×, others = 1×.
- Test split is the most recent year; all others are used for training during optimization.
