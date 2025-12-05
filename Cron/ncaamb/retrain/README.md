# Model Retraining Guide

This folder contains scripts to retrain all NCAAMB models from scratch using Bayesian optimization. Run these scripts **before every season** to ensure models are trained on the latest data.

---

## Prerequisites

Before running any retraining scripts, ensure you have:

1. **Feature files**: All `featuresYYYY.csv` files in the parent directory (`Cron/ncaamb/`)
   - Example: `features2021.csv`, `features2022.csv`, ..., `features2025.csv`
2. **Python packages**: All dependencies installed (polars, scikit-optimize, xgboost, lightgbm, catboost, etc.)

---

## Workflow Overview

The retraining process consists of 3 steps:

### Step 1: Generate Features (If Needed)

If you need to generate new feature files from raw game data:

```bash
# 1. Build parquet file from API data
python retrain/build_parquet.py

# 2. Generate features from parquet
python retrain/generate_features_and_validate.py
```

This creates `featuresYYYY.csv` files in the parent directory.

### Step 2: Optimize Base Models

Run these 5 scripts **in any order** (they're independent). Each runs 250 iterations of Bayesian optimization:

```bash
# Over/Under models (3 models)
python retrain/ou_xgb_optimize_and_save.py      # ~15-30 min
python retrain/ou_lgb_optimize_and_save.py      # ~15-30 min
python retrain/ou_catboost_optimize_and_save.py # ~15-30 min

# Moneyline models (2 models)
python retrain/ml_xgb_optimize_and_save.py      # ~15-30 min
python retrain/ml_lgb_optimize_and_save.py      # ~15-30 min
```

**What each script does:**
- Loads all `featuresYYYY.csv` files
- Splits data: oldest year for validation, newer years for training
- Weights recent data: most recent year 4x, second most recent 2x, others 1x
- Runs 250 iterations of Bayesian optimization
- Trains final model on ALL data with best hyperparameters
- Saves model to `models/*/saved/*.pkl`
- Saves hyperparameters to `models/*/saved/*_hyperparameters.txt`

### Step 3: Find Optimal Ensemble Weights

**IMPORTANT**: Run these scripts **AFTER** Step 2 is complete. They find the best way to combine model predictions.

```bash
# Find optimal weights for O/U models
python retrain/ou_find_ensemble_weights.py  # ~2-5 min

# Find optimal weights for ML models (optional if not using ML ensemble)
python retrain/ml_find_ensemble_weights.py  # ~1-3 min
```

**What each script does:**
- Trains models with saved hyperparameters
- Grid searches all possible weight combinations (step=0.01)
- For O/U: Finds weights that minimize MAE
- For ML: Finds weights that maximize accuracy
- Saves optimal weights to `ensemble_weights.txt` or `ml_ensemble_weights.txt`

**Example output:**
```
Optimal Ensemble Weights:
  XGBoost:  0.441 (44.1%)
  LightGBM: 0.466 (46.6%)
  CatBoost: 0.093 (9.3%)

Ensemble MAE: 8.42
```

### Step 4: Optimize Good Bets Model

**IMPORTANT**: Run this script **AFTER** Step 3 is complete. It uses the ensemble weights found in Step 3.

```bash
python retrain/ou_good_bets_optimize_and_save.py  # ~20-40 min
```

**What it does:**
- Reads optimized hyperparameters from the 3 O/U model txt files
- Reads optimal ensemble weights from `ensemble_weights.txt`
- Trains XGB, LGB, CatBoost on training data using those hyperparameters (no data leakage)
- Generates predictions on train and test sets separately
- Creates Good Bets features using optimal ensemble weights
- Optimizes Good Bets Random Forest model (250 iterations)
- Retrains all 4 models on ALL data
- Saves Good Bets model to `models/overunder/saved/ou_good_bets_final.pkl`

---

## Output Files

After running all scripts, you'll have:

### Over/Under Models
```
models/overunder/saved/
├── xgboost_model.pkl
├── xgboost_hyperparameters.txt
├── lightgbm_model.pkl
├── lightgbm_hyperparameters.txt
├── catboost_model.pkl
├── catboost_hyperparameters.txt
├── ensemble_weights.txt              # NEW: Optimal ensemble weights
├── ou_good_bets_final.pkl
└── ou_good_bets_final_hyperparameters.txt
```

### Moneyline Models
```
models/moneyline/saved/
├── xgboost_model_final.pkl
├── xgboost_final_hyperparameters.txt
├── lightgbm_model_final.pkl
├── lightgbm_final_hyperparameters.txt
└── ml_ensemble_weights.txt           # NEW: Optimal ensemble weights
```

---

## Full Retraining Procedure (Start to Finish)

Run these commands in order:

```bash
# Navigate to retrain folder
cd Cron/ncaamb/retrain

# Step 1: Generate features (if needed)
python build_parquet.py
python generate_features_and_validate.py

# Step 2: Optimize base models (can run in parallel)
python ou_xgb_optimize_and_save.py &
python ou_lgb_optimize_and_save.py &
python ou_catboost_optimize_and_save.py &
python ml_xgb_optimize_and_save.py &
python ml_lgb_optimize_and_save.py &
wait

# Step 3: Find optimal ensemble weights (MUST run after Step 2)
python ou_find_ensemble_weights.py
python ml_find_ensemble_weights.py  # Optional

# Step 4: Optimize Good Bets model (MUST run after Step 3)
python ou_good_bets_optimize_and_save.py
```

**Total time**: ~1.5-3 hours depending on hardware and data size.

---

## Training Strategy

All scripts use the following strategy to prioritize recent data:

### Optimization Phase (Finding Best Hyperparameters)
- **Test set**: **Most recent year** (e.g., 2026)
  - Validates against current season patterns
  - Ensures hyperparameters generalize to latest data
- **Training set**: All older years (e.g., 2021-2025)
- **Sample weighting during training**:
  - **(Most recent - 1)** year: **4x weight** (e.g., 2025=4x)
  - **(Most recent - 2)** year: **2x weight** (e.g., 2024=2x)
  - All other years: 1x weight

This ensures hyperparameters are optimized for recent season patterns while preventing overfitting to a single year.

### Final Production Model
After finding optimal hyperparameters, each script trains on **ALL available data**:
- **Training set**: **ALL years** (e.g., 2021-2026)
  - Includes the most recent year that was held out during optimization
  - Maximizes training data for production model
- **Sample weighting**:
  - **Most recent year**: **4x weight** (e.g., 2026=4x)
  - **Second most recent**: **2x weight** (e.g., 2025=2x)
  - All other years: 1x weight

This final model is what gets deployed to production and has seen the maximum amount of data possible.

---

## Validation

Each script prints:
- Year range trained on (e.g., "Training on features2021-2026")
- Train/test split details
- Optimization progress (250 iterations)
- Final model performance metrics (MAE, R², Accuracy, F1, etc.)

Review these outputs to ensure:
- All expected years are loaded
- Performance metrics are reasonable
- No errors or warnings occurred

---

## Troubleshooting

**"No features files found!"**
- Ensure `featuresYYYY.csv` files exist in parent directory (`Cron/ncaamb/`)
- Run `build_parquet.py` and `generate_features_and_validate.py` first

**"XGBoost hyperparameters not found" (Good Bets script)**
- Run the 3 O/U optimization scripts first
- Check that `models/overunder/saved/*_hyperparameters.txt` files exist

**"Out of memory" errors**
- Reduce `n_calls` from 250 to 100 or 50 in the script
- Close other applications
- Use a machine with more RAM

**Models perform poorly**
- Check feature quality in `featuresYYYY.csv` files
- Ensure odds data is accurate and complete
- Review feature engineering in parent scripts

---

## Notes

- **Bayesian Optimization**: Each script runs 250 iterations to find optimal hyperparameters. This is computationally expensive but results in better models.
- **No Data Leakage**: Good Bets script ensures base models don't see test data during meta-feature generation.
- **Dynamic Year Detection**: Scripts automatically detect available years from filenames - no hardcoding.
- **Production Ready**: Final models are trained on ALL data and ready for deployment.

---

## Questions?

Review the individual script docstrings for implementation details, or check the main production scripts (`main.py`, `ou_main.py`) for usage examples.
