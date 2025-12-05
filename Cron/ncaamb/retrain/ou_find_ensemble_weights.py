#!/usr/bin/env python3
"""
Over/Under Ensemble Weight Optimizer
Finds optimal weights for combining XGB, LGB, CatBoost predictions

Must run AFTER ou_xgb, ou_lgb, ou_catboost optimization scripts.
Must run BEFORE ou_good_bets script.

Grid searches all possible weight combinations (step=0.01) to minimize MAE.
Saves optimal weights to ensemble_weights.txt for Good Bets script to use.
"""

import os
import sys
from pathlib import Path
import polars as pl
import numpy as np
from xgboost import XGBRegressor
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error

# Add parent directory to path
ncaamb_dir = Path(__file__).parent.parent
sys.path.insert(0, str(ncaamb_dir))

print("="*80)
print("O/U ENSEMBLE WEIGHT OPTIMIZER")
print("="*80 + "\n")


def load_all_features():
    """Load all featuresYYYY.csv files"""
    features_dir = ncaamb_dir
    all_features = []

    print("Loading all features files...")
    for features_file in sorted(features_dir.glob("features*.csv")):
        year = features_file.stem.replace("features", "")
        if year.isdigit():
            print(f"  Loading {features_file.name}...")
            try:
                df = pl.read_csv(features_file)
                print(f"    ✓ Loaded {len(df)} games")
                all_features.append(df)
            except Exception as e:
                print(f"    ✗ Error: {e}")

    if not all_features:
        print("✗ No features files found!")
        return None

    combined_df = pl.concat(all_features)
    print(f"✓ Combined: {len(combined_df)} total games\n")
    return combined_df


def identify_feature_columns(df: pl.DataFrame) -> list:
    """Identify numeric feature columns for O/U model"""
    metadata_cols = {
        'game_id', 'date', 'season', 'team_1', 'team_2',
        'team_1_score', 'team_2_score', 'actual_total',
        'team_1_conference', 'team_2_conference',
        'team_1_is_home', 'team_2_is_home', 'location',
        'start_time', 'game_odds', 'ou_target',
        'betonline_ou_line', 'ou_line_variance'
    }

    feature_cols = []
    for col in df.columns:
        if col not in metadata_cols:
            dtype = df[col].dtype
            if dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]:
                feature_cols.append(col)

    return feature_cols


def prepare_train_test_split(df: pl.DataFrame, feature_cols: list):
    """Split data: train on oldest to 2nd-most-recent, test on most recent"""
    df = df.filter(pl.col('actual_total').is_not_null())

    df = df.with_columns(pl.col('date').str.slice(0, 4).alias('year'))
    years = sorted(df.select(pl.col('year')).unique().to_series().to_list())
    most_recent_year = years[-1]

    train_df = df.filter(pl.col('year') != most_recent_year)
    test_df = df.filter(pl.col('year') == most_recent_year)

    X_train = train_df.select(feature_cols).fill_null(0).to_numpy()
    y_train = train_df.select('actual_total').to_numpy().ravel()

    X_test = test_df.select(feature_cols).fill_null(0).to_numpy()
    y_test = test_df.select('actual_total').to_numpy().ravel()

    print(f"Train/Test Split:")
    print(f"  Test year (most recent): {most_recent_year}")
    print(f"  Train: {len(X_train)} games")
    print(f"  Test:  {len(X_test)} games\n")

    return X_train, y_train, X_test, y_test, years


def read_hyperparameters_from_file(filepath):
    """Read hyperparameters from txt file"""
    params = {}
    with open(filepath, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if ':' in line and not line.startswith('=') and not line.startswith('Final'):
                parts = line.strip().split(':')
                if len(parts) == 2:
                    key = parts[0].strip()
                    value = parts[1].strip()
                    try:
                        if '.' in value:
                            params[key] = float(value)
                        else:
                            params[key] = int(value)
                    except:
                        params[key] = value
    return params


def load_saved_hyperparameters():
    """Load optimized hyperparameters from saved txt files"""
    saved_dir = ncaamb_dir / "models" / "overunder" / "saved"

    print("Loading saved hyperparameters...")

    xgb_params_file = saved_dir / "xgboost_hyperparameters.txt"
    if not xgb_params_file.exists():
        print(f"  ✗ XGBoost hyperparameters not found")
        print("     Run ou_xgb_optimize_and_save.py first!")
        return None, None, None

    lgb_params_file = saved_dir / "lightgbm_hyperparameters.txt"
    if not lgb_params_file.exists():
        print(f"  ✗ LightGBM hyperparameters not found")
        print("     Run ou_lgb_optimize_and_save.py first!")
        return None, None, None

    cat_params_file = saved_dir / "catboost_hyperparameters.txt"
    if not cat_params_file.exists():
        print(f"  ✗ CatBoost hyperparameters not found")
        print("     Run ou_catboost_optimize_and_save.py first!")
        return None, None, None

    xgb_params = read_hyperparameters_from_file(xgb_params_file)
    lgb_params = read_hyperparameters_from_file(lgb_params_file)
    cat_params = read_hyperparameters_from_file(cat_params_file)

    print(f"  ✓ Loaded all hyperparameters\n")

    return xgb_params, lgb_params, cat_params


def train_models_and_predict(X_train, y_train, X_test, xgb_params, lgb_params, cat_params):
    """Train models with saved params and generate predictions"""
    print("Training models with saved hyperparameters...")

    # Train XGBoost
    print("  Training XGBoost...")
    xgb_model = XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1, **xgb_params)
    xgb_model.fit(X_train, y_train, verbose=False)
    xgb_preds = xgb_model.predict(X_test)

    # Train LightGBM
    print("  Training LightGBM...")
    train_data = lgb.Dataset(X_train, label=y_train)
    lgb_model = lgb.train({'objective': 'regression', 'metric': 'mae', 'verbose': -1,
                           'force_row_wise': True, **lgb_params},
                          train_data, num_boost_round=lgb_params['n_estimators'])
    lgb_preds = lgb_model.predict(X_test)

    # Train CatBoost
    print("  Training CatBoost...")
    cat_model = CatBoostRegressor(loss_function='MAE', random_state=42, verbose=False, **cat_params)
    cat_model.fit(X_train, y_train)
    cat_preds = cat_model.predict(X_test)

    print(f"  ✓ Generated predictions\n")

    return xgb_preds, lgb_preds, cat_preds


def find_optimal_weights(xgb_preds, lgb_preds, cat_preds, y_test):
    """Grid search for optimal ensemble weights"""
    print("Searching for optimal ensemble weights (step=0.01)...")
    print("  This may take a few minutes...\n")

    best_mae = float('inf')
    best_weights = None
    total_combinations = 0

    # Grid search: w1, w2, w3 where w1+w2+w3=1.0
    for w_xgb_pct in range(0, 101):  # 0% to 100%
        w_xgb = w_xgb_pct / 100.0
        for w_lgb_pct in range(0, 101 - w_xgb_pct):  # Ensure w_xgb + w_lgb <= 1.0
            w_lgb = w_lgb_pct / 100.0
            w_cat = 1.0 - w_xgb - w_lgb

            total_combinations += 1

            # Calculate ensemble prediction
            ensemble_preds = w_xgb * xgb_preds + w_lgb * lgb_preds + w_cat * cat_preds
            mae = mean_absolute_error(y_test, ensemble_preds)

            if mae < best_mae:
                best_mae = mae
                best_weights = (w_xgb, w_lgb, w_cat)

    print(f"  Tested {total_combinations} weight combinations\n")
    return best_weights, best_mae


if __name__ == "__main__":
    # Load data
    print("STEP 1: Loading Data")
    print("-"*80)
    df = load_all_features()
    if df is None:
        sys.exit(1)

    # Filter for O/U data
    print("STEP 2: Filtering for O/U Data")
    print("-"*80)
    df = df.filter(
        pl.col('actual_total').is_not_null() &
        pl.col('betonline_ou_line').is_not_null()
    )
    print(f"Kept {len(df)} games with O/U data\n")

    # Identify features
    print("STEP 3: Identifying Features")
    print("-"*80)
    feature_cols = identify_feature_columns(df)
    print(f"Found {len(feature_cols)} feature columns\n")

    # Split data
    print("STEP 4: Train/Test Split")
    print("-"*80)
    X_train, y_train, X_test, y_test, years = prepare_train_test_split(df, feature_cols)

    # Load hyperparameters
    print("STEP 5: Loading Hyperparameters")
    print("-"*80)
    xgb_params, lgb_params, cat_params = load_saved_hyperparameters()
    if xgb_params is None:
        sys.exit(1)

    # Train models and get predictions
    print("STEP 6: Training Models")
    print("-"*80)
    xgb_preds, lgb_preds, cat_preds = train_models_and_predict(
        X_train, y_train, X_test, xgb_params, lgb_params, cat_params)

    # Individual model MAE
    xgb_mae = mean_absolute_error(y_test, xgb_preds)
    lgb_mae = mean_absolute_error(y_test, lgb_preds)
    cat_mae = mean_absolute_error(y_test, cat_preds)

    print("Individual Model Performance:")
    print(f"  XGBoost MAE: {xgb_mae:.4f}")
    print(f"  LightGBM MAE: {lgb_mae:.4f}")
    print(f"  CatBoost MAE: {cat_mae:.4f}\n")

    # Find optimal weights
    print("STEP 7: Finding Optimal Ensemble Weights")
    print("-"*80)
    best_weights, best_mae = find_optimal_weights(xgb_preds, lgb_preds, cat_preds, y_test)

    # Results
    print("="*80)
    print("OPTIMIZATION COMPLETE")
    print("="*80)
    print(f"\nOptimal Ensemble Weights:")
    print(f"  XGBoost:  {best_weights[0]:.3f} ({best_weights[0]*100:.1f}%)")
    print(f"  LightGBM: {best_weights[1]:.3f} ({best_weights[1]*100:.1f}%)")
    print(f"  CatBoost: {best_weights[2]:.3f} ({best_weights[2]*100:.1f}%)")
    print(f"\nEnsemble MAE: {best_mae:.4f}")
    print(f"Improvement over best single model: {min(xgb_mae, lgb_mae, cat_mae) - best_mae:.4f}")

    # Save weights
    print("\n" + "="*80)
    print("STEP 8: Saving Ensemble Weights")
    print("="*80)

    save_dir = ncaamb_dir / "models" / "overunder" / "saved"
    save_path = save_dir / "ensemble_weights.txt"

    with open(save_path, 'w') as f:
        f.write("O/U Ensemble Weights (Optimized)\n")
        f.write("="*50 + "\n\n")
        f.write(f"xgb_weight: {best_weights[0]:.3f}\n")
        f.write(f"lgb_weight: {best_weights[1]:.3f}\n")
        f.write(f"cat_weight: {best_weights[2]:.3f}\n")
        f.write(f"\nEnsemble MAE: {best_mae:.4f}\n")
        f.write(f"XGBoost MAE: {xgb_mae:.4f}\n")
        f.write(f"LightGBM MAE: {lgb_mae:.4f}\n")
        f.write(f"CatBoost MAE: {cat_mae:.4f}\n")
        f.write(f"\nTested on features{years[0]}-{years[-1]} (test year: {years[-1]})\n")

    print(f"✓ Saved ensemble weights to: {save_path}")
    print("\nUse these weights in ou_good_bets_optimize_and_save.py:")
    print(f"  ensemble_conf = {best_weights[0]:.3f} * xgb_conf + {best_weights[1]:.3f} * lgb_conf + {best_weights[2]:.3f} * cat_conf")
    print("\n" + "="*80)
    print("DONE!")
    print("="*80)
