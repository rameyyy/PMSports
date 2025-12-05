#!/usr/bin/env python3
"""
Moneyline Ensemble Weight Optimizer
Finds optimal weights for combining XGB and LGB predictions

Must run AFTER ml_xgb and ml_lgb optimization scripts.

Grid searches all possible weight combinations (step=0.01) to maximize accuracy.
Saves optimal weights to ml_ensemble_weights.txt.
"""

import os
import sys
from pathlib import Path
import polars as pl
import numpy as np
from xgboost import XGBClassifier
import lightgbm as lgb
from sklearn.metrics import accuracy_score
import pickle

# Add parent directory to path
ncaamb_dir = Path(__file__).parent.parent
sys.path.insert(0, str(ncaamb_dir))

print("="*80)
print("MONEYLINE ENSEMBLE WEIGHT OPTIMIZER")
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


def create_target_variable(df: pl.DataFrame) -> pl.DataFrame:
    """Create binary target variable for moneyline (team_1 win)"""
    df_with_scores = df.filter(
        pl.col('team_1_score').is_not_null() &
        pl.col('team_2_score').is_not_null()
    )

    df_with_scores = df_with_scores.with_columns(
        pl.when(pl.col('team_1_score') > pl.col('team_2_score'))
            .then(1)
            .otherwise(0)
            .alias('ml_target')
    )

    print(f"Created target for {len(df_with_scores)} games")
    return df_with_scores


def identify_feature_columns(df: pl.DataFrame) -> list:
    """Identify numeric feature columns"""
    metadata_cols = {
        'game_id', 'date', 'season', 'team_1', 'team_2',
        'team_1_score', 'team_2_score', 'actual_total',
        'team_1_conference', 'team_2_conference',
        'team_1_is_home', 'team_2_is_home', 'location',
        'total_score_outcome', 'team_1_winloss',
        'team_1_leaderboard', 'team_2_leaderboard',
        'team_1_match_hist', 'team_2_match_hist',
        'start_time', 'game_odds', 'ml_target'
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
    df = df.with_columns(pl.col('date').str.slice(0, 4).alias('year'))
    years = sorted(df.select(pl.col('year')).unique().to_series().to_list())
    most_recent_year = years[-1]

    train_df = df.filter(pl.col('year') != most_recent_year)
    test_df = df.filter(pl.col('year') == most_recent_year)

    X_train = train_df.select(feature_cols).fill_null(0).to_numpy()
    y_train = train_df.select('ml_target').to_numpy().ravel()

    X_test = test_df.select(feature_cols).fill_null(0).to_numpy()
    y_test = test_df.select('ml_target').to_numpy().ravel()

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
    saved_dir = ncaamb_dir / "models" / "moneyline" / "saved"

    print("Loading saved hyperparameters...")

    xgb_params_file = saved_dir / "xgboost_final_hyperparameters.txt"
    if not xgb_params_file.exists():
        print(f"  ✗ XGBoost hyperparameters not found")
        print("     Run ml_xgb_optimize_and_save.py first!")
        return None, None

    lgb_params_file = saved_dir / "lightgbm_final_hyperparameters.txt"
    if not lgb_params_file.exists():
        print(f"  ✗ LightGBM hyperparameters not found")
        print("     Run ml_lgb_optimize_and_save.py first!")
        return None, None

    xgb_params = read_hyperparameters_from_file(xgb_params_file)
    lgb_params = read_hyperparameters_from_file(lgb_params_file)

    print(f"  ✓ Loaded all hyperparameters\n")

    return xgb_params, lgb_params


def train_models_and_predict(X_train, y_train, X_test, xgb_params, lgb_params):
    """Train models with saved params and generate probability predictions"""
    print("Training models with saved hyperparameters...")

    # Train XGBoost
    print("  Training XGBoost...")
    xgb_model = XGBClassifier(objective='binary:logistic', eval_metric='logloss',
                              use_label_encoder=False, random_state=42, n_jobs=-1, **xgb_params)
    xgb_model.fit(X_train, y_train, verbose=False)
    xgb_probs = xgb_model.predict_proba(X_test)[:, 1]  # Probability of class 1

    # Train LightGBM
    print("  Training LightGBM...")
    train_data = lgb.Dataset(X_train, label=y_train)
    lgb_model = lgb.train({'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1,
                           'force_row_wise': True, **lgb_params},
                          train_data, num_boost_round=lgb_params['n_estimators'])
    lgb_probs = lgb_model.predict(X_test)  # Already returns probabilities

    print(f"  ✓ Generated predictions\n")

    return xgb_probs, lgb_probs


def find_optimal_weights(xgb_probs, lgb_probs, y_test):
    """Grid search for optimal ensemble weights"""
    print("Searching for optimal ensemble weights (step=0.01)...")
    print("  This may take a minute...\n")

    best_accuracy = 0.0
    best_weights = None
    total_combinations = 0

    # Grid search: w1, w2 where w1+w2=1.0
    for w_xgb_pct in range(0, 101):  # 0% to 100%
        w_xgb = w_xgb_pct / 100.0
        w_lgb = 1.0 - w_xgb

        total_combinations += 1

        # Calculate ensemble probabilities
        ensemble_probs = w_xgb * xgb_probs + w_lgb * lgb_probs
        ensemble_preds = (ensemble_probs >= 0.5).astype(int)
        accuracy = accuracy_score(y_test, ensemble_preds)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_weights = (w_xgb, w_lgb)

    print(f"  Tested {total_combinations} weight combinations\n")
    return best_weights, best_accuracy


if __name__ == "__main__":
    # Load data
    print("STEP 1: Loading Data")
    print("-"*80)
    df = load_all_features()
    if df is None:
        sys.exit(1)

    # Create target
    print("STEP 2: Creating Target Variable")
    print("-"*80)
    df = create_target_variable(df)

    # Filter for ML data
    print("STEP 3: Filtering for Moneyline Data")
    print("-"*80)
    df = df.filter(
        pl.col('avg_ml_home').is_not_null() &
        pl.col('avg_ml_away').is_not_null()
    )
    print(f"Kept {len(df)} games with ML odds\n")

    # Identify features
    print("STEP 4: Identifying Features")
    print("-"*80)
    feature_cols = identify_feature_columns(df)
    print(f"Found {len(feature_cols)} feature columns\n")

    # Split data
    print("STEP 5: Train/Test Split")
    print("-"*80)
    X_train, y_train, X_test, y_test, years = prepare_train_test_split(df, feature_cols)

    # Load hyperparameters
    print("STEP 6: Loading Hyperparameters")
    print("-"*80)
    xgb_params, lgb_params = load_saved_hyperparameters()
    if xgb_params is None:
        sys.exit(1)

    # Train models and get predictions
    print("STEP 7: Training Models")
    print("-"*80)
    xgb_probs, lgb_probs = train_models_and_predict(X_train, y_train, X_test, xgb_params, lgb_params)

    # Individual model accuracy
    xgb_preds = (xgb_probs >= 0.5).astype(int)
    lgb_preds = (lgb_probs >= 0.5).astype(int)
    xgb_acc = accuracy_score(y_test, xgb_preds)
    lgb_acc = accuracy_score(y_test, lgb_preds)

    print("Individual Model Performance:")
    print(f"  XGBoost Accuracy: {xgb_acc:.4f}")
    print(f"  LightGBM Accuracy: {lgb_acc:.4f}\n")

    # Find optimal weights
    print("STEP 8: Finding Optimal Ensemble Weights")
    print("-"*80)
    best_weights, best_accuracy = find_optimal_weights(xgb_probs, lgb_probs, y_test)

    # Results
    print("="*80)
    print("OPTIMIZATION COMPLETE")
    print("="*80)
    print(f"\nOptimal Ensemble Weights:")
    print(f"  XGBoost:  {best_weights[0]:.3f} ({best_weights[0]*100:.1f}%)")
    print(f"  LightGBM: {best_weights[1]:.3f} ({best_weights[1]*100:.1f}%)")
    print(f"\nEnsemble Accuracy: {best_accuracy:.4f}")
    print(f"Improvement over best single model: {best_accuracy - max(xgb_acc, lgb_acc):.4f}")

    # Save weights
    print("\n" + "="*80)
    print("STEP 9: Saving Ensemble Weights")
    print("="*80)

    save_dir = ncaamb_dir / "models" / "moneyline" / "saved"
    save_path = save_dir / "ml_ensemble_weights.txt"

    with open(save_path, 'w') as f:
        f.write("Moneyline Ensemble Weights (Optimized)\n")
        f.write("="*50 + "\n\n")
        f.write(f"xgb_weight: {best_weights[0]:.3f}\n")
        f.write(f"lgb_weight: {best_weights[1]:.3f}\n")
        f.write(f"\nEnsemble Accuracy: {best_accuracy:.4f}\n")
        f.write(f"XGBoost Accuracy: {xgb_acc:.4f}\n")
        f.write(f"LightGBM Accuracy: {lgb_acc:.4f}\n")
        f.write(f"\nTested on features{years[0]}-{years[-1]} (test year: {years[-1]})\n")

    print(f"✓ Saved ensemble weights to: {save_path}")
    print("\nUse these weights for ML ensemble predictions:")
    print(f"  ensemble_prob = {best_weights[0]:.3f} * xgb_prob + {best_weights[1]:.3f} * lgb_prob")
    print("\n" + "="*80)
    print("DONE!")
    print("="*80)
