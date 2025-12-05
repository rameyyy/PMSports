#!/usr/bin/env python3
"""
Over/Under Good Bets Model - Bayesian Optimization and Training
NO DATA LEAKAGE VERSION

Prerequisites: Run ou_xgb, ou_lgb, ou_catboost optimization scripts first!
This script reads their saved hyperparameters.

Workflow:
1. Load all featuresYYYY.csv, split into train/test
2. Read optimized XGB, LGB, CatBoost hyperparameters from txt files
3. Train the 3 models on training data using those hyperparameters
4. Generate predictions on train and test sets (no leakage)
5. Create Good Bets features from those predictions
6. Optimize Good Bets model hyperparameters (250 iterations)
7. Retrain all 4 models on ALL data and save

Saves to models/overunder/saved/ou_good_bets_final.pkl
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, f1_score
from skopt import gp_minimize
from skopt.space import Integer
from skopt.utils import use_named_args
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
ncaamb_dir = Path(__file__).parent.parent
sys.path.insert(0, str(ncaamb_dir))

print("="*80)
print("O/U GOOD BETS MODEL - BAYESIAN OPTIMIZATION (NO DATA LEAKAGE)")
print("="*80 + "\n")


def load_all_features():
    """Load all featuresYYYY.csv files as Pandas DataFrame"""
    features_dir = ncaamb_dir
    all_features = []

    print("Loading all features files...")
    for features_file in sorted(features_dir.glob("features*.csv")):
        year = features_file.stem.replace("features", "")
        if year.isdigit():
            print(f"  Loading {features_file.name}...")
            try:
                df = pd.read_csv(features_file)
                print(f"    ✓ Loaded {len(df)} games")
                all_features.append(df)
            except Exception as e:
                print(f"    ✗ Error: {e}")

    if not all_features:
        print("✗ No features files found!")
        return None

    combined_df = pd.concat(all_features, ignore_index=True)

    # Convert numeric columns
    for col in combined_df.columns:
        if any(x in col.lower() for x in ['odds', 'line', 'spread', 'total', 'score', 'pace', 'efg', 'adj', 'avg']):
            combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')

    print(f"✓ Combined: {len(combined_df)} total games\n")
    return combined_df


def get_feature_columns(df):
    """Identify numeric feature columns for O/U models"""
    metadata_cols = {
        'game_id', 'date', 'season', 'team_1', 'team_2',
        'team_1_score', 'team_2_score', 'actual_total',
        'team_1_conference', 'team_2_conference',
        'team_1_is_home', 'team_2_is_home', 'location',
        'start_time', 'game_odds', 'ou_target'
    }

    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in metadata_cols]

    return feature_cols


def split_train_test(df):
    """Split for optimization: train on oldest to 2nd-most-recent, test on most recent"""
    df['year'] = pd.to_datetime(df['date']).dt.year
    years = sorted(df['year'].unique())
    most_recent_year = years[-1]
    second_most_recent = years[-2] if len(years) >= 2 else None
    third_most_recent = years[-3] if len(years) >= 3 else None

    # For optimization: test on MOST RECENT, train on all others
    train_df = df[df['year'] != most_recent_year].copy()
    test_df = df[df['year'] == most_recent_year].copy()

    # Weight training data: (most_recent - 1) = 4x, (most_recent - 2) = 2x, others = 1x
    sample_weights_train = np.ones(len(train_df))
    if second_most_recent:
        sample_weights_train[train_df['year'] == second_most_recent] = 4.0
    if third_most_recent:
        sample_weights_train[train_df['year'] == third_most_recent] = 2.0

    print(f"Optimization Train/Test Split:")
    print(f"  Years available: {years[0]}-{years[-1]}")
    print(f"  Test year (most recent): {most_recent_year}")
    print(f"  Train years: {sorted([y for y in years if y != most_recent_year])}")
    print(f"  Train: {len(train_df)} games")
    print(f"  Test:  {len(test_df)} games")
    if second_most_recent and third_most_recent:
        print(f"  Sample weights: {second_most_recent}=4x, {third_most_recent}=2x, others=1x\n")
    elif second_most_recent:
        print(f"  Sample weights: {second_most_recent}=4x, others=1x\n")

    return train_df, test_df, sample_weights_train, years


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
                    # Try to convert to appropriate type
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

    # Load XGBoost params
    xgb_params_file = saved_dir / "xgboost_hyperparameters.txt"
    if not xgb_params_file.exists():
        print(f"  ✗ XGBoost hyperparameters not found at {xgb_params_file}")
        print("     Run ou_xgb_optimize_and_save.py first!")
        return None, None, None, None

    xgb_params = read_hyperparameters_from_file(xgb_params_file)
    print(f"  ✓ Loaded XGBoost hyperparameters")

    # Load LightGBM params
    lgb_params_file = saved_dir / "lightgbm_hyperparameters.txt"
    if not lgb_params_file.exists():
        print(f"  ✗ LightGBM hyperparameters not found at {lgb_params_file}")
        print("     Run ou_lgb_optimize_and_save.py first!")
        return None, None, None, None

    lgb_params = read_hyperparameters_from_file(lgb_params_file)
    print(f"  ✓ Loaded LightGBM hyperparameters")

    # Load CatBoost params
    cat_params_file = saved_dir / "catboost_hyperparameters.txt"
    if not cat_params_file.exists():
        print(f"  ✗ CatBoost hyperparameters not found at {cat_params_file}")
        print("     Run ou_catboost_optimize_and_save.py first!")
        return None, None, None, None

    cat_params = read_hyperparameters_from_file(cat_params_file)
    print(f"  ✓ Loaded CatBoost hyperparameters")

    # Load ensemble weights
    weights_file = saved_dir / "ensemble_weights.txt"
    if not weights_file.exists():
        print(f"  ⚠ Ensemble weights not found - using default weights")
        print("     Run ou_find_ensemble_weights.py for optimal weights!")
        ensemble_weights = {'xgb_weight': 0.441, 'lgb_weight': 0.466, 'cat_weight': 0.093}
    else:
        ensemble_weights = read_hyperparameters_from_file(weights_file)
        print(f"  ✓ Loaded ensemble weights: XGB={ensemble_weights['xgb_weight']:.3f}, "
              f"LGB={ensemble_weights['lgb_weight']:.3f}, CAT={ensemble_weights['cat_weight']:.3f}")

    print()
    return xgb_params, lgb_params, cat_params, ensemble_weights


def train_ou_models_with_saved_params(train_df, test_df, feature_cols, sample_weights_train, xgb_params, lgb_params, cat_params):
    """
    Train XGB, LGB, CatBoost on training data using saved hyperparameters and sample weights
    Returns trained models and predictions for both train and test sets
    """
    from xgboost import XGBRegressor
    import lightgbm as lgb
    from catboost import CatBoostRegressor

    # Prepare data
    X_train = train_df[feature_cols].fillna(0).values
    y_train = train_df['actual_total'].values

    X_test = test_df[feature_cols].fillna(0).values
    y_test = test_df['actual_total'].values

    print(f"  Training data: {X_train.shape}")
    print(f"  Test data:     {X_test.shape}\n")

    # Train models with saved hyperparameters and sample weights
    print("  Training XGBoost on training data (with sample weights)...")
    xgb_model = XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1, **xgb_params)
    xgb_model.fit(X_train, y_train, sample_weight=sample_weights_train, verbose=False)

    print("  Training LightGBM on training data (with sample weights)...")
    train_data = lgb.Dataset(X_train, label=y_train, weight=sample_weights_train)
    lgb_model = lgb.train({'objective': 'regression', 'metric': 'mae', 'verbose': -1,
                           'force_row_wise': True, **lgb_params},
                          train_data, num_boost_round=lgb_params['n_estimators'])

    print("  Training CatBoost on training data (with sample weights)...")
    cat_model = CatBoostRegressor(loss_function='MAE', random_state=42, verbose=False, **cat_params)
    cat_model.fit(X_train, y_train, sample_weight=sample_weights_train)

    # Get predictions (NO DATA LEAKAGE)
    xgb_train_preds = xgb_model.predict(X_train)
    lgb_train_preds = lgb_model.predict(X_train)
    cat_train_preds = cat_model.predict(X_train)

    xgb_test_preds = xgb_model.predict(X_test)
    lgb_test_preds = lgb_model.predict(X_test)
    cat_test_preds = cat_model.predict(X_test)

    print(f"  ✓ Generated predictions for train and test sets\n")

    return (xgb_train_preds, lgb_train_preds, cat_train_preds,
            xgb_test_preds, lgb_test_preds, cat_test_preds)


def create_good_bets_features(df, xgb_preds, lgb_preds, cat_preds, ensemble_weights):
    """Create Good Bets features from ensemble predictions using optimal weights"""
    betonline_ou_line = df['betonline_ou_line'].values

    # Convert to confidence scores
    xgb_conf = 1.0 / (1.0 + np.exp(-(xgb_preds - betonline_ou_line) / 3.0))
    lgb_conf = 1.0 / (1.0 + np.exp(-(lgb_preds - betonline_ou_line) / 3.0))
    cat_conf = 1.0 / (1.0 + np.exp(-(cat_preds - betonline_ou_line) / 3.0))

    xgb_conf = np.clip(xgb_conf, 0.01, 0.99)
    lgb_conf = np.clip(lgb_conf, 0.01, 0.99)
    cat_conf = np.clip(cat_conf, 0.01, 0.99)

    # Use optimal ensemble weights
    w_xgb = ensemble_weights['xgb_weight']
    w_lgb = ensemble_weights['lgb_weight']
    w_cat = ensemble_weights['cat_weight']
    ensemble_conf = w_xgb * xgb_conf + w_lgb * lgb_conf + w_cat * cat_conf
    model_std = np.std([xgb_conf, lgb_conf, cat_conf], axis=0)

    features = pd.DataFrame({
        'xgb_confidence_over': xgb_conf,
        'lgb_confidence_over': lgb_conf,
        'cat_confidence_over': cat_conf,
        'ensemble_confidence_over': ensemble_conf,
        'model_std_dev': model_std,
        'betonline_ou_line': betonline_ou_line,
        'avg_ou_line': df.get('avg_ou_line', betonline_ou_line),
        'ou_line_variance': df.get('ou_line_variance', 0),
        'avg_over_odds': df.get('avg_over_odds', 0),
        'avg_under_odds': df.get('avg_under_odds', 0),
        'betonline_over_odds': df.get('betonline_over_odds', 0),
        'betonline_under_odds': df.get('betonline_under_odds', 0)
    })

    target = (df['actual_total'] > df['betonline_ou_line']).astype(int).values

    return features, target


# Good Bets hyperparameter space
rf_space = [
    Integer(100, 500, name='n_estimators'),
    Integer(5, 20, name='max_depth'),
    Integer(2, 10, name='min_samples_split'),
    Integer(1, 5, name='min_samples_leaf')
]


@use_named_args(rf_space)
def rf_objective(**params):
    """Objective for Good Bets Random Forest"""
    global X_gb_train, y_gb_train, X_gb_test, y_gb_test, gb_sample_weights, iteration_count

    iteration_count += 1
    print(f"\n[Iteration {iteration_count}/250]")

    model = RandomForestClassifier(random_state=42, n_jobs=-1, **params)
    model.fit(X_gb_train, y_gb_train, sample_weight=gb_sample_weights)

    test_f1 = f1_score(y_gb_test, model.predict(X_gb_test), zero_division=0)
    train_acc = accuracy_score(y_gb_train, model.predict(X_gb_train))
    test_acc = accuracy_score(y_gb_test, model.predict(X_gb_test))

    overfit_penalty = max(0, train_acc - test_acc - 0.03) * 2
    score = -test_f1 + overfit_penalty

    print(f"  Test F1: {test_f1:.4f}, Overfit: {train_acc - test_acc:.4f}, Score: {score:.4f}")

    return score


if __name__ == "__main__":
    # Load data
    print("STEP 1: Loading Data")
    print("-"*80)
    df = load_all_features()
    if df is None:
        sys.exit(1)

    # Filter for O/U data
    df = df.dropna(subset=['actual_total', 'betonline_ou_line'])
    print(f"Filtered to {len(df)} games with O/U data\n")

    # Split train/test
    print("STEP 2: Train/Test Split")
    print("-"*80)
    train_df, test_df, sample_weights_train, years = split_train_test(df)

    # Get feature columns
    feature_cols = get_feature_columns(df)
    print(f"Using {len(feature_cols)} feature columns\n")

    # Load saved hyperparameters and ensemble weights
    print("STEP 3: Loading Saved Hyperparameters and Ensemble Weights")
    print("-"*80)
    xgb_params, lgb_params, cat_params, ensemble_weights = load_saved_hyperparameters()
    if xgb_params is None:
        sys.exit(1)

    # Train base O/U models with saved params and sample weights
    print("STEP 4: Training XGB, LGB, CatBoost with Saved Hyperparameters")
    print("-"*80)
    (xgb_train_preds, lgb_train_preds, cat_train_preds,
     xgb_test_preds, lgb_test_preds, cat_test_preds) = train_ou_models_with_saved_params(
        train_df, test_df, feature_cols, sample_weights_train, xgb_params, lgb_params, cat_params)

    # Create Good Bets features using optimal ensemble weights
    print("STEP 5: Creating Good Bets Features (with optimal ensemble weights)")
    print("-"*80)
    X_gb_train, y_gb_train = create_good_bets_features(train_df, xgb_train_preds, lgb_train_preds, cat_train_preds, ensemble_weights)
    X_gb_test, y_gb_test = create_good_bets_features(test_df, xgb_test_preds, lgb_test_preds, cat_test_preds, ensemble_weights)

    X_gb_train = X_gb_train.fillna(0).values
    X_gb_test = X_gb_test.fillna(0).values

    # Use same sample weights for Good Bets model
    gb_sample_weights = sample_weights_train

    print(f"  Train features: {X_gb_train.shape}")
    print(f"  Test features:  {X_gb_test.shape}\n")

    # Optimize Good Bets model
    print("STEP 6: Optimizing Good Bets Model (250 iterations)")
    print("-"*80)
    iteration_count = 0

    rf_result = gp_minimize(rf_objective, rf_space, n_calls=250, random_state=42, verbose=False, n_jobs=1)

    print("\n" + "="*80)
    print("OPTIMIZATION COMPLETE")
    print("="*80)
    print(f"Best F1: {-rf_result.fun:.4f}")
    rf_best_params = {s.name: val for s, val in zip(rf_space, rf_result.x)}
    for k, v in rf_best_params.items():
        print(f"  {k}: {v}")

    # Retrain all models on ALL data
    print("\n" + "="*80)
    print("STEP 7: Retraining All Models on ALL Data")
    print("="*80)

    # Prepare full dataset with sample weights prioritizing most recent
    X_all = df[feature_cols].fillna(0).values
    y_all = df['actual_total'].values

    # Weight ALL data: most_recent=4x, second_most_recent=2x, others=1x
    all_years = df['year'].values
    weights_all = np.ones(len(all_years))
    most_recent = years[-1]
    second_most_recent = years[-2] if len(years) >= 2 else None
    weights_all[all_years == most_recent] = 4.0
    if second_most_recent:
        weights_all[all_years == second_most_recent] = 2.0

    print(f"  Training XGB on features{years[0]}-{years[-1]} ({len(X_all)} games)...")
    print(f"  Weights: {most_recent}=4x, {second_most_recent if second_most_recent else 'N/A'}=2x, others=1x")
    from xgboost import XGBRegressor
    xgb_final = XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1, **xgb_params)
    xgb_final.fit(X_all, y_all, sample_weight=weights_all, verbose=False)

    print(f"  Training LGB on features{years[0]}-{years[-1]} ({len(X_all)} games)...")
    import lightgbm as lgb
    train_data_all = lgb.Dataset(X_all, label=y_all, weight=weights_all)
    lgb_final = lgb.train({'objective': 'regression', 'metric': 'mae', 'verbose': -1,
                           'force_row_wise': True, **lgb_params},
                          train_data_all, num_boost_round=lgb_params['n_estimators'])

    print(f"  Training CatBoost on features{years[0]}-{years[-1]} ({len(X_all)} games)...")
    from catboost import CatBoostRegressor
    cat_final = CatBoostRegressor(loss_function='MAE', random_state=42, verbose=False, **cat_params)
    cat_final.fit(X_all, y_all, sample_weight=weights_all)

    # Get predictions on ALL data
    xgb_all_preds = xgb_final.predict(X_all)
    lgb_all_preds = lgb_final.predict(X_all)
    cat_all_preds = cat_final.predict(X_all)

    # Create Good Bets features from ALL data using optimal weights
    X_gb_all, y_gb_all = create_good_bets_features(df, xgb_all_preds, lgb_all_preds, cat_all_preds, ensemble_weights)
    X_gb_all = X_gb_all.fillna(0).values

    print(f"  Training Good Bets on features{years[0]}-{years[-1]} ({len(X_gb_all)} games)...")
    rf_final = RandomForestClassifier(random_state=42, n_jobs=-1, **rf_best_params)
    rf_final.fit(X_gb_all, y_gb_all, sample_weight=weights_all)

    final_acc = accuracy_score(y_gb_all, rf_final.predict(X_gb_all))
    final_f1 = f1_score(y_gb_all, rf_final.predict(X_gb_all))
    print(f"  Final Accuracy: {final_acc:.4f}")
    print(f"  Final F1:       {final_f1:.4f}")

    # Save Good Bets model
    print("\n" + "="*80)
    print("STEP 8: Saving Good Bets Model")
    print("="*80)

    save_dir = ncaamb_dir / "models" / "overunder" / "saved"
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / "ou_good_bets_final.pkl"

    with open(save_path, 'wb') as f:
        pickle.dump(rf_final, f)

    print(f"✓ Saved Good Bets model trained on features{years[0]}-{years[-1]}")
    print(f"  File: {save_path}")

    # Save hyperparameters
    params_path = save_dir / "ou_good_bets_final_hyperparameters.txt"
    with open(params_path, 'w') as f:
        f.write("O/U Good Bets Model - Final Hyperparameters\n")
        f.write("="*50 + "\n\n")
        f.write("Good Bets Random Forest:\n")
        for k, v in rf_best_params.items():
            f.write(f"  {k}: {v}\n")
        f.write(f"\nFinal Accuracy: {final_acc:.4f}\n")
        f.write(f"Final F1: {final_f1:.4f}\n")
        f.write(f"Training Games: {len(X_gb_all)}\n")

    print(f"✓ Saved hyperparameters to: {params_path}")
    print("\n" + "="*80)
    print("DONE!")
    print("="*80)
