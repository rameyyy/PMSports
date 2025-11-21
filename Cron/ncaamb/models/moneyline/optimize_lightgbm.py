#!/usr/bin/env python3
"""
Hyperparameter optimization for Moneyline LightGBM model
Uses Scikit-Optimize (Bayesian Optimization) to find best hyperparameters
Optimizes for test accuracy while penalizing overfitting
Saves best models and removes games without essential moneyline data
"""

import os
import sys
from pathlib import Path
import polars as pl
import numpy as np
import json
import lightgbm as lgb
from sklearn.metrics import accuracy_score
from skopt import gp_minimize

# Add parent directory to path
ncaamb_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ncaamb_dir))


def load_features_by_year(years: list) -> pl.DataFrame:
    """Load feature files from specified years"""
    features_dir = ncaamb_dir
    all_features = []

    print(f"Loading features for years: {years}")
    for year in years:
        features_file = features_dir / f"features{year}.csv"
        if features_file.exists():
            print(f"  Loading features{year}.csv...")
            try:
                df = pl.read_csv(features_file)
                print(f"    [OK] Loaded {len(df)} games")
                all_features.append(df)
            except Exception as e:
                print(f"    [ERR] Error loading {year}: {e}")
        else:
            print(f"    [ERR] File not found: {features_file}")

    if not all_features:
        return None

    # Handle schema mismatches for betting odds columns
    float_cols_to_fix = [
        'betonline_ml_team_1', 'betonline_ml_team_2',
        'betonline_spread_odds_team_1', 'betonline_spread_odds_team_2',
        'betonline_spread_pts_team_1', 'betonline_spread_pts_team_2',
        'fanduel_spread_odds_team_1', 'fanduel_spread_odds_team_2',
        'fanduel_spread_pts_team_1', 'fanduel_spread_pts_team_2',
        'mybookie_ml_team_1', 'mybookie_ml_team_2',
        'mybookie_spread_odds_team_1', 'mybookie_spread_odds_team_2',
        'mybookie_spread_pts_team_1', 'mybookie_spread_pts_team_2'
    ]

    all_features_fixed = []
    for df in all_features:
        for col in float_cols_to_fix:
            if col in df.columns:
                df = df.with_columns(pl.col(col).cast(pl.Float64, strict=False))
        all_features_fixed.append(df)

    combined_df = pl.concat(all_features_fixed)
    print(f"[OK] Combined: {len(combined_df)} total games\n")
    return combined_df


def filter_low_quality_games(df: pl.DataFrame, min_data_quality: float = 0.5) -> pl.DataFrame:
    """Filter out early season games"""
    before = len(df)

    if 'team_1_data_quality' in df.columns and 'team_2_data_quality' in df.columns:
        df = df.filter(
            (pl.col('team_1_data_quality') >= min_data_quality) &
            (pl.col('team_2_data_quality') >= min_data_quality)
        )

    after = len(df)
    removed = before - after
    print(f"Filtered low-quality games: removed {removed}, kept {after}\n")
    return df


def filter_missing_moneyline_data(df: pl.DataFrame) -> pl.DataFrame:
    """
    Remove games without essential moneyline data
    avg_ml_team_1 and avg_ml_team_2 are required
    """
    before = len(df)

    df = df.filter(
        pl.col('avg_ml_team_1').is_not_null() &
        pl.col('avg_ml_team_2').is_not_null()
    )

    after = len(df)
    removed = before - after
    print(f"Filtered missing moneyline data: removed {removed}, kept {after}\n")
    return df


def create_target_variable(df: pl.DataFrame) -> pl.DataFrame:
    """Create binary target variable"""
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
    print(f"  Team 1 wins: {df_with_scores.filter(pl.col('ml_target') == 1).height}")
    print(f"  Team 2 wins: {df_with_scores.filter(pl.col('ml_target') == 0).height}\n")
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
        'team_1_hist_count', 'team_2_hist_count',
        'start_time', 'game_odds', 'ml_target'
    }

    feature_cols = []
    for col in df.columns:
        if col not in metadata_cols:
            dtype = df[col].dtype
            if dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]:
                feature_cols.append(col)

    return feature_cols


def prepare_data(df: pl.DataFrame, feature_cols: list) -> tuple:
    """Prepare X and y"""
    X = df.select(feature_cols).fill_null(0).to_numpy()
    y = df.select('ml_target').to_numpy().ravel()
    return X, y


# Global variables for optimization
best_test_acc = 0
best_overall_score = float('-inf')
best_train_acc = 0
best_test_acc_for_overall = 0
best_overfitting = float('inf')
trial_count = 0


def objective(params):
    """Objective function for Scikit-Optimize (Bayesian Optimization)"""
    global best_test_acc, best_overall_score, best_train_acc, best_test_acc_for_overall, best_overfitting, trial_count

    trial_count += 1

    # Unpack parameters
    (num_leaves, learning_rate, feature_fraction, bagging_fraction,
     min_data_in_leaf, lambda_l1, lambda_l2) = params

    # Create LightGBM dataset
    train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols)

    # Train model
    params_dict = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'num_leaves': int(num_leaves),
        'learning_rate': learning_rate,
        'feature_fraction': feature_fraction,
        'bagging_fraction': bagging_fraction,
        'bagging_freq': 5,
        'min_data_in_leaf': int(min_data_in_leaf),
        'lambda_l1': lambda_l1,
        'lambda_l2': lambda_l2,
        'verbose': -1
    }

    model = lgb.train(
        params_dict,
        train_data,
        num_boost_round=200,
        callbacks=[lgb.log_evaluation(period=0)]
    )

    # Evaluate on both train and test
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_acc = accuracy_score(y_train, (train_pred > 0.5).astype(int))
    test_acc = accuracy_score(y_test, (test_pred > 0.5).astype(int))

    # Objective: maximize test accuracy while penalizing overfitting
    overfitting_gap = train_acc - test_acc
    penalty = overfitting_gap * 0.5
    overall_score = test_acc - penalty

    # Track best models
    if test_acc > best_test_acc:
        best_test_acc = test_acc

    if overall_score > best_overall_score:
        best_overall_score = overall_score
        best_train_acc = train_acc
        best_test_acc_for_overall = test_acc
        best_overfitting = overfitting_gap

    print(f"Trial {trial_count:3d}: Test={test_acc:.4f}, Train={train_acc:.4f}, Gap={overfitting_gap:.4f}, Score={overall_score:.4f}")

    # Return negative because gp_minimize minimizes
    return -overall_score


if __name__ == "__main__":
    print("\n")
    print("="*80)
    print("MONEYLINE MODEL - LIGHTGBM HYPERPARAMETER OPTIMIZATION (Bayesian)")
    print("="*80 + "\n")

    # Load training data (2021-2024)
    print("STEP 1: Loading Training Data (2021-2024)")
    print("-"*80 + "\n")
    train_df = load_features_by_year(['2021', '2022', '2023', '2024'])

    if train_df is None:
        print("Failed to load training features")
        sys.exit(1)

    train_df = filter_low_quality_games(train_df, min_data_quality=0.5)
    train_df = filter_missing_moneyline_data(train_df)
    train_df = create_target_variable(train_df)
    feature_cols = identify_feature_columns(train_df)

    X_train, y_train = prepare_data(train_df, feature_cols)
    print(f"Training data shape: X={X_train.shape}\n")

    # Load test data (2025)
    print("STEP 2: Loading Test Data (2025)")
    print("-"*80 + "\n")
    test_df = load_features_by_year(['2025'])

    if test_df is None:
        print("Failed to load test features")
        sys.exit(1)

    test_df = filter_low_quality_games(test_df, min_data_quality=0.5)
    test_df = filter_missing_moneyline_data(test_df)
    test_df = create_target_variable(test_df)
    X_test, y_test = prepare_data(test_df, feature_cols)
    print(f"Test data shape: X={X_test.shape}\n")

    # Define hyperparameter space for Bayesian Optimization
    print("STEP 3: Setting Up Bayesian Optimization")
    print("-"*80 + "\n")

    space_config = [
        (10, 100, 'uniform'),            # num_leaves
        (0.01, 0.3, 'log-uniform'),      # learning_rate
        (0.5, 1.0, 'uniform'),           # feature_fraction
        (0.5, 1.0, 'uniform'),           # bagging_fraction
        (5, 100, 'uniform'),             # min_data_in_leaf
        (0, 5, 'uniform'),               # lambda_l1
        (0, 5, 'uniform'),               # lambda_l2
    ]

    print(f"Hyperparameter search space configured")
    print(f"Running Bayesian Optimization with 300 iterations...\n")
    print(f"  50 initial random points + 250 Bayesian-guided points\n")

    # Run Bayesian optimization
    result = gp_minimize(
        objective,
        space_config,
        n_calls=300,
        n_initial_points=50,
        random_state=42,
        verbose=0
    )

    # Print results
    print("\n" + "="*80)
    print("OPTIMIZATION RESULTS - BAYESIAN (LIGHTGBM)")
    print("="*80 + "\n")

    best_params_list = result.x
    print("Best Hyperparameters (by overall score):")
    param_names = ['num_leaves', 'learning_rate', 'feature_fraction', 'bagging_fraction',
                   'min_data_in_leaf', 'lambda_l1', 'lambda_l2']
    best_params_dict = {}
    for i, name in enumerate(param_names):
        val = best_params_list[i]
        if name in ['num_leaves', 'min_data_in_leaf']:
            val = int(val)
        best_params_dict[name] = val
        print(f"  {name:20} = {val}")

    print("\n" + "-"*80)
    print("BEST MODEL STATISTICS")
    print("-"*80 + "\n")

    print(f"Best Overall Score:     {best_overall_score:.4f}")
    print(f"  Training Accuracy:    {best_train_acc:.4f}")
    print(f"  Test Accuracy:        {best_test_acc_for_overall:.4f}")
    print(f"  Overfitting Gap:      {best_overfitting:.4f}\n")

    print(f"Best Test Accuracy:     {best_test_acc:.4f}\n")

    # Train final model with best hyperparameters
    print("-"*80)
    print("Training final model with best hyperparameters...")
    print("-"*80 + "\n")

    train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols)

    final_params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'num_leaves': int(best_params_dict['num_leaves']),
        'learning_rate': best_params_dict['learning_rate'],
        'feature_fraction': best_params_dict['feature_fraction'],
        'bagging_fraction': best_params_dict['bagging_fraction'],
        'bagging_freq': 5,
        'min_data_in_leaf': int(best_params_dict['min_data_in_leaf']),
        'lambda_l1': best_params_dict['lambda_l1'],
        'lambda_l2': best_params_dict['lambda_l2'],
        'verbose': -1
    }

    final_model = lgb.train(
        final_params,
        train_data,
        num_boost_round=200,
        callbacks=[lgb.log_evaluation(period=0)]
    )

    final_train_pred = final_model.predict(X_train)
    final_test_pred = final_model.predict(X_test)

    final_train_acc = accuracy_score(y_train, (final_train_pred > 0.5).astype(int))
    final_test_acc = accuracy_score(y_test, (final_test_pred > 0.5).astype(int))
    final_gap = final_train_acc - final_test_acc

    print(f"Final Training Accuracy:  {final_train_acc:.4f}")
    print(f"Final Test Accuracy:      {final_test_acc:.4f}")
    print(f"Final Overfitting Gap:    {final_gap:.4f}\n")

    # Show top features
    print("Top 15 Feature Importances:")
    feature_importance = list(zip(feature_cols, final_model.feature_importance(importance_type='gain')))
    feature_importance.sort(key=lambda x: x[1], reverse=True)

    for i, (feat, importance) in enumerate(feature_importance[:15], 1):
        print(f"  {i:2}. {feat:40} {importance:10.2f}")

    print("\n" + "="*80)
    print("[OK] Optimization complete!")
    print("="*80 + "\n")

    # Save results
    results = {
        'best_overall_score': float(best_overall_score),
        'best_train_accuracy': float(best_train_acc),
        'best_test_accuracy': float(best_test_acc_for_overall),
        'best_overfitting_gap': float(best_overfitting),
        'best_test_accuracy_overall': float(best_test_acc),
        'hyperparameters': best_params_dict,
        'final_model_stats': {
            'train_accuracy': float(final_train_acc),
            'test_accuracy': float(final_test_acc),
            'overfitting_gap': float(final_gap)
        }
    }

    results_file = Path(__file__).parent / "optimization_results_lightgbm.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_file}\n")

    # Save best hyperparameters for reference
    params_file = Path(__file__).parent / "best_params_lightgbm.json"
    with open(params_file, 'w') as f:
        json.dump(best_params_dict, f, indent=2)
    print(f"Best parameters saved to {params_file}\n")

    # Save final model
    model_path = Path(__file__).parent / "lightgbm_model_optimized.txt"
    final_model.save_model(str(model_path))
    print(f"Final model saved to {model_path}\n")
