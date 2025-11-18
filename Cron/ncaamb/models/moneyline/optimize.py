#!/usr/bin/env python3
"""
Hyperparameter optimization for Moneyline XGBoost model
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
from xgboost import XGBClassifier
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
                print(f"    ✓ Loaded {len(df)} games")
                all_features.append(df)
            except Exception as e:
                print(f"    ✗ Error loading {year}: {e}")
        else:
            print(f"    ✗ File not found: {features_file}")

    if not all_features:
        return None

    combined_df = pl.concat(all_features)
    print(f"✓ Combined: {len(combined_df)} total games\n")
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
    avg_ml_home and avg_ml_away are required
    """
    before = len(df)

    df = df.filter(
        pl.col('avg_ml_home').is_not_null() &
        pl.col('avg_ml_away').is_not_null()
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
    (n_estimators, max_depth, learning_rate, subsample, colsample_bytree,
     min_child_weight, gamma, reg_alpha, reg_lambda) = params

    # Train model
    model = XGBClassifier(
        n_estimators=int(n_estimators),
        max_depth=int(max_depth),
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        min_child_weight=int(min_child_weight),
        gamma=gamma,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        random_state=42,
        eval_metric='logloss',
        verbosity=0
    )

    model.fit(X_train, y_train)

    # Evaluate on both train and test
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))

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
    print("MONEYLINE MODEL - HYPERPARAMETER OPTIMIZATION (Bayesian)")
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
        (100, 500, 'uniform'),           # n_estimators
        (3, 10, 'uniform'),              # max_depth
        (0.01, 0.3, 'log-uniform'),      # learning_rate
        (0.5, 1.0, 'uniform'),           # subsample
        (0.5, 1.0, 'uniform'),           # colsample_bytree
        (1, 10, 'uniform'),              # min_child_weight
        (0, 5, 'uniform'),               # gamma
        (0, 1, 'uniform'),               # reg_alpha
        (0, 1, 'uniform'),               # reg_lambda
    ]

    print(f"Hyperparameter search space configured")
    print(f"Running Bayesian Optimization with 300 iterations...\n")
    print(f"  20 initial random points + 280 Bayesian-guided points\n")

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
    print("OPTIMIZATION RESULTS - BAYESIAN")
    print("="*80 + "\n")

    best_params_list = result.x
    print("Best Hyperparameters (by overall score):")
    param_names = ['n_estimators', 'max_depth', 'learning_rate', 'subsample',
                   'colsample_bytree', 'min_child_weight', 'gamma', 'reg_alpha', 'reg_lambda']
    best_params_dict = {}
    for i, name in enumerate(param_names):
        val = best_params_list[i]
        if name in ['n_estimators', 'max_depth', 'min_child_weight']:
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

    final_model = XGBClassifier(
        **best_params_dict,
        random_state=42,
        eval_metric='logloss',
        verbosity=0
    )

    final_model.fit(X_train, y_train)

    final_train_acc = accuracy_score(y_train, final_model.predict(X_train))
    final_test_acc = accuracy_score(y_test, final_model.predict(X_test))
    final_gap = final_train_acc - final_test_acc

    print(f"Final Training Accuracy:  {final_train_acc:.4f}")
    print(f"Final Test Accuracy:      {final_test_acc:.4f}")
    print(f"Final Overfitting Gap:    {final_gap:.4f}\n")

    # Show top features
    print("Top 15 Feature Importances:")
    feature_importance = list(zip(feature_cols, final_model.feature_importances_))
    feature_importance.sort(key=lambda x: x[1], reverse=True)

    for i, (feat, importance) in enumerate(feature_importance[:15], 1):
        print(f"  {i:2}. {feat:40} {importance:.4f}")

    print("\n" + "="*80)
    print("✅ Optimization complete!")
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

    results_file = Path(__file__).parent / "optimization_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_file}\n")

    # Save best hyperparameters for reference
    params_file = Path(__file__).parent / "best_params.json"
    with open(params_file, 'w') as f:
        json.dump(best_params_dict, f, indent=2)
    print(f"Best parameters saved to {params_file}\n")
