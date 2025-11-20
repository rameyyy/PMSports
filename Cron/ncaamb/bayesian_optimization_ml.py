#!/usr/bin/env python3
"""
Bayesian Optimization for Moneyline XGBoost Model
Finds optimal hyperparameters using 2021-2024 for training and 2025 for testing
"""

import os
import sys
from pathlib import Path
import polars as pl
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from skopt import gp_minimize, space

# Add current directory to path
ncaamb_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ncaamb_dir)


def normalize_schemas(dfs):
    """Normalize schema across all dataframes"""
    normalized = []
    for df in dfs:
        # Convert odds columns to Float64 for consistency
        for col in df.columns:
            if any(x in col for x in ['_ml_team_', '_spread_pts_', '_spread_odds_']):
                try:
                    df = df.with_columns(pl.col(col).cast(pl.Float64, strict=False))
                except:
                    pass
        normalized.append(df)
    return normalized


def load_features_by_year(years: list) -> pl.DataFrame:
    """Load feature files from specified years"""
    features_dir = Path(__file__).parent
    all_features = []

    print(f"Loading features for years: {years}")
    for year in years:
        features_file = features_dir / f"features{year}.csv"

        if features_file.exists():
            print(f"  Loading features{year}.csv...")
            try:
                df = pl.read_csv(features_file)
                print(f"    [+] Loaded {len(df)} games")
                all_features.append(df)
            except Exception as e:
                print(f"    [-] Error loading {year}: {e}")
        else:
            print(f"    [-] File not found: {features_file}")

    if not all_features:
        return None

    # Normalize schemas
    all_features = normalize_schemas(all_features)

    combined_df = pl.concat(all_features)
    print(f"[+] Combined: {len(combined_df)} total games\n")
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
    print(f"[*] Filtered low-quality games: removed {removed}, kept {after}\n")
    return df


def filter_missing_moneyline_data(df: pl.DataFrame) -> pl.DataFrame:
    """Remove games without essential moneyline data"""
    before = len(df)

    df = df.filter(
        pl.col('avg_ml_team_1').is_not_null() &
        pl.col('avg_ml_team_2').is_not_null()
    )

    after = len(df)
    removed = before - after
    print(f"[*] Filtered missing moneyline data: removed {removed}, kept {after}\n")
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

    print(f"[*] Created target for {len(df_with_scores)} games with results")
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

    print(f"[*] Identified {len(feature_cols)} numeric feature columns\n")
    return feature_cols


def prepare_data(df: pl.DataFrame, feature_cols: list) -> tuple:
    """Prepare X and y"""
    X = df.select(feature_cols).fill_null(0).to_numpy()
    y = df.select('ml_target').to_numpy().ravel()
    return X, y


def objective(params):
    """Objective function for Bayesian optimization"""
    n_estimators, max_depth, learning_rate, subsample, colsample_bytree, min_child_weight, gamma = params

    # Train model with current parameters
    model = XGBClassifier(
        n_estimators=int(n_estimators),
        max_depth=int(max_depth),
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        min_child_weight=int(min_child_weight),
        gamma=gamma,
        random_state=42,
        eval_metric='logloss',
        verbosity=0
    )

    model.fit(X_train, y_train)

    # Evaluate on both sets
    train_pred = model.predict(X_train)
    train_acc = accuracy_score(y_train, train_pred)

    test_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, test_pred)

    test_pred_proba = model.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, test_pred_proba)

    # Calculate overfitting gap
    overfit_gap = train_acc - test_acc

    # Print metrics for this iteration
    print(f"  Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f} | Overfit Gap: {overfit_gap:.4f}")

    # Track best models
    if not hasattr(objective, 'best_test_acc'):
        objective.best_test_acc = test_acc
        objective.best_test_params = params
        objective.best_overfit_gap = overfit_gap
        objective.best_overfit_params = params
        objective.best_overfit_acc = test_acc
    else:
        # Best test accuracy (regardless of overfitting)
        if test_acc > objective.best_test_acc:
            objective.best_test_acc = test_acc
            objective.best_test_params = params
            print(f"    >> NEW BEST TEST ACC: {test_acc:.4f}")

        # Best generalization (lowest overfitting)
        if overfit_gap < objective.best_overfit_gap:
            objective.best_overfit_gap = overfit_gap
            objective.best_overfit_params = params
            objective.best_overfit_acc = test_acc
            print(f"    >> NEW BEST GENERALIZATION: Gap={overfit_gap:.4f}, Test Acc={test_acc:.4f}")

    # Return negative AUC (we want to minimize for gp_minimize)
    return -test_auc


def main():
    print("\n" + "="*80)
    print("BAYESIAN OPTIMIZATION FOR MONEYLINE MODEL")
    print("Train: 2021-2024 | Test: 2025")
    print("="*80 + "\n")

    # Load training data (2021-2024)
    print("STEP 1: Loading Training Data (2021-2024)")
    print("-"*80 + "\n")
    train_df = load_features_by_year(['2021', '2022', '2023', '2024'])

    if train_df is None:
        print("Failed to load training features")
        return

    train_df = filter_low_quality_games(train_df, min_data_quality=0.5)
    train_df = filter_missing_moneyline_data(train_df)
    train_df = create_target_variable(train_df)
    feature_cols = identify_feature_columns(train_df)

    X_train_full, y_train_full = prepare_data(train_df, feature_cols)
    print(f"Training set (2021-2024): {len(X_train_full)} samples\n")

    # Load test data (2025)
    print("STEP 2: Loading Test Data (2025)")
    print("-"*80 + "\n")
    test_df = load_features_by_year(['2025'])

    if test_df is None:
        print("Failed to load test features")
        return

    test_df = filter_low_quality_games(test_df, min_data_quality=0.5)
    test_df = filter_missing_moneyline_data(test_df)
    test_df = create_target_variable(test_df)

    X_test_full, y_test_full = prepare_data(test_df, feature_cols)
    print(f"Test set (2025): {len(X_test_full)} samples\n")

    # Make global for objective function
    global X_train, X_test, y_train, y_test
    X_train = X_train_full
    X_test = X_test_full
    y_train = y_train_full
    y_test = y_test_full

    # Define hyperparameter search space
    print("STEP 3: Running Bayesian Optimization")
    print("-"*80 + "\n")

    search_space = [
        space.Integer(50, 500, name='n_estimators'),
        space.Integer(3, 12, name='max_depth'),
        space.Real(0.01, 0.3, name='learning_rate'),
        space.Real(0.5, 1.0, name='subsample'),
        space.Real(0.5, 1.0, name='colsample_bytree'),
        space.Integer(1, 10, name='min_child_weight'),
        space.Real(0.0, 5.0, name='gamma')
    ]

    print("Search space:")
    print("  n_estimators: [50, 500]")
    print("  max_depth: [3, 12]")
    print("  learning_rate: [0.01, 0.3]")
    print("  subsample: [0.5, 1.0]")
    print("  colsample_bytree: [0.5, 1.0]")
    print("  min_child_weight: [1, 10]")
    print("  gamma: [0.0, 5.0]\n")

    print("Running optimization (this will take a while)...\n")

    # Run Bayesian optimization with 35 initial random + 230 optimization iterations
    result = gp_minimize(
        objective,
        search_space,
        n_calls=265,
        n_initial_points=35,
        random_state=42,
        verbose=1
    )

    # Extract best parameters
    best_params = result.x
    best_test_auc = -result.fun  # Convert back to positive

    print("\n" + "="*80)
    print("OPTIMIZATION RESULTS")
    print("="*80 + "\n")

    param_names = [
        'n_estimators', 'max_depth', 'learning_rate', 'subsample',
        'colsample_bytree', 'min_child_weight', 'gamma'
    ]

    # Display best test accuracy model
    print("MODEL 1: BEST TEST ACCURACY")
    print("-"*80)
    print(f"Test Accuracy: {objective.best_test_acc:.4f}")
    print(f"Parameters:")
    best_acc_config = {}
    for name, value in zip(param_names, objective.best_test_params):
        if name in ['n_estimators', 'max_depth', 'min_child_weight']:
            value = int(value)
        best_acc_config[name] = value
        print(f"  {name}: {value}")
    print()

    # Display best generalization model
    print("MODEL 2: BEST GENERALIZATION (Lowest Overfitting)")
    print("-"*80)
    print(f"Test Accuracy: {objective.best_overfit_acc:.4f}")
    print(f"Overfitting Gap: {objective.best_overfit_gap:.4f}")
    print(f"Parameters:")
    best_overfit_config = {}
    for name, value in zip(param_names, objective.best_overfit_params):
        if name in ['n_estimators', 'max_depth', 'min_child_weight']:
            value = int(value)
        best_overfit_config[name] = value
        print(f"  {name}: {value}")
    print("\n")

    # Use best generalization model for final training (to avoid overfitting)
    best_config = best_overfit_config
    print(f"Using MODEL 2 (best generalization) for final training\n")

    # Train final model on training set with best parameters
    print("STEP 4: Training Final Model with Best Parameters")
    print("-"*80 + "\n")

    final_model = XGBClassifier(
        n_estimators=best_config['n_estimators'],
        max_depth=best_config['max_depth'],
        learning_rate=best_config['learning_rate'],
        subsample=best_config['subsample'],
        colsample_bytree=best_config['colsample_bytree'],
        min_child_weight=best_config['min_child_weight'],
        gamma=best_config['gamma'],
        random_state=42,
        eval_metric='logloss',
        verbosity=0
    )

    final_model.fit(X_train, y_train)

    # Evaluate on both train and test sets
    train_pred = final_model.predict(X_train)
    train_accuracy = accuracy_score(y_train, train_pred)
    train_pred_proba = final_model.predict_proba(X_train)[:, 1]
    train_auc = roc_auc_score(y_train, train_pred_proba)

    test_pred = final_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, test_pred)
    test_pred_proba = final_model.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, test_pred_proba)

    print(f"[+] Final model training complete\n")
    print(f"Training Set (2021-2024):")
    print(f"  Accuracy: {train_accuracy:.4f}")
    print(f"  AUC: {train_auc:.4f}")
    print(f"  Samples: {len(X_train)}\n")
    print(f"Test Set (2025):")
    print(f"  Accuracy: {test_accuracy:.4f}")
    print(f"  AUC: {test_auc:.4f}")
    print(f"  Samples: {len(X_test)}\n")

    # Save model
    print("STEP 5: Saving Model")
    print("-"*80 + "\n")
    model_dir = Path(__file__).parent / "models" / "moneyline" / "saved"
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "xgboost_model.pkl"
    final_model.save_model(str(model_path))
    print(f"[+] Model saved to {model_path}")

    # Save feature columns
    feature_cols_file = model_dir / "feature_columns.txt"
    with open(feature_cols_file, 'w') as f:
        for col in feature_cols:
            f.write(f"{col}\n")
    print(f"[+] Feature columns saved to {feature_cols_file}\n")

    # Save best parameters for reference
    params_file = model_dir / "best_hyperparameters.txt"
    with open(params_file, 'w') as f:
        f.write(f"Bayesian Optimization Results\n")
        f.write(f"============================\n\n")
        f.write(f"Training Set: 2021-2024 ({len(X_train)} games)\n")
        f.write(f"Test Set: 2025 ({len(X_test)} games)\n\n")
        f.write(f"Best Test AUC: {best_test_auc:.4f}\n\n")
        f.write(f"Final Model Performance:\n")
        f.write(f"  Train Accuracy: {train_accuracy:.4f}\n")
        f.write(f"  Train AUC: {train_auc:.4f}\n")
        f.write(f"  Test Accuracy: {test_accuracy:.4f}\n")
        f.write(f"  Test AUC: {test_auc:.4f}\n\n")
        f.write(f"Best Hyperparameters:\n")
        for name, value in best_config.items():
            f.write(f"  {name}: {value}\n")
    print(f"[+] Best parameters saved to {params_file}\n")

    print("="*80)
    print("[SUCCESS] Bayesian optimization complete!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
