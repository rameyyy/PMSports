#!/usr/bin/env python3
"""
Optimize Good Bets Model - Bayesian Optimization
Trains XGBoost + LightGBM ONCE, builds features ONCE, then optimizes Random Forest hyperparameters
Efficient optimization - no retraining of base models or feature rebuilding
"""

import os
import sys
from pathlib import Path
import polars as pl
import numpy as np
import lightgbm as lgb
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from skopt import gp_minimize
import pickle

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

    # Handle schema mismatches
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


def create_target_variable(df: pl.DataFrame) -> pl.DataFrame:
    """Create binary target variable for moneyline"""
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


def prepare_training_data(df: pl.DataFrame, feature_cols: list) -> tuple:
    """Prepare X and y"""
    X = df.select(feature_cols).fill_null(0)
    y = df.select('ml_target')
    return X, y


def get_implied_probability(odds: float) -> float:
    """Convert American odds to implied probability"""
    if odds == 0:
        return 0.5
    if odds > 0:  # Underdog
        return 100 / (odds + 100)
    else:  # Favorite
        return abs(odds) / (abs(odds) + 100)


def calculate_ev(model_prob: float, american_odds: float) -> float:
    """Calculate expected value for a moneyline bet"""
    if american_odds == 0:
        return 0

    if american_odds > 0:
        decimal_odds = 1 + (american_odds / 100)
    else:
        decimal_odds = 1 + (100 / abs(american_odds))

    ev = (model_prob * decimal_odds) - 1
    return ev


def create_good_bets_data(df: pl.DataFrame, y: np.ndarray,
                         xgb_proba: np.ndarray, lgb_proba: np.ndarray) -> tuple:
    """
    Create good bets training data using vectorized Polars operations
    Creates 2 rows per game (one per team perspective)

    Returns:
        Tuple of (X_bets, y_bets) with 2 rows per game
    """
    # Add model predictions to dataframe
    df = df.with_columns([
        pl.lit(xgb_proba[:, 1]).alias('xgb_prob_team_1'),
        pl.lit(lgb_proba).alias('lgb_prob_team_1'),
        pl.lit(y).alias('game_result'),
    ])

    # Fill missing odds/spread data with 0
    for col in ['avg_ml_team_1', 'avg_ml_team_2', 'avg_spread_pts_team_1', 'avg_spread_pts_team_2',
                'avg_spread_odds_team_1', 'avg_spread_odds_team_2', 'month', 'team_1_adjoe',
                'team_1_adjde', 'team_2_adjoe', 'team_2_adjde']:
        if col in df.columns:
            df = df.with_columns(pl.col(col).fill_null(0))
        else:
            df = df.with_columns(pl.lit(0).alias(col))

    # Calculate implied probabilities using vectorized ops
    df = df.with_columns([
        pl.col('avg_ml_team_1').map_elements(
            lambda x: 0.5 if x == 0 else (100/(x+100) if x > 0 else abs(x)/(abs(x)+100)),
            return_dtype=pl.Float64
        ).alias('implied_prob_team_1'),
        pl.col('avg_ml_team_2').map_elements(
            lambda x: 0.5 if x == 0 else (100/(x+100) if x > 0 else abs(x)/(abs(x)+100)),
            return_dtype=pl.Float64
        ).alias('implied_prob_team_2'),
    ])

    # Calculate EV using vectorized ops
    df = df.with_columns([
        (pl.col('xgb_prob_team_1') * pl.when(pl.col('avg_ml_team_1') == 0).then(1.0)
            .when(pl.col('avg_ml_team_1') > 0).then(1 + (pl.col('avg_ml_team_1') / 100))
            .otherwise(1 + (100 / pl.col('avg_ml_team_1').abs())) - 1)
        .alias('ev_team_1'),
        ((1 - pl.col('xgb_prob_team_1')) * pl.when(pl.col('avg_ml_team_2') == 0).then(1.0)
            .when(pl.col('avg_ml_team_2') > 0).then(1 + (pl.col('avg_ml_team_2') / 100))
            .otherwise(1 + (100 / pl.col('avg_ml_team_2').abs())) - 1)
        .alias('ev_team_2'),
    ])

    # Calculate strength differentials
    df = df.with_columns([
        (pl.col('team_1_adjoe') - pl.col('team_2_adjoe')).abs().alias('strength_diff_1'),
        (pl.col('team_2_adjoe') - pl.col('team_1_adjoe')).abs().alias('strength_diff_2'),
    ])

    # Create team 1 perspective rows
    team_1_data = df.select([
        pl.col('xgb_prob_team_1'),
        pl.col('lgb_prob_team_1'),
        (pl.col('xgb_prob_team_1') - pl.col('lgb_prob_team_1')).abs().alias('model_disagreement'),
        pl.col('avg_ml_team_1').alias('moneyline_odds'),
        pl.col('implied_prob_team_1').alias('implied_prob'),
        pl.col('ev_team_1').alias('ev'),
        pl.col('avg_spread_pts_team_1').alias('spread_pts_self'),
        pl.col('avg_spread_pts_team_2').alias('spread_pts_opp'),
        pl.col('avg_spread_odds_team_1').alias('spread_odds_self'),
        pl.col('avg_spread_odds_team_2').alias('spread_odds_opp'),
        pl.col('month'),
        pl.col('strength_diff_1').alias('strength_differential'),
        (pl.col('game_result') == 1).cast(pl.Int32).alias('target'),
    ])

    # Create team 2 perspective rows
    team_2_data = df.select([
        (1 - pl.col('xgb_prob_team_1')).alias('xgb_prob_team_1'),
        (1 - pl.col('lgb_prob_team_1')).alias('lgb_prob_team_1'),
        (pl.col('xgb_prob_team_1') - pl.col('lgb_prob_team_1')).abs().alias('model_disagreement'),
        pl.col('avg_ml_team_2').alias('moneyline_odds'),
        pl.col('implied_prob_team_2').alias('implied_prob'),
        pl.col('ev_team_2').alias('ev'),
        pl.col('avg_spread_pts_team_2').alias('spread_pts_self'),
        pl.col('avg_spread_pts_team_1').alias('spread_pts_opp'),
        pl.col('avg_spread_odds_team_2').alias('spread_odds_self'),
        pl.col('avg_spread_odds_team_1').alias('spread_odds_opp'),
        pl.col('month'),
        pl.col('strength_diff_2').alias('strength_differential'),
        (pl.col('game_result') == 0).cast(pl.Int32).alias('target'),
    ])

    # Combine both perspectives
    all_bets = pl.concat([team_1_data, team_2_data])

    # Convert to numpy for sklearn
    feature_cols = ['xgb_prob_team_1', 'lgb_prob_team_1', 'model_disagreement', 'moneyline_odds',
                   'implied_prob', 'ev', 'spread_pts_self', 'spread_pts_opp',
                   'spread_odds_self', 'spread_odds_opp', 'month', 'strength_differential']

    X_bets = all_bets.select(feature_cols).to_numpy()
    y_bets = all_bets.select('target').to_numpy().ravel()

    print(f"Created {len(X_bets)} betting rows ({len(df)} games × 2 perspectives)")
    print(f"  Good bets (1): {np.sum(y_bets)}")
    print(f"  Bad bets (0):  {len(y_bets) - np.sum(y_bets)}\n")

    return X_bets, y_bets


# Global variables for optimization
best_test_acc = 0
best_overall_score = float('-inf')
best_train_acc = 0
best_test_acc_for_overall = 0
best_overfitting = float('inf')
trial_count = 0


def objective(params):
    """Objective function for Bayesian Optimization"""
    global best_test_acc, best_overall_score, best_train_acc, best_test_acc_for_overall, best_overfitting, trial_count

    trial_count += 1

    # Unpack Random Forest hyperparameters
    (n_estimators, max_depth, min_samples_split, min_samples_leaf) = params

    # Train Random Forest with current hyperparameters
    rf_model = RandomForestClassifier(
        n_estimators=int(n_estimators),
        max_depth=int(max_depth),
        min_samples_split=int(min_samples_split),
        min_samples_leaf=int(min_samples_leaf),
        random_state=42,
        n_jobs=-1,
        verbose=0
    )

    rf_model.fit(X_bets_train, y_bets_train)

    # Evaluate on both train and test
    train_acc = accuracy_score(y_bets_train, rf_model.predict(X_bets_train))
    test_acc = accuracy_score(y_bets_test, rf_model.predict(X_bets_test))

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


def main():
    global X_bets_train, y_bets_train, X_bets_test, y_bets_test

    print("\n")
    print("="*80)
    print("GOOD BETS MODEL - HYPERPARAMETER OPTIMIZATION (Bayesian)")
    print("Train base models ONCE, build features ONCE, optimize RF hyperparameters")
    print("="*80 + "\n")

    # Load training data (2021-2024)
    print("STEP 1: Loading Training Data (2021-2024)")
    print("-"*80 + "\n")
    train_df = load_features_by_year(['2021', '2022', '2023', '2024'])

    if train_df is None:
        print("Failed to load training features")
        return

    train_df = filter_low_quality_games(train_df, min_data_quality=0.5)
    train_df = create_target_variable(train_df)
    feature_cols = identify_feature_columns(train_df)

    X_train, y_train = prepare_training_data(train_df, feature_cols)
    X_train_np = X_train.to_numpy()
    y_train_np = y_train.to_numpy().ravel()

    print(f"Training data shape: X={X_train.shape}\n")

    # Train base models ONCE
    print("STEP 2: Training Base Models (XGBoost + LightGBM) - ONCE ONLY")
    print("-"*80 + "\n")

    print("Training XGBoost...")
    xgb_model = XGBClassifier(
        n_estimators=128,
        max_depth=3,
        learning_rate=0.01,
        subsample=0.82,
        colsample_bytree=1.0,
        random_state=42,
        eval_metric='logloss',
        verbosity=0
    )
    xgb_model.fit(X_train_np, y_train_np)
    print("[OK] XGBoost training complete\n")

    print("Training LightGBM...")
    train_data = lgb.Dataset(X_train_np, label=y_train_np, feature_name=feature_cols)

    lgb_params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'num_leaves': 10,
        'learning_rate': 0.011905546738777037,
        'feature_fraction': 0.5,
        'bagging_fraction': 0.6902301680678105,
        'bagging_freq': 5,
        'min_data_in_leaf': 100,
        'lambda_l1': 5,
        'lambda_l2': 5,
        'verbose': -1
    }

    lgb_model = lgb.train(
        lgb_params,
        train_data,
        num_boost_round=200,
        callbacks=[lgb.log_evaluation(period=0)]
    )
    print("[OK] LightGBM training complete\n")

    # Get predictions ONCE
    print("STEP 3: Generating Base Model Predictions - ONCE ONLY")
    print("-"*80 + "\n")

    xgb_proba_train = xgb_model.predict_proba(X_train_np)
    lgb_proba_train = lgb_model.predict(X_train_np)

    print("[OK] Training predictions generated\n")

    # Build features ONCE
    print("STEP 4: Building Good Bets Features - ONCE ONLY")
    print("-"*80 + "\n")

    X_bets_train, y_bets_train = create_good_bets_data(
        train_df, y_train_np, xgb_proba_train, lgb_proba_train
    )

    # Load test data
    print("STEP 5: Loading Test Data (2025)")
    print("-"*80 + "\n")

    test_df = load_features_by_year(['2025'])

    if test_df is None:
        print("Failed to load test features")
        return

    test_df = filter_low_quality_games(test_df, min_data_quality=0.5)
    test_df = create_target_variable(test_df)

    if len(test_df) == 0:
        print("No test games with results available")
        return

    X_test, y_test = prepare_training_data(test_df, feature_cols)
    X_test_np = X_test.to_numpy()
    y_test_np = y_test.to_numpy().ravel()

    # Get test predictions ONCE
    xgb_proba_test = xgb_model.predict_proba(X_test_np)
    lgb_proba_test = lgb_model.predict(X_test_np)

    # Build test features ONCE
    X_bets_test, y_bets_test = create_good_bets_data(
        test_df, y_test_np, xgb_proba_test, lgb_proba_test
    )

    # Define hyperparameter space for Random Forest optimization
    print("STEP 6: Setting Up Bayesian Optimization")
    print("-"*80 + "\n")

    space_config = [
        (50, 200, 'uniform'),           # n_estimators
        (3, 20, 'uniform'),             # max_depth
        (20, 100, 'uniform'),           # min_samples_split
        (5, 50, 'uniform'),             # min_samples_leaf
    ]

    print(f"Hyperparameter search space configured")
    print(f"Running Bayesian Optimization with 250 iterations...\n")
    print(f"  50 initial random points + 200 Bayesian-guided points\n")

    # Run Bayesian optimization
    result = gp_minimize(
        objective,
        space_config,
        n_calls=250,
        n_initial_points=50,
        random_state=42,
        verbose=0
    )

    # Print results
    print("\n" + "="*80)
    print("OPTIMIZATION RESULTS - BAYESIAN (GOOD BETS MODEL)")
    print("="*80 + "\n")

    best_params_list = result.x
    print("Best Hyperparameters (by overall score):")
    param_names = ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf']
    best_params_dict = {}
    for i, name in enumerate(param_names):
        val = best_params_list[i]
        if name in ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf']:
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

    final_model = RandomForestClassifier(
        **best_params_dict,
        random_state=42,
        n_jobs=-1,
        verbose=0
    )

    final_model.fit(X_bets_train, y_bets_train)

    final_train_acc = accuracy_score(y_bets_train, final_model.predict(X_bets_train))
    final_test_acc = accuracy_score(y_bets_test, final_model.predict(X_bets_test))
    final_gap = final_train_acc - final_test_acc

    print(f"Final Training Accuracy:  {final_train_acc:.4f}")
    print(f"Final Test Accuracy:      {final_test_acc:.4f}")
    print(f"Final Overfitting Gap:    {final_gap:.4f}\n")

    print("="*80)
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

    results_file = Path(__file__).parent / "saved" / "good_bets_optimization_results.json"
    results_file.parent.mkdir(parents=True, exist_ok=True)
    import json
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_file}\n")

    # Save best hyperparameters
    params_file = Path(__file__).parent / "saved" / "good_bets_best_params.json"
    with open(params_file, 'w') as f:
        json.dump(best_params_dict, f, indent=2)
    print(f"Best parameters saved to {params_file}\n")

    # Save final model
    model_path = Path(__file__).parent / "saved" / "good_bets_rf_model_optimized.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(final_model, f)
    print(f"Final model saved to {model_path}\n")


if __name__ == "__main__":
    main()
