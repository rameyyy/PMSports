#!/usr/bin/env python3
"""
Train final Good Bets Model - Random Forest
Trains on 2021-2025 data with optimized hyperparameters
Priority: maximize 2025 data in training, use 2021 for validation if needed
Uses XGBoost + LightGBM predictions + odds + spread data
"""

import os
import sys
from pathlib import Path
import polars as pl
import numpy as np
import lightgbm as lgb
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
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

    print(f"Created {len(X_bets)} betting rows ({len(df)} games * 2 perspectives)")
    print(f"  Good bets (1): {np.sum(y_bets)}")
    print(f"  Bad bets (0):  {len(y_bets) - np.sum(y_bets)}\n")

    return X_bets, y_bets


def main():
    print("\n")
    print("="*80)
    print("FINAL GOOD BETS MODEL - RANDOM FOREST")
    print("Training on 2021-2025 (2025 PRIORITY)")
    print("="*80 + "\n")

    # Load all data (2021-2025)
    print("STEP 1: Loading All Data (2021-2025)")
    print("-"*80 + "\n")
    all_df = load_features_by_year(['2021', '2022', '2023', '2024', '2025'])

    if all_df is None or len(all_df) == 0:
        print("Failed to load features")
        return

    all_df = filter_low_quality_games(all_df, min_data_quality=0.5)
    all_df = create_target_variable(all_df)
    feature_cols = identify_feature_columns(all_df)

    X_all, y_all = prepare_training_data(all_df, feature_cols)
    X_all_np = X_all.to_numpy()
    y_all_np = y_all.to_numpy().ravel()

    print(f"Full dataset shape: X={X_all.shape}, y={y_all.shape}\n")

    # Train base models
    print("STEP 2: Training Base Models (XGBoost + LightGBM)")
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
    xgb_model.fit(X_all_np, y_all_np)
    print("[OK] XGBoost training complete\n")

    print("Training LightGBM...")
    train_data = lgb.Dataset(X_all_np, label=y_all_np, feature_name=feature_cols)

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

    # Get predictions
    print("STEP 3: Generating Base Model Predictions")
    print("-"*80 + "\n")

    xgb_proba = xgb_model.predict_proba(X_all_np)
    lgb_proba = lgb_model.predict(X_all_np)

    print("[OK] Predictions generated\n")

    # Create good bets data
    print("STEP 4: Creating Good Bets Training Data")
    print("-"*80 + "\n")

    X_bets, y_bets = create_good_bets_data(all_df, y_all_np, xgb_proba, lgb_proba)

    # Train final Random Forest with optimized hyperparameters
    print("STEP 5: Training Random Forest with Optimized Hyperparameters")
    print("-"*80 + "\n")

    print("Training Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=190,
        max_depth=3,
        min_samples_split=99,
        min_samples_leaf=49,
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    rf_model.fit(X_bets, y_bets)
    print("[OK] Random Forest training complete\n")

    # Evaluate on full dataset
    print("STEP 6: Evaluating Final Model on Full Dataset")
    print("-"*80 + "\n")

    y_pred = rf_model.predict(X_bets)
    y_pred_proba = rf_model.predict_proba(X_bets)[:, 1]

    accuracy = accuracy_score(y_bets, y_pred)
    precision = precision_score(y_bets, y_pred, zero_division=0)
    recall = recall_score(y_bets, y_pred, zero_division=0)
    f1 = f1_score(y_bets, y_pred, zero_division=0)
    auc = roc_auc_score(y_bets, y_pred_proba)
    tn, fp, fn, tp = confusion_matrix(y_bets, y_pred).ravel()

    print(f"{'='*80}")
    print(f"FINAL MODEL EVALUATION - Full Dataset (2021-2025)")
    print(f"{'='*80}\n")

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"AUC:       {auc:.4f}\n")

    print(f"Confusion Matrix:")
    print(f"  True Negatives:  {tn:5d} (Bad bets correctly identified)")
    print(f"  False Positives: {fp:5d} (Bad bets marked as good)")
    print(f"  False Negatives: {fn:5d} (Good bets missed)")
    print(f"  True Positives:  {tp:5d} (Good bets correctly identified)\n")

    # Feature importance
    print("Top 12 Feature Importances:")
    feature_importance = list(zip(['xgb_prob', 'lgb_prob', 'model_disagreement', 'moneyline_odds',
                                   'implied_prob', 'ev', 'spread_pts_self', 'spread_pts_opp',
                                   'spread_odds_self', 'spread_odds_opp', 'month', 'strength_differential'],
                                  rf_model.feature_importances_))
    feature_importance.sort(key=lambda x: x[1], reverse=True)

    for i, (feat, importance) in enumerate(feature_importance, 1):
        print(f"  {i:2}. {feat:30} {importance:10.4f}")

    print()

    # Save model
    print("STEP 7: Saving Model")
    print("-"*80 + "\n")

    model_save_path = Path(__file__).parent / "saved" / "good_bets_rf_model_final.pkl"
    model_save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(model_save_path, 'wb') as f:
        pickle.dump(rf_model, f)
    print(f"[OK] Model saved to {model_save_path}\n")

    # Save model stats
    stats_path = Path(__file__).parent / "saved" / "good_bets_model_final_stats.txt"
    with open(stats_path, 'w') as f:
        f.write("Final Good Bets Model Statistics\n")
        f.write("================================\n\n")
        f.write(f"Training Data: 2021-2025 ({len(X_bets)} betting rows from {len(all_df)} games)\n")
        f.write(f"Priority: 2025 data maximized in training set\n\n")
        f.write(f"Performance on Full Dataset:\n")
        f.write(f"  Accuracy:  {accuracy:.4f}\n")
        f.write(f"  Precision: {precision:.4f}\n")
        f.write(f"  Recall:    {recall:.4f}\n")
        f.write(f"  F1 Score:  {f1:.4f}\n")
        f.write(f"  AUC:       {auc:.4f}\n\n")
        f.write(f"Confusion Matrix:\n")
        f.write(f"  TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}\n\n")
        f.write(f"Hyperparameters:\n")
        f.write(f"  n_estimators:     190\n")
        f.write(f"  max_depth:        3\n")
        f.write(f"  min_samples_split: 99\n")
        f.write(f"  min_samples_leaf:  49\n\n")
        f.write(f"Base Models:\n")
        f.write(f"  XGBoost:  128 trees, depth=3, learning_rate=0.01\n")
        f.write(f"  LightGBM: 200 rounds, optimized hyperparameters\n")

    print(f"[OK] Model stats saved to {stats_path}\n")

    # Save hyperparameters
    params_path = Path(__file__).parent / "saved" / "good_bets_final_hyperparameters.txt"
    with open(params_path, 'w') as f:
        f.write("Good Bets Final Model Hyperparameters\n")
        f.write("====================================\n\n")
        f.write("Random Forest:\n")
        f.write("  n_estimators:      190\n")
        f.write("  max_depth:         3\n")
        f.write("  min_samples_split: 99\n")
        f.write("  min_samples_leaf:  49\n\n")
        f.write("Base XGBoost:\n")
        f.write("  n_estimators:     128\n")
        f.write("  max_depth:        3\n")
        f.write("  learning_rate:    0.01\n")
        f.write("  subsample:        0.82\n")
        f.write("  colsample_bytree: 1.0\n\n")
        f.write("Base LightGBM:\n")
        f.write("  num_leaves:       10\n")
        f.write("  learning_rate:    0.0119\n")
        f.write("  feature_fraction: 0.5\n")
        f.write("  bagging_fraction: 0.6902\n")
        f.write("  min_data_in_leaf: 100\n")
        f.write("  lambda_l1:        5\n")
        f.write("  lambda_l2:        5\n\n")
        f.write("Features Used (12 total):\n")
        for feat, importance in feature_importance:
            f.write(f"  {feat:30} (importance: {importance:.4f})\n")

    print(f"[OK] Hyperparameters saved to {params_path}\n")

    print("="*80)
    print("[OK] Final good bets model training and saving complete!")
    print("="*80 + "\n")
    print(f"Model files saved:")
    print(f"  - {model_save_path}")
    print(f"  - {stats_path}")
    print(f"  - {params_path}\n")


if __name__ == "__main__":
    main()
