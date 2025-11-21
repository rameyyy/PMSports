#!/usr/bin/env python3
"""
Train final LightGBM model on 2021-2025 data
Priority: maximize 2025 data in training, use 2021 for validation if needed
Save as pickle/binary for deployment
"""

import os
import sys
from pathlib import Path
import polars as pl
import numpy as np
import lightgbm as lgb
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


def main():
    print("\n")
    print("="*80)
    print("FINAL LIGHTGBM MODEL - TRAIN ON 2021-2025 (2025 PRIORITY)")
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

    # Train LightGBM with optimized hyperparameters
    print("STEP 2: Training LightGBM with Optimized Hyperparameters")
    print("-"*80 + "\n")

    print("Training LightGBM on all 2021-2025 data...")

    train_data = lgb.Dataset(X_all_np, label=y_all_np, feature_name=feature_cols)

    # Optimized hyperparameters from Bayesian optimization
    params = {
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

    model = lgb.train(
        params,
        train_data,
        num_boost_round=200,
        callbacks=[lgb.log_evaluation(period=0)]
    )

    print("[OK] LightGBM training complete\n")

    # Get predictions on full dataset for evaluation
    print("STEP 3: Evaluating Model on Full Dataset")
    print("-"*80 + "\n")

    y_pred_proba = model.predict(X_all_np)
    y_pred = (y_pred_proba > 0.5).astype(int)

    accuracy = accuracy_score(y_all_np, y_pred)
    precision = precision_score(y_all_np, y_pred, zero_division=0)
    recall = recall_score(y_all_np, y_pred, zero_division=0)
    f1 = f1_score(y_all_np, y_pred, zero_division=0)
    auc = roc_auc_score(y_all_np, y_pred_proba)
    tn, fp, fn, tp = confusion_matrix(y_all_np, y_pred).ravel()

    print(f"{'='*80}")
    print(f"FINAL MODEL EVALUATION - Full Dataset (2021-2025)")
    print(f"{'='*80}\n")

    print(f"Accuracy:  {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1 Score:  {f1:.3f}")
    print(f"AUC:       {auc:.3f}\n")

    print(f"Confusion Matrix:")
    print(f"  True Negatives:  {tn:5d} (Team 2 wins, predicted Team 2)")
    print(f"  False Positives: {fp:5d} (Team 2 wins, predicted Team 1)")
    print(f"  False Negatives: {fn:5d} (Team 1 wins, predicted Team 2)")
    print(f"  True Positives:  {tp:5d} (Team 1 wins, predicted Team 1)\n")

    # Feature importance
    print("Top 15 Feature Importances:")
    feature_importance = list(zip(feature_cols, model.feature_importance(importance_type='gain')))
    feature_importance.sort(key=lambda x: x[1], reverse=True)

    for i, (feat, importance) in enumerate(feature_importance[:15], 1):
        print(f"  {i:2}. {feat:40} {importance:10.2f}")

    print()

    # Save model as pickle
    print("STEP 4: Saving Model")
    print("-"*80 + "\n")

    model_save_path = Path(__file__).parent / "saved" / "lightgbm_model_final.pkl"
    model_save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(model_save_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"[OK] Model saved to {model_save_path}\n")

    # Also save as LightGBM native format
    model_lgb_path = Path(__file__).parent / "saved" / "lightgbm_model_final.txt"
    model.save_model(str(model_lgb_path))
    print(f"[OK] Model also saved (LightGBM format) to {model_lgb_path}\n")

    # Save model stats
    stats_path = Path(__file__).parent / "saved" / "lightgbm_model_final_stats.txt"
    with open(stats_path, 'w') as f:
        f.write("Final LightGBM Model Statistics\n")
        f.write("==============================\n\n")
        f.write(f"Training Data: 2021-2025 ({len(X_all_np)} games after filtering)\n")
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
        f.write(f"  num_leaves: 10\n")
        f.write(f"  learning_rate: 0.011906\n")
        f.write(f"  feature_fraction: 0.5\n")
        f.write(f"  bagging_fraction: 0.6902\n")
        f.write(f"  min_data_in_leaf: 100\n")
        f.write(f"  lambda_l1: 5\n")
        f.write(f"  lambda_l2: 5\n")

    print(f"[OK] Model stats saved to {stats_path}\n")

    print("="*80)
    print("[OK] Final LightGBM model training and saving complete!")
    print("="*80 + "\n")
    print(f"Model files saved:")
    print(f"  - {model_save_path}")
    print(f"  - {model_lgb_path}")
    print(f"  - {stats_path}\n")


if __name__ == "__main__":
    main()
