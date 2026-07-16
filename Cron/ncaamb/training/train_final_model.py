#!/usr/bin/env python3
"""
Train final moneyline model using best hyperparameters from Bayesian optimization
Uses 2021-2025 data for training (all available data for production)
Model will be used to predict 2026 season
"""

import os
import sys
from pathlib import Path
import polars as pl
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

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


def main():
    print("\n" + "="*80)
    print("TRAIN FINAL MONEYLINE MODEL (2021-2025)")
    print("Using Best Hyperparameters from Bayesian Optimization")
    print("="*80 + "\n")

    # Load training data (2021-2025)
    print("STEP 1: Loading Training Data (2021-2025)")
    print("-"*80 + "\n")
    train_df = load_features_by_year(['2021', '2022', '2023', '2024', '2025'])

    if train_df is None:
        print("Failed to load training features")
        return

    train_df = filter_low_quality_games(train_df, min_data_quality=0.5)
    train_df = filter_missing_moneyline_data(train_df)
    train_df = create_target_variable(train_df)
    feature_cols = identify_feature_columns(train_df)

    X_train, y_train = prepare_data(train_df, feature_cols)
    print(f"Total training set (2021-2025): {len(X_train)} samples\n")

    # Best hyperparameters from Bayesian optimization (MODEL 2: Best Generalization)
    print("STEP 2: Training Model with Best Hyperparameters")
    print("-"*80 + "\n")

    best_hyperparams = {
        'n_estimators': 128,
        'max_depth': 3,
        'learning_rate': 0.01,
        'subsample': 0.8214041970901668,
        'colsample_bytree': 1.0,
        'min_child_weight': 7,
        'gamma': 1.3035520618205187,
    }

    print("Using hyperparameters:")
    for name, value in best_hyperparams.items():
        print(f"  {name}: {value}")
    print()

    model = XGBClassifier(
        n_estimators=best_hyperparams['n_estimators'],
        max_depth=best_hyperparams['max_depth'],
        learning_rate=best_hyperparams['learning_rate'],
        subsample=best_hyperparams['subsample'],
        colsample_bytree=best_hyperparams['colsample_bytree'],
        min_child_weight=best_hyperparams['min_child_weight'],
        gamma=best_hyperparams['gamma'],
        random_state=42,
        eval_metric='logloss',
        verbosity=0
    )

    model.fit(X_train, y_train)

    # Evaluate on training data
    train_pred = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, train_pred)
    train_pred_proba = model.predict_proba(X_train)[:, 1]
    train_auc = roc_auc_score(y_train, train_pred_proba)

    print(f"[+] Model training complete\n")
    print(f"Training Set (2021-2025):")
    print(f"  Accuracy: {train_accuracy:.4f}")
    print(f"  AUC: {train_auc:.4f}")
    print(f"  Samples: {len(X_train)}\n")

    # Save model
    print("STEP 3: Saving Model")
    print("-"*80 + "\n")
    model_dir = Path(__file__).parent / "models" / "moneyline" / "saved"
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "xgboost_model.pkl"
    model.save_model(str(model_path))
    print(f"[+] Model saved to {model_path}")

    # Save feature columns
    feature_cols_file = model_dir / "feature_columns.txt"
    with open(feature_cols_file, 'w') as f:
        for col in feature_cols:
            f.write(f"{col}\n")
    print(f"[+] Feature columns saved to {feature_cols_file}\n")

    # Save final hyperparameters info
    params_file = model_dir / "best_hyperparameters.txt"
    with open(params_file, 'w') as f:
        f.write(f"Final Production Model Hyperparameters\n")
        f.write(f"=====================================\n\n")
        f.write(f"Training Data: 2021-2025 ({len(X_train)} games)\n")
        f.write(f"Purpose: Predict 2026 season\n\n")
        f.write(f"Performance on Training Data (2021-2025):\n")
        f.write(f"  Accuracy: {train_accuracy:.4f}\n")
        f.write(f"  AUC: {train_auc:.4f}\n\n")
        f.write(f"Hyperparameters (from Bayesian Optimization - Model 2: Best Generalization):\n")
        for name, value in best_hyperparams.items():
            f.write(f"  {name}: {value}\n")
    print(f"[+] Hyperparameters saved to {params_file}\n")

    # Extract and save feature importance
    print("STEP 4: Extracting Feature Importance")
    print("-"*80 + "\n")

    feature_importance = model.get_booster().get_score(importance_type='weight')

    # Convert to list of (feature_index, importance) and sort
    feature_importance_list = []
    for feature_idx_str, importance in feature_importance.items():
        feature_idx = int(feature_idx_str.replace('f', ''))
        if feature_idx < len(feature_cols):
            feature_importance_list.append((feature_cols[feature_idx], importance))

    # Sort by importance descending
    feature_importance_list.sort(key=lambda x: x[1], reverse=True)

    # Save feature importance
    importance_file = model_dir / "feature_importance.txt"
    with open(importance_file, 'w') as f:
        f.write(f"Feature Importance Ranking\n")
        f.write(f"==========================\n\n")
        f.write(f"Total Features: {len(feature_cols)}\n")
        f.write(f"Features Used in Model: {len(feature_importance_list)}\n\n")
        f.write(f"Rank | Feature Name | Importance Score\n")
        f.write(f"-" * 80 + "\n")

        for rank, (feature_name, importance) in enumerate(feature_importance_list, 1):
            f.write(f"{rank:4d} | {feature_name:50s} | {importance:10.2f}\n")

    print(f"[+] Feature importance saved to {importance_file}")
    print(f"[+] Total features: {len(feature_cols)}")
    print(f"[+] Features used in model: {len(feature_importance_list)}\n")

    # Print top 20 features
    print("Top 20 Most Important Features:")
    print("-"*80)
    for rank, (feature_name, importance) in enumerate(feature_importance_list[:20], 1):
        print(f"{rank:2d}. {feature_name:50s} ({importance})")
    print()

    print("="*80)
    print("[SUCCESS] Final model trained and saved!")
    print("Model is ready for 2026 season predictions")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
