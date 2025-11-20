#!/usr/bin/env python3
"""
Train moneyline XGBoost model on 2021-2025 data
Save trained model to models/moneyline/saved/xgboost_model.pkl
"""

import os
import sys
from pathlib import Path
import polars as pl
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

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
    print("TRAIN MONEYLINE MODEL ON 2021-2025 DATA")
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
    print(f"Training data shape: X={X_train.shape}\n")

    # Train model
    print("STEP 2: Training Moneyline Model")
    print("-"*80 + "\n")

    print(f"Training on {len(X_train)} samples with {X_train.shape[1]} features...")

    model = XGBClassifier(
        n_estimators=200,
        max_depth=7,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss',
        verbosity=0
    )

    model.fit(X_train, y_train)
    print(f"[+] Model training complete\n")

    # Evaluate on training data
    y_train_pred = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    print(f"Training Accuracy: {train_accuracy:.3f}\n")

    # Save model
    print("STEP 3: Saving Model")
    print("-"*80 + "\n")
    model_dir = Path(__file__).parent / "models" / "moneyline" / "saved"
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "xgboost_model.pkl"
    model.save_model(str(model_path))
    print(f"[+] Model saved to {model_path}")
    print(f"    Training samples: {len(X_train)}")
    print(f"    Feature count: {X_train.shape[1]}")
    print(f"    Training accuracy: {train_accuracy:.3f}\n")

    # Save feature columns for reference
    feature_cols_file = model_dir / "feature_columns.txt"
    with open(feature_cols_file, 'w') as f:
        for col in feature_cols:
            f.write(f"{col}\n")
    print(f"[+] Feature columns saved to {feature_cols_file}\n")

    print("="*80)
    print("[SUCCESS] Moneyline model trained and saved!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
