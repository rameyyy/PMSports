#!/usr/bin/env python3
"""
Retrain 3-model ensemble (XGBoost + LightGBM + CatBoost) on corrected features 2021-2025
Save trained models to models/overunder/saved/
"""

import os
import sys
import polars as pl
from pathlib import Path

# Add models directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'models'))

from models.overunder.ensemble3_model import Ensemble3Model


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


def load_features():
    """Load and concatenate features from 2021-2025"""
    print("\n1. Loading features from 2021-2025...")
    dfs = []

    for year in [2021, 2022, 2023, 2024, 2025]:
        filename = f"features{year}.csv"
        try:
            df = pl.read_csv(filename)
            dfs.append(df)
            print(f"   Loaded {filename}: {len(df)} games")
        except FileNotFoundError:
            print(f"   ERROR: {filename} not found")

    if not dfs:
        raise FileNotFoundError("No feature files found")

    # Normalize schemas
    dfs = normalize_schemas(dfs)

    combined = pl.concat(dfs)
    print(f"   Total combined: {len(combined)} games\n")
    return combined


def apply_data_quality_filters(df, min_bookmakers=2):
    """Apply data quality filters"""
    df = df.filter(
        pl.col('team_1_adjoe').is_not_null() &
        pl.col('team_1_adjde').is_not_null() &
        pl.col('team_2_adjoe').is_not_null() &
        pl.col('team_2_adjde').is_not_null()
    )
    df = df.filter(pl.col('avg_ou_line').is_not_null())
    df = df.filter(pl.col('num_books_with_ou') >= min_bookmakers)
    return df


def remove_rows_with_too_many_nulls(df, null_threshold=0.2):
    """Remove rows where null percentage exceeds threshold"""
    metadata_cols = {'game_id', 'date', 'team_1', 'team_2'}
    feature_cols = [c for c in df.columns if c not in metadata_cols]

    null_counts = df.select(feature_cols).select([
        pl.sum_horizontal(pl.all().is_null()).alias('null_count')
    ])

    total_feature_cols = len(feature_cols)
    null_pct = null_counts['null_count'] / total_feature_cols if total_feature_cols > 0 else 0
    valid_rows = null_pct <= null_threshold
    return df.filter(valid_rows)


def main():
    print("\n" + "="*80)
    print("RETRAIN 3-MODEL ENSEMBLE ON CORRECTED FEATURES (2021-2025)")
    print("="*80)

    # Load features
    features_df = load_features()

    # Apply data quality filters
    print("2. Applying data quality filters...")
    features_df = apply_data_quality_filters(features_df, min_bookmakers=2)
    print(f"   Games after filters: {len(features_df)}")

    # Remove rows with too many nulls
    features_df = remove_rows_with_too_many_nulls(features_df, null_threshold=0.2)
    print(f"   Games after null filtering: {len(features_df)}")

    # Ensure actual_total is not null
    features_df = features_df.filter(pl.col('actual_total').is_not_null())
    print(f"   Games with valid target: {len(features_df)}\n")

    # Train 3-model ensemble
    print("3. Training 3-model ensemble...")
    ensemble = Ensemble3Model(xgb_weight=0.441, lgb_weight=0.466, cat_weight=0.093)

    optimized_xgb_params = {
        'learning_rate': 0.1356325569317646,
        'max_depth': 3,
        'n_estimators': 264,
        'min_child_weight': 10,
        'subsample': 0.9059553897628048,
        'colsample_bytree': 0.8651858536173023,
        'reg_alpha': 1.7596003894852836,
        'reg_lambda': 0.0687329597968497,
    }

    optimized_lgb_params = {
        'learning_rate': 0.1133837674716694,
        'max_depth': 3,
        'num_leaves': 49,
        'min_child_samples': 12,
        'subsample': 0.7991800060529038,
        'colsample_bytree': 0.8152595898952936,
        'reg_alpha': 0.8915908456370663,
        'reg_lambda': 0.2613802136226955,
    }

    optimized_cat_params = {
        'learning_rate': 0.076195,
        'depth': 3,
        'iterations': 238,
        'l2_leaf_reg': 9.63,
        'subsample': 0.859,
        'colsample_bylevel': 0.963,
    }

    metrics = ensemble.train(
        features_df,
        test_size=0.2,
        xgb_params=optimized_xgb_params,
        lgb_params=optimized_lgb_params,
        cat_params=optimized_cat_params,
    )

    print(f"   XGBoost Test MAE: {metrics['xgb_test_mae']:.4f}")
    print(f"   LightGBM Test MAE: {metrics['lgb_test_mae']:.4f}")
    print(f"   CatBoost Test MAE: {metrics['cat_test_mae']:.4f}")
    print(f"   3-Model Ensemble Test MAE: {metrics['ensemble_test_mae']:.4f}")
    print(f"   Test games: {metrics['n_test']}\n")

    # Save models
    print("4. Saving trained models...")
    model_dir = Path(__file__).parent / "models" / "overunder" / "saved"
    model_dir.mkdir(parents=True, exist_ok=True)

    # Save XGBoost
    xgb_path = model_dir / "xgboost_model.pkl"
    ensemble.xgb_model.save_model(str(xgb_path))
    print(f"   Saved XGBoost model to {xgb_path}")

    # Save LightGBM
    lgb_path = model_dir / "lightgbm_model.pkl"
    ensemble.lgb_model.save_model(str(lgb_path))
    print(f"   Saved LightGBM model to {lgb_path}")

    # Save CatBoost
    cat_path = model_dir / "catboost_model.pkl"
    ensemble.cat_model.save_model(str(cat_path))
    print(f"   Saved CatBoost model to {cat_path}\n")

    print("="*80)
    print("[SUCCESS] Models trained and saved!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
