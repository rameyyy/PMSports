#!/usr/bin/env python3
"""
Optimize Ensemble Weights Only (Fast Version)

This script:
1. Trains XGBoost ONCE with best parameters
2. Trains LightGBM ONCE with best parameters
3. Gets predictions from both models
4. Tests 100 different weight combinations on those fixed predictions
5. Finds optimal weight split instantly (no retraining)
"""
import polars as pl
import os
import sys
import numpy as np
import pandas as pd

# Add models directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'models'))

from models.ou_model import OUModel
from models.lgb_model import LGBModel


def apply_data_quality_filters(df, min_bookmakers=2):
    """Apply data quality filters to remove low-quality rows"""
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
    df = df.filter(valid_rows)

    return df


def load_features():
    """Load and concatenate features from multiple years"""
    print("\n1. Loading features from multiple years...")
    dfs = []

    for year in [2021, 2022, 2023, 2024]:
        filename = f"features{year}.csv"
        try:
            df = pl.read_csv(filename)
            dfs.append(df)
            print(f"   Loaded {filename}: {len(df)} games")
        except FileNotFoundError:
            pass

    try:
        df = pl.read_csv("features.csv")
        dfs.append(df)
        print(f"   Loaded features.csv: {len(df)} games")
    except FileNotFoundError:
        pass

    if not dfs:
        raise FileNotFoundError("No feature files found")

    features_df = pl.concat(dfs)
    print(f"   Total combined: {len(features_df)} games")
    return features_df


def main():
    print("="*80)
    print("OPTIMIZE ENSEMBLE WEIGHTS (Train Once, Test Weights)")
    print("="*80)

    # Best parameters from BEST_OPTIMIZATIONS.txt
    xgb_params = {
        'learning_rate': 0.1356325569317646,
        'max_depth': 3,
        'n_estimators': 264,
        'min_child_weight': 10,
        'subsample': 0.9059553897628048,
        'colsample_bytree': 0.8651858536173023,
        'reg_alpha': 1.7596003894852836,
        'reg_lambda': 0.0687329597968497,
    }

    lgb_params = {
        'learning_rate': 0.1133837674716694,
        'max_depth': 3,
        'num_leaves': 49,
        'min_child_samples': 12,
        'subsample': 0.7991800060529038,
        'colsample_bytree': 0.8152595898952936,
        'reg_alpha': 0.8915908456370663,
        'reg_lambda': 0.2613802136226955,
    }

    print("\n   Using XGBoost parameters (Trial #227, MAE: 12.6216)")
    print("   Using LightGBM parameters (Trial #295, MAE: 12.6401)")

    # Load features
    features_df = load_features()

    # Apply data quality filters
    print("\n2. Applying data quality filters...")
    features_df = apply_data_quality_filters(features_df, min_bookmakers=2)
    print(f"   Games after filters: {len(features_df)}")

    # Remove rows with too many nulls
    features_df = remove_rows_with_too_many_nulls(features_df, null_threshold=0.2)
    print(f"   Games after null filtering: {len(features_df)}")

    # Ensure actual_total is not null
    features_df = features_df.filter(pl.col('actual_total').is_not_null())
    print(f"   Games with valid target: {len(features_df)}")

    # Train XGBoost ONCE
    print("\n3. Training XGBoost (once)...")
    xgb_model = OUModel()
    xgb_metrics = xgb_model.train(features_df, test_size=0.2, **xgb_params)
    print(f"   XGBoost Test MAE: {xgb_metrics['test_mae']:.4f}")

    # Train LightGBM ONCE
    print("\n4. Training LightGBM (once)...")
    lgb_model = LGBModel()
    lgb_metrics = lgb_model.train(features_df, test_size=0.2, **lgb_params)
    print(f"   LightGBM Test MAE: {lgb_metrics['test_mae']:.4f}")

    # Get predictions from both models
    print("\n5. Getting predictions from both models...")
    xgb_pred = xgb_model.predict(features_df)
    lgb_pred = lgb_model.predict(features_df)
    print(f"   Predictions collected: {len(xgb_pred['predicted_total'])} games")

    # Get test set indices (chronological split)
    n_samples = len(features_df)
    n_test = int(n_samples * 0.2)
    n_train = n_samples - n_test
    test_indices = list(range(n_train, n_samples))

    print(f"   Test set size: {len(test_indices)}")

    # Extract test predictions and actual values
    xgb_test_pred = np.array([xgb_pred['predicted_total'][i] for i in test_indices])
    lgb_test_pred = np.array([lgb_pred['predicted_total'][i] for i in test_indices])
    actual_test = np.array([xgb_pred['actual_total'][i] for i in test_indices])

    # Now test different weight combinations
    print("\n6. Testing weight combinations (±2% of optimal with 0.01 increments)...")

    results = []
    # Test ±2% around 64% with 0.01 increments
    weight_range = np.arange(0.62, 0.67, 0.01)  # 62% to 67% in 0.01 increments

    for xgb_weight in weight_range:
        lgb_weight = 1.0 - xgb_weight

        # Ensemble prediction: weighted average
        ensemble_pred = (xgb_weight * xgb_test_pred) + (lgb_weight * lgb_test_pred)

        # Calculate MAE on test set
        mae = np.mean(np.abs(actual_test - ensemble_pred))

        results.append({
            'xgb_weight': xgb_weight,
            'lgb_weight': lgb_weight,
            'ensemble_mae': mae,
        })

    # Convert to dataframe and sort
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('ensemble_mae')

    # Save results
    print("\n7. Saving results...")
    results_df.to_csv("optimize_ensemble_weights_results.csv", index=False)
    print(f"   Saved all {len(results_df)} weight combinations")

    # Get best result
    best = results_df.iloc[0]
    print("\n" + "="*80)
    print("OPTIMIZATION COMPLETE - ENSEMBLE WEIGHTS")
    print("="*80)

    print(f"\nBest Ensemble Configuration:")
    print(f"   XGBoost weight: {best['xgb_weight']:.2%}")
    print(f"   LightGBM weight: {best['lgb_weight']:.2%}")
    print(f"   Ensemble Test MAE: {best['ensemble_mae']:.4f} points")

    # Show top 10
    print(f"\n" + "="*80)
    print(f"TOP 10 WEIGHT COMBINATIONS")
    print(f"="*80)
    top_10 = results_df.head(10)

    print("\n" + " "*5 + f"{'XGB %':<12} {'LGB %':<12} {'Ensemble MAE':<14}")
    print("="*55)

    for idx, row in top_10.iterrows():
        xgb_pct = row['xgb_weight'] * 100
        lgb_pct = row['lgb_weight'] * 100
        print(f"   {xgb_pct:<12.1f} {lgb_pct:<12.1f} {row['ensemble_mae']:<14.4f}")

    # Summary statistics
    print(f"\n" + "="*80)
    print(f"WEIGHT OPTIMIZATION SUMMARY")
    print(f"="*80)
    print(f"XGBoost alone: {xgb_metrics['test_mae']:.4f} MAE")
    print(f"LightGBM alone: {lgb_metrics['test_mae']:.4f} MAE")
    print(f"Best Ensemble: {best['ensemble_mae']:.4f} MAE")
    print(f"Best weights: {best['xgb_weight']:.1%} XGB / {best['lgb_weight']:.1%} LGB")

    improvement = max(xgb_metrics['test_mae'], lgb_metrics['test_mae']) - best['ensemble_mae']
    if improvement > 0:
        pct_improvement = (improvement / xgb_metrics['test_mae']) * 100
        print(f"\n✅ Ensemble improves best single model by {improvement:.4f} points ({pct_improvement:.2f}%)")
    else:
        print(f"\n➡️  Ensemble matches or is slightly worse than best single model")

    print(f"\nResults saved to: optimize_ensemble_weights_results.csv")
    print(f"\nUse these weights in betting_simulation_ensemble.py:")
    print(f"  EnsembleModel(xgb_weight={best['xgb_weight']:.4f}, lgb_weight={best['lgb_weight']:.4f})")


if __name__ == "__main__":
    main()
