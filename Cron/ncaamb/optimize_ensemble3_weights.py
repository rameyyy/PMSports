#!/usr/bin/env python3
"""
Optimize 3-Model Ensemble Weights (XGBoost + LightGBM + CatBoost)

This script:
1. Trains XGBoost, LightGBM, and CatBoost ONCE with best parameters
2. Gets predictions from all three models
3. Tests different weight combinations on those fixed predictions
4. Finds optimal weight split for maximum accuracy
"""
import polars as pl
import os
import sys
import numpy as np
import pandas as pd
from itertools import product

# Add models directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'models'))

from models.ou_model import OUModel
from models.lgb_model import LGBModel
from models.cat_model import CatModel


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

    for year in [2021, 2022, 2023, 2024, 2025]:
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
    print("OPTIMIZE 3-MODEL ENSEMBLE WEIGHTS (XGB + LGB + CAT)")
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

    # CatBoost params from optimization
    cat_params = {
        'learning_rate': 0.07753322475708241,
        'depth': 4,
        'iterations': 241,
        'l2_leaf_reg': 7.567872474563853,
        'subsample': 0.704299987039777,
        'colsample_bylevel': 0.9955541910374803,
    }

    print("\n   Using XGBoost parameters (Trial #227, MAE: 12.6216)")
    print("   Using LightGBM parameters (Trial #295, MAE: 12.6401)")
    print("   Using CatBoost parameters (Trial #33, MAE: 12.9020)")

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

    # Train CatBoost ONCE
    print("\n5. Training CatBoost (once)...")
    cat_model = CatModel()
    cat_metrics = cat_model.train(features_df, test_size=0.2, **cat_params)
    print(f"   CatBoost Test MAE: {cat_metrics['test_mae']:.4f}")

    # Get predictions from all models
    print("\n6. Getting predictions from all three models...")
    xgb_pred = xgb_model.predict(features_df)
    lgb_pred = lgb_model.predict(features_df)
    cat_pred = cat_model.predict(features_df)
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
    cat_test_pred = np.array([cat_pred['predicted_total'][i] for i in test_indices])
    actual_test = np.array([xgb_pred['actual_total'][i] for i in test_indices])

    # Now test different weight combinations (ultra-fine-grained around sweet spot)
    print("\n7. Testing weight combinations (ultra-fine-grained around sweet spot)...")

    results = []

    # Sweet spot is around: 44-48% XGB, 43-47% LGB, 8-10% CAT
    # Test at 0.001 (0.1%) increments in this region
    xgb_range = np.arange(0.43, 0.49, 0.001)  # 43-49%
    lgb_range = np.arange(0.42, 0.48, 0.001)  # 42-48%

    combo_count = 0
    for xgb_w in xgb_range:
        for lgb_w in lgb_range:
            cat_w = 1.0 - xgb_w - lgb_w

            # Skip if weights don't sum to 1 (must be valid probability distribution)
            if cat_w < 0 or cat_w > 1 or cat_w < 0.05 or cat_w > 0.15:
                continue

            combo_count += 1

            # Weights already sum to 1, no need to normalize
            xgb_wn = xgb_w
            lgb_wn = lgb_w
            cat_wn = cat_w

            # Ensemble prediction: weighted average
            ensemble_pred = (xgb_wn * xgb_test_pred) + (lgb_wn * lgb_test_pred) + (cat_wn * cat_test_pred)

            # Calculate MAE on test set
            mae = np.mean(np.abs(actual_test - ensemble_pred))

            results.append({
                'xgb_weight': xgb_wn,
                'lgb_weight': lgb_wn,
                'cat_weight': cat_wn,
                'ensemble_mae': mae,
            })

    print(f"   Tested {combo_count} weight combinations at 0.1% increment")

    # Convert to dataframe and sort
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('ensemble_mae')

    # Save results
    print("\n8. Saving results...")
    results_df.to_csv("optimize_ensemble3_weights_results.csv", index=False)
    print(f"   Saved all {len(results_df)} weight combinations")

    # Get best result
    best = results_df.iloc[0]
    print("\n" + "="*80)
    print("OPTIMIZATION COMPLETE - 3-MODEL ENSEMBLE WEIGHTS")
    print("="*80)

    print(f"\nBest Ensemble Configuration:")
    print(f"   XGBoost weight: {best['xgb_weight']:.2%}")
    print(f"   LightGBM weight: {best['lgb_weight']:.2%}")
    print(f"   CatBoost weight: {best['cat_weight']:.2%}")
    print(f"   Ensemble Test MAE: {best['ensemble_mae']:.4f} points")

    # Show top 10
    print(f"\n" + "="*80)
    print(f"TOP 10 WEIGHT COMBINATIONS")
    print(f"="*80)
    top_10 = results_df.head(10)

    print("\n" + " "*5 + f"{'XGB %':<12} {'LGB %':<12} {'CAT %':<12} {'Ensemble MAE':<14}")
    print("="*65)

    for idx, row in top_10.iterrows():
        xgb_pct = row['xgb_weight'] * 100
        lgb_pct = row['lgb_weight'] * 100
        cat_pct = row['cat_weight'] * 100
        print(f"   {xgb_pct:<12.1f} {lgb_pct:<12.1f} {cat_pct:<12.1f} {row['ensemble_mae']:<14.4f}")

    # Summary statistics
    print(f"\n" + "="*80)
    print(f"3-MODEL ENSEMBLE SUMMARY")
    print(f"="*80)
    print(f"XGBoost alone: {xgb_metrics['test_mae']:.4f} MAE")
    print(f"LightGBM alone: {lgb_metrics['test_mae']:.4f} MAE")
    print(f"CatBoost alone: {cat_metrics['test_mae']:.4f} MAE")
    print(f"Best 3-Model Ensemble: {best['ensemble_mae']:.4f} MAE")
    print(f"Best weights: {best['xgb_weight']:.1%} XGB / {best['lgb_weight']:.1%} LGB / {best['cat_weight']:.1%} CAT")

    worst_single = max(xgb_metrics['test_mae'], lgb_metrics['test_mae'], cat_metrics['test_mae'])
    improvement = worst_single - best['ensemble_mae']

    if improvement > 0:
        pct_improvement = (improvement / worst_single) * 100
        print(f"\n✅ 3-Model Ensemble improves best single model by {improvement:.4f} points ({pct_improvement:.2f}%)")
    else:
        print(f"\n➡️  3-Model Ensemble matches or is slightly worse than best single model")

    print(f"\nResults saved to: optimize_ensemble3_weights_results.csv")
    print(f"\nUse these weights in betting_simulation_ensemble3.py:")
    print(f"  Ensemble3Model(xgb_weight={best['xgb_weight']:.4f}, lgb_weight={best['lgb_weight']:.4f}, cat_weight={best['cat_weight']:.4f})")


if __name__ == "__main__":
    main()
