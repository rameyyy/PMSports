#!/usr/bin/env python3
"""
Comprehensive parameter optimization for XGBoost Over/Under model
Tests up to 300 different parameter combinations
"""
import polars as pl
import os
import sys
import itertools
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'models'))

from models.ou_model import OUModel


def apply_data_quality_filters(df):
    """Apply data quality filters"""
    initial_count = len(df)
    df = df.filter(pl.col('avg_ou_line').is_not_null())
    df = df.filter(pl.col('num_books_with_ou') >= 2)
    return df


def test_parameters(features_df, param_combo, test_size=0.2):
    """Test a single parameter combination"""
    try:
        model = OUModel()

        metrics = model.train(
            features_df,
            test_size=test_size,
            learning_rate=param_combo['learning_rate'],
            max_depth=param_combo['max_depth'],
            min_child_weight=param_combo['min_child_weight'],
            subsample=param_combo['subsample'],
            colsample_bytree=param_combo['colsample_bytree'],
            gamma=param_combo['gamma'],
            n_estimators=param_combo['n_estimators'],
            reg_alpha=param_combo['reg_alpha'],
            reg_lambda=param_combo['reg_lambda'],
        )

        return {
            'params': param_combo,
            'train_mae': metrics['train_mae'],
            'test_mae': metrics['test_mae'],
            'train_rmse': metrics['train_rmse'],
            'test_rmse': metrics['test_rmse'],
            'n_test': metrics['n_test'],
            'success': True
        }
    except Exception as e:
        return {
            'params': param_combo,
            'error': str(e),
            'success': False
        }


def main():
    print("=" * 100)
    print("COMPREHENSIVE XGBOOST PARAMETER OPTIMIZATION")
    print("=" * 100)

    # Load and filter features
    print("\n1. Loading features...")
    features_df = pl.read_csv("ou_features.csv")
    features_df = apply_data_quality_filters(features_df)
    print(f"   Loaded {len(features_df)} games with {len(features_df.columns)} features")

    # Define parameter search space
    print("\n2. Defining parameter search space...")

    param_grid = {
        'learning_rate': [0.01, 0.02, 0.03, 0.05, 0.07, 0.1],
        'max_depth': [4, 5, 6, 7, 8, 9, 10],
        'min_child_weight': [1, 2, 3, 4, 5],
        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
        'gamma': [0, 0.1, 0.2, 0.3],
        'n_estimators': [50, 100, 150, 200],
        'reg_alpha': [0, 0.1, 0.5, 1.0],
        'reg_lambda': [1, 1.5, 2.0],
    }

    # Generate all combinations
    keys = param_grid.keys()
    values = param_grid.values()
    all_combos = list(itertools.product(*values))

    print(f"   Total possible combinations: {len(all_combos)}")
    print(f"   Will test: min(300, {len(all_combos)})")

    # Limit to 300 combinations
    test_combos = all_combos[:300]
    num_tests = len(test_combos)

    # Convert to parameter dicts
    param_combos = []
    for combo in test_combos:
        param_dict = {k: v for k, v in zip(keys, combo)}
        param_combos.append(param_dict)

    # Test parameters
    print(f"\n3. Testing {num_tests} parameter combinations...")
    print("   " + "-" * 96)

    results = []
    successful = 0

    for i, param_combo in enumerate(param_combos):
        if (i + 1) % 20 == 0 or (i + 1) == num_tests:
            print(f"   Progress: {i+1}/{num_tests} combinations tested")

        result = test_parameters(features_df, param_combo)
        if result['success']:
            results.append(result)
            successful += 1

    print(f"\n   Successfully tested: {successful}/{num_tests} combinations")

    # Sort results by test MAE
    results.sort(key=lambda x: x['test_mae'])

    # Display top 30 results
    print("\n" + "=" * 100)
    print("TOP 30 PARAMETER COMBINATIONS (by lowest Test MAE)")
    print("=" * 100)
    print(f"\n{'Rank':<5} {'Test MAE':<10} {'Train MAE':<10} {'Test RMSE':<10} {'Parameters':<60}")
    print("-" * 100)

    for i, result in enumerate(results[:30]):
        params = result['params']
        param_str = f"lr={params['learning_rate']:.2f} d={params['max_depth']} ss={params['subsample']:.1f} cs={params['colsample_bytree']:.1f} ne={params['n_estimators']} g={params['gamma']:.1f}"
        print(f"{i+1:<5} {result['test_mae']:<10.2f} {result['train_mae']:<10.2f} {result['test_rmse']:<10.2f} {param_str:<60}")

    # Best parameters
    best = results[0]
    best_params = best['params']

    print("\n" + "=" * 100)
    print("BEST PARAMETERS FOUND")
    print("=" * 100)
    print(f"\nTest MAE: {best['test_mae']:.2f} points (lowest)")
    print(f"Train MAE: {best['train_mae']:.2f} points")
    print(f"Test RMSE: {best['test_rmse']:.2f} points")
    print(f"Overfitting ratio: {best['test_mae']/best['train_mae']:.2f}x")

    print(f"\nOptimal Parameters:")
    print(f"  learning_rate: {best_params['learning_rate']}")
    print(f"  max_depth: {best_params['max_depth']}")
    print(f"  min_child_weight: {best_params['min_child_weight']}")
    print(f"  subsample: {best_params['subsample']}")
    print(f"  colsample_bytree: {best_params['colsample_bytree']}")
    print(f"  gamma: {best_params['gamma']}")
    print(f"  n_estimators: {best_params['n_estimators']}")
    print(f"  reg_alpha: {best_params['reg_alpha']}")
    print(f"  reg_lambda: {best_params['reg_lambda']}")

    # Save results
    print(f"\n4. Saving results...")
    results_df = pl.DataFrame([
        {
            'rank': i+1,
            'test_mae': r['test_mae'],
            'train_mae': r['train_mae'],
            'test_rmse': r['test_rmse'],
            'train_rmse': r['train_rmse'],
            'learning_rate': r['params']['learning_rate'],
            'max_depth': r['params']['max_depth'],
            'min_child_weight': r['params']['min_child_weight'],
            'subsample': r['params']['subsample'],
            'colsample_bytree': r['params']['colsample_bytree'],
            'gamma': r['params']['gamma'],
            'n_estimators': r['params']['n_estimators'],
            'reg_alpha': r['params']['reg_alpha'],
            'reg_lambda': r['params']['reg_lambda'],
        }
        for i, r in enumerate(results)
    ])

    results_df.write_csv("parameter_optimization_results.csv")
    print("   Saved to parameter_optimization_results.csv")

    # Train final model with best parameters
    print(f"\n5. Training final model with best parameters...")
    model = OUModel()
    final_metrics = model.train(
        features_df,
        test_size=0.2,
        learning_rate=best_params['learning_rate'],
        max_depth=best_params['max_depth'],
        min_child_weight=best_params['min_child_weight'],
        subsample=best_params['subsample'],
        colsample_bytree=best_params['colsample_bytree'],
        gamma=best_params['gamma'],
        n_estimators=best_params['n_estimators'],
        reg_alpha=best_params['reg_alpha'],
        reg_lambda=best_params['reg_lambda'],
    )

    print(f"   Final model MAE: {final_metrics['test_mae']:.2f}")

    # Save final model
    model.save_model("ou_model.pkl")
    print(f"   Model saved to ou_model.pkl")

    # Save predictions
    predictions = model.predict(features_df)
    pred_df = pl.DataFrame({
        'game_id': predictions['game_id'],
        'date': predictions['date'],
        'team_1': predictions['team_1'],
        'team_2': predictions['team_2'],
        'actual_total': predictions['actual_total'],
        'predicted_total': predictions['predicted_total'],
        'prediction_error': predictions['prediction_error'],
    })
    pred_df.write_csv("ou_predictions.csv")
    print(f"   Predictions saved to ou_predictions.csv")

    print("\n" + "=" * 100)
    print("OPTIMIZATION COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()
