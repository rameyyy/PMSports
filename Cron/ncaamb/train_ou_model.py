#!/usr/bin/env python3
"""
Train Over/Under model on feature set

This script:
1. Loads ou_features.csv
2. Trains XGBoost model to predict total points
3. Evaluates on test set
4. Shows feature importance
5. Saves trained model
"""
import polars as pl
import os
import sys

# Add models directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'models'))

from models.ou_model import OUModel


def apply_data_quality_filters(df):
    """
    Apply data quality filters to remove low-quality rows

    Filters (per user requirement):
    1. Remove rows where avg_ou_line is null (no odds data)
    2. Remove rows where num_books_with_ou < 2 (need at least 2 bookmakers)

    Args:
        df: Polars DataFrame with features

    Returns:
        Filtered DataFrame and filter statistics
    """
    initial_count = len(df)
    stats = {'initial': initial_count}

    # Filter 1: Remove null avg_ou_line (no odds data)
    df = df.filter(pl.col('avg_ou_line').is_not_null())
    stats['after_avg_ou_line'] = len(df)

    # Filter 2: Remove rows with < 2 bookmakers for O/U
    df = df.filter(pl.col('num_books_with_ou') >= 2)
    stats['after_num_books'] = len(df)

    # Calculate filtered out
    stats['filtered_out'] = initial_count - len(df)
    stats['percent_retained'] = (len(df) / initial_count * 100) if initial_count > 0 else 0

    return df, stats


def main():
    print("="*80)
    print("TRAINING OVER/UNDER MODEL")
    print("="*80)

    # Load features
    print("\n1. Loading features...")
    features_df = pl.read_csv("ou_features.csv")
    print(f"   Loaded {len(features_df)} games with {len(features_df.columns)} features")

    # Apply data quality filters
    print("\n2. Applying data quality filters...")
    features_df, filter_stats = apply_data_quality_filters(features_df)
    print(f"   Initial games: {filter_stats['initial']}")
    print(f"   After removing null avg_ou_line: {filter_stats['after_avg_ou_line']}")
    print(f"   After requiring >=2 bookmakers: {filter_stats['after_num_books']}")
    print(f"   Final games: {len(features_df)}")
    print(f"   Filtered out: {filter_stats['filtered_out']} ({100-filter_stats['percent_retained']:.1f}%)")
    print(f"   Retained: {filter_stats['percent_retained']:.1f}%")

    # Initialize model
    print("\n3. Initializing XGBoost model...")
    model = OUModel()

    # Train model
    print("\n4. Training model...")
    metrics = model.train(
        features_df,
        test_size=0.2,
        learning_rate=0.05,
        max_depth=7,
        n_estimators=150,
    )

    print("\n   Training Results:")
    print(f"   - Training MAE: {metrics['train_mae']:.2f} points")
    print(f"   - Test MAE: {metrics['test_mae']:.2f} points")
    print(f"   - Training RMSE: {metrics['train_rmse']:.2f} points")
    print(f"   - Test RMSE: {metrics['test_rmse']:.2f} points")
    print(f"   - Train games: {metrics['n_train']}")
    print(f"   - Test games: {metrics['n_test']}")

    # Feature importance
    print("\n5. Top 20 Most Important Features:")
    importance = model.get_feature_importance(top_n=20)

    # Map feature indices to actual names
    feature_name_map = {f'f{i}': model.feature_names[i] for i in range(len(model.feature_names))}

    for i, (feat, score) in enumerate(importance.items(), 1):
        feat_name = feature_name_map.get(feat, feat)
        print(f"   {i:2d}. {feat_name:<50s} {score:>8.1f}")

    # Save model
    print("\n6. Saving model...")
    model_path = "ou_model.pkl"
    model.save_model(model_path)

    # Make predictions on all data
    print("\n7. Making predictions on filtered games...")
    predictions = model.predict(features_df)

    # Show sample predictions
    print("\n   Sample Predictions (first 10 games):")
    print("   " + "-"*100)
    print(f"   {'Game ID':<35} {'Team 1':<15} {'Team 2':<15} {'Actual':<8} {'Predicted':<10} {'Error':<8}")
    print("   " + "-"*100)

    for i in range(min(10, len(predictions['game_id']))):
        game_id = predictions['game_id'][i]
        t1 = predictions['team_1'][i][:12]
        t2 = predictions['team_2'][i][:12]
        actual = predictions['actual_total'][i]
        pred = predictions['predicted_total'][i]
        error = predictions['prediction_error'][i]

        if actual is not None and not np.isnan(actual):
            print(f"   {game_id:<35} {t1:<15} {t2:<15} {actual:>7.1f} {pred:>9.1f} {error:>7.1f}")
        else:
            print(f"   {game_id:<35} {t1:<15} {t2:<15} {'N/A':>7} {pred:>9.1f} {'N/A':>7}")

    # Save predictions
    print("\n8. Saving predictions to CSV...")
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
    print("   Saved to ou_predictions.csv")

    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"\nModel: ou_model.pkl")
    print(f"Predictions: ou_predictions.csv")


if __name__ == "__main__":
    import numpy as np
    main()
