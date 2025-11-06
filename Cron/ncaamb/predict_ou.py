#!/usr/bin/env python3
"""
Predict Over/Under totals for games

This script:
1. Builds features from flat game data (or loads existing features)
2. Loads trained model
3. Makes predictions for upcoming/past games
4. Shows Over/Under signals based on Vegas lines
"""
import polars as pl
import os
import sys

# Add models directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'models'))

from models.ou_model import OUModel
from models.build_ou_features import build_ou_features


def predict_from_features(features_path: str = "ou_features.csv", model_path: str = "ou_model.pkl"):
    """
    Make predictions using pre-computed features

    Args:
        features_path: Path to CSV with computed features
        model_path: Path to trained model
    """
    print("="*100)
    print("PREDICTING OVER/UNDER TOTALS")
    print("="*100)

    # Load model
    print("\n1. Loading trained model...")
    model = OUModel(model_path)
    print(f"   Model loaded with {len(model.feature_names)} features")

    # Load features
    print("\n2. Loading features...")
    features_df = pl.read_csv(features_path)
    print(f"   Loaded {len(features_df)} games")

    # Make predictions
    print("\n3. Making predictions...")
    predictions = model.predict(features_df)

    # Create output dataframe
    output_df = pl.DataFrame({
        'game_id': predictions['game_id'],
        'date': predictions['date'],
        'team_1': predictions['team_1'],
        'team_2': predictions['team_2'],
        'actual_total': predictions['actual_total'],
        'predicted_total': [round(x, 1) for x in predictions['predicted_total']],
        'prediction_error': [round(x, 1) if x is not None else None for x in predictions['prediction_error']],
    })

    # Sort by date
    output_df = output_df.sort('date')

    # Display results
    print("\n4. Predictions by Date:")
    print("-" * 120)
    print(f"{'Date':<12} {'Team 1':<20} {'Team 2':<20} {'Actual':<10} {'Predicted':<12} {'Error':<10}")
    print("-" * 120)

    for row in output_df.iter_rows(named=True):
        date = str(row['date'])
        t1 = row['team_1'][:18]
        t2 = row['team_2'][:18]
        actual = row['actual_total']
        pred = row['predicted_total']
        err = row['prediction_error']

        if actual is not None:
            print(f"{date:<12} {t1:<20} {t2:<20} {actual:>9.1f} {pred:>11.1f} {err:>9.1f}")
        else:
            print(f"{date:<12} {t1:<20} {t2:<20} {'N/A':>9} {pred:>11.1f} {'N/A':>9}")

    # Save predictions
    print("\n5. Saving predictions...")
    output_df.write_csv("ou_predictions_detailed.csv")
    print("   Saved to ou_predictions_detailed.csv")

    print("\n" + "="*100)
    print("PREDICTION COMPLETE")
    print("="*100)

    return output_df


def predict_single_game(features_dict: dict, market_line: float, model_path: str = "ou_model.pkl") -> dict:
    """
    Predict total for a single game

    Args:
        features_dict: Dictionary with all feature values
        market_line: Vegas O/U line
        model_path: Path to trained model

    Returns:
        Dictionary with prediction and O/U signal
    """
    model = OUModel(model_path)
    predicted = model.predict_single(features_dict)
    ou_pred = model.get_ou_prediction(predicted, market_line)

    return ou_pred


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Make O/U predictions")
    parser.add_argument("--features", default="ou_features.csv", help="Path to features CSV")
    parser.add_argument("--model", default="ou_model.pkl", help="Path to trained model")

    args = parser.parse_args()

    # Make predictions
    results = predict_from_features(args.features, args.model)

    # Summary statistics
    print("\nSummary Statistics:")
    print("-" * 40)

    games_with_actual = results.filter(pl.col('actual_total').is_not_null())
    if len(games_with_actual) > 0:
        mae = (games_with_actual['prediction_error'].abs().sum() / len(games_with_actual))
        print(f"Mean Absolute Error: {mae:.2f} points")
        print(f"Games evaluated: {len(games_with_actual)}")

        # Count over/under hits
        over_hit = games_with_actual.filter(pl.col('prediction_error') < 0)
        under_hit = games_with_actual.filter(pl.col('prediction_error') > 0)

        print(f"Over predictions (error < 0): {len(over_hit)} games")
        print(f"Under predictions (error > 0): {len(under_hit)} games")
