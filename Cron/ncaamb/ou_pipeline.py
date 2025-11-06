#!/usr/bin/env python3
"""
Complete Over/Under prediction pipeline

Workflow:
1. Build flat dataset (from test.py if needed)
2. Build features (from test_features_ou.py)
3. Load trained model
4. Make predictions
5. Show Over/Under recommendations

This is the complete end-to-end pipeline for O/U prediction.
"""
import polars as pl
import os
import sys
from datetime import datetime

# Add models directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'models'))

from models.ou_model import OUModel
from models.build_ou_features import build_ou_features
from models.build_flat_df import build_flat_df
from models.utils import fetch_games, fetch_teams, fetch_leaderboard, fetch_player_stats

from scrapes.sqlconn import create_connection, fetch


def fetch_odds_batch(game_ids):
    """Fetch odds for multiple games in a single query"""
    conn = create_connection()
    if not conn:
        return {}

    try:
        placeholders = ','.join(['%s'] * len(game_ids))
        query = f"""
            SELECT game_id, bookmaker, ml_home, ml_away, spread_home, spread_pts_home,
                   spread_away, spread_pts_away, over_odds, under_odds, over_point, under_point, start_time
            FROM odds
            WHERE game_id IN ({placeholders})
            ORDER BY game_id, bookmaker, start_time
        """
        results = fetch(conn, query, tuple(game_ids))
        conn.close()

        odds_dict = {}
        for row in results:
            game_id = row['game_id']
            if game_id not in odds_dict:
                odds_dict[game_id] = []
            odds_dict[game_id].append(row)

        return odds_dict
    except Exception as e:
        print(f"Error fetching odds: {e}")
        if conn:
            conn.close()
        return {}


def run_pipeline(start_date: str, end_date: str, use_saved_features: bool = True,
                 model_path: str = "ou_model.pkl") -> pl.DataFrame:
    """
    Run complete O/U prediction pipeline

    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        use_saved_features: If True, use saved ou_features.csv; if False rebuild features
        model_path: Path to trained model

    Returns:
        DataFrame with predictions
    """
    print("="*100)
    print("OVER/UNDER PREDICTION PIPELINE")
    print("="*100)

    # Step 1: Build flat dataset
    if not use_saved_features:
        print("\n1. Loading data...")
        season = 2025
        games_df = fetch_games(season=season)
        teams_df = fetch_teams(season=season)
        leaderboard_df = fetch_leaderboard()
        player_stats_df = fetch_player_stats(season=season)

        # Step 2: Fetch odds for all games (before building flat_df)
        print("\n2. Fetching odds...")
        game_ids = games_df['game_id'].to_list()
        odds_dict = fetch_odds_batch(game_ids)
        print(f"   Retrieved odds for {len(odds_dict)} games")

        # Step 3: Build flat dataset with odds_dict
        print("\n3. Building flat dataset...")
        flat_df = build_flat_df(
            season=season,
            target_date_start=start_date,
            target_date_end=end_date,
            games_df=games_df,
            teams_df=teams_df,
            leaderboard_df=leaderboard_df,
            player_stats_df=player_stats_df,
            odds_dict=odds_dict
        )
        print(f"   Built flat dataset with {len(flat_df)} games")

        # Add odds as column (for feature engineering)
        print("\n4. Adding odds to dataset...")
        game_ids_in_range = flat_df['game_id'].to_list()
        odds_list = [odds_dict.get(game_id, []) for game_id in game_ids_in_range]
        flat_df = flat_df.with_columns(pl.Series("game_odds", odds_list))

        # Step 5: Build features
        print("\n5. Building features...")
        features_df = build_ou_features(flat_df)
        features_df.write_csv("ou_features_pipeline.csv")
        print(f"   Built {len(features_df.columns)} features for {len(features_df)} games")
    else:
        print("\n1. Loading saved features...")
        features_df = pl.read_csv("ou_features.csv")
        print(f"   Loaded {len(features_df)} games with {len(features_df.columns)} features")

    # Step 6: Load model and make predictions
    print("\n6. Loading model and making predictions...")
    model = OUModel(model_path)
    predictions = model.predict(features_df)

    # Step 7: Create output with O/U signals
    print("\n7. Generating O/U signals...")

    output = pl.DataFrame({
        'game_id': predictions['game_id'],
        'date': predictions['date'],
        'team_1': predictions['team_1'],
        'team_2': predictions['team_2'],
        'predicted_total': [round(x, 1) for x in predictions['predicted_total']],
        'actual_total': predictions['actual_total'],
        'market_line': features_df['avg_ou_line'].to_list(),
    })

    # Add O/U signal
    ou_signals = []
    for i in range(len(output)):
        pred = output[i, 'predicted_total']
        market = output[i, 'market_line']

        if market is None or market == 0:
            signal = "No Line"
        elif pred > market:
            edge = round(pred - market, 1)
            signal = f"Over (+{edge})"
        else:
            edge = round(market - pred, 1)
            signal = f"Under (+{edge})"

        ou_signals.append(signal)

    output = output.with_columns(pl.Series("ou_signal", ou_signals))

    # Step 8: Display results
    print("\n8. Results:")
    print("-" * 130)
    print(f"{'Date':<12} {'Team 1':<20} {'Team 2':<20} {'Line':<8} {'Predicted':<12} {'Signal':<15}")
    print("-" * 130)

    for row in output.iter_rows(named=True):
        date = str(row['date'])
        t1 = row['team_1'][:18]
        t2 = row['team_2'][:18]
        line = f"{row['market_line']:.1f}" if row['market_line'] else "N/A"
        pred = f"{row['predicted_total']:.1f}"
        signal = row['ou_signal']

        print(f"{date:<12} {t1:<20} {t2:<20} {line:>7} {pred:>11} {signal:<15}")

    # Save results
    print("\n9. Saving results...")
    output.write_csv("ou_predictions_with_signals.csv")
    print("   Saved to ou_predictions_with_signals.csv")

    print("\n" + "="*100)
    print("PIPELINE COMPLETE")
    print("="*100)

    return output


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run O/U prediction pipeline")
    parser.add_argument("--start", default="2025-02-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2025-02-08", help="End date (YYYY-MM-DD)")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild features from DB")
    parser.add_argument("--model", default="ou_model.pkl", help="Path to trained model")

    args = parser.parse_args()

    # Run pipeline
    results = run_pipeline(
        args.start,
        args.end,
        use_saved_features=not args.rebuild,
        model_path=args.model
    )

    # Summary
    print("\nSummary:")
    print(f"- Total games: {len(results)}")
    print(f"- Over signals: {len(results.filter(pl.col('ou_signal').str.starts_with('Over')))}")
    print(f"- Under signals: {len(results.filter(pl.col('ou_signal').str.starts_with('Under')))}")
