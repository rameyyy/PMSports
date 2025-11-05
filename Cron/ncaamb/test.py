#!/usr/bin/env python3
"""
Test script for building flat dataset with historical match data and odds
"""
from models.build_flat_df import build_flat_df
import json
import sys
import os
import polars as pl
import numpy as np

# Add current directory to path for imports
ncaamb_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ncaamb_dir)

from scrapes.sqlconn import create_connection, fetch
from models.build_features import build_ml_features
from models.train_xgboost import train_xgboost

def get_game_odds(game_id):
    """Fetch all odds for a specific game_id from odds table"""
    conn = create_connection()
    if not conn:
        return []

    try:
        query = """
            SELECT *
            FROM odds
            WHERE game_id = %s
            ORDER BY bookmaker, start_time
        """
        results = fetch(conn, query, (game_id,))
        conn.close()
        return results if results else []
    except Exception as e:
        print(f"Error fetching odds for game_id {game_id}: {e}")
        if conn:
            conn.close()
        return []


if __name__ == "__main__":
    print("="*100)
    print("Testing build_flat_df with games from 2019-11-15 (season 2020)")
    print("="*100 + "\n")

    try:
        # Build flat dataset for specific date (returns polars DataFrame)
        flat_df = build_flat_df(season=2025, target_date="2024-11-24")

        print("\n" + "="*100)
        print("RESULTS")
        print("="*100)
        print(f"\nTotal games: {len(flat_df)}")
        print(f"Flat DataFrame shape: {flat_df.shape}")

        print("\n" + "-"*100)
        print("First game details:")
        print("-"*100)
        first_game = flat_df.row(0, named=True)
        print(f"  Game ID: {first_game['game_id']}")
        print(f"  Date: {first_game['date']}")
        print(f"  Season: {first_game['season']}")
        print(f"  Team 1: {first_game['team_1']}")
        print(f"  Team 1 Conference: {first_game['team_1_conference']}")
        print(f"  Team 1 Score: {first_game['team_1_score']}")
        print(f"  Team 1 Rank: {first_game['team_1_rank']}")
        print(f"  Team 1 W-L: {first_game['team_1_wins']}-{first_game['team_1_losses']}")
        print(f"  Team 1 Historical Games: {first_game['team_1_hist_count']}")
        print(f"  Team 2: {first_game['team_2']}")
        print(f"  Team 2 Conference: {first_game['team_2_conference']}")
        print(f"  Team 2 Score: {first_game['team_2_score']}")
        print(f"  Team 2 Rank: {first_game['team_2_rank']}")
        print(f"  Team 2 W-L: {first_game['team_2_wins']}-{first_game['team_2_losses']}")
        print(f"  Team 2 Historical Games: {first_game['team_2_hist_count']}")
        print(f"  Total Score: {first_game['total_score_outcome']}")
        print(f"  Team 1 Win/Loss: {first_game['team_1_winloss']} ({'Win' if first_game['team_1_winloss'] == 1 else 'Loss'})")

        if first_game['team_1_hist_count'] > 0:
            print(f"\n  Team 1 has {first_game['team_1_hist_count']} historical games")

        # Fetch odds for the first game
        print("\n  Odds for this game:")
        print(f"    Looking for game_id: {first_game['game_id']}")
        game_odds = get_game_odds(first_game['game_id'])
        if game_odds:
            print(f"    Found {len(game_odds)} odds records")
            # Add odds to first_game
            first_game['game_odds'] = game_odds
            for odd in game_odds[:3]:  # Print first 3
                print(f"    {odd.get('bookmaker')}: ML H:{odd.get('ml_home')} A:{odd.get('ml_away')} | Spread H:{odd.get('spread_home')}({odd.get('spread_pts_home')}) A:{odd.get('spread_away')}({odd.get('spread_pts_away')})")
        else:
            print(f"    No odds found for game_id: {first_game['game_id']}")

        # Add odds data to all games
        print("\n  Adding odds data to all games...")
        odds_dict = {}
        for i in range(len(flat_df)):
            game_id = flat_df.row(i, named=True)['game_id']
            game_odds = get_game_odds(game_id)
            odds_dict[game_id] = game_odds if game_odds else []

        print(f"  Retrieved odds for {len(odds_dict)} games")

        # Build ML feature DataFrame (convert flat_df back to list of dicts for compatibility)
        print("\n" + "="*100)
        print("BUILDING ML FEATURE DATAFRAME")
        print("="*100)

        # Convert polars df to list of dicts and add odds
        data_with_odds = []
        for i in range(len(flat_df)):
            row_dict = flat_df.row(i, named=True)
            game_id = row_dict['game_id']
            row_dict['game_odds'] = odds_dict.get(game_id, [])
            data_with_odds.append(row_dict)

        ml_df = build_ml_features(data_with_odds)
        print(f"\nFeature DataFrame shape: {ml_df.shape}")
        print(f"Columns: {ml_df.columns}")
        print(f"\nData types:\n{ml_df.schema}")
        print(f"\nFirst 5 rows:\n{ml_df.head()}")

        # Save to CSV and parquet for ML
        output_csv = "ml_features.csv"
        output_parquet = "ml_features.parquet"

        ml_df.write_csv(output_csv)
        ml_df.write_parquet(output_parquet)

        print(f"\n✓ Feature DataFrame saved to {output_csv}")
        print(f"✓ Feature DataFrame saved to {output_parquet}")

        # Show some stats
        print("\n" + "-"*100)
        print("FEATURE STATISTICS")
        print("-"*100)
        numeric_cols = [col for col in ml_df.columns if col not in ['game_id', 'date', 'season', 'team_1', 'team_2', 'same_conference', 'team_1_win']]
        print(ml_df.select(numeric_cols).describe())

        # Train XGBoost model
        print("\n" + "="*100)
        print("TRAINING XGBOOST MODEL")
        print("="*100)
        xgb_results = train_xgboost(ml_df, test_size=0.33, random_state=42)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
