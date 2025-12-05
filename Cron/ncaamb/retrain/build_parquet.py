#!/usr/bin/env python3
"""
Build flat dataset with historical match data and odds
Outputs to sampleYYYY.parquet based on current season
"""
from models.build_flat_df import build_flat_df
from models.utils import fetch_games, fetch_teams, fetch_leaderboard, fetch_player_stats
import json
import sys
import os
import polars as pl
from datetime import datetime, timedelta

# Add current directory to path for imports
ncaamb_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ncaamb_dir)

from scrapes.sqlconn import create_connection, fetch


def fetch_odds_batch(game_ids):
    """Fetch odds for multiple games in a single query for better performance"""
    conn = create_connection()
    if not conn:
        return {}

    try:
        # Create placeholders for SQL IN clause
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

        # Group results by game_id
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


def get_current_season():
    """
    Determine the current basketball season based on current date.
    If month >= 10 (October or later), season = current_year + 1
    Otherwise, season = current_year

    Examples:
    - October 2024 -> season 2025
    - September 2024 -> season 2024
    - January 2025 -> season 2025
    """
    now = datetime.now()
    if now.month >= 10:
        return now.year + 1
    else:
        return now.year


if __name__ == "__main__":
    print("Building flat dataset with odds...\n")

    try:
        # Determine current season automatically
        season = get_current_season()
        print(f"Current season: {season}\n")

        # Configuration - adjust dates based on your needs
        start_date = f"{season-1}-11-01"  # Start in November of previous year
        end_date = f"{season}-04-05"      # End in April of current year

        print(f"Date range: {start_date} to {end_date}\n")

        # Load data once
        print("Loading data...")
        games_df = fetch_games(season=season)
        teams_df = fetch_teams(season=season)
        leaderboard_df = fetch_leaderboard()
        player_stats_df = fetch_player_stats(season=season)
        print(f"   ✓ Data loaded\n")

        # Fetch odds for all games first (before building flat_df)
        print("1. Fetching odds for all games...")
        game_ids = games_df['game_id'].to_list()
        odds_dict = fetch_odds_batch(game_ids)
        print(f"   ✓ Retrieved odds for {len(odds_dict)} games")

        # Build flat dataset for date range with odds_dict
        print("2. Building flat DataFrame...")
        flat_df = build_flat_df(season=season, target_date_start=start_date, target_date_end=end_date,
                                games_df=games_df, teams_df=teams_df, leaderboard_df=leaderboard_df,
                                player_stats_df=player_stats_df, odds_dict=odds_dict)
        print(f"   ✓ Built flat dataset with {len(flat_df)} games")

        if len(flat_df) == 0:
            print("   No games found for date range")
        else:
            # Add odds as a column (for feature engineering)
            print("3. Adding odds column...")
            game_ids_in_range = flat_df['game_id'].to_list()
            odds_list = [odds_dict.get(game_id, []) for game_id in game_ids_in_range]
            df_with_odds = flat_df.with_columns(pl.Series("game_odds", odds_list))
            print(f"   ✓ DataFrame shape: {df_with_odds.shape}")

            # Save to parquet with season-specific filename
            output_file = f"sample{season}.parquet"
            print(f"4. Saving to {output_file}...")
            df_with_odds.write_parquet(output_file)
            print(f"   ✓ Saved to {output_file}")

            print("\n✓ Complete!")


    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
