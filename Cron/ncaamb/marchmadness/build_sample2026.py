#!/usr/bin/env python3
"""
Build flat dataset for 2026 season (completed games only).
Outputs sample2026.parquet to the ncaamb/ parent directory.

Run from ncaamb/ directory:
    python marchmadness/build_sample2026.py
"""

import sys
import os
from pathlib import Path
import polars as pl

# Add ncaamb/ to path so models/ and scrapes/ packages are found
ncaamb_dir = Path(__file__).parent.parent
sys.path.insert(0, str(ncaamb_dir))

from models.build_flat_df import build_flat_df
from models.utils import fetch_games, fetch_teams, fetch_leaderboard, fetch_player_stats
from scrapes.sqlconn import create_connection, fetch as sql_fetch

SEASON = 2026
START_DATE = f"{SEASON - 1}-11-01"
END_DATE   = f"{SEASON}-04-07"
OUTPUT_FILE = ncaamb_dir / f"sample{SEASON}.parquet"


def fetch_odds_batch(game_ids: list) -> dict:
    """Fetch odds for multiple games in a single query."""
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
        results = sql_fetch(conn, query, tuple(game_ids))
        conn.close()

        odds_dict = {}
        for row in results:
            gid = row['game_id']
            if gid not in odds_dict:
                odds_dict[gid] = []
            odds_dict[gid].append(row)
        return odds_dict
    except Exception as e:
        print(f"Error fetching odds: {e}")
        if conn:
            conn.close()
        return {}


if __name__ == "__main__":
    print("=" * 80)
    print(f"BUILDING SAMPLE{SEASON}.PARQUET (COMPLETED GAMES ONLY)")
    print("=" * 80)
    print(f"Season:     {SEASON}")
    print(f"Date range: {START_DATE} to {END_DATE}")
    print(f"Output:     {OUTPUT_FILE}\n")

    print("1. Loading data from database...")
    games_df        = fetch_games(season=SEASON)
    teams_df        = fetch_teams(season=SEASON)
    leaderboard_df  = fetch_leaderboard()
    player_stats_df = fetch_player_stats(season=SEASON)
    print(f"   Total games in DB for season {SEASON}: {len(games_df)}")

    # Filter to completed games only (both scores must be present)
    completed = games_df.filter(
        pl.col('team_1_score').is_not_null() &
        pl.col('team_2_score').is_not_null()
    )
    print(f"   Completed games (both scores present): {len(completed)}\n")

    print("2. Fetching odds for completed games...")
    game_ids = completed['game_id'].to_list()
    odds_dict = fetch_odds_batch(game_ids)
    print(f"   Retrieved odds for {len(odds_dict)} games\n")

    print("3. Building flat DataFrame (completed games only)...")
    flat_df = build_flat_df(
        season=SEASON,
        target_date_start=START_DATE,
        target_date_end=END_DATE,
        games_df=completed,          # pass only completed games
        teams_df=teams_df,
        leaderboard_df=leaderboard_df,
        player_stats_df=player_stats_df,
        odds_dict=odds_dict,
    )
    print(f"   Built flat dataset: {len(flat_df)} games\n")

    if len(flat_df) == 0:
        print("No completed games found in date range — nothing to save.")
        sys.exit(0)

    # Double-check: drop any rows still missing scores (shouldn't happen, but be safe)
    before = len(flat_df)
    flat_df = flat_df.filter(
        pl.col('team_1_score').is_not_null() &
        pl.col('team_2_score').is_not_null()
    )
    if len(flat_df) < before:
        print(f"   Dropped {before - len(flat_df)} rows still missing scores after build.")

    print("4. Attaching odds column...")
    odds_list = [odds_dict.get(gid, []) for gid in flat_df['game_id'].to_list()]
    df_with_odds = flat_df.with_columns(pl.Series("game_odds", odds_list))
    print(f"   Final shape: {df_with_odds.shape}\n")

    print(f"5. Saving to {OUTPUT_FILE}...")
    df_with_odds.write_parquet(str(OUTPUT_FILE))
    print(f"   ✅ Saved {len(df_with_odds)} completed games to {OUTPUT_FILE.name}")
    print("\nDone!")
