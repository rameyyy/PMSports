#!/usr/bin/env python3
"""
Quick diagnostic script to check if games are loaded for a specific season
"""
import mysql.connector
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv('.env')


def check_games_for_season(season):
    """
    Check how many games are loaded for a specific season

    Args:
        season: Season year to check (e.g., 2022)
    """
    try:
        conn = mysql.connector.connect(
            host=os.getenv("DB_HOST"),
            port=int(os.getenv("DB_PORT")),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            database=os.getenv("NCAAMB_DB"),
        )

        cursor = conn.cursor()

        # Check total games for season
        cursor.execute("SELECT COUNT(*) FROM games WHERE season = %s", (season,))
        total_games = cursor.fetchone()[0]

        # Check date range
        cursor.execute("""
            SELECT MIN(date), MAX(date)
            FROM games
            WHERE season = %s
        """, (season,))
        date_range = cursor.fetchone()

        # Get sample game_ids
        cursor.execute("""
            SELECT game_id
            FROM games
            WHERE season = %s
            ORDER BY date
            LIMIT 5
        """, (season,))
        sample_games = [row[0] for row in cursor.fetchall()]

        cursor.close()
        conn.close()

        print("="*80)
        print(f"Games loaded for season {season}:")
        print(f"  - Total games: {total_games}")

        if total_games > 0:
            print(f"  - Date range: {date_range[0]} to {date_range[1]}")
            print(f"\nSample game_ids:")
            for game_id in sample_games:
                print(f"    - {game_id}")
        else:
            print(f"\n⚠️  WARNING: No games found for season {season}!")
            print(f"You need to run the game loading script first:")
            print(f"  python main.py  (update to use year='{season}', season={season})")

        print("="*80)

        return total_games

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 0


if __name__ == "__main__":
    import sys

    # Get season from command line or use default
    season = int(sys.argv[1]) if len(sys.argv) > 1 else 2022

    check_games_for_season(season)
