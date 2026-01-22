"""
Custom Rating System Builder

Start by pulling leaderboard + moneyline data for a specific date
to understand the data structure and build ratings.
"""

import os
from dotenv import load_dotenv
import mysql.connector
import pandas as pd

# Load environment variables
ncaamb_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
env_path = os.path.join(ncaamb_dir, '.env')
load_dotenv(env_path)


def get_connection():
    """Create MySQL connection"""
    return mysql.connector.connect(
        host=os.getenv("DB_HOST"),
        port=int(os.getenv("DB_PORT")),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("NCAAMB_DB"),
    )


def fetch_leaderboard_for_date(date: str) -> pd.DataFrame:
    """Fetch leaderboard data for a specific date"""
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    query = """
        SELECT * FROM leaderboard
        WHERE date = %s
        ORDER BY `rank`
    """
    cursor.execute(query, (date,))
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return pd.DataFrame(rows)


def fetch_moneyline_for_date(date: str) -> pd.DataFrame:
    """Fetch moneyline data with best book odds for a specific date"""
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    query = """
        SELECT
            game_id,
            team_1,
            team_2,
            team_predicted_to_win,
            ensemble_prob_team_1,
            ensemble_prob_team_2,
            best_book_odds_team_1,
            best_book_odds_team_2,
            best_book_team_1,
            best_book_team_2,
            winning_team,
            game_date
        FROM moneyline
        WHERE game_date = %s
    """
    cursor.execute(query, (date,))
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return pd.DataFrame(rows)


def main():
    target_date = "2025-12-02"

    print("=" * 80)
    print(f"FETCHING DATA FOR {target_date}")
    print("=" * 80)

    # Fetch leaderboard
    print("\nFetching leaderboard data...")
    leaderboard_df = fetch_leaderboard_for_date(target_date)
    print(f"   Found {len(leaderboard_df)} teams in leaderboard")

    if len(leaderboard_df) > 0:
        print("\n   Top 10 teams:")
        print(leaderboard_df[['rank', 'team', 'adjoe', 'adjde', 'barthag', 'wins', 'losses']].head(10).to_string(index=False))

        print("\n   Columns available:")
        print(f"   {list(leaderboard_df.columns)}")

    # Fetch moneyline
    print("\n" + "=" * 80)
    print("Fetching moneyline data...")
    moneyline_df = fetch_moneyline_for_date(target_date)
    print(f"   Found {len(moneyline_df)} games")

    if len(moneyline_df) > 0:
        print("\n   Sample games:")
        print(moneyline_df[['team_1', 'team_2', 'best_book_odds_team_1', 'best_book_odds_team_2', 'winning_team']].head(5).to_string(index=False))

        print("\n   Columns available:")
        print(f"   {list(moneyline_df.columns)}")

    print("\n" + "=" * 80)
    print("Data fetched successfully!")
    print("=" * 80)

    return leaderboard_df, moneyline_df


if __name__ == "__main__":
    leaderboard_df, moneyline_df = main()
