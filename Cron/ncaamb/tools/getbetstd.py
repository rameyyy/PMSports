"""
Extract ensemble predictions from gamestd.txt and compare with BetOnline.ag odds.
Print games where (ensemble - over_point) > 2.29
"""

import os
import re
from pathlib import Path
from dotenv import load_dotenv
from scrapes.sqlconn import create_connection, fetch

# Load environment variables
ncaamb_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(ncaamb_dir, '.env')
load_dotenv(env_path)

# File path for gamestd.txt
GAMESTD_FILE = os.path.join(ncaamb_dir, 'gamestd.txt')


def parse_gamestd() -> list:
    """
    Parse gamestd.txt and extract game_id and ensemble predictions.

    Returns:
        List of dicts with keys: game_id, ensemble_value
    """
    games = []

    if not os.path.exists(GAMESTD_FILE):
        print(f"❌ File not found: {GAMESTD_FILE}")
        return games

    with open(GAMESTD_FILE, 'r') as f:
        content = f.read()

    # Pattern to find game blocks
    # Looking for "Game ID: YYYYMMDD_Team1_Team2" followed by ensemble value
    game_blocks = re.split(r'\n\s*\n', content.strip())

    for block in game_blocks:
        lines = block.strip().split('\n')
        if len(lines) < 2:
            continue

        # Find Game ID line
        game_id = None
        ensemble = None

        for line in lines:
            # Extract game_id
            if 'Game ID:' in line:
                match = re.search(r'Game ID:\s*(\d{8}_.*)', line)
                if match:
                    game_id = match.group(1)

            # Extract ensemble value
            if 'Ensemble:' in line:
                match = re.search(r'Ensemble:\s*([\d.]+)', line)
                if match:
                    ensemble = float(match.group(1))

        if game_id and ensemble is not None:
            games.append({
                'game_id': game_id,
                'ensemble': ensemble
            })

    return games


def get_odds_for_game(conn, game_id: str) -> dict:
    """
    Fetch odds for a specific game from ncaamb.odds table for BetOnline.ag.

    Args:
        conn: Database connection
        game_id: Game ID string (YYYYMMDD_Team1_Team2)

    Returns:
        Dict with over_point value, or empty dict if not found
    """
    query = """
        SELECT over_point
        FROM odds
        WHERE game_id = %s
        AND LOWER(bookmaker) = 'betonlineag'
        LIMIT 1
    """

    results = fetch(conn, query, (game_id,))

    if results:
        return results[0]

    return {}


def check_odds_data_available(conn):
    """Check if odds data is available in the database."""
    query = """
        SELECT COUNT(*) as count
        FROM odds
        WHERE game_id LIKE '20251112_%'
        AND LOWER(bookmaker) = 'betonlineag'
    """
    results = fetch(conn, query, ())

    if results and results[0].get('count', 0) > 0:
        print(f"[INFO] Found {results[0].get('count')} BetOnline odds records for today")
        return True
    else:
        print("[WARNING] No BetOnline odds found for today's games")
        print("[INFO] Odds data may not be populated yet")
        return False


def main():
    """Main execution."""

    # Parse gamestd.txt
    print("[INFO] Parsing gamestd.txt...")
    games = parse_gamestd()

    if not games:
        print("[ERROR] No games found in gamestd.txt")
        return

    print(f"[SUCCESS] Found {len(games)} games")
    print(f"   Sample game IDs: {games[0]['game_id'] if games else 'N/A'}")

    # Connect to database
    print("\n[INFO] Connecting to database...")
    conn = create_connection()

    if not conn:
        print("[ERROR] Could not connect to database")
        return

    # Check if odds data is available
    if not check_odds_data_available(conn):
        conn.close()
        return

    # Process each game
    print("\n[INFO] Checking games with ensemble value > (BetOnline over_point + 2.29)...\n")

    qualifying_games = []

    for game in games:
        game_id = game['game_id']
        ensemble = game['ensemble']

        # Get odds from database
        odds_data = get_odds_for_game(conn, game_id)

        if not odds_data:
            # No odds data found for this game
            continue

        over_point = odds_data.get('over_point')

        if over_point is None:
            continue

        # Convert to float if needed (database returns Decimal)
        over_point = float(over_point)

        # Calculate difference: ensemble - over_point
        difference = ensemble - over_point

        # Check if difference > 2.29
        if difference > 2.3:
            qualifying_games.append({
                'game_id': game_id,
                'ensemble': ensemble,
                'over_point': over_point,
                'difference': difference
            })

            print(f"[MATCH] {game_id}")
            print(f"   Ensemble: {ensemble}")
            print(f"   BetOnline Over: {over_point}")
            print(f"   Difference: {difference:.2f}")
            print()

    conn.close()

    # Summary
    if qualifying_games:
        print(f"\n[SUMMARY] Found {len(qualifying_games)} qualifying games")
        print("=" * 70)
        for game in qualifying_games:
            print(f"{game['game_id']}: {game['difference']:.2f} (Ensemble: {game['ensemble']}, Over: {game['over_point']})")
    else:
        print("\n[INFO] No games found with difference > 2.29")


if __name__ == '__main__':
    main()
