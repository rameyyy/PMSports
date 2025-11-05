"""
Map game_ids from ncaamb.games table to odds records.
Reads team_mappings.csv, constructs game_ids, and updates odds table.
"""

import sys
import os
from datetime import datetime
import csv

# Add parent directory to path for imports
ncaamb_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ncaamb_dir)

from scrapes.sqlconn import create_connection, fetch, execute_query

# Path to team mappings CSV
MAPPINGS_CSV = os.path.join(os.path.dirname(__file__), 'team_mappings.csv')


def load_team_mappings():
    """Load team name mappings from CSV file"""
    mappings = {}
    try:
        with open(MAPPINGS_CSV, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get('from_odds_team_name') and row.get('my_team_name'):
                    mappings[row['from_odds_team_name']] = row['my_team_name']
        print(f"Loaded {len(mappings)} team mappings from {MAPPINGS_CSV}")
        return mappings
    except Exception as e:
        print(f"Error loading team mappings: {e}")
        return {}


def map_team_name(odds_team_name, mappings):
    """Map odds team name to ncaamb.games team name using mappings"""
    return mappings.get(odds_team_name, odds_team_name)


def format_game_id(start_time, team1, team2):
    """
    Format game_id as YYYYMMDD_team1_team2 with teams in alphabetical order.

    Args:
        start_time: datetime string (YYYY-MM-DD HH:MM:SS)
        team1: First team name
        team2: Second team name

    Returns:
        List of possible game_ids (main date ±1 day) to account for date discrepancies
    """
    # Parse datetime and extract date
    try:
        dt = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
    except Exception as e:
        print(f"Error parsing datetime {start_time}: {e}")
        return []

    # Sort teams alphabetically
    teams = sorted([team1, team2])
    team_str = f"{teams[0]}_{teams[1]}"

    # Generate game_ids for ±1 day to account for date discrepancies
    from datetime import timedelta
    game_ids = []
    for day_offset in [-1, 0, 1]:
        date = dt + timedelta(days=day_offset)
        date_str = date.strftime('%Y%m%d')
        game_ids.append(f"{date_str}_{team_str}")

    return game_ids


def load_all_game_ids():
    """Load all game_ids from ncaamb.games table into memory for fast lookup"""
    conn = create_connection()
    if not conn:
        return set()

    try:
        query = "SELECT DISTINCT game_id FROM games"
        results = fetch(conn, query)
        game_ids = {row['game_id'] for row in results if row['game_id']}
        print(f"Loaded {len(game_ids)} game_ids from ncaamb.games table")
        conn.close()
        return game_ids
    except Exception as e:
        print(f"Error loading game_ids: {e}")
        if conn:
            conn.close()
        return set()


def get_odds_without_game_id():
    """Get all odds records without game_id"""
    conn = create_connection()
    if not conn:
        return []

    try:
        query = """
            SELECT DISTINCT
                home_team,
                away_team,
                DATE_FORMAT(start_time, '%Y-%m-%d %H:%i:%s') as start_time
            FROM odds
            WHERE game_id IS NULL
            ORDER BY start_time DESC
        """
        results = fetch(conn, query)
        conn.close()
        return results
    except Exception as e:
        print(f"Error fetching odds records: {e}")
        if conn:
            conn.close()
        return []


def update_odds_with_game_id(odds_records, mappings, available_game_ids):
    """
    Map game_ids and update odds table.

    Args:
        odds_records: List of dicts with home_team, away_team, start_time
        mappings: Dict of team name mappings
        available_game_ids: Set of game_ids that exist in ncaamb.games
    """
    conn = create_connection()
    if not conn:
        print("Could not connect to database")
        return False

    cursor = conn.cursor()
    updated = 0
    skipped = 0
    skipped_game_ids = []
    update_batches = []

    for record in odds_records:
        home_team = record.get('home_team')
        away_team = record.get('away_team')
        start_time = record.get('start_time')

        # Map team names using the mappings file
        mapped_home = map_team_name(home_team, mappings)
        mapped_away = map_team_name(away_team, mappings)

        # Format game_ids (list of ±1 day variations)
        possible_game_ids = format_game_id(start_time, mapped_home, mapped_away)
        if not possible_game_ids:
            print(f"  Skipped: Could not format game_id for {home_team} vs {away_team} on {start_time}")
            skipped += 1
            continue

        # Find first matching game_id from available game_ids
        matching_game_id = None
        for gid in possible_game_ids:
            if gid in available_game_ids:
                matching_game_id = gid
                break

        if not matching_game_id:
            skipped_game_ids.append({
                'game_ids_tried': possible_game_ids,
                'home_team': home_team,
                'away_team': away_team,
                'mapped_home': mapped_home,
                'mapped_away': mapped_away,
                'start_time': start_time
            })
            skipped += 1
            continue

        # Add to batch for update
        update_batches.append((matching_game_id, home_team, away_team, start_time))

    # Execute all updates in batch
    if update_batches:
        try:
            update_query = """
                UPDATE odds
                SET game_id = %s
                WHERE home_team = %s AND away_team = %s AND start_time = %s
            """
            cursor.executemany(update_query, update_batches)
            updated = cursor.rowcount
            conn.commit()
        except Exception as e:
            print(f"Error executing batch update: {e}")
            conn.rollback()

    if skipped_game_ids:
        print(f"\nSkipped {len(skipped_game_ids)} records (game_id not found in ncaamb.games):")
        for item in skipped_game_ids:
            print(f"  Tried game_ids: {item['game_ids_tried']}")
            print(f"    Odds teams: {item['home_team']} vs {item['away_team']}")
            print(f"    Mapped teams: {item['mapped_home']} vs {item['mapped_away']}")
            print(f"    Start time: {item['start_time']}")
            print()

    print(f"\nUpdated {updated} odds records with game_ids")

    cursor.close()
    conn.close()
    return True


def main():
    """Main function"""
    print("Loading team mappings...")
    mappings = load_team_mappings()

    if not mappings:
        print("Error: Could not load team mappings from team_mappings.csv")
        return False

    print("\nLoading all game_ids from ncaamb.games into memory...")
    available_game_ids = load_all_game_ids()

    if not available_game_ids:
        print("Error: Could not load game_ids from ncaamb.games")
        return False

    print("\nFetching odds records without game_id...")
    odds_records = get_odds_without_game_id()
    print(f"Found {len(odds_records)} distinct game combinations without game_id")

    if not odds_records:
        print("No odds records to update.")
        return True

    print("\nMatching game_ids and updating odds table...")
    success = update_odds_with_game_id(odds_records, mappings, available_game_ids)

    if success:
        print("\nGame ID mapping complete!")
    else:
        print("\nError during game ID mapping.")

    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
