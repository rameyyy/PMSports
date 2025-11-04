import requests
import json
import mysql.connector
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv('.env')

def get_teams_from_player_stats(year='2025'):
    """Extract unique teams from player stats"""
    url = f"https://barttorvik.com/{year}_all_advgames.json.gz"
    response = requests.get(url, timeout=60)
    data = json.loads(response.text)

    teams = set()
    for record in data:
        if isinstance(record, list) and len(record) > 47:
            team = record[47]  # tt (this team)
            if team and isinstance(team, str):
                teams.add(team)

    return sorted(list(teams))


def get_teams_in_db(season=2025):
    """Get all team names in teams table for a season"""
    try:
        conn = mysql.connector.connect(
            host=os.getenv("DB_HOST"),
            port=int(os.getenv("DB_PORT")),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            database=os.getenv("NCAAMB_DB"),
        )

        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT team_name FROM teams WHERE season = %s", (season,))
        results = cursor.fetchall()
        cursor.close()
        conn.close()

        return set([row[0] for row in results])
    except Exception as e:
        print(f"Error querying database: {e}")
        return set()


def scrape_missing_teams(missing_teams, year='2025', season=2025):
    """Scrape game history for missing teams"""
    from scrapes.gamehistory import scrape_game_history
    from scrapes import sqlconn

    print(f"\nScraping {len(missing_teams)} missing teams...")
    print("=" * 100)

    successful = 0
    failed = 0

    for idx, team in enumerate(missing_teams, 1):
        print(f"[{idx}/{len(missing_teams)}] Scraping {team}...")
        try:
            df = scrape_game_history(year=year, team=team)
            if df is not None and len(df) > 0:
                sqlconn.push_to_teams(df, season)
                sqlconn.push_to_games(df, season, year)
                print(f"  ✓ Success - {len(df)} games")
                successful += 1
            else:
                print(f"  ✗ No games found")
                failed += 1
        except Exception as e:
            print(f"  ✗ Error: {e}")
            failed += 1

    print("\n" + "=" * 100)
    print(f"Scraping complete: {successful} successful, {failed} failed")


if __name__ == "__main__":
    print("Getting teams from player stats...")
    player_stats_teams = get_teams_from_player_stats(year='2025')
    print(f"Found {len(player_stats_teams)} teams in player stats\n")

    print("Getting teams from database...")
    db_teams = get_teams_in_db(season=2025)
    print(f"Found {len(db_teams)} teams in database\n")

    # Find missing teams
    missing_teams = [t for t in player_stats_teams if t not in db_teams]

    if missing_teams:
        print(f"Missing {len(missing_teams)} teams from database:")
        print("-" * 50)
        for team in missing_teams:
            print(f"  {team}")

        # Scrape missing teams
        scrape_missing_teams(missing_teams, year='2025', season=2025)
    else:
        print("All teams from player stats are in the database!")
