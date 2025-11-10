"""
Push today's games from Barttorvik super_sked to ncaamb.games table
"""

import pandas as pd
from datetime import datetime
from . import sqlconn


def parse_date_to_mysql(date_str: str) -> str:
    """
    Convert M/D/YY format to MySQL DATE format (YYYY-MM-DD).

    Args:
        date_str: Date in M/D/YY format (e.g., "11/10/25")

    Returns:
        Date string in YYYY-MM-DD format
    """
    try:
        dt = pd.to_datetime(date_str, format='%m/%d/%y')
        return dt.strftime('%Y-%m-%d')
    except Exception as e:
        print(f"Error parsing date '{date_str}': {e}")
        return None


def get_team_1_and_2(team_a_name: str, team_b_name: str):
    """
    Determine team_1 and team_2 based on alphabetical order.
    team_1 is always the alphabetically first team.

    Args:
        team_a_name: First team name from Barttorvik
        team_b_name: Second team name from Barttorvik

    Returns:
        Tuple of (team_1, team_2) in alphabetical order
    """
    teams = sorted([team_a_name, team_b_name])
    return teams[0], teams[1]


def get_location(team_a_name: str, team_b_name: str, matchup_str: str):
    """
    Determine location based on matchup string.

    The matchup string contains venue info:
    - "team_a at team_b" means team_a is away, team_b is home
    - "team_a vs. team_b" means neutral site

    Args:
        team_a_name: Team A name (team1 in Barttorvik data)
        team_b_name: Team B name (team2 in Barttorvik data)
        matchup_str: Matchup string from Barttorvik (e.g., "281 Iowa St. vs. 38 Mississippi St.")

    Returns:
        Location string (home team name or 'N' for neutral)
    """
    matchup_lower = str(matchup_str).lower()

    # Get alphabetical order
    teams = sorted([team_a_name, team_b_name])
    team_1, team_2 = teams[0], teams[1]

    # Check if neutral site (contains "vs.")
    if ' vs.' in matchup_lower or ' vs ' in matchup_lower:
        return 'N'

    # Check if team_a is away (contains "at")
    if ' at ' in matchup_lower:
        # team_a is away, so team_b is home
        home_team = team_b_name
        return home_team

    # Default to neutral if we can't determine
    return 'N'


def get_game_type(team_1: str, team_2: str, season: int, cursor):
    """
    Determine game type based on whether teams are in same conference.

    Args:
        team_1: First team name (alphabetically ordered)
        team_2: Second team name (alphabetically ordered)
        season: Season year
        cursor: Database cursor

    Returns:
        str: 'Regular Season Conference' if same conference, else 'Regular Season Non-Conference'
    """
    try:
        # Query conference for team_1
        query_1 = "SELECT conference FROM teams WHERE season = %s AND team_name = %s LIMIT 1"
        cursor.execute(query_1, (season, team_1))
        result_1 = cursor.fetchone()

        # Query conference for team_2
        cursor.execute(query_1, (season, team_2))
        result_2 = cursor.fetchone()

        if result_1 and result_2:
            conf_1 = result_1[0] if isinstance(result_1, tuple) else result_1.get('conference')
            conf_2 = result_2[0] if isinstance(result_2, tuple) else result_2.get('conference')

            if conf_1 and conf_2 and conf_1 == conf_2:
                return 'Regular Season Conference'
            else:
                return 'Regular Season Non-Conference'
        else:
            # If we can't find teams, default to non-conference
            return 'Regular Season Non-Conference'

    except Exception as e:
        print(f"    Error determining game type: {e}")
        return 'Regular Season Non-Conference'


def insert_teams_for_games(df, season, cursor):
    """
    Insert teams into the teams table if they don't exist for the given season.

    Args:
        df: DataFrame with team1 and team2 columns
        season: Season year
        cursor: Database cursor

    Returns:
        Number of teams inserted
    """
    teams_inserted = 0

    # Get all unique teams from the games
    teams = set()
    for idx, row in df.iterrows():
        teams.add(str(row['team1']).strip())
        teams.add(str(row['team2']).strip())

    for team in sorted(teams):
        try:
            # Check if team already exists for this season
            check_query = "SELECT team_name FROM teams WHERE season = %s AND team_name = %s LIMIT 1"
            cursor.execute(check_query, (season, team))
            result = cursor.fetchone()

            if not result:
                # Insert team with null conference (will be populated later)
                insert_query = "INSERT INTO teams (season, team_name, conference) VALUES (%s, %s, NULL)"
                cursor.execute(insert_query, (season, team))
                teams_inserted += 1
        except Exception as e:
            print(f"  Warning: Error checking/inserting team {team}: {e}")

    return teams_inserted


def push_todays_games_to_db(df):
    """
    Push today's games from Barttorvik super_sked to the ncaamb.games table.

    Args:
        df: DataFrame from scrape_barttorvik_schedule with today's games

    Returns:
        bool: True if successful, False otherwise
    """
    if df is None or len(df) == 0:
        print("No games to push")
        return False

    try:
        conn = sqlconn.create_connection()
        if not conn:
            print("Could not connect to database")
            return False

        cursor = conn.cursor(dictionary=True)
        games_inserted = 0
        games_skipped = 0

        # First, determine the season from the first game's date
        if len(df) > 0:
            first_date = pd.to_datetime(df.iloc[0]['date'], format='%m/%d/%y')
            if first_date.month > 10:
                season = first_date.year + 1
            else:
                season = first_date.year

            # Insert teams for this season
            print(f"\nInserting teams for season {season}...")
            teams_count = insert_teams_for_games(df, season, cursor)
            if teams_count > 0:
                conn.commit()
                print(f"  Inserted {teams_count} new teams")
            else:
                print(f"  All teams already exist")

        for idx, row in df.iterrows():
            try:
                # Extract team names
                team_a = str(row['team1']).strip()
                team_b = str(row['team2']).strip()

                # Get alphabetically ordered team names
                team_1, team_2 = get_team_1_and_2(team_a, team_b)

                # Generate game_id: YYYYMMDD_team1_team2
                date_obj = pd.to_datetime(row['date'], format='%m/%d/%y')
                date_yyyymmdd = date_obj.strftime('%Y%m%d')
                game_id = f"{date_yyyymmdd}_{team_1}_{team_2}"

                # Get location (home team or neutral) - parse from matchup string
                location = get_location(team_a, team_b, str(row['matchup']))

                # Convert date to MySQL format
                mysql_date = parse_date_to_mysql(row['date'])

                # Check if game already exists
                check_query = "SELECT game_id FROM games WHERE game_id = %s"
                cursor.execute(check_query, (game_id,))
                game_exists = cursor.fetchone()

                if game_exists:
                    print(f"  Game {game_id} already exists, skipping")
                    games_skipped += 1
                    continue

                # Determine season: if month > 10 (Nov-Dec), season = year+1, else season = year
                if date_obj.month > 10:
                    season = date_obj.year + 1
                else:
                    season = date_obj.year

                # Determine game type based on conference
                game_type = get_game_type(team_1, team_2, season, cursor)

                # Insert new game
                insert_query = """
                    INSERT INTO games (
                        game_id, season, date, game_type, location,
                        team_1, team_2
                    ) VALUES (
                        %s, %s, %s, %s, %s,
                        %s, %s
                    )
                """

                params = (
                    game_id,
                    season,
                    mysql_date,
                    game_type,
                    location,
                    team_1,
                    team_2
                )

                cursor.execute(insert_query, params)
                games_inserted += 1
                print(f"  Inserted: {team_1} @ {location} vs {team_2} ({mysql_date}) - {game_type}")

            except Exception as e:
                print(f"  Error processing game {idx}: {e}")
                games_skipped += 1
                continue

        conn.commit()
        cursor.close()
        conn.close()

        print(f"\nSummary: {games_inserted} games inserted, {games_skipped} skipped")
        return True

    except Exception as e:
        print(f"Error pushing games to database: {e}")
        return False


if __name__ == "__main__":
    from barttorvik_schedule import scrape_barttorvik_schedule
    from datetime import datetime

    print("Testing push_todays_games...")

    # Get today's games
    season = '2026'
    df = scrape_barttorvik_schedule(season)

    if df is not None:
        # Filter for today
        today = datetime.now()
        month = str(today.month)
        day = str(today.day)
        year = today.strftime('%y')
        today_str = f"{month}/{day}/{year}"

        todays_games = df[df['date'].astype(str) == today_str].copy()

        if len(todays_games) > 0:
            print(f"\nFound {len(todays_games)} games for today ({today_str})")
            print("Pushing to database...\n")
            push_todays_games_to_db(todays_games)
        else:
            print(f"No games found for today ({today_str})")
