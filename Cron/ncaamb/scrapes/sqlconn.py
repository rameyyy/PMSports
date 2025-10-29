import mysql.connector
from mysql.connector import Error
from dotenv import load_dotenv
import os
import pandas as pd
from sqlalchemy import create_engine
from typing import Optional, Union

# Load environment variables from ncaamb/.env
ncaamb_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
env_path = os.path.join(ncaamb_dir, '.env')
load_dotenv(env_path)

def create_connection():
    """Create and return a MySQL database connection"""
    try:
        conn = mysql.connector.connect(
            host=os.getenv("DB_HOST"),
            port=int(os.getenv("DB_PORT")),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            database=os.getenv("NCAAMB_DB"),
        )
        if conn.is_connected():
            return conn
    except mysql.connector.Error as e:
        print(f"❌ Error: {e}")
        return None


def fetch(connection, query, params=None):
    """Fetch results from a SELECT query"""
    cursor = connection.cursor(dictionary=True)  # dict rows
    try:
        cursor.execute(query, params or ())
        return cursor.fetchall()
    except Error as e:
        print(f"❌ Error fetching query: {e}")
        return []
    finally:
        cursor.close()


def execute_query(connection=None, query: Optional[str] = None, params=None,
                  df: Optional[pd.DataFrame] = None, table_name: Optional[str] = None,
                  if_exists: str = 'append'):
    """
    Execute a query or insert a pandas DataFrame into the database.

    Args:
        connection: MySQL connection object (required if using query, optional if using df)
        query: SQL query string to execute
        params: Parameters for the query
        df: pandas DataFrame to insert
        table_name: Target table name (required if df is provided)
        if_exists: How to behave if table exists ('fail', 'replace', 'append')

    Returns:
        bool: True if successful, False otherwise
    """
    # If DataFrame is provided, use SQLAlchemy to push data
    if df is not None:
        if table_name is None:
            print("❌ Error: table_name must be provided when using DataFrame")
            return False

        try:
            # Create SQLAlchemy engine
            db_host = os.getenv("DB_HOST")
            db_port = os.getenv("DB_PORT")
            db_user = os.getenv("DB_USER")
            db_password = os.getenv("DB_PASSWORD")
            db_name = os.getenv("NCAAMB_DB")

            engine = create_engine(
                f"mysql+mysqlconnector://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
            )

            # Push DataFrame to database
            df.to_sql(name=table_name, con=engine, if_exists=if_exists, index=False)
            engine.dispose()
            print(f"✅ Successfully inserted {len(df)} rows into {table_name}")
            return True

        except Exception as e:
            print(f"❌ Error inserting DataFrame: {e}")
            return False

    # Otherwise, execute the query using the connection
    if query is None or connection is None:
        print("❌ Error: query and connection must be provided if not using DataFrame")
        return False

    cursor = connection.cursor()
    try:
        cursor.execute(query, params or ())
        connection.commit()
        return True
    except Error as e:
        if e.errno == 1062:  # duplicate entry
            return True
        print(f"❌ Error running query: {e}")
        return False
    finally:
        cursor.close()


def push_to_teams(df, season):
    """
    Push team data to the teams table.

    Args:
        df: DataFrame with game history data
        season: Season year

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        conn = create_connection()
        if not conn:
            print("❌ Could not connect to database")
            return False

        cursor = conn.cursor()

        # Get unique team-coach-conference combinations from the dataframe
        teams_data = []

        # Add team 1 (the team that was scraped)
        for _, row in df.iterrows():
            team_name = row['team']
            head_coach = row['team_coach']
            conference = row['conference']
            teams_data.append((season, team_name, head_coach, conference))

        # Add opponent teams
        for _, row in df.iterrows():
            opp_name = row['opponent']
            opp_coach = row['opp_coach']
            opp_conference = row['opp_conference']
            teams_data.append((season, opp_name, opp_coach, opp_conference))

        # Remove duplicates while preserving order
        seen = set()
        unique_teams = []
        for team in teams_data:
            if team not in seen:
                seen.add(team)
                unique_teams.append(team)

        # Insert into teams table (ignore if exists due to unique key)
        insert_query = """
            INSERT IGNORE INTO teams (season, team_name, head_coach, conference)
            VALUES (%s, %s, %s, %s)
        """

        for team in unique_teams:
            try:
                cursor.execute(insert_query, team)
            except Error as e:
                if e.errno != 1062:  # ignore duplicate entry errors
                    print(f"⚠️ Warning inserting team {team[1]}: {e}")

        conn.commit()
        print(f"✅ Teams data pushed to database ({len(unique_teams)} unique teams)")
        cursor.close()
        conn.close()
        return True

    except Exception as e:
        print(f"❌ Error pushing teams data: {e}")
        return False


def push_to_games(df, season, year):
    """
    Push game data to the games table (one row per game).
    Creates or updates games with team_1 as the scraped team.

    Args:
        df: DataFrame with game history data (one row per game from team's perspective)
        season: Season year
        year: Year for date context

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        conn = create_connection()
        if not conn:
            print("❌ Could not connect to database")
            return False

        cursor = conn.cursor()

        games_inserted = 0
        games_updated = 0

        for _, row in df.iterrows():
            # Generate game_id using alphabetically sorted team names
            teams = sorted([row['team'], row['opponent']])
            game_id = f"{row['date'].replace('-', '')}_{teams[0]}_{teams[1]}"

            # Determine which team is team_1 and team_2
            if row['team'] == teams[0]:
                # This team is team_1
                team_1 = row['team']
                team_2 = row['opponent']
                team_1_is_scraped = True
            else:
                # This team is team_2
                team_1 = teams[0]
                team_2 = teams[1]
                team_1_is_scraped = False

            # Check if game already exists
            check_query = "SELECT game_id FROM games WHERE game_id = %s"
            cursor.execute(check_query, (game_id,))
            game_exists = cursor.fetchone()

            if game_exists:
                # Update existing game with team data
                if team_1_is_scraped:
                    # Update team_1 columns
                    update_query = """
                        UPDATE games SET
                            season = %s,
                            team_1 = %s,
                            team_1_score = %s,
                            team_1_adjoe = %s,
                            team_1_adjde = %s,
                            team_1_eff = %s,
                            team_1_efg_pct = %s,
                            team_1_to_pct = %s,
                            team_1_or_pct = %s,
                            team_1_ftr = %s,
                            team_1_def_eff = %s,
                            team_1_def_efg_pct = %s,
                            team_1_def_to_pct = %s,
                            team_1_def_or_pct = %s,
                            team_1_def_ftr = %s,
                            team_1_g_sc = %s,
                            team_1_fgm = %s,
                            team_1_fga = %s,
                            team_1_2pm = %s,
                            team_1_2pa = %s,
                            team_1_3pm = %s,
                            team_1_3pa = %s,
                            team_1_ftm = %s,
                            team_1_fta = %s,
                            team_1_oreb = %s,
                            team_1_dreb = %s,
                            team_1_treb = %s,
                            team_1_ast = %s,
                            team_1_to = %s,
                            team_1_stl = %s,
                            team_1_blk = %s,
                            team_1_pf = %s
                        WHERE game_id = %s
                    """
                    params = (
                        season,
                        row['team'],
                        row['team_score'],
                        row['team_adjoe'],
                        row['team_adjde'],
                        row['team_eff'],
                        row['team_efg_pct'],
                        row['team_to_pct'],
                        row['team_or_pct'],
                        row['team_ftr'],
                        row['team_def_eff'],
                        row['team_def_efg_pct'],
                        row['team_def_to_pct'],
                        row['team_def_or_pct'],
                        row['team_def_ftr'],
                        row['team_g_sc'],
                        row['team_fgm'],
                        row['team_fga'],
                        row['team_2pm'],
                        row['team_2pa'],
                        row['team_3pm'],
                        row['team_3pa'],
                        row['team_ftm'],
                        row['team_fta'],
                        row['team_oreb'],
                        row['team_dreb'],
                        row['team_treb'],
                        row['team_ast'],
                        row['team_to'],
                        row['team_stl'],
                        row['team_blk'],
                        row['team_pf'],
                        game_id
                    )
                else:
                    # Update team_2 columns
                    update_query = """
                        UPDATE games SET
                            season = %s,
                            team_2 = %s,
                            team_2_score = %s,
                            team_2_adjoe = %s,
                            team_2_adjde = %s,
                            team_2_eff = %s,
                            team_2_efg_pct = %s,
                            team_2_to_pct = %s,
                            team_2_or_pct = %s,
                            team_2_ftr = %s,
                            team_2_def_eff = %s,
                            team_2_def_efg_pct = %s,
                            team_2_def_to_pct = %s,
                            team_2_def_or_pct = %s,
                            team_2_def_ftr = %s,
                            team_2_g_sc = %s,
                            team_2_fgm = %s,
                            team_2_fga = %s,
                            team_2_2pm = %s,
                            team_2_2pa = %s,
                            team_2_3pm = %s,
                            team_2_3pa = %s,
                            team_2_ftm = %s,
                            team_2_fta = %s,
                            team_2_oreb = %s,
                            team_2_dreb = %s,
                            team_2_treb = %s,
                            team_2_ast = %s,
                            team_2_to = %s,
                            team_2_stl = %s,
                            team_2_blk = %s,
                            team_2_pf = %s
                        WHERE game_id = %s
                    """
                    params = (
                        season,
                        row['opponent'],
                        row['opp_score'],
                        row['opp_adjoe'] if 'opp_adjoe' in df.columns else None,
                        row['opp_adjde'] if 'opp_adjde' in df.columns else None,
                        row['opp_eff'] if 'opp_eff' in df.columns else None,
                        row['opp_efg_pct'] if 'opp_efg_pct' in df.columns else None,
                        row['opp_to_pct'] if 'opp_to_pct' in df.columns else None,
                        row['opp_or_pct'] if 'opp_or_pct' in df.columns else None,
                        row['opp_ftr'] if 'opp_ftr' in df.columns else None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        row['opp_fgm'],
                        row['opp_fga'],
                        row['opp_2pm'],
                        row['opp_2pa'],
                        row['opp_3pm'],
                        row['opp_3pa'],
                        row['opp_ftm'],
                        row['opp_fta'],
                        row['opp_oreb'],
                        row['opp_dreb'],
                        row['opp_treb'],
                        row['opp_ast'],
                        row['opp_to'],
                        row['opp_stl'],
                        row['opp_blk'],
                        row['opp_pf'],
                        game_id
                    )

                cursor.execute(update_query, params)
                games_updated += 1
            else:
                # Insert new game
                insert_query = """
                    INSERT INTO games (
                        game_id, season, date, game_type, location, plus_minus, ot,
                        team_1, team_1_score,
                        team_1_adjoe, team_1_adjde, team_1_eff,
                        team_1_efg_pct, team_1_to_pct, team_1_or_pct, team_1_ftr,
                        team_1_def_eff, team_1_def_efg_pct, team_1_def_to_pct, team_1_def_or_pct, team_1_def_ftr, team_1_g_sc,
                        team_1_fgm, team_1_fga, team_1_2pm, team_1_2pa, team_1_3pm, team_1_3pa,
                        team_1_ftm, team_1_fta, team_1_oreb, team_1_dreb, team_1_treb,
                        team_1_ast, team_1_to, team_1_stl, team_1_blk, team_1_pf,
                        team_2, team_2_score
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s,
                        %s, %s,
                        %s, %s, %s,
                        %s, %s, %s, %s,
                        %s, %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s,
                        %s, %s
                    )
                """
                params = (
                    game_id, season, row['date'], row['game_type'], row['location'], row['plus_minus'], row['ot'],
                    row['team'], row['team_score'],
                    row['team_adjoe'], row['team_adjde'], row['team_eff'],
                    row['team_efg_pct'], row['team_to_pct'], row['team_or_pct'], row['team_ftr'],
                    row['team_def_eff'], row['team_def_efg_pct'], row['team_def_to_pct'], row['team_def_or_pct'], row['team_def_ftr'], row['team_g_sc'],
                    row['team_fgm'], row['team_fga'], row['team_2pm'], row['team_2pa'], row['team_3pm'], row['team_3pa'],
                    row['team_ftm'], row['team_fta'], row['team_oreb'], row['team_dreb'], row['team_treb'],
                    row['team_ast'], row['team_to'], row['team_stl'], row['team_blk'], row['team_pf'],
                    row['opponent'], row['opp_score']
                )

                cursor.execute(insert_query, params)
                games_inserted += 1

        conn.commit()
        print(f"✅ Games data pushed to database ({games_inserted} inserted, {games_updated} updated)")
        cursor.close()
        conn.close()
        return True

    except Exception as e:
        print(f"❌ Error pushing games data: {e}")
        return False
