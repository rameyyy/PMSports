"""
Utility functions for fetching and processing NCAAMB data using Polars
"""
import polars as pl
from dotenv import load_dotenv
import os
from typing import Optional, List

# Load environment variables from ncaamb/.env
ncaamb_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
env_path = os.path.join(ncaamb_dir, '.env')
load_dotenv(env_path)


def get_db_connection_string() -> str:
    """Get MySQL connection string from environment variables"""
    return (
        f"mysql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
        f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('NCAAMB_DB')}"
    )


def fetch_games(season: Optional[int] = None) -> pl.DataFrame:
    """
    Fetch games data from database

    Args:
        season: Optional season filter

    Returns:
        Polars DataFrame with games data
    """
    conn_str = get_db_connection_string()

    query = "SELECT * FROM games"
    if season:
        query += f" WHERE season = {season}"
    query += " ORDER BY date, game_id"

    return pl.read_database_uri(query, conn_str)


def fetch_teams(season: Optional[int] = None) -> pl.DataFrame:
    """
    Fetch teams data from database

    Args:
        season: Optional season filter

    Returns:
        Polars DataFrame with teams data
    """
    conn_str = get_db_connection_string()

    query = "SELECT season, team_name, conference FROM teams"
    if season:
        query += f" WHERE season = {season}"

    return pl.read_database_uri(query, conn_str)


def fetch_leaderboard(season: Optional[int] = None) -> pl.DataFrame:
    """
    Fetch leaderboard data from database

    Args:
        season: Optional season filter (not directly available, but can filter by date)

    Returns:
        Polars DataFrame with leaderboard data
    """
    conn_str = get_db_connection_string()

    query = "SELECT * FROM leaderboard ORDER BY date, team"

    return pl.read_database_uri(query, conn_str)


def fetch_player_stats(season: Optional[int] = None, game_ids: Optional[List[str]] = None) -> pl.DataFrame:
    """
    Fetch player stats data from database

    Args:
        season: Optional season filter
        game_ids: Optional list of game_ids to filter

    Returns:
        Polars DataFrame with player stats data
    """
    conn_str = get_db_connection_string()

    query = "SELECT * FROM player_stats"
    conditions = []

    if season:
        conditions.append(f"season = {season}")
    if game_ids:
        game_ids_str = "', '".join(game_ids)
        conditions.append(f"game_id IN ('{game_ids_str}')")

    if conditions:
        query += " WHERE " + " AND ".join(conditions)

    return pl.read_database_uri(query, conn_str)


def get_team_historical_games(team_name: str, before_date: str, season: int, games_df: pl.DataFrame) -> pl.DataFrame:
    """
    Get all games for a team before a specific date in a season

    Args:
        team_name: Name of the team
        before_date: Date string (YYYY-MM-DD format)
        season: Season year
        games_df: DataFrame containing all games

    Returns:
        Polars DataFrame with historical games for the team
    """
    historical = games_df.filter(
        (pl.col("season") == season) &
        (pl.col("date") < before_date) &
        ((pl.col("team_1") == team_name) | (pl.col("team_2") == team_name))
    ).sort("date")

    return historical


def add_team_conferences(games_df: pl.DataFrame, teams_df: pl.DataFrame) -> pl.DataFrame:
    """
    Add conference information for both teams to games dataframe

    Args:
        games_df: DataFrame with games data
        teams_df: DataFrame with teams data (must have season, team_name, conference)

    Returns:
        Games DataFrame with team_1_conference and team_2_conference columns
    """
    # Join for team_1 conference
    games_with_conf = games_df.join(
        teams_df.select(["season", "team_name", "conference"]),
        left_on=["season", "team_1"],
        right_on=["season", "team_name"],
        how="left"
    ).rename({"conference": "team_1_conference"})

    # Join for team_2 conference
    games_with_conf = games_with_conf.join(
        teams_df.select(["season", "team_name", "conference"]),
        left_on=["season", "team_2"],
        right_on=["season", "team_name"],
        how="left"
    ).rename({"conference": "team_2_conference"})

    return games_with_conf


def get_team_perspective(game_row: dict, team_name: str) -> dict:
    """
    Get game data from a specific team's perspective

    Args:
        game_row: Dictionary containing game data
        team_name: Name of the team

    Returns:
        Dictionary with data from team's perspective (team vs opponent)
    """
    is_team_1 = game_row["team_1"] == team_name

    if is_team_1:
        return {
            "team": game_row["team_1"],
            "opponent": game_row["team_2"],
            "team_score": game_row["team_1_score"],
            "opp_score": game_row["team_2_score"],
            "is_home": game_row["location"] == game_row["team_1"],
            # Add all team_1 stats with "team_" prefix
            **{f"team_{k.replace('team_1_', '')}": v for k, v in game_row.items() if k.startswith("team_1_")},
            # Add all team_2 stats with "opp_" prefix
            **{f"opp_{k.replace('team_2_', '')}": v for k, v in game_row.items() if k.startswith("team_2_")},
        }
    else:
        return {
            "team": game_row["team_2"],
            "opponent": game_row["team_1"],
            "team_score": game_row["team_2_score"],
            "opp_score": game_row["team_1_score"],
            "is_home": game_row["location"] == game_row["team_2"],
            # Add all team_2 stats with "team_" prefix
            **{f"team_{k.replace('team_2_', '')}": v for k, v in game_row.items() if k.startswith("team_2_")},
            # Add all team_1 stats with "opp_" prefix
            **{f"opp_{k.replace('team_1_', '')}": v for k, v in game_row.items() if k.startswith("team_1_")},
        }
