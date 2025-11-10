"""
Build flat dataframe for NCAAMB games with historical match data
"""
import polars as pl
from typing import Optional, List
from datetime import timedelta
from .utils import (
    fetch_games,
    fetch_teams,
    fetch_leaderboard,
    fetch_player_stats,
    add_team_conferences,
    get_team_historical_games
)


def build_flat_row_for_game(game_row: dict, games_df: pl.DataFrame, leaderboard_df: pl.DataFrame, player_stats_df: pl.DataFrame, odds_dict: dict = None) -> dict:
    """
    Build a flat row for a single game with raw historical match data

    Args:
        game_row: Dictionary containing game data (the current game)
        games_df: DataFrame with all games
        leaderboard_df: DataFrame with leaderboard data
        player_stats_df: DataFrame with player stats data
        odds_dict: Dictionary mapping game_id to odds data (optional)

    Returns:
        Dictionary with game info, outcome data, and historical matches for both teams
    """
    game_id = game_row["game_id"]
    season = game_row["season"]
    date = game_row["date"]
    team_1 = game_row["team_1"]
    team_2 = game_row["team_2"]

    # Calculate the day before the game
    day_before = date - timedelta(days=1)

    # Start with base game info
    team_1_score = game_row.get("team_1_score")
    team_2_score = game_row.get("team_2_score")

    # Calculate outcome fields
    total_score_outcome = team_1_score + team_2_score if team_1_score is not None and team_2_score is not None else None
    team_1_winloss = 1 if (team_1_score is not None and team_2_score is not None and team_1_score > team_2_score) else 0

    # Get leaderboard data from the day before
    team_1_leaderboard_data = leaderboard_df.filter(
        (pl.col("team") == team_1) & (pl.col("date") == day_before)
    )
    team_2_leaderboard_data = leaderboard_df.filter(
        (pl.col("team") == team_2) & (pl.col("date") == day_before)
    )

    # Convert to dict or None if not found
    team_1_leaderboard = team_1_leaderboard_data.to_dicts()[0] if len(team_1_leaderboard_data) > 0 else None
    team_2_leaderboard = team_2_leaderboard_data.to_dicts()[0] if len(team_2_leaderboard_data) > 0 else None

    # Determine home/away status
    # location is a team name (meaning that team is at home) or 'N' (neutral site)
    location = game_row.get("location")

    if location == 'N':
        team_1_is_home = None  # Neutral site
        team_2_is_home = None
    elif location == team_1:
        team_1_is_home = True
        team_2_is_home = False
    elif location == team_2:
        team_1_is_home = False
        team_2_is_home = True
    else:
        # Default: unknown location
        team_1_is_home = None
        team_2_is_home = None

    # Extract start_time from odds data if available
    start_time = None
    if odds_dict and game_id in odds_dict:
        odds_data = odds_dict[game_id]
        if isinstance(odds_data, list) and len(odds_data) > 0:
            # Get start_time from first sportsbook (should be same for all)
            start_time = odds_data[0].get("start_time")

    flat_row = {
        "game_id": game_id,
        "season": season,
        "date": date,
        "start_time": start_time,
        "team_1": team_1,
        "team_2": team_2,
        "team_1_conference": game_row.get("team_1_conference"),
        "team_2_conference": game_row.get("team_2_conference"),
        "team_1_is_home": team_1_is_home,
        "team_2_is_home": team_2_is_home,
        "location": location,
        "team_1_score": team_1_score,
        "team_2_score": team_2_score,
        "total_score_outcome": total_score_outcome,
        "team_1_winloss": team_1_winloss,
        "team_1_leaderboard": team_1_leaderboard,
        "team_2_leaderboard": team_2_leaderboard,
    }

    # Get historical games for both teams (before this date)
    team_1_history = get_team_historical_games(team_1, date, season, games_df)
    team_2_history = get_team_historical_games(team_2, date, season, games_df)

    # Convert historical games to list of dicts and add winner + leaderboard data
    team_1_matches = []
    if len(team_1_history) > 0:
        for game in team_1_history.to_dicts():
            # Add winner field
            if game['team_1_score'] is not None and game['team_2_score'] is not None:
                game['winner'] = game['team_1'] if game['team_1_score'] > game['team_2_score'] else game['team_2']
            else:
                game['winner'] = None

            # Add leaderboard data from day before this historical game
            hist_game_date = game['date']
            hist_day_before = hist_game_date - timedelta(days=1)

            hist_team_1_lb = leaderboard_df.filter(
                (pl.col("team") == game['team_1']) & (pl.col("date") == hist_day_before)
            )
            hist_team_2_lb = leaderboard_df.filter(
                (pl.col("team") == game['team_2']) & (pl.col("date") == hist_day_before)
            )

            game['team_1_leaderboard'] = hist_team_1_lb.to_dicts()[0] if len(hist_team_1_lb) > 0 else None
            game['team_2_leaderboard'] = hist_team_2_lb.to_dicts()[0] if len(hist_team_2_lb) > 0 else None

            # Get player stats for this historical game
            hist_game_id = game['game_id']
            hist_player_stats = player_stats_df.filter(pl.col("game_id") == hist_game_id)

            if len(hist_player_stats) > 0:
                # Drop unwanted columns
                cols_to_drop = ['id', 'game_id', 'season', 'player_id', 'numdate', 'datetext']
                hist_player_stats_clean = hist_player_stats.drop([col for col in cols_to_drop if col in hist_player_stats.columns])

                # Split by team
                team_1_players = hist_player_stats_clean.filter(pl.col("team") == game['team_1'])
                team_2_players = hist_player_stats_clean.filter(pl.col("team") == game['team_2'])

                game['team_1_player_stats'] = team_1_players.to_dicts() if len(team_1_players) > 0 else []
                game['team_2_player_stats'] = team_2_players.to_dicts() if len(team_2_players) > 0 else []
            else:
                game['team_1_player_stats'] = []
                game['team_2_player_stats'] = []

            team_1_matches.append(game)

    team_2_matches = []
    if len(team_2_history) > 0:
        for game in team_2_history.to_dicts():
            # Add winner field
            if game['team_1_score'] is not None and game['team_2_score'] is not None:
                game['winner'] = game['team_1'] if game['team_1_score'] > game['team_2_score'] else game['team_2']
            else:
                game['winner'] = None

            # Add leaderboard data from day before this historical game
            hist_game_date = game['date']
            hist_day_before = hist_game_date - timedelta(days=1)

            hist_team_1_lb = leaderboard_df.filter(
                (pl.col("team") == game['team_1']) & (pl.col("date") == hist_day_before)
            )
            hist_team_2_lb = leaderboard_df.filter(
                (pl.col("team") == game['team_2']) & (pl.col("date") == hist_day_before)
            )

            game['team_1_leaderboard'] = hist_team_1_lb.to_dicts()[0] if len(hist_team_1_lb) > 0 else None
            game['team_2_leaderboard'] = hist_team_2_lb.to_dicts()[0] if len(hist_team_2_lb) > 0 else None

            # Get player stats for this historical game
            hist_game_id = game['game_id']
            hist_player_stats = player_stats_df.filter(pl.col("game_id") == hist_game_id)

            if len(hist_player_stats) > 0:
                # Drop unwanted columns
                cols_to_drop = ['id', 'game_id', 'season', 'player_id', 'numdate', 'datetext']
                hist_player_stats_clean = hist_player_stats.drop([col for col in cols_to_drop if col in hist_player_stats.columns])

                # Split by team
                team_1_players = hist_player_stats_clean.filter(pl.col("team") == game['team_1'])
                team_2_players = hist_player_stats_clean.filter(pl.col("team") == game['team_2'])

                game['team_1_player_stats'] = team_1_players.to_dicts() if len(team_1_players) > 0 else []
                game['team_2_player_stats'] = team_2_players.to_dicts() if len(team_2_players) > 0 else []
            else:
                game['team_1_player_stats'] = []
                game['team_2_player_stats'] = []

            team_2_matches.append(game)

    flat_row["team_1_match_hist"] = team_1_matches
    flat_row["team_2_match_hist"] = team_2_matches
    flat_row["team_1_hist_count"] = len(team_1_matches)
    flat_row["team_2_hist_count"] = len(team_2_matches)

    return flat_row


def build_flat_df(season: Optional[int] = None, limit: Optional[int] = None, target_date: Optional[str] = None,
                  target_date_start: Optional[str] = None, target_date_end: Optional[str] = None,
                  games_df: Optional[pl.DataFrame] = None, teams_df: Optional[pl.DataFrame] = None,
                  leaderboard_df: Optional[pl.DataFrame] = None, player_stats_df: Optional[pl.DataFrame] = None,
                  odds_dict: Optional[dict] = None, filter_incomplete_data: bool = False,
                  min_match_history: int = 1, require_leaderboard: bool = True) -> List[dict]:
    """
    Build flat dataset with raw historical match data for all games

    Args:
        season: Optional season filter
        limit: Optional limit on number of games to process (for testing)
        target_date: Optional specific date to process (format: YYYY-MM-DD)
        games_df: Optional pre-loaded games DataFrame (skips fetch if provided)
        teams_df: Optional pre-loaded teams DataFrame (skips fetch if provided)
        leaderboard_df: Optional pre-loaded leaderboard DataFrame (skips fetch if provided)
        player_stats_df: Optional pre-loaded player stats DataFrame (skips fetch if provided)
        odds_dict: Optional dictionary mapping game_id to odds data (skips fetch if provided)
        filter_incomplete_data: If True, skip games missing leaderboard or insufficient match history
        min_match_history: Minimum number of games each team must have played (default: 1)
        require_leaderboard: If True, both teams must have leaderboard data from day before (default: True)

    Returns:
        List of dictionaries, each containing game info and historical matches
    """
    if games_df is None:
        print(f"Loading games data{f' for season {season}' if season else ''}...")
        games_df = fetch_games(season=season)
        print(f"Loaded {len(games_df)} games")

    if teams_df is None:
        print("Loading teams data...")
        teams_df = fetch_teams(season=season)
        print(f"Loaded {len(teams_df)} teams")

    if leaderboard_df is None:
        print("Loading leaderboard data...")
        leaderboard_df = fetch_leaderboard()
        print(f"Loaded {len(leaderboard_df)} leaderboard records")

    if player_stats_df is None:
        print("Loading player stats data...")
        player_stats_df = fetch_player_stats(season=season)
        print(f"Loaded {len(player_stats_df)} player stats records")

    print("Adding conference information...")
    games_df = add_team_conferences(games_df, teams_df)

    print("Building flat dataset...")
    flat_rows = []

    # Process games - filter by date if provided
    if target_date:
        # Convert string date to polars date
        from datetime import datetime
        target_date_obj = datetime.strptime(target_date, "%Y-%m-%d").date()
        games_to_process = games_df.filter(pl.col("date") == target_date_obj)
        print(f"Filtered to {len(games_to_process)} games on {target_date}")
    elif target_date_start and target_date_end:
        # Convert string dates to polars dates
        from datetime import datetime
        start_date_obj = datetime.strptime(target_date_start, "%Y-%m-%d").date()
        end_date_obj = datetime.strptime(target_date_end, "%Y-%m-%d").date()
        games_to_process = games_df.filter(
            (pl.col("date") >= start_date_obj) & (pl.col("date") <= end_date_obj)
        )
        print(f"Filtered to {len(games_to_process)} games between {target_date_start} and {target_date_end}")
    elif limit:
        games_to_process = games_df.head(limit)
    else:
        games_to_process = games_df

    skipped_games = 0
    for i, row in enumerate(games_to_process.iter_rows(named=True)):
        if i % 100 == 0:
            print(f"Processing game {i+1}/{len(games_to_process)}...")

        # Build the flat row
        flat_row = build_flat_row_for_game(row, games_df, leaderboard_df, player_stats_df, odds_dict)

        # Apply filtering if enabled
        if filter_incomplete_data:
            team_1 = flat_row['team_1']
            team_2 = flat_row['team_2']
            team_1_hist_count = flat_row['team_1_hist_count']
            team_2_hist_count = flat_row['team_2_hist_count']
            team_1_leaderboard = flat_row['team_1_leaderboard']
            team_2_leaderboard = flat_row['team_2_leaderboard']

            # Check match history threshold
            if team_1_hist_count < min_match_history or team_2_hist_count < min_match_history:
                skipped_games += 1
                continue

            # Check leaderboard requirement
            if require_leaderboard and (team_1_leaderboard is None or team_2_leaderboard is None):
                skipped_games += 1
                continue

        flat_rows.append(flat_row)

    print(f"\nFlat dataset built with {len(flat_rows)} games")
    if filter_incomplete_data and skipped_games > 0:
        print(f"Skipped {skipped_games} games due to incomplete data (filter_incomplete_data=True)")

    # Convert to polars DataFrame for efficient processing
    # Keep all nested structures (match history with player stats, leaderboard dicts)
    flattened_rows = []
    for row in flat_rows:
        flat_row = {
            'game_id': row.get('game_id'),
            'date': row.get('date'),
            'start_time': row.get('start_time'),
            'season': row.get('season'),
            'team_1': row.get('team_1'),
            'team_2': row.get('team_2'),
            'team_1_conference': row.get('team_1_conference'),
            'team_2_conference': row.get('team_2_conference'),
            'team_1_is_home': row.get('team_1_is_home'),
            'team_2_is_home': row.get('team_2_is_home'),
            'location': row.get('location'),
            'team_1_score': row.get('team_1_score'),
            'team_2_score': row.get('team_2_score'),
            'total_score_outcome': row.get('total_score_outcome'),
            'team_1_winloss': row.get('team_1_winloss'),
            'team_1_leaderboard': row.get('team_1_leaderboard'),
            'team_2_leaderboard': row.get('team_2_leaderboard'),
            'team_1_match_hist': row.get('team_1_match_hist', []),
            'team_2_match_hist': row.get('team_2_match_hist', []),
            'team_1_hist_count': row.get('team_1_hist_count', 0),
            'team_2_hist_count': row.get('team_2_hist_count', 0),
        }
        flattened_rows.append(flat_row)

    # Convert to Polars DataFrame
    df = pl.DataFrame(flattened_rows)
    return df


if __name__ == "__main__":
    import json

    # Test with specific date
    print("Testing build_flat_df with games from 2019-11-15 (season 2020)...")
    data = build_flat_df(season=2020, target_date="2019-11-15")

    print(f"\nTotal games: {len(data)}")
    if len(data) > 0:
        print("\nFirst game sample:")
        print(f"  Game ID: {data[0]['game_id']}")
        print(f"  Date: {data[0]['date']}")
        print(f"  Team 1: {data[0]['team_1']} (Conference: {data[0]['team_1_conference']})")
        print(f"  Team 2: {data[0]['team_2']} (Conference: {data[0]['team_2_conference']})")
        print(f"  Team 1 historical games: {data[0]['team_1_hist_count']}")
        print(f"  Team 2 historical games: {data[0]['team_2_hist_count']}")

        # Save to JSON for inspection
        output_file = "flat_df_sample.json"
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        print(f"\nSample saved to {output_file}")
    else:
        print("No games found for that date!")
