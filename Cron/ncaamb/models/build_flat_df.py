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


def build_flat_row_for_game(game_row: dict, games_df: pl.DataFrame, leaderboard_df: pl.DataFrame, player_stats_df: pl.DataFrame) -> dict:
    """
    Build a flat row for a single game with raw historical match data

    Args:
        game_row: Dictionary containing game data (the current game)
        games_df: DataFrame with all games
        leaderboard_df: DataFrame with leaderboard data
        player_stats_df: DataFrame with player stats data

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

    flat_row = {
        "game_id": game_id,
        "season": season,
        "date": date,
        "team_1": team_1,
        "team_2": team_2,
        "team_1_conference": game_row.get("team_1_conference"),
        "team_2_conference": game_row.get("team_2_conference"),
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


def build_flat_df(season: Optional[int] = None, limit: Optional[int] = None, target_date: Optional[str] = None) -> List[dict]:
    """
    Build flat dataset with raw historical match data for all games

    Args:
        season: Optional season filter
        limit: Optional limit on number of games to process (for testing)
        target_date: Optional specific date to process (format: YYYY-MM-DD)

    Returns:
        List of dictionaries, each containing game info and historical matches
    """
    print(f"Loading games data{f' for season {season}' if season else ''}...")
    games_df = fetch_games(season=season)
    print(f"Loaded {len(games_df)} games")

    print("Loading teams data...")
    teams_df = fetch_teams(season=season)
    print(f"Loaded {len(teams_df)} teams")

    print("Loading leaderboard data...")
    leaderboard_df = fetch_leaderboard()
    print(f"Loaded {len(leaderboard_df)} leaderboard records")

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
    elif limit:
        games_to_process = games_df.head(limit)
    else:
        games_to_process = games_df

    for i, row in enumerate(games_to_process.iter_rows(named=True)):
        if i % 100 == 0:
            print(f"Processing game {i+1}/{len(games_to_process)}...")

        flat_row = build_flat_row_for_game(row, games_df, leaderboard_df, player_stats_df)
        flat_rows.append(flat_row)

    print(f"\nFlat dataset built with {len(flat_rows)} games")

    # Convert to polars DataFrame for efficient processing
    # Flatten nested structures for DataFrame conversion
    flattened_rows = []
    for row in flat_rows:
        flat_row = {
            'game_id': row.get('game_id'),
            'date': row.get('date'),
            'season': row.get('season'),
            'team_1': row.get('team_1'),
            'team_2': row.get('team_2'),
            'team_1_conference': row.get('team_1_conference'),
            'team_2_conference': row.get('team_2_conference'),
            'team_1_score': row.get('team_1_score'),
            'team_2_score': row.get('team_2_score'),
            'total_score_outcome': row.get('total_score_outcome'),
            'team_1_winloss': row.get('team_1_winloss'),
            'team_1_rank': (row.get('team_1_leaderboard') or {}).get('rank'),
            'team_1_wins': (row.get('team_1_leaderboard') or {}).get('wins'),
            'team_1_losses': (row.get('team_1_leaderboard') or {}).get('losses'),
            'team_1_barthag': (row.get('team_1_leaderboard') or {}).get('barthag'),
            'team_2_rank': (row.get('team_2_leaderboard') or {}).get('rank'),
            'team_2_wins': (row.get('team_2_leaderboard') or {}).get('wins'),
            'team_2_losses': (row.get('team_2_leaderboard') or {}).get('losses'),
            'team_2_barthag': (row.get('team_2_leaderboard') or {}).get('barthag'),
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
