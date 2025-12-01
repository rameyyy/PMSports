"""
Build comprehensive Over/Under features from flat dataset using Polars operations
Includes: rolling windows, rank-based features, leaderboard differentials, player stats, odds data,
          contextual features (rest/home/away), and interaction terms
"""
import polars as pl
import mysql.connector
from typing import Optional, Dict, List
from dotenv import load_dotenv
import os
from .ou_feature_build_utils import (
    get_rolling_windows,
    calculate_simple_average,
    calculate_variance,
    calculate_trend,
    find_closest_rank_games,
    calculate_weighted_average,
    assess_data_quality,
    american_to_decimal,
    decimal_to_american,
    average_american_odds
)
from .ou_advanced_features import (
    calculate_rest_features,
    calculate_venue_features,
    calculate_conference_features,
    calculate_game_time_features
)


def normalize_bookmaker_name(bookmaker: str) -> str:
    """
    Normalize bookmaker name to lowercase format

    Args:
        bookmaker: Bookmaker name from database

    Returns:
        Normalized bookmaker name (e.g., 'BetOnline.ag' -> 'betonline')
    """
    if not bookmaker:
        return ''

    # Mapping of database bookmaker names to normalized names
    normalizations = {
        'BetOnline.ag': 'betonline',
        'betonlineag': 'betonline',  # Handle variant without dot
        'MyBookie.ag': 'mybookie',
        'mybookieag': 'mybookie',  # Handle variant without dot
        'LowVig.ag': 'lowvig',
        'lowvig': 'lowvig',  # Handle lowercase variant
        'BetMGM': 'betmgm',
        'Bovada': 'bovada',
        'DraftKings': 'draftkings',
        'FanDuel': 'fanduel',
        'Caesars': 'caesars',  # Add Caesars/Caesars
        'Bookmaker': 'bookmaker'  # Add Bookmaker
    }

    return normalizations.get(bookmaker, bookmaker.lower().replace('.ag', '').replace(' ', ''))


def process_odds_data(odds_list: list, team_1: str = None, team_2: str = None, team_mappings: dict = None) -> dict:
    """
    Process odds data from multiple sportsbooks for a game
    Creates individual columns for each sportsbook's O/U line + over/under odds
    Fills nulls with average for that metric
    Maps home_team/away_team from odds to team_1/team_2 using team_mappings

    Args:
        odds_list: List of odds dictionaries from database with home_team/away_team fields
        team_1: Name of team_1 (internal team naming)
        team_2: Name of team_2 (internal team naming)
        team_mappings: Dict mapping from_odds_team_name -> my_team_name

    Returns:
        Dictionary with odds features including:
        - Per-bookmaker columns: betonline_ml_team_1, betonline_ml_team_2, etc.
        - Average columns: avg_ml_team_1, avg_ml_team_2, etc.
        - O/U columns: betonline_ou_line, betonline_over_odds, etc.
        - Spread columns: betonline_spread_pts_team_1, etc.
    """
    # Standard sportsbooks (normalized names)
    sportsbooks = ['betmgm', 'betonline', 'bovada', 'draftkings', 'fanduel', 'lowvig', 'mybookie']

    odds_features = {}

    # Create columns for each sportsbook
    ou_lines_by_book = {}
    over_odds_by_book = {}
    under_odds_by_book = {}
    ml_team_1_by_book = {}
    ml_team_2_by_book = {}
    spread_pts_team_1_by_book = {}
    spread_odds_team_1_by_book = {}
    spread_pts_team_2_by_book = {}
    spread_odds_team_2_by_book = {}

    for book in sportsbooks:
        odds_features[f'{book}_ou_line'] = None
        odds_features[f'{book}_over_odds'] = None
        odds_features[f'{book}_under_odds'] = None
        odds_features[f'{book}_ml_team_1'] = None
        odds_features[f'{book}_ml_team_2'] = None
        odds_features[f'{book}_spread_pts_team_1'] = None
        odds_features[f'{book}_spread_odds_team_1'] = None
        odds_features[f'{book}_spread_pts_team_2'] = None
        odds_features[f'{book}_spread_odds_team_2'] = None

    # Add average columns
    odds_features['avg_ou_line'] = None
    odds_features['ou_line_variance'] = None
    odds_features['avg_over_odds'] = None
    odds_features['avg_under_odds'] = None
    odds_features['num_books_with_ou'] = 0

    # Spread and ML features (now mapped to team_1/team_2)
    odds_features['avg_spread_pts_team_1'] = None
    odds_features['avg_spread_pts_team_2'] = None
    odds_features['spread_variance'] = None
    odds_features['avg_spread_odds_team_1'] = None
    odds_features['avg_spread_odds_team_2'] = None
    odds_features['avg_ml_team_1'] = None
    odds_features['avg_ml_team_2'] = None
    odds_features['num_books_with_spread'] = 0
    odds_features['num_books_with_ml'] = 0

    if not odds_list or len(odds_list) == 0:
        return odds_features

    # Extract data by sportsbook
    ou_lines_all = []
    over_odds_all = []
    under_odds_all = []

    spread_pts_team_1_all = []
    spread_pts_team_2_all = []
    spread_odds_team_1_all = []
    spread_odds_team_2_all = []
    ml_for_team_1 = []
    ml_for_team_2 = []

    for odds in odds_list:
        if not isinstance(odds, dict):
            continue

        bookmaker_raw = odds.get('bookmaker')
        bookmaker = normalize_bookmaker_name(bookmaker_raw)

        # O/U line and odds
        ou_point = odds.get('over_point') or odds.get('under_point')
        over_odds = odds.get('over_odds')
        under_odds = odds.get('under_odds')

        if ou_point is not None:
            ou_point_float = float(ou_point)
            ou_lines_all.append(ou_point_float)
            if bookmaker in sportsbooks:
                ou_lines_by_book[bookmaker] = ou_point_float

        if over_odds is not None:
            over_odds_float = float(over_odds)
            over_odds_all.append(over_odds_float)
            if bookmaker in sportsbooks:
                over_odds_by_book[bookmaker] = over_odds_float

        if under_odds is not None:
            under_odds_float = float(under_odds)
            under_odds_all.append(under_odds_float)
            if bookmaker in sportsbooks:
                under_odds_by_book[bookmaker] = under_odds_float

        # Get home_team and away_team from odds data
        home_team_odds = odds.get('home_team')
        away_team_odds = odds.get('away_team')

        # Map odds team names to internal team names using team_mappings
        home_team_mapped = home_team_odds
        away_team_mapped = away_team_odds

        if team_mappings:
            home_team_mapped = team_mappings.get(home_team_odds, home_team_odds)
            away_team_mapped = team_mappings.get(away_team_odds, away_team_odds)

        # Determine which team is team_1 and which is team_2
        home_is_team_1 = (home_team_mapped == team_1)
        away_is_team_1 = (away_team_mapped == team_1)

        # Process spread data
        spread_pts_home = odds.get('spread_pts_home')
        spread_home = odds.get('spread_home')
        spread_pts_away = odds.get('spread_pts_away')
        spread_away = odds.get('spread_away')

        # Map spread to team_1 and team_2
        if home_is_team_1:
            # home is team_1, away is team_2
            if spread_pts_home is not None:
                spread_pts_team_1_val = float(spread_pts_home)
                spread_pts_team_1_all.append(spread_pts_team_1_val)
                if bookmaker in sportsbooks:
                    spread_pts_team_1_by_book[bookmaker] = spread_pts_team_1_val

            if spread_pts_away is not None:
                spread_pts_team_2_val = float(spread_pts_away)
                spread_pts_team_2_all.append(spread_pts_team_2_val)
                if bookmaker in sportsbooks:
                    spread_pts_team_2_by_book[bookmaker] = spread_pts_team_2_val

            if spread_home is not None:
                spread_odds_team_1_val = float(spread_home)
                spread_odds_team_1_all.append(spread_odds_team_1_val)
                if bookmaker in sportsbooks:
                    spread_odds_team_1_by_book[bookmaker] = spread_odds_team_1_val

            if spread_away is not None:
                spread_odds_team_2_val = float(spread_away)
                spread_odds_team_2_all.append(spread_odds_team_2_val)
                if bookmaker in sportsbooks:
                    spread_odds_team_2_by_book[bookmaker] = spread_odds_team_2_val

        elif away_is_team_1:
            # away is team_1, home is team_2
            if spread_pts_away is not None:
                spread_pts_team_1_val = float(spread_pts_away)
                spread_pts_team_1_all.append(spread_pts_team_1_val)
                if bookmaker in sportsbooks:
                    spread_pts_team_1_by_book[bookmaker] = spread_pts_team_1_val

            if spread_pts_home is not None:
                spread_pts_team_2_val = float(spread_pts_home)
                spread_pts_team_2_all.append(spread_pts_team_2_val)
                if bookmaker in sportsbooks:
                    spread_pts_team_2_by_book[bookmaker] = spread_pts_team_2_val

            if spread_away is not None:
                spread_odds_team_1_val = float(spread_away)
                spread_odds_team_1_all.append(spread_odds_team_1_val)
                if bookmaker in sportsbooks:
                    spread_odds_team_1_by_book[bookmaker] = spread_odds_team_1_val

            if spread_home is not None:
                spread_odds_team_2_val = float(spread_home)
                spread_odds_team_2_all.append(spread_odds_team_2_val)
                if bookmaker in sportsbooks:
                    spread_odds_team_2_by_book[bookmaker] = spread_odds_team_2_val

        # Process moneyline data
        ml_home = odds.get('ml_home')
        ml_away = odds.get('ml_away')

        # Map moneyline to team_1 and team_2
        if home_is_team_1 and ml_home is not None:
            ml_team_1_val = float(ml_home)
            ml_for_team_1.append(ml_team_1_val)
            if bookmaker in sportsbooks:
                ml_team_1_by_book[bookmaker] = ml_team_1_val

        if home_is_team_1 and ml_away is not None:
            ml_team_2_val = float(ml_away)
            ml_for_team_2.append(ml_team_2_val)
            if bookmaker in sportsbooks:
                ml_team_2_by_book[bookmaker] = ml_team_2_val

        if away_is_team_1 and ml_away is not None:
            ml_team_1_val = float(ml_away)
            ml_for_team_1.append(ml_team_1_val)
            if bookmaker in sportsbooks:
                ml_team_1_by_book[bookmaker] = ml_team_1_val

        if away_is_team_1 and ml_home is not None:
            ml_team_2_val = float(ml_home)
            ml_for_team_2.append(ml_team_2_val)
            if bookmaker in sportsbooks:
                ml_team_2_by_book[bookmaker] = ml_team_2_val

    # Calculate averages for O/U
    if ou_lines_all:
        avg_ou = float(sum(ou_lines_all) / len(ou_lines_all))
        odds_features['avg_ou_line'] = avg_ou
        odds_features['num_books_with_ou'] = len(ou_lines_all)

        if len(ou_lines_all) > 1:
            variance = sum((x - avg_ou) ** 2 for x in ou_lines_all) / len(ou_lines_all)
            odds_features['ou_line_variance'] = float(variance ** 0.5)

    if over_odds_all:
        # Using decimal odds conversion to properly handle mixed +/- odds
        avg_over_odds = average_american_odds(over_odds_all)
        odds_features['avg_over_odds'] = float(avg_over_odds) if avg_over_odds is not None else None

    if under_odds_all:
        # Using decimal odds conversion to properly handle mixed +/- odds
        avg_under_odds = average_american_odds(under_odds_all)
        odds_features['avg_under_odds'] = float(avg_under_odds) if avg_under_odds is not None else None

    # Fill individual sportsbook columns, use average for nulls (O/U only)
    for book in sportsbooks:
        if book in ou_lines_by_book:
            odds_features[f'{book}_ou_line'] = ou_lines_by_book[book]
        elif odds_features['avg_ou_line'] is not None:
            odds_features[f'{book}_ou_line'] = odds_features['avg_ou_line']

        if book in over_odds_by_book:
            odds_features[f'{book}_over_odds'] = over_odds_by_book[book]
        elif odds_features['avg_over_odds'] is not None:
            odds_features[f'{book}_over_odds'] = odds_features['avg_over_odds']

        if book in under_odds_by_book:
            odds_features[f'{book}_under_odds'] = under_odds_by_book[book]
        elif odds_features['avg_under_odds'] is not None:
            odds_features[f'{book}_under_odds'] = odds_features['avg_under_odds']

        # Moneyline by book (not filled with average, leave as null if missing)
        if book in ml_team_1_by_book:
            odds_features[f'{book}_ml_team_1'] = ml_team_1_by_book[book]
        if book in ml_team_2_by_book:
            odds_features[f'{book}_ml_team_2'] = ml_team_2_by_book[book]

        # Spread by book (not filled with average, leave as null if missing)
        if book in spread_pts_team_1_by_book:
            odds_features[f'{book}_spread_pts_team_1'] = spread_pts_team_1_by_book[book]
        if book in spread_odds_team_1_by_book:
            odds_features[f'{book}_spread_odds_team_1'] = spread_odds_team_1_by_book[book]
        if book in spread_pts_team_2_by_book:
            odds_features[f'{book}_spread_pts_team_2'] = spread_pts_team_2_by_book[book]
        if book in spread_odds_team_2_by_book:
            odds_features[f'{book}_spread_odds_team_2'] = spread_odds_team_2_by_book[book]

    # Calculate spread averages
    if spread_pts_team_1_all:
        odds_features['num_books_with_spread'] = len(spread_pts_team_1_all)
        odds_features['avg_spread_pts_team_1'] = float(sum(spread_pts_team_1_all) / len(spread_pts_team_1_all))

    if spread_pts_team_2_all:
        odds_features['avg_spread_pts_team_2'] = float(sum(spread_pts_team_2_all) / len(spread_pts_team_2_all))

    if spread_pts_team_1_all and len(spread_pts_team_1_all) > 1:
        avg_spread = odds_features['avg_spread_pts_team_1']
        variance = sum((x - avg_spread) ** 2 for x in spread_pts_team_1_all) / len(spread_pts_team_1_all)
        odds_features['spread_variance'] = float(variance ** 0.5)

    if spread_odds_team_1_all:
        # Using decimal odds conversion to properly handle mixed +/- odds
        avg_spread_odds_team_1 = average_american_odds(spread_odds_team_1_all)
        odds_features['avg_spread_odds_team_1'] = float(avg_spread_odds_team_1) if avg_spread_odds_team_1 is not None else None

    if spread_odds_team_2_all:
        # Using decimal odds conversion to properly handle mixed +/- odds
        avg_spread_odds_team_2 = average_american_odds(spread_odds_team_2_all)
        odds_features['avg_spread_odds_team_2'] = float(avg_spread_odds_team_2) if avg_spread_odds_team_2 is not None else None

    # Moneyline averages - now correctly mapped to team_1/team_2
    # Using decimal odds conversion to properly handle mixed +/- odds
    if ml_for_team_1:
        odds_features['num_books_with_ml'] = len(ml_for_team_1)
        avg_ml_team_1 = average_american_odds(ml_for_team_1)
        odds_features['avg_ml_team_1'] = float(avg_ml_team_1) if avg_ml_team_1 is not None else None

    if ml_for_team_2:
        avg_ml_team_2 = average_american_odds(ml_for_team_2)
        odds_features['avg_ml_team_2'] = float(avg_ml_team_2) if avg_ml_team_2 is not None else None

    return odds_features


def build_rolling_window_features(team_hist: List[dict], team_prefix: str, stat_key: str,
                                  windows: List[int]) -> Dict[str, Optional[float]]:
    """
    Build rolling window features for a given stat across multiple window sizes
    Uses fallback: if not enough data for larger window, uses smaller window values

    Args:
        team_hist: List of historical games
        team_prefix: 'team_1' or 'team_2'
        stat_key: Base stat key (e.g., 'score')
        windows: List of window sizes to calculate

    Returns:
        Dictionary with keys like 'stat_last3', 'stat_last5', etc.
        Falls back to smaller windows if not enough data
    """
    features = {}
    full_key = f'{team_prefix}_{stat_key}'
    computed_values = {}

    for window in windows:
        if window == 0:
            actual_window = len(team_hist)
        else:
            actual_window = min(window, len(team_hist))

        # Get games to use
        games_to_use = team_hist[:actual_window]

        # Calculate if we have enough data (at least 2 games)
        if len(games_to_use) >= 2:
            avg_val = calculate_simple_average(games_to_use, full_key)
            var_val = None
            trend_val = None

            if stat_key in ['score', 'total', 'efg_pct', 'pace']:
                var_val = calculate_variance(games_to_use, full_key)
                trend_val = calculate_trend(games_to_use, full_key)

            computed_values[window] = {
                'avg': avg_val,
                'variance': var_val,
                'trend': trend_val
            }
        else:
            # Not enough data - use fallback from smaller window
            fallback_window = None
            for prev_window in sorted([w for w in windows if w < window], reverse=True):
                if prev_window in computed_values:
                    fallback_window = prev_window
                    break

            if fallback_window:
                computed_values[window] = computed_values[fallback_window]
            else:
                computed_values[window] = None

    # Assign features from computed values
    for window in windows:
        if computed_values[window]:
            features[f'{team_prefix}_{stat_key}_last{window}'] = computed_values[window]['avg']
            if stat_key in ['score', 'total', 'efg_pct', 'pace']:
                features[f'{team_prefix}_{stat_key}_variance_last{window}'] = computed_values[window]['variance']
                features[f'{team_prefix}_{stat_key}_trend_last{window}'] = computed_values[window]['trend']
        else:
            features[f'{team_prefix}_{stat_key}_last{window}'] = None
            if stat_key in ['score', 'total', 'efg_pct', 'pace']:
                features[f'{team_prefix}_{stat_key}_variance_last{window}'] = None
                features[f'{team_prefix}_{stat_key}_trend_last{window}'] = None

    return features


def build_closest_rank_features(team_hist: List[dict], opp_rank: Optional[int],
                               team_prefix: str, team_name: str, windows: List[int] = [3, 5, 7]) -> Dict[str, Optional[float]]:
    """
    Build features based on closest rank historical games

    Args:
        team_hist: Match history for a team
        opp_rank: Opponent's rank in current game
        team_prefix: 'team_1' or 'team_2'
        team_name: Name of the team (to find correct leaderboard in historical games)
        windows: List of window sizes (default: [3, 5, 7] for top 3, 5, 7 closest rank games)

    Returns:
        Dictionary with closest rank features for each window size
    """
    features = {}
    base_features = ['score', 'score_allowed', 'total', 'margin', 'adjoe', 'adjde']

    if not opp_rank or not team_hist:
        # Return None values if we can't calculate
        for window in windows:
            for feat in base_features:
                features[f'{team_prefix}_{feat}_closest{window}rank'] = None
        return features

    # Find closest rank games (use max window size)
    max_window = max(windows)
    closest_games, weights = find_closest_rank_games(team_hist, opp_rank, top_n=max_window, rank_window=50, team_name=team_name)

    if not closest_games:
        # Return None values if no close rank games found
        for window in windows:
            for feat in base_features:
                features[f'{team_prefix}_{feat}_closest{window}rank'] = None
        return features

    # Extract all stats from closest games
    all_scores = []
    all_scores_allowed = []
    all_totals = []
    all_margins = []
    all_adjoe_values = []
    all_adjde_values = []

    for game in closest_games:
        if not isinstance(game, dict):
            continue

        t_score = game.get(f'{team_prefix}_score')
        opp_score = game.get(f'{"team_2" if team_prefix == "team_1" else "team_1"}_score')

        if t_score is not None:
            all_scores.append(float(t_score))
        else:
            all_scores.append(None)

        if opp_score is not None:
            all_scores_allowed.append(float(opp_score))
        else:
            all_scores_allowed.append(None)

        if t_score is not None and opp_score is not None:
            all_totals.append(float(t_score) + float(opp_score))
            all_margins.append(float(t_score) - float(opp_score))
        else:
            all_totals.append(None)
            all_margins.append(None)

        # Extract adjoe and adjde from leaderboard snapshot
        # Determine which team this is in the historical game
        if game.get('team_1') == team_name:
            lb = game.get('team_1_leaderboard')
        elif game.get('team_2') == team_name:
            lb = game.get('team_2_leaderboard')
        else:
            lb = None

        if lb and isinstance(lb, dict):
            adjoe = lb.get('adjoe')
            adjde = lb.get('adjde')

            if adjoe is not None:
                try:
                    all_adjoe_values.append(float(adjoe))
                except (ValueError, TypeError):
                    all_adjoe_values.append(None)
            else:
                all_adjoe_values.append(None)

            if adjde is not None:
                try:
                    all_adjde_values.append(float(adjde))
                except (ValueError, TypeError):
                    all_adjde_values.append(None)
            else:
                all_adjde_values.append(None)
        else:
            all_adjoe_values.append(None)
            all_adjde_values.append(None)

    # Calculate weighted averages for each window size
    # Store computed values for fallback (use smaller window if larger not available)
    computed_values = {}

    for window in windows:
        # Get subset for this window
        window_size = min(window, len(closest_games))

        # Filter out None values and get corresponding weights
        scores = [v for v in all_scores[:window_size] if v is not None]
        scores_allowed = [v for v in all_scores_allowed[:window_size] if v is not None]
        totals = [v for v in all_totals[:window_size] if v is not None]
        margins = [v for v in all_margins[:window_size] if v is not None]
        adjoe_values = [v for v in all_adjoe_values[:window_size] if v is not None]
        adjde_values = [v for v in all_adjde_values[:window_size] if v is not None]

        # If we have data for this window, compute features
        if len(scores) >= min(window, 2):  # Need at least 2 games or whatever we can get
            computed_values[window] = {
                'score': calculate_weighted_average(scores, weights[:len(scores)]),
                'score_allowed': calculate_weighted_average(scores_allowed, weights[:len(scores_allowed)]),
                'total': calculate_weighted_average(totals, weights[:len(totals)]),
                'margin': calculate_weighted_average(margins, weights[:len(margins)]),
                'adjoe': calculate_weighted_average(adjoe_values, weights[:len(adjoe_values)]),
                'adjde': calculate_weighted_average(adjde_values, weights[:len(adjde_values)])
            }
        else:
            # Not enough data for this window - use fallback from smaller window
            # Find the largest window smaller than this one that has data
            fallback_window = None
            for prev_window in sorted([w for w in windows if w < window], reverse=True):
                if prev_window in computed_values:
                    fallback_window = prev_window
                    break

            if fallback_window:
                computed_values[window] = computed_values[fallback_window]
            else:
                computed_values[window] = None

    # Assign features from computed values
    for window in windows:
        if computed_values[window]:
            for feat in base_features:
                features[f'{team_prefix}_{feat}_closest{window}rank'] = computed_values[window].get(feat)
        else:
            for feat in base_features:
                features[f'{team_prefix}_{feat}_closest{window}rank'] = None

    return features


def build_leaderboard_differentials(team1_lb: Optional[dict], team2_lb: Optional[dict]) -> Dict[str, Optional[float]]:
    """
    Build differential features comparing team leaderboard stats

    Args:
        team1_lb: Team 1's leaderboard dict
        team2_lb: Team 2's leaderboard dict

    Returns:
        Dictionary with differential features
    """
    features = {}

    if not team1_lb or not team2_lb:
        return {
            'rank_differential': None,
            'barthag_differential': None,
            'adj_tempo_differential': None,
            'adjoe_differential': None,
            'adjde_differential': None,
            'efg_off_differential': None,
            'efg_def_differential': None,
            'orb_rate_differential': None,
            'tor_rate_differential': None,
            '3p_rate_differential': None,
            'ftr_differential': None,
        }

    # Extract key fields with safe defaults
    def safe_float(val):
        try:
            return float(val) if val is not None else None
        except (ValueError, TypeError):
            return None

    # Rank differential
    t1_rank = safe_float(team1_lb.get('rank'))
    t2_rank = safe_float(team2_lb.get('rank'))
    if t1_rank and t2_rank:
        features['rank_differential'] = t1_rank - t2_rank
    else:
        features['rank_differential'] = None

    # Barthag differential (power rating)
    t1_barthag = safe_float(team1_lb.get('barthag'))
    t2_barthag = safe_float(team2_lb.get('barthag'))
    if t1_barthag and t2_barthag:
        features['barthag_differential'] = t1_barthag - t2_barthag
    else:
        features['barthag_differential'] = None

    # Tempo differential
    t1_tempo = safe_float(team1_lb.get('adj_t'))
    t2_tempo = safe_float(team2_lb.get('adj_t'))
    if t1_tempo and t2_tempo:
        features['adj_tempo_differential'] = t1_tempo - t2_tempo
    else:
        features['adj_tempo_differential'] = None

    # Adjusted Offensive Efficiency differential
    t1_adjoe = safe_float(team1_lb.get('adjoe'))
    t2_adjoe = safe_float(team2_lb.get('adjoe'))
    if t1_adjoe is not None and t2_adjoe is not None:
        features['adjoe_differential'] = t1_adjoe - t2_adjoe
    else:
        features['adjoe_differential'] = None

    # Adjusted Defensive Efficiency differential
    t1_adjde = safe_float(team1_lb.get('adjde'))
    t2_adjde = safe_float(team2_lb.get('adjde'))
    if t1_adjde is not None and t2_adjde is not None:
        features['adjde_differential'] = t1_adjde - t2_adjde
    else:
        features['adjde_differential'] = None

    # Four factors differentials
    t1_efg_off = safe_float(team1_lb.get('efg_off_prcnt'))
    t2_efg_off = safe_float(team2_lb.get('efg_off_prcnt'))
    if t1_efg_off and t2_efg_off:
        features['efg_off_differential'] = t1_efg_off - t2_efg_off
    else:
        features['efg_off_differential'] = None

    t1_efg_def = safe_float(team1_lb.get('efg_def_prcnt'))
    t2_efg_def = safe_float(team2_lb.get('efg_def_prcnt'))
    if t1_efg_def and t2_efg_def:
        features['efg_def_differential'] = t1_efg_def - t2_efg_def
    else:
        features['efg_def_differential'] = None

    t1_orb = safe_float(team1_lb.get('orb'))
    t2_orb = safe_float(team2_lb.get('orb'))
    if t1_orb and t2_orb:
        features['orb_rate_differential'] = t1_orb - t2_orb
    else:
        features['orb_rate_differential'] = None

    t1_tor = safe_float(team1_lb.get('tord'))
    t2_tor = safe_float(team2_lb.get('tord'))
    if t1_tor and t2_tor:
        features['tor_rate_differential'] = t1_tor - t2_tor
    else:
        features['tor_rate_differential'] = None

    t1_3p_rate = safe_float(team1_lb.get('3pr'))
    t2_3p_rate = safe_float(team2_lb.get('3pr'))
    if t1_3p_rate and t2_3p_rate:
        features['3p_rate_differential'] = t1_3p_rate - t2_3p_rate
    else:
        features['3p_rate_differential'] = None

    t1_ftr = safe_float(team1_lb.get('ftr'))
    t2_ftr = safe_float(team2_lb.get('ftr'))
    if t1_ftr and t2_ftr:
        features['ftr_differential'] = t1_ftr - t2_ftr
    else:
        features['ftr_differential'] = None

    return features


def build_leaderboard_rolling_features(team_hist: List[dict], team_prefix: str, stat_key: str,
                                       windows: List[int], team_name: str) -> Dict[str, Optional[float]]:
    """
    Build rolling window features for leaderboard-based stats (adjoe, adjde, adj_t)
    Extracts stats from historical game leaderboard snapshots

    Args:
        team_hist: List of historical games
        team_prefix: 'team_1' or 'team_2'
        stat_key: Leaderboard stat key (e.g., 'adjoe', 'adjde', 'adj_t')
        windows: List of window sizes to calculate
        team_name: Name of the team (to find correct leaderboard in historical games)

    Returns:
        Dictionary with keys like 'team_1_adjoe_last3', 'team_1_adjoe_last5', etc.
    """
    features = {}

    # Extract the stat values from historical game leaderboards
    stat_values = []
    for game in team_hist:
        if not isinstance(game, dict):
            continue

        # Determine which team this is in the historical game
        # and get the corresponding leaderboard
        if game.get('team_1') == team_name:
            lb = game.get('team_1_leaderboard')
        elif game.get('team_2') == team_name:
            lb = game.get('team_2_leaderboard')
        else:
            continue

        if lb and isinstance(lb, dict):
            val = lb.get(stat_key)
            if val is not None:
                try:
                    stat_values.append(float(val))
                except (ValueError, TypeError):
                    pass

    # Calculate rolling averages for each window with fallback
    computed_values = {}

    for window in windows:
        if window == 0:
            actual_window = len(stat_values)
        else:
            actual_window = min(window, len(stat_values))

        # Get values for this window
        window_values = stat_values[:actual_window]

        # Calculate average if we have enough data
        if len(window_values) >= 2:
            avg_val = sum(window_values) / len(window_values)
            mean = avg_val
            variance = sum((x - mean) ** 2 for x in window_values) / len(window_values)
            variance_val = variance ** 0.5 if len(window_values) > 1 else None

            computed_values[window] = {
                'avg': avg_val,
                'variance': variance_val
            }
        else:
            # Not enough data - use fallback from smaller window
            fallback_window = None
            for prev_window in sorted([w for w in windows if w < window], reverse=True):
                if prev_window in computed_values:
                    fallback_window = prev_window
                    break

            if fallback_window:
                computed_values[window] = computed_values[fallback_window]
            else:
                computed_values[window] = None

    # Assign features from computed values
    for window in windows:
        if computed_values[window]:
            features[f'{team_prefix}_{stat_key}_last{window}'] = computed_values[window]['avg']
            features[f'{team_prefix}_{stat_key}_variance_last{window}'] = computed_values[window]['variance']
        else:
            features[f'{team_prefix}_{stat_key}_last{window}'] = None
            features[f'{team_prefix}_{stat_key}_variance_last{window}'] = None

    return features


def build_ou_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Build comprehensive Over/Under features from flat dataset

    Args:
        df: Polars DataFrame with game data, match_hist, leaderboard, and odds columns

    Returns:
        Polars DataFrame with all O/U features
    """
    features_list = []
    rolling_windows = get_rolling_windows()

    # Load environment variables for database connection
    load_dotenv()

    # Batch load odds data for all games upfront (much faster than per-game queries)
    db_connection = None
    all_odds_by_game_id = {}
    try:
        db_connection = mysql.connector.connect(
            host=os.getenv('DB_HOST'),
            port=int(os.getenv('DB_PORT', 3306)),
            user=os.getenv('DB_USER'),
            password=os.getenv('DB_PASSWORD'),
            database=os.getenv('NCAAMB_DB')
        )
        print("[+] Database connection established, loading odds data...")

        # Load ALL odds records (much more efficient than per-game queries)
        cursor = db_connection.cursor(dictionary=True)
        cursor.execute("""
            SELECT game_id, home_team, away_team, bookmaker,
                   ml_home, ml_away,
                   spread_home, spread_pts_home, spread_away, spread_pts_away,
                   over_odds, under_odds, over_point, under_point,
                   start_time
            FROM odds
            ORDER BY game_id
        """)

        # Organize odds by game_id for fast lookup
        for row in cursor.fetchall():
            game_id = row['game_id']
            if game_id not in all_odds_by_game_id:
                all_odds_by_game_id[game_id] = []
            all_odds_by_game_id[game_id].append(row)

        cursor.close()
        print(f"[+] Loaded odds data for {len(all_odds_by_game_id)} games")
    except Exception as e:
        print(f"[!] Warning: Could not load odds from database: {e}")
        all_odds_by_game_id = {}

    # Load team mappings for moneyline odds mapping
    team_mappings = {}
    try:
        import csv
        mappings_path = os.path.join(os.path.dirname(__file__), '..', '..', 'bookmaker', 'team_mappings.csv')
        with open(mappings_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get('from_odds_team_name') and row.get('my_team_name'):
                    team_mappings[row['from_odds_team_name']] = row['my_team_name']
        print(f"[+] Loaded team mappings for {len(team_mappings)} teams")
    except Exception as e:
        print(f"[!] Warning: Could not load team_mappings: {e}")

    for row in df.iter_rows(named=True):
        features = {
            'game_id': row['game_id'],
            'date': row['date'],
            'team_1': row['team_1'],
            'team_2': row['team_2'],
            'team_1_score': row['team_1_score'],
            'team_2_score': row['team_2_score'],
            'actual_total': (row['team_1_score'] + row['team_2_score']) if row['team_1_score'] and row['team_2_score'] else None,
        }

        t1_hist = row.get('team_1_match_hist', [])
        t2_hist = row.get('team_2_match_hist', [])
        t1_lb = row.get('team_1_leaderboard')
        t2_lb = row.get('team_2_leaderboard')

        # Get opponent ranks from current game leaderboard data
        t1_rank = None
        t2_rank = None
        if t1_lb and isinstance(t1_lb, dict):
            try:
                t1_rank = int(t1_lb.get('rank', 0)) or None
            except (ValueError, TypeError):
                pass
        if t2_lb and isinstance(t2_lb, dict):
            try:
                t2_rank = int(t2_lb.get('rank', 0)) or None
            except (ValueError, TypeError):
                pass

        # Extract current adjoe/adjde values for both teams
        def safe_float(val):
            try:
                return float(val) if val is not None else None
            except (ValueError, TypeError):
                return None

        if t1_lb and isinstance(t1_lb, dict):
            features['team_1_adjoe'] = safe_float(t1_lb.get('adjoe'))
            features['team_1_adjde'] = safe_float(t1_lb.get('adjde'))
        else:
            features['team_1_adjoe'] = None
            features['team_1_adjde'] = None

        if t2_lb and isinstance(t2_lb, dict):
            features['team_2_adjoe'] = safe_float(t2_lb.get('adjoe'))
            features['team_2_adjde'] = safe_float(t2_lb.get('adjde'))
        else:
            features['team_2_adjoe'] = None
            features['team_2_adjde'] = None

        # ============================================
        # TEAM-LEVEL ROLLING WINDOW FEATURES
        # ============================================
        # Game-level stats (from match history)
        team_stats_to_roll = ['score', 'efg_pct']

        for stat in team_stats_to_roll:
            t1_features = build_rolling_window_features(t1_hist, 'team_1', stat, rolling_windows)
            t2_features = build_rolling_window_features(t2_hist, 'team_2', stat, rolling_windows)
            features.update(t1_features)
            features.update(t2_features)

        # Leaderboard-based stats (from historical leaderboard snapshots)
        leaderboard_stats_to_roll = ['adjoe', 'adjde', 'adj_t']

        for stat in leaderboard_stats_to_roll:
            t1_lb_features = build_leaderboard_rolling_features(t1_hist, 'team_1', stat, rolling_windows, row['team_1'])
            t2_lb_features = build_leaderboard_rolling_features(t2_hist, 'team_2', stat, rolling_windows, row['team_2'])
            features.update(t1_lb_features)
            features.update(t2_lb_features)

        # ============================================
        # RANK-BASED CLOSEST MATCH FEATURES
        # ============================================
        t1_closest = build_closest_rank_features(t1_hist, t2_rank, 'team_1', row['team_1'])
        t2_closest = build_closest_rank_features(t2_hist, t1_rank, 'team_2', row['team_2'])
        features.update(t1_closest)
        features.update(t2_closest)

        # Derived: Combined features from closest rank games (for each window)
        rank_windows = [3, 5, 7]
        for window in rank_windows:
            # Combined expected total
            t1_score_closest = t1_closest.get(f'team_1_score_closest{window}rank')
            t2_score_closest = t2_closest.get(f'team_2_score_closest{window}rank')
            if t1_score_closest and t2_score_closest:
                features[f'combined_expected_total_closest{window}rank'] = t1_score_closest + t2_score_closest
            else:
                features[f'combined_expected_total_closest{window}rank'] = None

            # Combined adjoe/adjde from closest rank games
            t1_adjoe_closest = t1_closest.get(f'team_1_adjoe_closest{window}rank')
            t2_adjoe_closest = t2_closest.get(f'team_2_adjoe_closest{window}rank')
            if t1_adjoe_closest is not None and t2_adjoe_closest is not None:
                features[f'combined_adjoe_closest{window}rank'] = (t1_adjoe_closest + t2_adjoe_closest) / 2
                features[f'adjoe_differential_closest{window}rank'] = t1_adjoe_closest - t2_adjoe_closest
            else:
                features[f'combined_adjoe_closest{window}rank'] = None
                features[f'adjoe_differential_closest{window}rank'] = None

            t1_adjde_closest = t1_closest.get(f'team_1_adjde_closest{window}rank')
            t2_adjde_closest = t2_closest.get(f'team_2_adjde_closest{window}rank')
            if t1_adjde_closest is not None and t2_adjde_closest is not None:
                features[f'combined_adjde_closest{window}rank'] = (t1_adjde_closest + t2_adjde_closest) / 2
                features[f'adjde_differential_closest{window}rank'] = t1_adjde_closest - t2_adjde_closest
            else:
                features[f'combined_adjde_closest{window}rank'] = None
                features[f'adjde_differential_closest{window}rank'] = None

        # ============================================
        # LEADERBOARD DIFFERENTIALS
        # ============================================
        lb_diffs = build_leaderboard_differentials(t1_lb, t2_lb)
        features.update(lb_diffs)

        # Derived adjoe/adjde features
        # Combined offensive/defensive efficiency (for total scoring prediction)
        t1_adjoe = features.get('team_1_adjoe')
        t2_adjoe = features.get('team_2_adjoe')
        t1_adjde = features.get('team_1_adjde')
        t2_adjde = features.get('team_2_adjde')

        if t1_adjoe is not None and t2_adjoe is not None:
            features['combined_adjoe'] = (t1_adjoe + t2_adjoe) / 2
        else:
            features['combined_adjoe'] = None

        if t1_adjde is not None and t2_adjde is not None:
            features['combined_adjde'] = (t1_adjde + t2_adjde) / 2
        else:
            features['combined_adjde'] = None

        # ============================================
        # ODDS DATA FEATURES
        # ============================================
        # Get pre-loaded odds for this game
        game_id = row.get('game_id')
        team_1 = row.get('team_1')
        team_2 = row.get('team_2')
        game_odds = all_odds_by_game_id.get(game_id, [])

        odds_features = process_odds_data(game_odds, team_1=team_1, team_2=team_2, team_mappings=team_mappings)
        features.update(odds_features)

        # Implied scores from spread and O/U line
        # Use team_1 spread points for calculation
        avg_spread_team_1 = odds_features.get('avg_spread_pts_team_1')
        avg_ou_line = odds_features.get('avg_ou_line')

        if avg_ou_line and avg_spread_team_1:
            # If team_1 spread is negative, they're favored
            # team_1 implied = (total + spread) / 2, team_2 implied = (total - spread) / 2
            implied_team_1_score = (avg_ou_line + avg_spread_team_1) / 2
            implied_team_2_score = (avg_ou_line - avg_spread_team_1) / 2
            features['implied_team_1_score'] = implied_team_1_score
            features['implied_team_2_score'] = implied_team_2_score
            features['spread_ou_agreement'] = abs(implied_team_1_score - implied_team_2_score)
        else:
            features['implied_team_1_score'] = None
            features['implied_team_2_score'] = None
            features['spread_ou_agreement'] = None

        # ============================================
        # DATA QUALITY FEATURES
        # ============================================
        t1_with_lb = sum(1 for g in t1_hist if isinstance(g, dict) and g.get('team_1_leaderboard'))
        t2_with_lb = sum(1 for g in t2_hist if isinstance(g, dict) and g.get('team_2_leaderboard'))

        t1_quality = assess_data_quality(len(t1_hist), t1_with_lb, len(t1_hist))
        t2_quality = assess_data_quality(len(t2_hist), t2_with_lb, len(t2_hist))

        features['team_1_games_available'] = len(t1_hist)
        features['team_2_games_available'] = len(t2_hist)
        features['team_1_data_quality'] = t1_quality['overall_data_quality']
        features['team_2_data_quality'] = t2_quality['overall_data_quality']

        # ============================================
        # CONTEXTUAL FEATURES - REST & FATIGUE
        # ============================================
        t1_rest = calculate_rest_features(row['date'], t1_hist, row['team_1'])
        t2_rest = calculate_rest_features(row['date'], t2_hist, row['team_2'])

        # Prefix rest features with team names
        for key, val in t1_rest.items():
            features[f'team_1_{key}'] = val
        for key, val in t2_rest.items():
            features[f'team_2_{key}'] = val

        # ============================================
        # CONTEXTUAL FEATURES - VENUE & HOME/AWAY
        # ============================================
        # Determine home/away from location field
        location = row.get('location')
        team_1 = row.get('team_1')
        team_2 = row.get('team_2')

        # Determine if team_1 is home
        team_1_is_home = None
        if location == 'N':
            team_1_is_home = None  # Neutral site
        elif location == team_1:
            team_1_is_home = True
        elif location == team_2:
            team_1_is_home = False
        else:
            team_1_is_home = None  # Unknown location

        # Calculate venue features for team_1
        venue_features_t1 = calculate_venue_features(team_1_is_home, None)
        for key, val in venue_features_t1.items():
            features[f'team_1_{key}'] = val

        # Calculate venue features for team_2 (opposite of team_1)
        team_2_is_home = None
        if team_1_is_home is not None:
            team_2_is_home = not team_1_is_home
        elif location == 'N':
            team_2_is_home = None

        venue_features_t2 = calculate_venue_features(team_2_is_home, None)
        for key, val in venue_features_t2.items():
            features[f'team_2_{key}'] = val

        # ============================================
        # CONFERENCE FEATURES
        # ============================================
        t1_conf = t1_lb.get('conference') if isinstance(t1_lb, dict) else None
        t2_conf = t2_lb.get('conference') if isinstance(t2_lb, dict) else None
        conf_features = calculate_conference_features(t1_conf, t2_conf, row['team_1'], row['team_2'], t1_hist)
        features.update(conf_features)

        # ============================================
        # GAME TIME FEATURES
        # ============================================
        start_time = row.get('start_time')
        game_time = None
        if start_time:
            # Extract hour from start_time (format: YYYY-MM-DD HH:MM:SS)
            try:
                if hasattr(start_time, 'hour'):
                    # If it's a datetime object
                    game_time = f"{start_time.hour:02d}:00"
                elif isinstance(start_time, str):
                    # If it's a string, extract the hour
                    parts = start_time.split(' ')
                    if len(parts) >= 2:
                        time_parts = parts[1].split(':')
                        if len(time_parts) >= 1:
                            game_time = f"{time_parts[0]}:00"
            except (AttributeError, IndexError, ValueError):
                pass

        time_features = calculate_game_time_features(game_time)
        features.update(time_features)

        # ============================================
        # ODDS AVAILABILITY TO GAME START TIME DIFFERENCE
        # ============================================
        # Odds available at 9AM EST on game date
        # Calculate hours from odds availability to game start
        features['hours_until_game_from_odds'] = None

        game_date = row.get('date')
        start_time_val = row.get('start_time')

        if game_date and start_time_val:
            try:
                from datetime import datetime, timezone
                import pytz

                # Parse game date (format: YYYY-MM-DD or date object)
                if isinstance(game_date, str):
                    date_obj = datetime.strptime(game_date, '%Y-%m-%d')
                elif hasattr(game_date, 'year'):
                    date_obj = datetime(game_date.year, game_date.month, game_date.day)
                else:
                    date_obj = None

                if date_obj:
                    # Create odds timestamp: 9AM EST on game date
                    est = pytz.timezone('US/Eastern')
                    odds_time_est = est.localize(date_obj.replace(hour=9, minute=0, second=0))
                    odds_time_utc = odds_time_est.astimezone(timezone.utc)

                    # Parse start_time (assumed to be UTC)
                    if isinstance(start_time_val, str):
                        # Try parsing as datetime string
                        try:
                            start_dt = datetime.strptime(start_time_val, '%Y-%m-%d %H:%M:%S')
                        except ValueError:
                            try:
                                start_dt = datetime.strptime(start_time_val, '%Y-%m-%d %H:%M')
                            except ValueError:
                                start_dt = None

                        if start_dt:
                            # Assume it's UTC
                            start_dt_utc = start_dt.replace(tzinfo=timezone.utc)
                    elif hasattr(start_time_val, 'hour'):
                        # It's already a datetime object
                        if start_time_val.tzinfo is None:
                            start_dt_utc = start_time_val.replace(tzinfo=timezone.utc)
                        else:
                            start_dt_utc = start_time_val.astimezone(timezone.utc)
                    else:
                        start_dt_utc = None

                    # Calculate difference in hours
                    if start_dt_utc:
                        time_diff = start_dt_utc - odds_time_utc
                        hours_diff = time_diff.total_seconds() / 3600
                        features['hours_until_game_from_odds'] = float(hours_diff)
            except Exception:
                # If any error occurs, leave as None
                features['hours_until_game_from_odds'] = None

        features_list.append(features)

    # Close database connection
    if db_connection:
        try:
            db_connection.close()
            print("Database connection closed")
        except Exception as e:
            print(f"Warning: Error closing database connection: {e}")

    # Convert to DataFrame with proper type inference
    # Use infer_schema_length to check more rows for consistent schema
    df = pl.DataFrame(features_list, infer_schema_length=5000)

    # Ensure key columns are string type and not null
    # Fill any nulls in identifiers with empty strings, then convert to string
    if 'game_id' in df.columns:
        df = df.with_columns(pl.col('game_id').fill_null('').cast(pl.Utf8))
    if 'team_1' in df.columns:
        df = df.with_columns(pl.col('team_1').fill_null('').cast(pl.Utf8))
    if 'team_2' in df.columns:
        df = df.with_columns(pl.col('team_2').fill_null('').cast(pl.Utf8))
    if 'date' in df.columns:
        df = df.with_columns(pl.col('date').cast(pl.Utf8))

    return df
