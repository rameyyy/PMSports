"""
Build comprehensive Over/Under features from flat dataset using Polars operations
Includes: rolling windows, rank-based features, leaderboard differentials, player stats, odds data,
          contextual features (rest/home/away), and interaction terms
"""
import polars as pl
from typing import Optional, Dict, List
from .ou_feature_build_utils import (
    get_rolling_windows,
    calculate_simple_average,
    calculate_variance,
    calculate_trend,
    find_closest_rank_games,
    calculate_weighted_average,
    assess_data_quality
)
from .ou_advanced_features import (
    calculate_rest_features,
    calculate_venue_features,
    calculate_conference_features,
    calculate_game_time_features
)


def process_odds_data(odds_list: list) -> dict:
    """
    Process odds data from multiple sportsbooks for a game
    Creates individual columns for each sportsbook's O/U line + over/under odds
    Fills nulls with average for that metric
    Also includes: spread points/odds, moneyline odds
    """
    # Standard sportsbooks
    sportsbooks = ['BetMGM', 'BetOnline.ag', 'Bovada', 'DraftKings', 'FanDuel', 'LowVig.ag', 'MyBookie.ag']

    odds_features = {}

    # Create columns for each sportsbook's O/U line and odds
    ou_lines_by_book = {}
    over_odds_by_book = {}
    under_odds_by_book = {}

    for book in sportsbooks:
        odds_features[f'{book}_ou_line'] = None
        odds_features[f'{book}_over_odds'] = None
        odds_features[f'{book}_under_odds'] = None

    # Add average columns
    odds_features['avg_ou_line'] = None
    odds_features['ou_line_variance'] = None
    odds_features['avg_over_odds'] = None
    odds_features['avg_under_odds'] = None
    odds_features['num_books_with_ou'] = 0

    # Spread and ML features (averages only for now)
    odds_features['avg_spread'] = None
    odds_features['spread_variance'] = None
    odds_features['avg_spread_home_odds'] = None
    odds_features['avg_spread_away_odds'] = None
    odds_features['avg_ml_home'] = None
    odds_features['avg_ml_away'] = None
    odds_features['num_books_with_spread'] = 0
    odds_features['num_books_with_ml'] = 0

    if not odds_list or len(odds_list) == 0:
        return odds_features

    # Extract data by sportsbook
    ou_lines_all = []
    over_odds_all = []
    under_odds_all = []

    spreads = []
    spread_home_odds = []
    spread_away_odds = []
    ml_home = []
    ml_away = []

    for odds in odds_list:
        if not isinstance(odds, dict):
            continue

        bookmaker = odds.get('bookmaker')

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

        # Spread data
        spread_pts = odds.get('spread_pts_home') or odds.get('spread_pts_away')
        if spread_pts is not None:
            spreads.append(abs(float(spread_pts)))

        spread_home = odds.get('spread_home')
        if spread_home is not None:
            spread_home_odds.append(float(spread_home))

        spread_away = odds.get('spread_away')
        if spread_away is not None:
            spread_away_odds.append(float(spread_away))

        # Moneyline data
        ml_h = odds.get('ml_home')
        if ml_h is not None:
            ml_home.append(float(ml_h))

        ml_a = odds.get('ml_away')
        if ml_a is not None:
            ml_away.append(float(ml_a))

    # Calculate averages for O/U
    if ou_lines_all:
        avg_ou = float(sum(ou_lines_all) / len(ou_lines_all))
        odds_features['avg_ou_line'] = avg_ou
        odds_features['num_books_with_ou'] = len(ou_lines_all)

        if len(ou_lines_all) > 1:
            variance = sum((x - avg_ou) ** 2 for x in ou_lines_all) / len(ou_lines_all)
            odds_features['ou_line_variance'] = float(variance ** 0.5)

    if over_odds_all:
        odds_features['avg_over_odds'] = float(sum(over_odds_all) / len(over_odds_all))

    if under_odds_all:
        odds_features['avg_under_odds'] = float(sum(under_odds_all) / len(under_odds_all))

    # Fill individual sportsbook columns, use average for nulls
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

    # Calculate spread averages
    if spreads:
        odds_features['num_books_with_spread'] = len(spreads)
        avg_spread = float(sum(spreads) / len(spreads))
        odds_features['avg_spread'] = avg_spread

        if len(spreads) > 1:
            variance = sum((x - avg_spread) ** 2 for x in spreads) / len(spreads)
            odds_features['spread_variance'] = float(variance ** 0.5)

    if spread_home_odds:
        odds_features['avg_spread_home_odds'] = float(sum(spread_home_odds) / len(spread_home_odds))

    if spread_away_odds:
        odds_features['avg_spread_away_odds'] = float(sum(spread_away_odds) / len(spread_away_odds))

    # Moneyline averages
    if ml_home:
        odds_features['num_books_with_ml'] = len(ml_home)
        odds_features['avg_ml_home'] = float(sum(ml_home) / len(ml_home))

    if ml_away:
        odds_features['avg_ml_away'] = float(sum(ml_away) / len(ml_away))

    return odds_features


def build_rolling_window_features(team_hist: List[dict], team_prefix: str, stat_key: str,
                                  windows: List[int]) -> Dict[str, Optional[float]]:
    """
    Build rolling window features for a given stat across multiple window sizes
    For larger windows (>= 5), if not enough history, falls back to using all available games
    For smaller windows (2-4), always uses the requested window or available games

    Args:
        team_hist: List of historical games
        team_prefix: 'team_1' or 'team_2'
        stat_key: Base stat key (e.g., 'score')
        windows: List of window sizes to calculate

    Returns:
        Dictionary with keys like 'stat_last3', 'stat_last5', etc.
        For large windows without enough data, returns None
    """
    features = {}
    full_key = f'{team_prefix}_{stat_key}'

    for window in windows:
        if window == 0:
            # 0 means use all available
            actual_window = len(team_hist)
        elif window <= 4:
            # Small windows: use requested window or available, whichever is smaller
            actual_window = min(window, len(team_hist))
        else:
            # Large windows (5+): need enough history to create the window
            # If not enough games, use the largest available <= 4, otherwise None
            if len(team_hist) >= window:
                actual_window = window
            else:
                # Fall back to last3 if available
                actual_window = min(3, len(team_hist))
                if actual_window < 2:
                    # Not enough data even for fallback
                    features[f'{team_prefix}_{stat_key}_last{window}'] = None
                    if stat_key in ['score', 'total', 'efg_pct', 'pace']:
                        features[f'{team_prefix}_{stat_key}_variance_last{window}'] = None
                        features[f'{team_prefix}_{stat_key}_trend_last{window}'] = None
                    continue

        # Get games to use
        games_to_use = team_hist[:actual_window]

        # Calculate average
        avg_val = calculate_simple_average(games_to_use, full_key)

        # Store with requested window name (not actual window used)
        feature_key = f'{team_prefix}_{stat_key}_last{window}'
        features[feature_key] = avg_val

        # Also add variance and trend for key stats
        if stat_key in ['score', 'total', 'efg_pct', 'pace']:
            var_val = calculate_variance(games_to_use, full_key)
            features[f'{team_prefix}_{stat_key}_variance_last{window}'] = var_val

            trend_val = calculate_trend(games_to_use, full_key)
            features[f'{team_prefix}_{stat_key}_trend_last{window}'] = trend_val

    return features


def build_closest_rank_features(team_hist: List[dict], opp_rank: Optional[int],
                               team_prefix: str) -> Dict[str, Optional[float]]:
    """
    Build features based on closest rank historical games

    Args:
        team_hist: Match history for a team
        opp_rank: Opponent's rank in current game
        team_prefix: 'team_1' or 'team_2'

    Returns:
        Dictionary with closest rank features
    """
    features = {}

    if not opp_rank or not team_hist:
        # Return None values if we can't calculate
        base_features = ['score', 'score_allowed', 'total', 'margin']
        for feat in base_features:
            features[f'{team_prefix}_{feat}_closest3rank'] = None
        return features

    # Find closest rank games
    closest_games, weights = find_closest_rank_games(team_hist, opp_rank, top_n=3, rank_window=50)

    if not closest_games:
        # Return None values if no close rank games found
        base_features = ['score', 'score_allowed', 'total', 'margin']
        for feat in base_features:
            features[f'{team_prefix}_{feat}_closest3rank'] = None
        return features

    # Calculate weighted averages from closest games
    scores = []
    scores_allowed = []
    totals = []
    margins = []

    for game in closest_games:
        if not isinstance(game, dict):
            continue

        t_score = game.get(f'{team_prefix}_score')
        opp_score = game.get(f'{"team_2" if team_prefix == "team_1" else "team_1"}_score')

        if t_score is not None:
            scores.append(float(t_score))

        if opp_score is not None:
            scores_allowed.append(float(opp_score))

        if t_score is not None and opp_score is not None:
            totals.append(float(t_score) + float(opp_score))
            margins.append(float(t_score) - float(opp_score))

    # Calculate weighted averages
    features[f'{team_prefix}_score_closest3rank'] = calculate_weighted_average(scores, weights[:len(scores)])
    features[f'{team_prefix}_score_allowed_closest3rank'] = calculate_weighted_average(scores_allowed, weights[:len(scores_allowed)])
    features[f'{team_prefix}_total_closest3rank'] = calculate_weighted_average(totals, weights[:len(totals)])
    features[f'{team_prefix}_margin_closest3rank'] = calculate_weighted_average(margins, weights[:len(margins)])

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
            'adj_oe_differential': None,
            'adj_de_differential': None,
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

        # ============================================
        # TEAM-LEVEL ROLLING WINDOW FEATURES
        # ============================================
        # Note: pace only exists in leaderboard snapshot (adj_t), not in match_hist
        # So we cannot build rolling windows for pace
        team_stats_to_roll = ['score', 'efg_pct']

        for stat in team_stats_to_roll:
            t1_features = build_rolling_window_features(t1_hist, 'team_1', stat, rolling_windows)
            t2_features = build_rolling_window_features(t2_hist, 'team_2', stat, rolling_windows)
            features.update(t1_features)
            features.update(t2_features)

        # ============================================
        # RANK-BASED CLOSEST MATCH FEATURES
        # ============================================
        t1_closest = build_closest_rank_features(t1_hist, t2_rank, 'team_1')
        t2_closest = build_closest_rank_features(t2_hist, t1_rank, 'team_2')
        features.update(t1_closest)
        features.update(t2_closest)

        # Derived: Combined expected total from closest rank games
        t1_score_closest = t1_closest.get('team_1_score_closest3rank')
        t2_score_closest = t2_closest.get('team_2_score_closest3rank')
        if t1_score_closest and t2_score_closest:
            features['combined_expected_total_closest_rank'] = t1_score_closest + t2_score_closest
        else:
            features['combined_expected_total_closest_rank'] = None

        # ============================================
        # LEADERBOARD DIFFERENTIALS
        # ============================================
        lb_diffs = build_leaderboard_differentials(t1_lb, t2_lb)
        features.update(lb_diffs)

        # ============================================
        # ODDS DATA FEATURES
        # ============================================
        game_odds = row.get('game_odds', [])
        odds_features = process_odds_data(game_odds)
        features.update(odds_features)

        # Implied scores from spread and O/U line
        if odds_features['avg_ou_line'] and odds_features['avg_spread']:
            implied_fav_score = (odds_features['avg_ou_line'] + odds_features['avg_spread']) / 2
            implied_dog_score = (odds_features['avg_ou_line'] - odds_features['avg_spread']) / 2
            features['implied_fav_score'] = implied_fav_score
            features['implied_dog_score'] = implied_dog_score
            features['spread_ou_agreement'] = abs(implied_fav_score - implied_dog_score)
        else:
            features['implied_fav_score'] = None
            features['implied_dog_score'] = None
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

        features_list.append(features)

    # Convert to DataFrame
    df = pl.DataFrame(features_list)

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
