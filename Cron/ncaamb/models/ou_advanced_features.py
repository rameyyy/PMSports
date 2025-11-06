"""
Advanced Over/Under features: contextual, situational, and market dynamics
Includes: rest/fatigue, home/away, time of game, market movement, momentum
"""
from typing import Optional, Dict, List
from datetime import datetime, timedelta
import statistics


def calculate_rest_features(game_date, team_hist: List[dict], team_name: str) -> Dict[str, Optional[float]]:
    """
    Calculate rest/fatigue features

    Args:
        game_date: Date of current game (can be str or date object)
        team_hist: Historical games for team (sorted most recent first)
        team_name: Name of team (for logging)

    Returns:
        Dictionary with rest features
    """
    features = {
        'days_rest': None,
        'back_to_back_flag': 0,
        'games_in_last_5_days': 0,
    }

    if not game_date or not team_hist or len(team_hist) == 0:
        return features

    # Convert game_date to datetime
    try:
        if isinstance(game_date, str):
            current_date = datetime.strptime(game_date, '%Y-%m-%d')
        else:
            # Assume it's already a date object
            current_date = datetime.combine(game_date, datetime.min.time())
    except (ValueError, TypeError, AttributeError):
        return features

    # Get most recent game date
    most_recent_game = None
    for game in team_hist:
        if isinstance(game, dict) and game.get('date'):
            try:
                hist_date = game['date']
                if isinstance(hist_date, str):
                    game_dt = datetime.strptime(hist_date, '%Y-%m-%d')
                else:
                    game_dt = datetime.combine(hist_date, datetime.min.time())

                if most_recent_game is None or game_dt > most_recent_game:
                    most_recent_game = game_dt
            except (ValueError, TypeError, AttributeError):
                continue

    if most_recent_game:
        days_rest = (current_date - most_recent_game).days
        features['days_rest'] = float(days_rest)

        # Back-to-back flag (0 days rest)
        if days_rest == 0:
            features['back_to_back_flag'] = 1

    # Games in last 5 days
    five_days_ago = current_date - timedelta(days=5)
    games_in_window = 0
    for game in team_hist:
        if isinstance(game, dict) and game.get('date'):
            try:
                hist_date = game['date']
                if isinstance(hist_date, str):
                    game_dt = datetime.strptime(hist_date, '%Y-%m-%d')
                else:
                    game_dt = datetime.combine(hist_date, datetime.min.time())

                if five_days_ago <= game_dt < current_date:
                    games_in_window += 1
            except (ValueError, TypeError, AttributeError):
                continue

    features['games_in_last_5_days'] = float(games_in_window)

    return features


def calculate_venue_features(is_home_game: Optional[bool], venue_info: Optional[dict]) -> Dict[str, Optional[float]]:
    """
    Calculate home/away and venue-related features

    Args:
        is_home_game: Whether this is home team (True/False/None)
        venue_info: Dictionary with venue data (optional)

    Returns:
        Dictionary with venue features
    """
    features = {
        'is_home_game': float(is_home_game) if is_home_game is not None else 0.5,
        'home_court_advantage': 0.0,  # Will be populated if venue_info available
        'is_neutral_site': 0.0,
    }

    if venue_info and isinstance(venue_info, dict):
        # Neutral site indicator
        if venue_info.get('is_neutral'):
            features['is_neutral_site'] = 1.0

    # Home court advantage adjustment (typical: +1.5 pts, +2% eFG)
    if features['is_home_game'] > 0.5 and not features['is_neutral_site']:
        features['home_court_advantage'] = 1.5

    return features


def calculate_game_time_features(game_time: Optional[str]) -> Dict[str, Optional[float]]:
    """
    Calculate game time bucket features
    Early games (<11am) tend under, late games (>9pm) tend over

    Args:
        game_time: Game start time (HH:MM format, 24-hour)

    Returns:
        Dictionary with game time features
    """
    features = {
        'is_early_game': 0.0,      # Morning games (before 11am)
        'is_afternoon_game': 0.0,  # 11am-5pm
        'is_evening_game': 0.0,    # 5pm-9pm
        'is_late_game': 0.0,       # After 9pm
        'hour_of_day': None,
    }

    if not game_time:
        features['is_afternoon_game'] = 1.0  # Default assumption
        return features

    try:
        hour = int(game_time.split(':')[0])
        features['hour_of_day'] = float(hour)

        if hour < 11:
            features['is_early_game'] = 1.0
        elif hour < 17:
            features['is_afternoon_game'] = 1.0
        elif hour < 21:
            features['is_evening_game'] = 1.0
        else:
            features['is_late_game'] = 1.0
    except (ValueError, IndexError, TypeError):
        features['is_afternoon_game'] = 1.0

    return features


def calculate_conference_features(team1_conf: Optional[str], team2_conf: Optional[str],
                                 team1_name: Optional[str], team2_name: Optional[str],
                                 team_hist: Optional[List[dict]]) -> Dict[str, Optional[float]]:
    """
    Calculate conference-related features

    Args:
        team1_conf: Conference of team 1
        team2_conf: Conference of team 2
        team1_name: Name of team 1
        team2_name: Name of team 2
        team_hist: Historical games (to check for rematch)

    Returns:
        Dictionary with conference features
    """
    features = {
        'is_conference_game': 0.0,
        'is_conference_rematch': 0.0,
    }

    # Conference game flag
    if team1_conf and team2_conf and team1_conf == team2_conf:
        features['is_conference_game'] = 1.0

    # Conference rematch (in-conference same opponent within last 30 days)
    if features['is_conference_game'] and team_hist and team2_name:
        for game in team_hist[:10]:  # Check last 10 games
            if isinstance(game, dict):
                opponent = game.get('opponent_name') or game.get('team_2')
                if opponent and opponent.lower() == team2_name.lower():
                    features['is_conference_rematch'] = 1.0
                    break

    return features


def calculate_momentum_features(team_hist: List[dict], stat_key: str = 'score',
                               window: int = 3) -> Dict[str, Optional[float]]:
    """
    Calculate momentum features (recent form, consistency, over/under trends)

    Args:
        team_hist: Historical games (sorted most recent first)
        stat_key: Statistic to analyze ('score', 'total', etc)
        window: Number of recent games to analyze

    Returns:
        Dictionary with momentum features
    """
    features = {
        'recent_volatility': None,
        'momentum_score': None,
        'consecutive_overs_last3': None,
        'recent_avg_diff_from_trend': None,
    }

    if not team_hist or len(team_hist) < window:
        return features

    # Get recent scores/totals
    recent_values = []
    for game in team_hist[:window]:
        if isinstance(game, dict):
            val = game.get(stat_key)
            if val is not None:
                try:
                    recent_values.append(float(val))
                except (ValueError, TypeError):
                    pass

    if not recent_values or len(recent_values) < 2:
        return features

    # Volatility (std dev of recent games)
    try:
        features['recent_volatility'] = float(statistics.stdev(recent_values))
    except (statistics.StatisticsError, ValueError):
        pass

    # Momentum (slope trend)
    if len(recent_values) >= 2:
        trend = recent_values[0] - recent_values[-1]
        features['momentum_score'] = float(trend)

    # For games with actual_total, check over/under trend
    if stat_key == 'score' and len(recent_values) >= 3:
        try:
            avg_recent = statistics.mean(recent_values[:3])
            earlier_avg = statistics.mean(recent_values) if len(recent_values) > 3 else avg_recent
            features['recent_avg_diff_from_trend'] = float(avg_recent - earlier_avg)
        except (statistics.StatisticsError, ValueError):
            pass

    return features


def calculate_market_dynamics_features(current_ou_line: Optional[float],
                                      opening_ou_line: Optional[float],
                                      current_spread: Optional[float],
                                      opening_spread: Optional[float]) -> Dict[str, Optional[float]]:
    """
    Calculate market movement features (sharp action indicators)

    Args:
        current_ou_line: Current O/U line
        opening_ou_line: Opening O/U line
        current_spread: Current spread
        opening_spread: Opening spread

    Returns:
        Dictionary with market movement features
    """
    features = {
        'ou_line_movement': None,
        'ou_move_direction': None,  # 1 for up (over favored), -1 for down
        'ou_move_magnitude': None,
        'spread_movement': None,
        'spread_move_direction': None,
        'line_volatility_signal': 0.0,  # Flag for high movement
    }

    # O/U line movement
    if current_ou_line and opening_ou_line:
        try:
            movement = float(current_ou_line) - float(opening_ou_line)
            features['ou_line_movement'] = movement
            features['ou_move_magnitude'] = abs(movement)

            if movement > 0:
                features['ou_move_direction'] = 1.0  # Over got worse (line up)
            elif movement < 0:
                features['ou_move_direction'] = -1.0  # Under got better (line down)

            # Flag high volatility (>1.5 points movement = sharp action)
            if abs(movement) > 1.5:
                features['line_volatility_signal'] = 1.0
        except (ValueError, TypeError):
            pass

    # Spread movement
    if current_spread and opening_spread:
        try:
            movement = abs(float(current_spread)) - abs(float(opening_spread))
            features['spread_movement'] = movement

            if movement > 0:
                features['spread_move_direction'] = 1.0  # Spread widening
            elif movement < 0:
                features['spread_move_direction'] = -1.0  # Spread tightening
        except (ValueError, TypeError):
            pass

    return features


def calculate_interaction_features(pace_diff: Optional[float],
                                  efg_def_diff: Optional[float],
                                  rank_diff: Optional[float],
                                  avg_tempo: Optional[float],
                                  avg_ou_line: Optional[float]) -> Dict[str, Optional[float]]:
    """
    Calculate interaction terms for non-linear effects

    Args:
        pace_diff: Pace differential between teams
        efg_def_diff: eFG% defensive differential
        rank_diff: Rank differential
        avg_tempo: Average tempo
        avg_ou_line: Vegas O/U line

    Returns:
        Dictionary with interaction features
    """
    features = {
        'pace_x_def_efficiency': None,        # High pace + bad defense = high totals
        'rank_x_tempo': None,                 # Rank gap × tempo interaction
        'line_x_momentum': None,              # Will be filled by caller with momentum
        'implied_total_ppp': None,            # Implied points per possession
    }

    # Interaction 1: Pace × Defensive Efficiency
    if pace_diff and efg_def_diff:
        try:
            features['pace_x_def_efficiency'] = float(pace_diff) * float(efg_def_diff)
        except (ValueError, TypeError):
            pass

    # Interaction 2: Rank Differential × Tempo
    if rank_diff and avg_tempo:
        try:
            features['rank_x_tempo'] = float(rank_diff) * float(avg_tempo) / 100.0
        except (ValueError, TypeError):
            pass

    # Interaction 3: Implied PPP (points per possession)
    if avg_ou_line and avg_tempo:
        try:
            # Rough estimate: total points / (expected possessions)
            # Avg game ~70 possessions per team
            features['implied_total_ppp'] = float(avg_ou_line) / (float(avg_tempo) * 0.7)
        except (ValueError, TypeError, ZeroDivisionError):
            pass

    return features
