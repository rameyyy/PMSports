"""
Helper functions for feature engineering - rolling windows, rank matching, weighting
"""
import math
from typing import List, Dict, Optional, Tuple


def get_rolling_windows():
    """Return standard rolling window sizes"""
    # Start from 2: can't compute variance/trend from just 1 game
    # Skip 7 (too many nulls in early season), keep other windows for trend analysis
    return [2, 3, 4, 5, 9, 11, 13, 15]


def exponential_decay_weight(games_ago: int, decay_rate: float = 0.90) -> float:
    """
    Calculate exponential decay weight for a game based on how long ago it was

    Args:
        games_ago: How many games in the past (0 = most recent)
        decay_rate: Decay rate per game (0.90 = 10% decay per game)

    Returns:
        Weight between 0 and 1
    """
    return decay_rate ** games_ago


def calculate_weighted_stat(games_list: List[dict], stat_key: str, decay_rate: float = 0.90) -> Optional[float]:
    """
    Calculate weighted average of a stat across games with exponential decay

    Args:
        games_list: List of game dicts in reverse chronological order (most recent first)
        stat_key: Key to extract from each game dict
        decay_rate: Exponential decay rate

    Returns:
        Weighted average or None if no data
    """
    values = []
    weights = []

    for games_ago, game in enumerate(games_list):
        if not isinstance(game, dict):
            continue

        val = game.get(stat_key)
        if val is not None:
            try:
                val = float(val)
                weight = exponential_decay_weight(games_ago, decay_rate)
                values.append(val)
                weights.append(weight)
            except (ValueError, TypeError):
                continue

    if not values:
        return None

    return sum(v * w for v, w in zip(values, weights)) / sum(weights)


def calculate_simple_average(games_list: List[dict], stat_key: str, limit: Optional[int] = None) -> Optional[float]:
    """
    Calculate simple average of a stat across games

    Args:
        games_list: List of game dicts
        stat_key: Key to extract from each game dict
        limit: Optional limit on number of games to use

    Returns:
        Average or None if no data
    """
    values = []

    for i, game in enumerate(games_list):
        if limit and i >= limit:
            break

        if not isinstance(game, dict):
            continue

        val = game.get(stat_key)
        if val is not None:
            try:
                values.append(float(val))
            except (ValueError, TypeError):
                continue

    if not values:
        return None

    return sum(values) / len(values)


def calculate_variance(games_list: List[dict], stat_key: str, limit: Optional[int] = None) -> Optional[float]:
    """
    Calculate standard deviation of a stat across games
    """
    values = []

    for i, game in enumerate(games_list):
        if limit and i >= limit:
            break

        if not isinstance(game, dict):
            continue

        val = game.get(stat_key)
        if val is not None:
            try:
                values.append(float(val))
            except (ValueError, TypeError):
                continue

    if len(values) < 2:
        return None

    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return variance ** 0.5


def calculate_trend(games_list: List[dict], stat_key: str, limit: Optional[int] = None) -> Optional[float]:
    """
    Calculate linear regression slope (trend) for a stat across games
    Positive slope = improving, negative = declining
    """
    values = []

    for i, game in enumerate(games_list):
        if limit and i >= limit:
            break

        if not isinstance(game, dict):
            continue

        val = game.get(stat_key)
        if val is not None:
            try:
                values.append(float(val))
            except (ValueError, TypeError):
                continue

    if len(values) < 2:
        return None

    # Simple linear regression: x = game number (0, 1, 2...), y = stat value
    n = len(values)
    x_vals = list(range(n))

    x_mean = sum(x_vals) / n
    y_mean = sum(values) / n

    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_vals, values))
    denominator = sum((x - x_mean) ** 2 for x in x_vals)

    if denominator == 0:
        return None

    return numerator / denominator


def find_closest_rank_games(team_hist: List[dict], opponent_rank: Optional[int],
                           top_n: int = 3, rank_window: int = 50, team_name: Optional[str] = None) -> Tuple[List[dict], List[float]]:
    """
    Find games in team history with closest opponent rank to current game
    Uses opponent barthag (power rating) to find similar matchups

    Args:
        team_hist: List of historical games for a team (team can be team_1 OR team_2 in these games)
        opponent_rank: Opponent's rank in current game (None if unranked)
        top_n: Number of closest games to return
        rank_window: Initial rank window (±rank_window)
        team_name: Name of the team (to identify which side they're on in historical games)

    Returns:
        Tuple of (closest games list, similarity weights list)
    """
    if not team_hist or opponent_rank is None:
        return [], []

    # Extract opponent ranks from historical games
    games_with_ranks = []
    for game in team_hist:
        if not isinstance(game, dict):
            continue

        # Determine which team in this historical game is OUR team and which is the opponent
        # The team could be team_1 or team_2 in their match history
        our_team_is_team1 = game.get('team_1') == team_name if team_name else True

        # Get OPPONENT's leaderboard (not our team's leaderboard!)
        if our_team_is_team1:
            # Our team is team_1, so opponent is team_2
            opp_lb = game.get('team_2_leaderboard')
        else:
            # Our team is team_2, so opponent is team_1
            opp_lb = game.get('team_1_leaderboard')

        if not opp_lb or not isinstance(opp_lb, dict):
            continue

        # Try to extract opponent rank
        try:
            opp_rank = int(opp_lb.get('rank', 0))
            if opp_rank is None or opp_rank == 0:
                continue

            rank_diff = abs(opp_rank - opponent_rank)
            games_with_ranks.append({
                'game': game,
                'rank_diff': rank_diff,
                'opp_rank': opp_rank
            })
        except (ValueError, TypeError):
            continue

    if not games_with_ranks:
        return [], []

    # Sort by rank difference
    games_with_ranks.sort(key=lambda x: x['rank_diff'])

    # Take top N games
    closest = games_with_ranks[:top_n]

    # Calculate similarity weights (closer rank = higher weight)
    max_diff = max(g['rank_diff'] for g in closest) if closest else 1

    result_games = []
    weights = []

    for item in closest:
        similarity = 1.0 - (item['rank_diff'] / max(max_diff, 1))
        result_games.append(item['game'])
        weights.append(similarity)

    return result_games, weights


def calculate_weighted_average(values: List[float], weights: List[float]) -> Optional[float]:
    """
    Calculate weighted average given values and weights
    """
    if not values or not weights or len(values) != len(weights):
        return None

    total_weight = sum(weights)
    if total_weight == 0:
        return None

    return sum(v * w for v, w in zip(values, weights)) / total_weight


def get_top_n_players_by_minutes(game_player_stats: List[dict], top_n: int = 5) -> List[dict]:
    """
    Get top N players by minutes played from game player stats

    Args:
        game_player_stats: List of player stat dicts from a single game
        top_n: Number of top players to return

    Returns:
        List of top N player stats
    """
    if not game_player_stats:
        return []

    # Extract players with minutes
    players_with_minutes = []
    for player in game_player_stats:
        if not isinstance(player, dict):
            continue

        min_played = player.get('min_per')
        if min_played is not None:
            try:
                min_played = float(min_played)
                players_with_minutes.append((player, min_played))
            except (ValueError, TypeError):
                continue

    # Sort by minutes descending
    players_with_minutes.sort(key=lambda x: x[1], reverse=True)

    # Return top N
    return [p[0] for p in players_with_minutes[:top_n]]


def aggregate_top_players_stat(match_hist: List[dict], player_stat_key: str,
                               team_name: str, top_n: int = 5, limit: Optional[int] = None) -> Optional[float]:
    """
    Aggregate a stat across top N players in recent games

    Args:
        match_hist: Match history list with team_1_player_stats/team_2_player_stats
        player_stat_key: Key to extract from player stat dicts (e.g., 'pts', 'bpm')
        team_name: Name of the team (to match players by their 'team' field)
        top_n: Number of top players by minutes to aggregate
        limit: Limit on number of games to use

    Returns:
        Average stat across top players and games
    """
    all_values = []

    for i, game in enumerate(match_hist):
        if limit and i >= limit:
            break

        if not isinstance(game, dict):
            continue

        # Get ALL player stats from the game (both teams)
        all_players = []
        team_1_players = game.get('team_1_player_stats', [])
        team_2_players = game.get('team_2_player_stats', [])

        if isinstance(team_1_players, list):
            all_players.extend(team_1_players)
        if isinstance(team_2_players, list):
            all_players.extend(team_2_players)

        # Filter to only OUR team's players by matching the 'team' field
        our_team_players = []
        for player in all_players:
            if isinstance(player, dict) and player.get('team') == team_name:
                our_team_players.append(player)

        # Get top N players by minutes
        top_players = get_top_n_players_by_minutes(our_team_players, top_n)

        for player in top_players:
            stat_val = player.get(player_stat_key)
            if stat_val is not None:
                try:
                    all_values.append(float(stat_val))
                except (ValueError, TypeError):
                    continue

    if not all_values:
        return None

    return sum(all_values) / len(all_values)


def assess_data_quality(team_hist_count: int, games_with_leaderboard: int,
                       games_with_similar_rank: int) -> Dict[str, float]:
    """
    Assess quality of available data for feature calculation

    Returns:
        Dict with quality scores (0-1)
    """
    return {
        'hist_games_quality': min(1.0, team_hist_count / 5.0),  # 5+ games = perfect
        'leaderboard_quality': min(1.0, games_with_leaderboard / team_hist_count) if team_hist_count > 0 else 0.0,
        'rank_match_quality': min(1.0, games_with_similar_rank / 3.0) if team_hist_count > 0 else 0.0,
        'overall_data_quality': (min(1.0, team_hist_count / 5.0) +
                                 (min(1.0, games_with_leaderboard / team_hist_count) if team_hist_count > 0 else 0.0)) / 2.0
    }


def build_player_aggregated_features(match_hist: List[dict], team_prefix: str, team_name: str,
                                     windows: List[int], top_n_players: int = 5) -> Dict[str, Optional[float]]:
    """
    Build aggregated player-level features from match history

    Extracts top N players by minutes played in recent games and aggregates their stats
    across multiple rolling windows.

    Args:
        match_hist: List of historical games with player_stats embedded
        team_prefix: 'team_1' or 'team_2'
        team_name: Name of the team (to match players by their 'team' field)
        windows: List of window sizes [1,2,3,5,7...]
        top_n_players: Number of top players to aggregate

    Returns:
        Dictionary with aggregated player features for each window
    """
    features = {}

    if not match_hist:
        # Return empty feature set if no history
        for window in windows:
            window_label = f'last{window}' if window <= len(match_hist) else 'alltime'
            for stat in ['ppg', 'bpm', 'usage', 'efg_pct', 'ast_rate', 'to_rate', 'min_per']:
                features[f'{team_prefix}_top{top_n_players}_{stat}_{window_label}'] = None
        return features

    # Build features for each window
    for window in windows:
        games_to_use = match_hist[:window] if window > 0 else match_hist
        window_label = f'last{window}' if window <= len(match_hist) else 'alltime'

        # Aggregate stats across top players in these games
        ppg_vals = []
        bpm_vals = []
        usage_vals = []
        efg_vals = []
        ast_rate_vals = []
        to_rate_vals = []
        min_per_vals = []

        for game in games_to_use:
            if not isinstance(game, dict):
                continue

            # Get ALL player stats from the game (both teams)
            all_players = []
            team_1_players = game.get('team_1_player_stats', [])
            team_2_players = game.get('team_2_player_stats', [])

            if isinstance(team_1_players, list):
                all_players.extend(team_1_players)
            if isinstance(team_2_players, list):
                all_players.extend(team_2_players)

            # Filter to only OUR team's players by matching the 'team' field
            our_team_players = []
            for player in all_players:
                if isinstance(player, dict) and player.get('team') == team_name:
                    our_team_players.append(player)

            if not our_team_players:
                continue

            # Get top players by minutes in this game
            top_players = get_top_n_players_by_minutes(our_team_players, top_n_players)

            for player in top_players:
                if not isinstance(player, dict):
                    continue

                # Extract player stats safely
                ppg = player.get('pts')
                if ppg is not None:
                    ppg_vals.append(float(ppg))

                bpm = player.get('bpm')
                if bpm is not None:
                    bpm_vals.append(float(bpm))

                usage = player.get('usage')
                if usage is not None:
                    usage_vals.append(float(usage))

                efg = player.get('efg_pct')
                if efg is not None:
                    efg_vals.append(float(efg))

                ast_rate = player.get('ast_rate')
                if ast_rate is not None:
                    ast_rate_vals.append(float(ast_rate))

                to_rate = player.get('to_rate')
                if to_rate is not None:
                    to_rate_vals.append(float(to_rate))

                min_per = player.get('min_per')
                if min_per is not None:
                    min_per_vals.append(float(min_per))

        # Calculate averages for each stat
        features[f'{team_prefix}_top{top_n_players}_ppg_{window_label}'] = sum(ppg_vals) / len(ppg_vals) if ppg_vals else None
        features[f'{team_prefix}_top{top_n_players}_bpm_{window_label}'] = sum(bpm_vals) / len(bpm_vals) if bpm_vals else None
        features[f'{team_prefix}_top{top_n_players}_usage_{window_label}'] = sum(usage_vals) / len(usage_vals) if usage_vals else None
        features[f'{team_prefix}_top{top_n_players}_efg_pct_{window_label}'] = sum(efg_vals) / len(efg_vals) if efg_vals else None
        features[f'{team_prefix}_top{top_n_players}_ast_rate_{window_label}'] = sum(ast_rate_vals) / len(ast_rate_vals) if ast_rate_vals else None
        features[f'{team_prefix}_top{top_n_players}_to_rate_{window_label}'] = sum(to_rate_vals) / len(to_rate_vals) if to_rate_vals else None
        features[f'{team_prefix}_top{top_n_players}_min_per_{window_label}'] = sum(min_per_vals) / len(min_per_vals) if min_per_vals else None

    return features


def american_to_decimal(american_odds: float) -> Optional[float]:
    """
    Convert American odds to decimal odds format

    American odds format:
    - Positive (e.g., +100): underdog odds
    - Negative (e.g., -110): favorite odds

    Decimal conversion:
    - Positive: Decimal = 1 + (American / 100)
    - Negative: Decimal = (100 / |American|) + 1

    Examples:
    - +100 → 2.0 (bet $100, win $100, total return $200)
    - -110 → 1.909 (bet $110, win $100, total return $210)
    - +105 → 2.05
    - -120 → 1.833

    Args:
        american_odds: American odds value (e.g., 100, -110)

    Returns:
        Decimal odds or None if invalid
    """
    if american_odds is None:
        return None

    try:
        odds = float(american_odds)

        if odds > 0:
            # Positive odds: +100 → 2.0
            return 1.0 + (odds / 100.0)
        else:
            # Negative odds: -110 → 1.909...
            return 1.0 + (100.0 / abs(odds))
    except (ValueError, TypeError, ZeroDivisionError):
        return None


def decimal_to_american(decimal_odds: float) -> Optional[float]:
    """
    Convert decimal odds back to American odds format

    Args:
        decimal_odds: Decimal odds (e.g., 2.0, 1.909)

    Returns:
        American odds or None if invalid
    """
    if decimal_odds is None:
        return None

    try:
        decimal = float(decimal_odds)

        if decimal >= 2.0:
            # Decimal >= 2.0: 2.0 → +100 (positive odds)
            return (decimal - 1.0) * 100.0
        elif decimal > 1.0:
            # 1.0 < Decimal < 2.0: 1.909 → -110 (negative odds)
            # decimal = 1 + (100 / |american|)
            # decimal - 1 = 100 / |american|
            # |american| = 100 / (decimal - 1)
            # american = -(100 / (decimal - 1))
            return -(100.0 / (decimal - 1.0))
        else:
            return None
    except (ValueError, TypeError, ZeroDivisionError):
        return None


def average_american_odds(odds_list: List[float]) -> Optional[float]:
    """
    Calculate average of American odds by converting to decimal, averaging, then converting back

    This handles the problem where simple arithmetic average of mixed +/- odds doesn't work.
    For example: [+100, -110, +105, -120] cannot be simply averaged.

    Args:
        odds_list: List of American odds values

    Returns:
        Average odds in American format, or None if no valid odds
    """
    if not odds_list:
        return None

    # Convert all to decimal
    decimal_odds = []
    for odds in odds_list:
        dec = american_to_decimal(odds)
        if dec is not None:
            decimal_odds.append(dec)

    if not decimal_odds:
        return None

    # Average in decimal format
    avg_decimal = sum(decimal_odds) / len(decimal_odds)

    # Convert back to American
    return decimal_to_american(avg_decimal)
