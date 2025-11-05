"""
Build ML features from raw flat DataFrame
Takes raw game data and creates engineered features for ML models
"""

import polars as pl
import numpy as np
from typing import List, Dict, Any


def extract_odds_features(game_odds: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Extract features from game odds data.

    Args:
        game_odds: List of odds records from bookmakers

    Returns:
        Dict with odds features (avg moneyline, spread, consensus, etc.)
    """
    if not game_odds:
        return {
            'avg_ml_home': None,
            'avg_ml_away': None,
            'avg_spread_home': None,
            'avg_spread_away': None,
            'consensus_ml_home': None,
            'num_books': 0,
            'spread_variance': None,
        }

    ml_homes = []
    ml_aways = []
    spreads_home = []
    spreads_away = []

    for odd in game_odds:
        if odd.get('ml_home'):
            ml_homes.append(float(odd['ml_home']))
        if odd.get('ml_away'):
            ml_aways.append(float(odd['ml_away']))
        if odd.get('spread_home'):
            spreads_home.append(float(odd['spread_home']))
        if odd.get('spread_away'):
            spreads_away.append(float(odd['spread_away']))

    # Calculate consensus (home team implied win probability from moneyline)
    # ML of -110 = 52.4% win prob, ML of +100 = 50% win prob
    def ml_to_prob(ml):
        if ml < 0:
            return abs(ml) / (abs(ml) + 100)
        else:
            return 100 / (ml + 100)

    home_probs = [ml_to_prob(ml) for ml in ml_homes] if ml_homes else []

    return {
        'avg_ml_home': float(np.mean(ml_homes)) if ml_homes else None,
        'avg_ml_away': float(np.mean(ml_aways)) if ml_aways else None,
        'avg_spread_home': float(np.mean(spreads_home)) if spreads_home else None,
        'avg_spread_away': float(np.mean(spreads_away)) if spreads_away else None,
        'consensus_ml_home': float(np.mean(home_probs)) if home_probs else None,
        'num_books': len(game_odds),
        'spread_variance': float(np.var(spreads_home)) if len(spreads_home) > 1 else None,
    }


def build_ml_features(raw_data: List[Dict[str, Any]]) -> pl.DataFrame:
    """
    Build ML features from raw flat game data.

    Args:
        raw_data: List of dicts from build_flat_df

    Returns:
        Polars DataFrame with engineered features
    """
    features = []

    for game in raw_data:
        # Extract odds features
        odds_features = extract_odds_features(game.get('game_odds', []))

        # Build feature row
        row = {
            'game_id': game.get('game_id'),
            'date': game.get('date'),
            'season': game.get('season'),
            'team_1': game.get('team_1'),
            'team_2': game.get('team_2'),

            # Team info
            'team_1_conference': game.get('team_1_conference'),
            'team_2_conference': game.get('team_2_conference'),

            # Scores and outcome
            'team_1_score': game.get('team_1_score'),
            'team_2_score': game.get('team_2_score'),
            'total_score': (game.get('team_1_score') or 0) + (game.get('team_2_score') or 0),
            'team_1_win': game.get('team_1_winloss'),

            # Leaderboard (pre-game strength)
            'team_1_rank': (game.get('team_1_leaderboard') or {}).get('rank'),
            'team_2_rank': (game.get('team_2_leaderboard') or {}).get('rank'),
            'team_1_wins': (game.get('team_1_leaderboard') or {}).get('wins'),
            'team_1_losses': (game.get('team_1_leaderboard') or {}).get('losses'),
            'team_2_wins': (game.get('team_2_leaderboard') or {}).get('wins'),
            'team_2_losses': (game.get('team_2_leaderboard') or {}).get('losses'),
            'team_1_barthag': (game.get('team_1_leaderboard') or {}).get('barthag'),
            'team_2_barthag': (game.get('team_2_leaderboard') or {}).get('barthag'),

            # Historical performance
            'team_1_hist_count': game.get('team_1_hist_count', 0),
            'team_2_hist_count': game.get('team_2_hist_count', 0),

            # Odds features
            'avg_ml_home': odds_features['avg_ml_home'],
            'avg_ml_away': odds_features['avg_ml_away'],
            'avg_spread_home': odds_features['avg_spread_home'],
            'avg_spread_away': odds_features['avg_spread_away'],
            'consensus_ml_home': odds_features['consensus_ml_home'],
            'num_books': odds_features['num_books'],
            'spread_variance': odds_features['spread_variance'],
        }

        features.append(row)

    # Convert to DataFrame
    df = pl.DataFrame(features)

    # Feature engineering
    df = df.with_columns([
        # Win-loss record as features
        (pl.col('team_1_wins') / (pl.col('team_1_wins') + pl.col('team_1_losses'))).alias('team_1_winpct'),
        (pl.col('team_2_wins') / (pl.col('team_2_wins') + pl.col('team_2_losses'))).alias('team_2_winpct'),

        # Rank difference (lower rank = better, so handle nulls carefully)
        (pl.col('team_2_rank') - pl.col('team_1_rank')).alias('rank_diff'),

        # Win-loss difference
        ((pl.col('team_2_wins') - pl.col('team_1_wins')) - (pl.col('team_2_losses') - pl.col('team_1_losses'))).alias('record_diff'),

        # Barthag difference (strength rating)
        (pl.col('team_2_barthag') - pl.col('team_1_barthag')).alias('barthag_diff'),

        # Score differential (team_1 - team_2)
        (pl.col('team_1_score') - pl.col('team_2_score')).alias('score_diff'),

        # Home team implied probability from consensus
        (1.0 - pl.col('consensus_ml_home')).alias('implied_away_prob'),

        # Spread impact (did home team cover the spread?)
        (pl.col('team_1_score') - pl.col('team_2_score') - pl.col('avg_spread_home')).alias('home_cover'),

        # Conference matchup (same or different)
        (pl.col('team_1_conference') == pl.col('team_2_conference')).alias('same_conference'),
    ])

    # Fill nulls with strategy
    df = df.with_columns([
        # Ranks: fill with 400 (worst rank)
        pl.col(['team_1_rank', 'team_2_rank']).fill_null(400),

        # Win pct: fill with 0.5 (neutral)
        pl.col(['team_1_winpct', 'team_2_winpct']).fill_null(0.5),

        # Barthag: fill with 0.5 (neutral)
        pl.col(['team_1_barthag', 'team_2_barthag']).fill_null(0.5),

        # Moneyline consensus: fill with 0.5 (pick 'em)
        pl.col('consensus_ml_home').fill_null(0.5),

        # Spread variance: fill with 0 (no variance)
        pl.col('spread_variance').fill_null(0),

        # Spread: fill with 0 (pick 'em)
        pl.col(['avg_spread_home', 'avg_spread_away']).fill_null(0),

        # Moneylines: fill with -110 (neutral)
        pl.col(['avg_ml_home', 'avg_ml_away']).fill_null(-110),
    ])

    # Select and order columns for ML
    ml_columns = [
        'game_id', 'date', 'season', 'team_1', 'team_2',
        'team_1_rank', 'team_2_rank', 'rank_diff',
        'team_1_winpct', 'team_2_winpct', 'record_diff',
        'team_1_barthag', 'team_2_barthag', 'barthag_diff',
        'team_1_hist_count', 'team_2_hist_count',
        'consensus_ml_home', 'implied_away_prob',
        'avg_spread_home', 'spread_variance',
        'num_books',
        'same_conference',
        'team_1_score', 'team_2_score', 'score_diff', 'total_score',
        'home_cover',
        'team_1_win',  # Target variable
    ]

    return df.select(ml_columns)
