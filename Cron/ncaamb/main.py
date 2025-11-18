#!/usr/bin/env python3
"""
Futures Report: Combined ML and OU Betting Recommendations
Generates daily betting report with:
- Moneyline (ML) bets where EV > 9%
- Over/Under (OU) bets where ensemble - over_point > 2.3
Only considers odds from: BetOnline.ag, Bovada, MyBookie.ag
"""

import os
import sys
from pathlib import Path
import polars as pl
import numpy as np
from xgboost import XGBClassifier

# Add current directory to path
ncaamb_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ncaamb_dir)

from ou_main import main as ou_main
from scrapes import sqlconn


def load_model():
    """Load pre-trained moneyline model from saved folder"""
    model_path = Path(__file__).parent / "models" / "moneyline" / "saved" / "xgboost_model.pkl"

    if not model_path.exists():
        print(f"[-] Model not found at {model_path}")
        print(f"    Please run export_2025_predictions.py first to train and save the model\n")
        return None

    model = XGBClassifier()
    model.load_model(str(model_path))
    print(f"[+] Model loaded from {model_path}\n")
    return model


def identify_feature_columns(df: pl.DataFrame) -> list:
    """Identify numeric feature columns"""
    metadata_cols = {
        'game_id', 'date', 'season', 'team_1', 'team_2',
        'team_1_score', 'team_2_score', 'actual_total',
        'team_1_conference', 'team_2_conference',
        'team_1_is_home', 'team_2_is_home', 'location',
        'total_score_outcome', 'team_1_winloss',
        'team_1_leaderboard', 'team_2_leaderboard',
        'team_1_match_hist', 'team_2_match_hist',
        'team_1_hist_count', 'team_2_hist_count',
        'start_time', 'game_odds', 'ml_target',
        'avg_ml_home', 'avg_ml_away'
    }

    feature_cols = []
    for col in df.columns:
        if col not in metadata_cols:
            dtype = df[col].dtype
            if dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]:
                feature_cols.append(col)

    return feature_cols


def prepare_data(df: pl.DataFrame, feature_cols: list) -> np.ndarray:
    """Prepare X features"""
    X = df.select(feature_cols).fill_null(0).to_numpy()
    return X


def american_to_decimal(american_odds: float) -> float:
    """Convert American odds to decimal odds"""
    american_odds = float(american_odds)
    if american_odds >= 0:
        return float((american_odds / 100) + 1)
    else:
        return float((100 / abs(american_odds)) + 1)


def get_best_odds_from_bookmakers(all_odds: list, team_position: str, allowed_bookmakers: set) -> tuple:
    """
    Get best odds for a team from allowed bookmakers only
    Returns (best_odds, decimal, bookmaker) or (None, None, None)
    """
    if not all_odds:
        return None, None, None

    odds_key = 'ml_home' if team_position == 'home' else 'ml_away'

    # Filter for allowed bookmakers
    valid_odds = [
        (o[odds_key], o['bookmaker'])
        for o in all_odds
        if o[odds_key] is not None and o['bookmaker'] in allowed_bookmakers
    ]

    if not valid_odds:
        return None, None, None

    # Find best odds (highest positive or least negative)
    best_odds, best_bookmaker = max(valid_odds, key=lambda x: american_to_decimal(x[0]))
    best_decimal = american_to_decimal(best_odds)

    return best_odds, best_decimal, best_bookmaker


def load_odds_for_games(game_ids: list) -> dict:
    """Load odds from ncaamb.odds table for given game_ids"""
    if not game_ids:
        return {}

    try:
        conn = sqlconn.create_connection()
        if not conn:
            print("[-] Failed to connect to database")
            return {}

        # Build query for all odds
        placeholders = ','.join(['%s'] * len(game_ids))
        query = f"""
            SELECT game_id, bookmaker, ml_home, ml_away
            FROM odds
            WHERE game_id IN ({placeholders})
            ORDER BY game_id, bookmaker
        """

        results = sqlconn.fetch(conn, query, tuple(game_ids))
        conn.close()

        # Group by game_id
        odds_dict = {}
        for row in results:
            gid = row['game_id']
            if gid not in odds_dict:
                odds_dict[gid] = []
            odds_dict[gid].append({
                'bookmaker': row['bookmaker'],
                'ml_home': row['ml_home'],
                'ml_away': row['ml_away']
            })

        print(f"[+] Loaded odds for {len(odds_dict)} games\n")
        return odds_dict

    except Exception as e:
        print(f"[-] Error loading odds: {e}\n")
        return {}


def calculate_ev(win_prob: float, american_odds: int, stake: float = 100) -> float:
    """
    Calculate expected value as a percentage based on actual profit

    EV = (win_prob * profit_if_win) - ((1 - win_prob) * stake) / stake

    For negative odds (favorite): profit = stake * (100 / |odds|)
    For positive odds (underdog): profit = stake * (odds / 100)
    """
    win_prob = float(win_prob)
    american_odds = int(american_odds)

    # Calculate profit if bet wins
    if american_odds < 0:
        # Favorite: profit = stake * (100 / |odds|)
        profit_if_win = stake * (100 / abs(american_odds))
    else:
        # Underdog: profit = stake * (odds / 100)
        profit_if_win = stake * (american_odds / 100)

    # Expected value = (win_prob * profit) - (loss_prob * stake)
    ev = (win_prob * profit_if_win) - ((1 - win_prob) * stake)

    # Return as percentage (EV / stake)
    return float(ev / stake)


def get_ml_bets(features_df: pl.DataFrame, model: XGBClassifier, allowed_bookmakers: set, ev_threshold: float = 0.09) -> list:
    """
    Get moneyline bets with EV > threshold using allowed bookmakers only
    """
    ml_bets = []

    # Identify features and prepare data
    feature_cols = identify_feature_columns(features_df)
    X = prepare_data(features_df, feature_cols)

    # Make predictions
    pred_proba = model.predict_proba(X)

    # Load odds
    game_ids = features_df['game_id'].to_list()
    odds_dict = load_odds_for_games(game_ids)

    # Evaluate each game
    for idx, game in enumerate(features_df.iter_rows(named=True)):
        game_id = game.get('game_id')
        team_1 = game.get('team_1')
        team_2 = game.get('team_2')
        team_1_is_home = game.get('team_1_is_home_game')
        date = game.get('date')

        # Get odds
        all_odds = odds_dict.get(game_id, [])
        if not all_odds:
            continue

        # Get probabilities
        team_1_prob = float(pred_proba[idx, 1])
        team_2_prob = float(pred_proba[idx, 0])

        # Determine home/away
        if team_1_is_home == 1:
            team_1_odds, team_1_decimal, team_1_bm = get_best_odds_from_bookmakers(all_odds, 'home', allowed_bookmakers)
            team_2_odds, team_2_decimal, team_2_bm = get_best_odds_from_bookmakers(all_odds, 'away', allowed_bookmakers)
            team_1_home_away = 'HOME'
            team_2_home_away = 'AWAY'
        elif team_1_is_home == 0:
            team_1_odds, team_1_decimal, team_1_bm = get_best_odds_from_bookmakers(all_odds, 'away', allowed_bookmakers)
            team_2_odds, team_2_decimal, team_2_bm = get_best_odds_from_bookmakers(all_odds, 'home', allowed_bookmakers)
            team_1_home_away = 'AWAY'
            team_2_home_away = 'HOME'
        else:
            # Neutral
            team_1_odds, team_1_decimal, team_1_bm = get_best_odds_from_bookmakers(all_odds, 'home', allowed_bookmakers)
            team_2_odds, team_2_decimal, team_2_bm = get_best_odds_from_bookmakers(all_odds, 'away', allowed_bookmakers)
            team_1_home_away = 'NEUTRAL'
            team_2_home_away = 'NEUTRAL'

        # Check team 1
        if team_1_odds is not None:
            team_1_ev = calculate_ev(team_1_prob, team_1_odds, stake=100)
            if team_1_ev > ev_threshold:
                ml_bets.append({
                    'type': 'ML',
                    'game_id': game_id,
                    'date': date,
                    'team': team_1,
                    'opponent': team_2,
                    'position': team_1_home_away,
                    'odds': int(team_1_odds),
                    'decimal': round(team_1_decimal, 4),
                    'win_prob': round(team_1_prob, 4),
                    'ev': round(team_1_ev, 4),
                    'ev_percent': round(team_1_ev * 100, 2),
                    'bookmaker': team_1_bm
                })

        # Check team 2
        if team_2_odds is not None:
            team_2_ev = calculate_ev(team_2_prob, team_2_odds, stake=100)
            if team_2_ev > ev_threshold:
                ml_bets.append({
                    'type': 'ML',
                    'game_id': game_id,
                    'date': date,
                    'team': team_2,
                    'opponent': team_1,
                    'position': team_2_home_away,
                    'odds': int(team_2_odds),
                    'decimal': round(team_2_decimal, 4),
                    'win_prob': round(team_2_prob, 4),
                    'ev': round(team_2_ev, 4),
                    'ev_percent': round(team_2_ev * 100, 2),
                    'bookmaker': team_2_bm
                })

    return ml_bets


def get_ou_bets(ou_predictions_df: pl.DataFrame, difference_threshold: float = 2.3, allowed_bookmakers: set = None) -> list:
    """
    Extract OU bets where (ensemble - over_point) > threshold
    """
    ou_bets = []

    if ou_predictions_df is None or len(ou_predictions_df) == 0:
        return ou_bets

    # Try to get allowed bookmakers from odds table
    if allowed_bookmakers is None:
        allowed_bookmakers = {'betonlineag', 'BetOnline.ag', 'Bovada', 'MyBookie.ag'}

    try:
        conn = sqlconn.create_connection()
        if not conn:
            print("[-] Failed to connect to database for OU odds")
            return ou_bets

        for row in ou_predictions_df.iter_rows(named=True):
            game_id = row.get('game_id')
            ensemble = row.get('ensemble_pred', row.get('prediction'))
            date = row.get('date')

            if game_id is None or ensemble is None:
                continue

            # Get odds from allowed bookmakers
            query = """
                SELECT over_point, bookmaker
                FROM odds
                WHERE game_id = %s
                AND bookmaker IN (%s, %s, %s, %s)
                ORDER BY bookmaker
            """

            results = sqlconn.fetch(conn, query, (
                game_id,
                'BetOnline.ag',
                'Bovada',
                'MyBookie.ag',
                'betonlineag'
            ))

            if results:
                over_point = results[0].get('over_point')
                bookmaker = results[0].get('bookmaker')

                if over_point is not None:
                    difference = ensemble - over_point
                    if difference > difference_threshold:
                        ou_bets.append({
                            'type': 'OU',
                            'game_id': game_id,
                            'date': date,
                            'ensemble': round(ensemble, 2),
                            'over_point': round(over_point, 2),
                            'difference': round(difference, 2),
                            'bookmaker': bookmaker,
                            'bet': 'OVER'
                        })

        conn.close()

    except Exception as e:
        print(f"[-] Error processing OU bets: {e}")

    return ou_bets


def main():
    print("\n")
    print("="*80)
    print("FUTURES REPORT - ML & OU BETTING RECOMMENDATIONS")
    print("="*80 + "\n")

    # Define allowed bookmakers
    allowed_bookmakers = {'BetOnline.ag', 'Bovada', 'MyBookie.ag', 'betonlineag'}

    # Step 1: Get OU predictions
    print("STEP 1: Loading OU Predictions")
    print("-"*80 + "\n")
    ou_features_df, ou_predictions_df = ou_main()

    if ou_features_df is None or ou_predictions_df is None:
        print("[-] Failed to get OU predictions")
        return

    print(f"[+] Loaded {len(ou_predictions_df)} OU predictions\n")

    # Step 2: Load ML model
    print("STEP 2: Loading Moneyline Model")
    print("-"*80 + "\n")
    model = load_model()
    if model is None:
        return

    # Step 3: Get ML predictions and filter for EV > 9%
    print("STEP 3: Getting ML Predictions (EV > 9%)")
    print("-"*80 + "\n")
    ml_bets = get_ml_bets(ou_features_df, model, allowed_bookmakers, ev_threshold=0.09)
    print(f"[+] Found {len(ml_bets)} ML bets with EV > 9%\n")

    # Step 4: Get OU predictions and filter for difference > 2.3
    print("STEP 4: Getting OU Predictions (Difference > 2.3)")
    print("-"*80 + "\n")
    ou_bets = get_ou_bets(ou_predictions_df, difference_threshold=2.3, allowed_bookmakers=allowed_bookmakers)
    print(f"[+] Found {len(ou_bets)} OU bets with difference > 2.3\n")

    # Step 5: Print combined futures report
    print("\n" + "="*80)
    print("FUTURES REPORT - BETTING RECOMMENDATIONS")
    print("="*80 + "\n")

    if ml_bets or ou_bets:
        # Print ML bets
        if ml_bets:
            print(f"MONEYLINE BETS (EV > 9%, Bookmakers: {', '.join(sorted(allowed_bookmakers))})")
            print("-"*80 + "\n")

            # Sort by EV descending
            ml_bets_sorted = sorted(ml_bets, key=lambda x: x['ev_percent'], reverse=True)

            for bet in ml_bets_sorted:
                print(f"Date: {bet['date']}")
                print(f"Game ID: {bet['game_id']}")
                print(f"Matchup: {bet['team']} ({bet['position']}) vs {bet['opponent']}")
                print(f"  Odds: {bet['odds']:+d}  |  Decimal: {bet['decimal']}")
                print(f"  Win Prob: {bet['win_prob']*100:.2f}%")
                print(f"  EV: {bet['ev_percent']:.2f}%")
                print(f"  Bookmaker: {bet['bookmaker']}")
                print()

        # Print OU bets
        if ou_bets:
            print(f"\nOVER/UNDER BETS (Difference > 2.3, Bookmakers: {', '.join(sorted(allowed_bookmakers))})")
            print("-"*80 + "\n")

            # Sort by difference descending
            ou_bets_sorted = sorted(ou_bets, key=lambda x: x['difference'], reverse=True)

            for bet in ou_bets_sorted:
                print(f"Date: {bet['date']}")
                print(f"Game ID: {bet['game_id']}")
                print(f"  Ensemble: {bet['ensemble']}  |  Over Point: {bet['over_point']}")
                print(f"  Difference: {bet['difference']} (BET: {bet['bet']})")
                print(f"  Bookmaker: {bet['bookmaker']}")
                print()

        # Summary
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80 + "\n")
        print(f"Total ML Bets (EV > 9%): {len(ml_bets)}")
        print(f"Total OU Bets (Diff > 2.3): {len(ou_bets)}")
        print(f"Total Bets: {len(ml_bets) + len(ou_bets)}\n")

    else:
        print("No bets meet criteria")
        print("  ML: EV > 9% from allowed bookmakers")
        print("  OU: Difference > 2.3\n")

    print("="*80 + "\n")


if __name__ == "__main__":
    main()
