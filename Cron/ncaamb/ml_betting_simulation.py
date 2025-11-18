#!/usr/bin/env python3
"""
Moneyline Betting Simulation
Trains on 2021-2024, tests on 2025
Uses odds from ncaamb.odds table (all bookmakers)
Calculates EV and ROI based on moneyline odds
"""

import os
import sys
from pathlib import Path
import polars as pl
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# Add current directory to path
ncaamb_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ncaamb_dir)

from scrapes import sqlconn


def load_features_by_year(years: list) -> pl.DataFrame:
    """Load feature files from specified years"""
    features_dir = Path(__file__).parent
    all_features = []

    print(f"Loading features for years: {years}")
    for year in years:
        features_file = features_dir / f"features{year}.csv"

        if features_file.exists():
            print(f"  Loading features{year}.csv...")
            try:
                df = pl.read_csv(features_file)
                print(f"    [+] Loaded {len(df)} games")
                all_features.append(df)
            except Exception as e:
                print(f"    [-] Error loading {year}: {e}")
        else:
            print(f"    [-] File not found: {features_file}")

    if not all_features:
        return None

    combined_df = pl.concat(all_features)
    print(f"[+] Combined: {len(combined_df)} total games\n")
    return combined_df


def load_team_mappings() -> dict:
    """Load team_mappings.csv to map team names"""
    mappings_file = Path(__file__).parent / "bookmaker" / "team_mappings.csv"

    if not mappings_file.exists():
        print(f"Warning: team_mappings.csv not found at {mappings_file}")
        return {}

    df = pl.read_csv(mappings_file)
    # Create mapping from home_team/away_team to team_1/team_2
    mapping = {}
    for row in df.iter_rows(named=True):
        mapping[(row['home_team'], row['away_team'])] = (row['team_1'], row['team_2'])

    return mapping


def load_odds_for_games(game_ids: list) -> dict:
    """Load odds from ncaamb.odds table for given game_ids"""
    if not game_ids:
        return {}

    try:
        conn = sqlconn.create_connection()
        if not conn:
            print("Failed to connect to database")
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


def filter_low_quality_games(df: pl.DataFrame, min_data_quality: float = 0.5) -> pl.DataFrame:
    """Filter out early season games"""
    before = len(df)

    if 'team_1_data_quality' in df.columns and 'team_2_data_quality' in df.columns:
        df = df.filter(
            (pl.col('team_1_data_quality') >= min_data_quality) &
            (pl.col('team_2_data_quality') >= min_data_quality)
        )

    after = len(df)
    removed = before - after
    print(f"[*] Filtered low-quality games: removed {removed}, kept {after}\n")
    return df


def filter_missing_moneyline_data(df: pl.DataFrame) -> pl.DataFrame:
    """Remove games without essential moneyline data"""
    before = len(df)

    df = df.filter(
        pl.col('avg_ml_home').is_not_null() &
        pl.col('avg_ml_away').is_not_null()
    )

    after = len(df)
    removed = before - after
    print(f"[*] Filtered missing moneyline data: removed {removed}, kept {after}\n")
    return df


def create_target_variable(df: pl.DataFrame) -> pl.DataFrame:
    """Create binary target variable"""
    df_with_scores = df.filter(
        pl.col('team_1_score').is_not_null() &
        pl.col('team_2_score').is_not_null()
    )

    df_with_scores = df_with_scores.with_columns(
        pl.when(pl.col('team_1_score') > pl.col('team_2_score'))
            .then(1)
            .otherwise(0)
            .alias('ml_target')
    )

    print(f"[*] Created target for {len(df_with_scores)} games with results")
    print(f"  Team 1 wins: {df_with_scores.filter(pl.col('ml_target') == 1).height}")
    print(f"  Team 2 wins: {df_with_scores.filter(pl.col('ml_target') == 0).height}\n")

    return df_with_scores


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


def prepare_data(df: pl.DataFrame, feature_cols: list) -> tuple:
    """Prepare X and y"""
    X = df.select(feature_cols).fill_null(0).to_numpy()
    y = df.select('ml_target').to_numpy().ravel()
    return X, y


def american_to_decimal(american_odds: float) -> float:
    """Convert American odds to decimal odds"""
    american_odds = float(american_odds)  # Ensure float conversion
    if american_odds >= 0:
        return float((american_odds / 100) + 1)
    else:
        return float((100 / abs(american_odds)) + 1)


def calculate_ev(win_prob: float, decimal_odds: float) -> float:
    """
    Calculate expected value as percentage
    EV% = (win_prob * (decimal_odds - 1)) - (1 - win_prob)
    """
    win_prob = float(win_prob)
    decimal_odds = float(decimal_odds)
    ev = (win_prob * (decimal_odds - 1)) - (1 - win_prob)
    return float(ev)


def get_best_odds(all_odds: list, team_position: str) -> tuple:
    """
    Get best odds for a team from all bookmakers
    team_position: 'home' or 'away'
    Returns (best_odds, decimal, bookmaker)
    """
    if not all_odds:
        return None, None, None

    odds_key = 'ml_home' if team_position == 'home' else 'ml_away'
    valid_odds = [o[odds_key] for o in all_odds if o[odds_key] is not None]

    if not valid_odds:
        return None, None, None

    # Find best odds (highest positive or least negative)
    best_odds = max(valid_odds, key=lambda x: american_to_decimal(x))
    best_decimal = american_to_decimal(best_odds)

    # Find bookmaker with best odds
    best_bookmaker = None
    for o in all_odds:
        if o[odds_key] == best_odds:
            best_bookmaker = o['bookmaker']
            break

    return best_odds, best_decimal, best_bookmaker


def simulate_betting(test_df: pl.DataFrame, predictions: np.ndarray, pred_proba: np.ndarray,
                    odds_dict: dict, feature_cols: list, stake: float = 10.0) -> dict:
    """
    Simulate betting on all games with best available odds
    """
    results = {
        'games': [],
        'roi_by_threshold': {},
        'debug_games': []
    }

    for idx, row in enumerate(test_df.iter_rows(named=True)):
        game_id = row.get('game_id')

        # Get all odds for this game from database
        all_odds = odds_dict.get(game_id, [])

        if not all_odds:
            continue

        team_1_win_prob = pred_proba[idx, 1]
        team_2_win_prob = pred_proba[idx, 0]

        team_1_is_home = row.get('team_1_is_home_game')

        # Determine home/away for team_1 and team_2
        if team_1_is_home == 1:
            # Team 1 is home
            team_1_odds, team_1_decimal, team_1_bookmaker = get_best_odds(all_odds, 'home')
            team_2_odds, team_2_decimal, team_2_bookmaker = get_best_odds(all_odds, 'away')
        elif team_1_is_home == 0:
            # Team 1 is away
            team_1_odds, team_1_decimal, team_1_bookmaker = get_best_odds(all_odds, 'away')
            team_2_odds, team_2_decimal, team_2_bookmaker = get_best_odds(all_odds, 'home')
        else:
            # Neutral - try both
            team_1_odds_h, team_1_decimal_h, team_1_bm_h = get_best_odds(all_odds, 'home')
            team_1_odds_a, team_1_decimal_a, team_1_bm_a = get_best_odds(all_odds, 'away')

            if team_1_decimal_h and team_1_decimal_a:
                if team_1_decimal_h > team_1_decimal_a:
                    team_1_odds, team_1_decimal, team_1_bookmaker = team_1_odds_h, team_1_decimal_h, team_1_bm_h
                    team_2_odds, team_2_decimal, team_2_bookmaker = get_best_odds(all_odds, 'away')
                else:
                    team_1_odds, team_1_decimal, team_1_bookmaker = team_1_odds_a, team_1_decimal_a, team_1_bm_a
                    team_2_odds, team_2_decimal, team_2_bookmaker = get_best_odds(all_odds, 'home')
            else:
                team_1_odds, team_1_decimal, team_1_bookmaker = get_best_odds(all_odds, 'home')
                team_2_odds, team_2_decimal, team_2_bookmaker = get_best_odds(all_odds, 'away')

        if team_1_odds is None or team_2_odds is None:
            continue

        # Calculate EV for both sides
        team_1_ev = calculate_ev(team_1_win_prob, team_1_decimal)
        team_2_ev = calculate_ev(team_2_win_prob, team_2_decimal)

        # Determine best bet - only bet if EV is positive
        if team_1_ev > team_2_ev and team_1_ev > 0:
            best_bet = 'team_1'
            ev = team_1_ev
            odds = team_1_odds
            decimal = team_1_decimal
            bookmaker = team_1_bookmaker
            win_prob = team_1_win_prob
        elif team_2_ev > 0:
            best_bet = 'team_2'
            ev = team_2_ev
            odds = team_2_odds
            decimal = team_2_decimal
            bookmaker = team_2_bookmaker
            win_prob = team_2_win_prob
        else:
            # Both sides have negative EV, skip this game
            best_bet = None
            ev = max(team_1_ev, team_2_ev)  # Track best option even if both negative

        # Skip if no positive EV bet found
        if best_bet is None:
            continue

        # Store debug info for first 5 games
        if len(results['debug_games']) < 5:
            results['debug_games'].append({
                'game_id': game_id,
                'team_1': row['team_1'],
                'team_2': row['team_2'],
                'team_1_is_home': team_1_is_home,
                'team_1_win_prob': team_1_win_prob,
                'team_2_win_prob': team_2_win_prob,
                'team_1_odds': team_1_odds,
                'team_2_odds': team_2_odds,
                'team_1_decimal': team_1_decimal,
                'team_2_decimal': team_2_decimal,
                'team_1_ev': team_1_ev,
                'team_2_ev': team_2_ev,
                'best_bet': best_bet,
                'win_prob': win_prob,
                'odds': odds,
                'bookmaker': bookmaker,
                'decimal': decimal,
                'ev': ev
            })

        # Get actual result
        actual_winner = 1 if row['team_1_score'] > row['team_2_score'] else 0
        actual_winner_str = 'team_1' if actual_winner == 1 else 'team_2'
        correct = (best_bet == actual_winner_str)

        # Calculate P&L
        if correct:
            pnl = stake * (decimal - 1)
        else:
            pnl = -stake

        results['games'].append({
            'game_id': game_id,
            'date': row.get('date'),
            'team_1': row.get('team_1'),
            'team_2': row.get('team_2'),
            'best_bet': best_bet,
            'bookmaker': bookmaker,
            'odds': odds,
            'ev_percent': ev * 100,
            'ev': ev,
            'win_prob': win_prob,
            'actual_result': actual_winner_str,
            'correct': correct,
            'pnl': pnl,
            'stake': stake
        })

    # Calculate ROI by EV threshold
    ev_thresholds = list(range(-5, int(max([g['ev_percent'] for g in results['games']]) + 1)))

    for threshold in ev_thresholds:
        bets = [g for g in results['games'] if g['ev_percent'] >= threshold]

        if len(bets) == 0:
            continue

        total_stake = len(bets) * stake
        total_pnl = sum(b['pnl'] for b in bets)
        roi = (total_pnl / total_stake) * 100 if total_stake > 0 else 0
        win_rate = sum(b['correct'] for b in bets) / len(bets) * 100

        results['roi_by_threshold'][threshold] = {
            'num_bets': len(bets),
            'wins': sum(b['correct'] for b in bets),
            'losses': len(bets) - sum(b['correct'] for b in bets),
            'total_stake': total_stake,
            'total_pnl': total_pnl,
            'roi_percent': roi,
            'win_rate': win_rate
        }

    return results


def main():
    print("\n")
    print("="*80)
    print("MONEYLINE BETTING SIMULATION")
    print("="*80 + "\n")

    # Load training data (2021-2024)
    print("STEP 1: Loading Training Data (2021-2024)")
    print("-"*80 + "\n")
    train_df = load_features_by_year(['2021', '2022', '2023', '2024'])

    if train_df is None:
        print("Failed to load training features")
        return

    train_df = filter_low_quality_games(train_df, min_data_quality=0.5)
    train_df = filter_missing_moneyline_data(train_df)
    train_df = create_target_variable(train_df)
    feature_cols = identify_feature_columns(train_df)

    X_train, y_train = prepare_data(train_df, feature_cols)
    print(f"Training data shape: X={X_train.shape}\n")

    # Load pre-trained model from saved folder
    print("STEP 2: Loading Pre-trained Model")
    print("-"*80 + "\n")
    model_path = Path(__file__).parent / "models" / "moneyline" / "saved" / "xgboost_model.pkl"

    if model_path.exists():
        model = XGBClassifier()
        model.load_model(str(model_path))
        print(f"[+] Model loaded from {model_path}\n")
    else:
        print(f"[-] Model not found at {model_path}")
        print(f"    Please run export_2025_predictions.py first to train and save the model\n")
        return

    # Load test data (2025)
    print("STEP 3: Loading Test Data (2025)")
    print("-"*80 + "\n")

    # Don't train model, just prepare training data for feature columns
    # Training data is only used to extract feature columns, not for training
    test_df = load_features_by_year(['2025'])

    if test_df is None:
        print("Failed to load test features")
        return

    test_df = filter_low_quality_games(test_df, min_data_quality=0.5)
    test_df = filter_missing_moneyline_data(test_df)
    test_df_with_target = create_target_variable(test_df)

    X_test, y_test = prepare_data(test_df_with_target, feature_cols)
    print(f"Test data shape: X={X_test.shape}\n")

    # Load odds from database
    print("STEP 4: Loading Odds from Database")
    print("-"*80 + "\n")
    game_ids = test_df_with_target['game_id'].to_list()
    odds_dict = load_odds_for_games(game_ids)

    # Make predictions
    print("STEP 5: Making Predictions")
    print("-"*80 + "\n")
    predictions = model.predict(X_test)
    pred_proba = model.predict_proba(X_test)

    accuracy = accuracy_score(y_test, predictions)
    print(f"Model Accuracy on 2025: {accuracy:.3f}\n")

    # Check probability calibration
    print("STEP 6: Probability Analysis")
    print("-"*80 + "\n")
    team_1_probs = pred_proba[:, 1]
    print(f"Team 1 win probability distribution:")
    print(f"  Min: {team_1_probs.min():.6f}")
    print(f"  Max: {team_1_probs.max():.6f}")
    print(f"  Mean: {team_1_probs.mean():.6f}")
    print(f"  Median: {np.median(team_1_probs):.6f}")
    print(f"  Std Dev: {team_1_probs.std():.6f}")
    print(f"  < 0.1: {(team_1_probs < 0.1).sum()} games")
    print(f"  0.1-0.4: {((team_1_probs >= 0.1) & (team_1_probs < 0.4)).sum()} games")
    print(f"  0.4-0.6: {((team_1_probs >= 0.4) & (team_1_probs < 0.6)).sum()} games")
    print(f"  0.6-0.9: {((team_1_probs >= 0.6) & (team_1_probs < 0.9)).sum()} games")
    print(f"  >= 0.9: {(team_1_probs >= 0.9).sum()} games\n")

    # Run betting simulation
    print("STEP 7: Running Betting Simulation")
    print("-"*80 + "\n")
    results = simulate_betting(test_df_with_target, predictions, pred_proba, odds_dict, feature_cols)

    # Display results
    print("\n" + "="*80)
    print("BETTING SIMULATION RESULTS")
    print("="*80 + "\n")

    print("DEBUG: Detailed EV Calculation for First 5 Games (with probability analysis):")
    print("-"*80)
    for i, game in enumerate(results['debug_games']):
        home_away_str = "Team 1 is HOME" if game['team_1_is_home'] == 1 else "Team 1 is AWAY" if game['team_1_is_home'] == 0 else "NEUTRAL"
        bet_team = game['team_1'] if game['best_bet'] == 'team_1' else game['team_2']
        print(f"\nGame {i+1}: {game['team_1']:20} vs {game['team_2']:20} ({home_away_str})")
        print(f"  Game ID: {game['game_id']}")
        print(f"  {game['team_1']:20} Win Prob: {game['team_1_win_prob']:.4f} ({game['team_1_win_prob']*100:.2f}%)")
        print(f"  {game['team_2']:20} Win Prob: {game['team_2_win_prob']:.4f} ({game['team_2_win_prob']*100:.2f}%)")
        print(f"  {game['team_1']:20} Best Odds: {game['team_1_odds']:.0f} | Decimal: {game['team_1_decimal']:.4f}")
        print(f"  {game['team_2']:20} Best Odds: {game['team_2_odds']:.0f} | Decimal: {game['team_2_decimal']:.4f}")
        print(f"  {game['team_1']:20} EV: {game['team_1_ev']:.4f} ({game['team_1_ev']*100:.2f}%)")
        print(f"  {game['team_2']:20} EV: {game['team_2_ev']:.4f} ({game['team_2_ev']*100:.2f}%)")
        print(f"  [BEST] Best Bet: {bet_team:20} @ {game['bookmaker']:15} | Odds: {game['odds']:.0f} | EV: {game['ev']*100:.2f}%")

    print("\n" + "-"*80)
    print("EV and Odds Distribution:")
    if results['games']:
        print(f"  Total games with odds: {len(results['games'])}")
        print(f"  Min EV: {min(g['ev_percent'] for g in results['games']):.2f}%")
        print(f"  Max EV: {max(g['ev_percent'] for g in results['games']):.2f}%")
        print(f"  Avg EV: {sum(g['ev_percent'] for g in results['games']) / len(results['games']):.2f}%")

        # Count games by EV ranges
        ev_ranges = {
            'Extreme (>100%)': len([g for g in results['games'] if g['ev_percent'] > 100]),
            'Very High (50-100%)': len([g for g in results['games'] if 50 <= g['ev_percent'] <= 100]),
            'High (10-50%)': len([g for g in results['games'] if 10 <= g['ev_percent'] < 50]),
            'Positive (0-10%)': len([g for g in results['games'] if 0 <= g['ev_percent'] < 10]),
            'Negative': len([g for g in results['games'] if g['ev_percent'] < 0]),
        }
        print(f"\n  EV Distribution:")
        for range_name, count in ev_ranges.items():
            print(f"    {range_name:25} {count:5d} games ({count/len(results['games'])*100:5.1f}%)")

    print("ROI by EV Threshold ($10 per bet):\n")
    print(f"{'EV Threshold':>15} {'Bets':>8} {'Wins':>8} {'ROI':>10} {'Win Rate':>12}")
    print("-"*60)

    # Only show thresholds from -5 to 10 to avoid spam
    for threshold in sorted([t for t in results['roi_by_threshold'].keys() if t <= 10]):
        stats = results['roi_by_threshold'][threshold]
        print(f"{threshold:>14}% {stats['num_bets']:>8} {stats['wins']:>8} "
              f"{stats['roi_percent']:>9.2f}% {stats['win_rate']:>11.1f}%")

    # Find best threshold
    if results['roi_by_threshold']:
        best_threshold = max(results['roi_by_threshold'].items(),
                             key=lambda x: x[1]['roi_percent'])
        threshold, stats = best_threshold

        print("\n" + "-"*60)
        print(f"\nBest EV Threshold: {threshold}%")
        print(f"  Number of Bets: {stats['num_bets']}")
        print(f"  Wins: {stats['wins']}")
        print(f"  Losses: {stats['losses']}")
        print(f"  Total Stake: ${stats['total_stake']:.2f}")
        print(f"  Total P&L: ${stats['total_pnl']:.2f}")
        print(f"  ROI: {stats['roi_percent']:.2f}%")
        print(f"  Win Rate: {stats['win_rate']:.1f}%\n")

    # Sample positive EV bets
    print("-"*60)
    print("\nSample Positive EV Bets:\n")
    positive_ev_bets = [g for g in results['games'] if g['ev_percent'] > 0]
    for bet in positive_ev_bets[:10]:
        result_str = "[WIN]" if bet['correct'] else "[LOSS]"
        bet_team = bet['team_1'] if bet['best_bet'] == 'team_1' else bet['team_2']
        print(f"{bet['team_1']:20} vs {bet['team_2']:20}")
        print(f"  Game ID: {bet['game_id']}")
        print(f"  Bet: {bet_team:20} @ {bet['bookmaker']:15} | Odds: {bet['odds']:7.0f} | EV: {bet['ev_percent']:6.2f}% {result_str}")
        print(f"  P&L: ${bet['pnl']:+.2f}\n")

    print("="*80)
    print("[SUCCESS] Betting simulation complete!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
