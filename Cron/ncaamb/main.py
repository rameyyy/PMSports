#!/usr/bin/env python3
"""
Futures Report: Combined ML and OU Betting Recommendations
Generates daily betting report with:
- Moneyline (ML) bets using Ensemble (18% LGB + 82% XGB): EV > 0%, Probability > 0.50
- Over/Under (OU) bets where ensemble - over_point > 2.3
Only considers odds from: BetOnline.ag, Bovada, MyBookie.ag
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import polars as pl
import pandas as pd
import numpy as np
import lightgbm as lgb
from xgboost import XGBClassifier
import pickle

# Add current directory to path
ncaamb_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ncaamb_dir)

from scrapes import sqlconn
from ou_main import build_todays_games_df, generate_features


def load_team_mappings() -> dict:
    """Load team mappings from CSV (odds_team_name -> my_team_name)"""
    try:
        mappings_file = Path(__file__).parent / "bookmaker" / "team_mappings.csv"
        if not mappings_file.exists():
            print(f"[-] Team mappings file not found: {mappings_file}")
            return {}

        df = pl.read_csv(str(mappings_file))
        # Create dict mapping from_odds_team_name -> my_team_name
        mapping_dict = {}
        for row in df.iter_rows(named=True):
            mapping_dict[row['from_odds_team_name']] = row['my_team_name']

        return mapping_dict
    except Exception as e:
        print(f"[-] Error loading team mappings: {e}")
        return {}


def get_todays_date():
    """Get today's date in YYYY-MM-DD format"""
    return datetime.now().strftime('%Y-%m-%d')


def get_todays_date_yyyymmdd():
    """Get today's date in YYYYMMDD format"""
    return datetime.now().strftime('%Y%m%d')


def load_ml_models() -> tuple:
    """Load LGB and XGB models for Ensemble implementation (18% LGB + 82% XGB)"""
    saved_dir = Path(__file__).parent / "models" / "moneyline" / "saved"

    # Load LightGBM model
    lgb_path = saved_dir / "lightgbm_model_final.pkl"
    if not lgb_path.exists():
        print(f"[-] LGB model not found at {lgb_path}\n")
        return None, None

    try:
        lgb_model = pickle.load(open(lgb_path, 'rb'))
        print(f"[+] LGB model loaded from {lgb_path}")
    except Exception as e:
        print(f"[-] Error loading LGB model: {e}\n")
        return None, None

    # Load XGBoost model
    xgb_path = saved_dir / "xgboost_model_final.pkl"
    if not xgb_path.exists():
        print(f"[-] XGB model not found at {xgb_path}\n")
        return lgb_model, None

    try:
        xgb_model = pickle.load(open(xgb_path, 'rb'))
        print(f"[+] XGB model loaded from {xgb_path}\n")
    except Exception as e:
        print(f"[-] Error loading XGB model: {e}\n")
        return lgb_model, None

    return lgb_model, xgb_model


def get_metadata_cols() -> set:
    """Get metadata columns (same as train_final_ensemble_new_odds.py)"""
    return {
        'game_id', 'date', 'season', 'team_1', 'team_2',
        'team_1_score', 'team_2_score', 'actual_total',
        'team_1_conference', 'team_2_conference',
        'team_1_is_home', 'team_2_is_home', 'location',
        'total_score_outcome', 'team_1_winloss',
        'team_1_leaderboard', 'team_2_leaderboard',
        'team_1_match_hist', 'team_2_match_hist',
        'team_1_hist_count', 'team_2_hist_count',
        'start_time', 'game_odds', 'ml_target',
        'team_1_data_quality', 'team_2_data_quality'
    }


def identify_feature_columns(df: pl.DataFrame) -> list:
    """Identify numeric feature columns (same as in ou_main.py)"""
    metadata_cols = get_metadata_cols()

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
    # Replace any NaN values that might remain (e.g., from None values in training)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X


def load_expected_feature_columns() -> list:
    """Load the list of expected feature columns from file (tab-separated index and name)"""
    expected_file = Path(__file__).parent / "models" / "moneyline" / "saved" / "feature_columns.txt"

    if expected_file.exists():
        feature_cols = []
        with open(expected_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    # Parse "index\tcolumn_name" format
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        feature_cols.append(parts[1])
        return feature_cols if feature_cols else None

    return None


def align_features_to_model(df: pl.DataFrame) -> tuple:
    """
    Align features using the exact list from training (feature_columns.txt)
    Convert American odds to decimal (same as training).
    Returns (X array, feature_cols list) or (None, None) if alignment fails.
    """
    # Load expected feature columns from training
    expected_feature_cols = load_expected_feature_columns()
    if not expected_feature_cols:
        print("[-] Could not load expected feature columns from training")
        return None, None

    # Convert polars to pandas
    df_pd = df.to_pandas()

    print(f"[+] Feature alignment:")
    print(f"    Expected features from training: {len(expected_feature_cols)}")

    # Check which expected columns are available
    available_cols = [col for col in expected_feature_cols if col in df_pd.columns]
    missing_cols = [col for col in expected_feature_cols if col not in df_pd.columns]

    if len(available_cols) < len(expected_feature_cols):
        print(f"    WARNING: Missing {len(missing_cols)} columns")

    # Convert all odds columns from American to decimal (same as training)
    odds_cols = [
        'betonline_ml_team_1', 'betonline_ml_team_2',
        'bovada_ml_team_1', 'bovada_ml_team_2',
        'betmgm_ml_team_1', 'betmgm_ml_team_2',
        'draftkings_ml_team_1', 'draftkings_ml_team_2',
        'fanduel_ml_team_1', 'fanduel_ml_team_2',
        'lowvig_ml_team_1', 'lowvig_ml_team_2',
        'mybookie_ml_team_1', 'mybookie_ml_team_2',
        'avg_ml_team_1', 'avg_ml_team_2',
    ]

    for col in odds_cols:
        if col in df_pd.columns:
            # Convert American odds to decimal
            df_pd[col] = df_pd[col].apply(lambda x: american_to_decimal(x) if x is not None else None)

    # Use the expected feature columns (as they were used in training)
    feature_cols = expected_feature_cols

    # For missing columns, add zeros
    for col in missing_cols:
        if col not in df_pd.columns:
            df_pd[col] = 0.0

    # Fill null values with 0
    df_pd[feature_cols] = df_pd[feature_cols].fillna(0)

    # Convert to numpy
    try:
        X = df_pd[feature_cols].to_numpy(dtype=np.float64)
        # Replace any NaN/inf values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    except Exception as e:
        print(f"[-] Error preparing feature matrix: {e}\n")
        return None, None

    print(f"[+] Feature matrix shape: {X.shape}\n")

    return X, feature_cols


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

        # Build query for all odds - include home_team and away_team names
        placeholders = ','.join(['%s'] * len(game_ids))
        query = f"""
            SELECT game_id, bookmaker, home_team, away_team, ml_home, ml_away
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
                'home_team': row['home_team'],
                'away_team': row['away_team'],
                'ml_home': row['ml_home'],
                'ml_away': row['ml_away']
            })

        print(f"[+] Loaded odds for {len(odds_dict)} games\n")
        return odds_dict

    except Exception as e:
        print(f"[-] Error loading odds: {e}\n")
        return {}


def calculate_ev(win_prob: float, american_odds: int, stake: float = 10) -> float:
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




def get_ml_bets(features_df: pl.DataFrame, lgb_model, xgb_model, allowed_bookmakers: set, team_mappings: dict = None) -> list:
    """
    Get moneyline bets using Ensemble predictions: 18% LGB + 82% XGB
    Uses Clay's optimal betting rules:
    - PRIMARY VOLUME: 0.52 <= prob < 0.57 AND 0% < EV < 10%
    - HIGH ROI TARGETS: 0.59 <= prob < 0.60 AND 0% < EV < 10%
    - HIDDEN GEMS: 0.74 <= prob < 0.76 AND 0% < EV < 10%
    - HIDDEN GEMS: 0.80 <= prob < 0.84 AND 0% < EV < 10%
    - AVOID: EV <= 0%, EV >= 10%, 0.55 <= prob < 0.59
    """
    ml_bets = []

    if lgb_model is None or xgb_model is None:
        print("[-] LGB or XGB model not loaded\n")
        return ml_bets

    print("STEP 2.5: Getting ML Predictions (Ensemble: 18% LGB + 82% XGB)")
    print("-"*80 + "\n")

    # Get LGB predictions for all games
    try:
        X, feature_cols = align_features_to_model(features_df)
        if X is None:
            print("[-] Could not align features to LGB model\n")
            return ml_bets

        # LGB returns probabilities as 1D array (probability of class 1)
        lgb_proba = lgb_model.predict(X)
        print(f"[+] Made LGB predictions for {len(X)} games\n")
    except Exception as e:
        print(f"[-] Error during LGB prediction: {e}\n")
        return ml_bets

    # Get XGB predictions
    try:
        xgb_proba = xgb_model.predict_proba(X)[:, 1]  # Get probability of class 1
        print(f"[+] Made XGB predictions for {len(X)} games\n")
    except Exception as e:
        print(f"[-] Error during XGB prediction: {e}\n")
        return ml_bets

    # Create ensemble predictions: 18% LGB + 82% XGB
    ensemble_proba = 0.18 * lgb_proba + 0.82 * xgb_proba
    print(f"[+] Created ensemble predictions (18% LGB + 82% XGB)\n")

    # Add ensemble probabilities to features dataframe
    features_df = features_df.with_columns([
        pl.lit(ensemble_proba).alias('ensemble_prob_team_1'),
    ])

    # Load odds
    game_ids = features_df['game_id'].to_list()
    odds_dict = load_odds_for_games(game_ids)

    print(f"[*] Evaluating bets with Clay's Optimal Rules:")
    print(f"    - PRIMARY VOLUME: 0.52 <= prob < 0.57 AND 0% < EV < 10%")
    print(f"    - HIGH ROI TARGETS: 0.59 <= prob < 0.60 AND 0% < EV < 10%")
    print(f"    - HIDDEN GEMS: 0.74 <= prob < 0.76 AND 0% < EV < 10%")
    print(f"    - HIDDEN GEMS: 0.80 <= prob < 0.84 AND 0% < EV < 10%")
    print(f"    - AVOID: EV <= 0%, EV >= 10%, 0.55 <= prob < 0.59\n")

    def matches_ml_rule(prob: float, ev: float) -> bool:
        """Check if probability and EV match any profitable ML betting rule"""
        # Avoid: EV <= 0%, EV >= 10%, 0.55 <= prob < 0.59
        if ev <= 0 or ev >= 10 or (0.55 <= prob < 0.59):
            return False

        # PRIMARY VOLUME: 0.52 <= prob < 0.57 AND 0% < EV < 10%
        if 0.52 <= prob < 0.57 and 0 < ev < 10:
            return True

        # HIGH ROI TARGETS: 0.59 <= prob < 0.60 AND 0% < EV < 10%
        if 0.59 <= prob < 0.60 and 0 < ev < 10:
            return True

        # HIDDEN GEMS: 0.74 <= prob < 0.76 AND 0% < EV < 10%
        if 0.74 <= prob < 0.76 and 0 < ev < 10:
            return True

        # HIDDEN GEMS: 0.80 <= prob < 0.84 AND 0% < EV < 10%
        if 0.80 <= prob < 0.84 and 0 < ev < 10:
            return True

        return False

    # Process each game
    for i, game_row in enumerate(features_df.iter_rows(named=True)):
        game_id = game_row.get('game_id')
        team_1 = game_row.get('team_1')
        team_2 = game_row.get('team_2')
        date = game_row.get('date')
        ensemble_prob_t1 = ensemble_proba[i]
        ensemble_prob_t2 = 1 - ensemble_prob_t1

        # Get odds for this game
        all_odds = odds_dict.get(game_id, [])
        if not all_odds:
            continue

        # Process Team 1
        # Find best ML odds for team_1
        best_odds_t1 = None
        best_bm_t1 = None

        for odds_rec in all_odds:
            bm_name = odds_rec.get('bookmaker')
            team_a_odds_name = odds_rec.get('home_team')
            team_b_odds_name = odds_rec.get('away_team')
            ml_team_a = odds_rec.get('ml_home')
            ml_team_b = odds_rec.get('ml_away')

            # Map odds team names
            team_a_mapped = team_a_odds_name
            team_b_mapped = team_b_odds_name
            if team_mappings:
                team_a_mapped = team_mappings.get(team_a_odds_name, team_a_odds_name)
                team_b_mapped = team_mappings.get(team_b_odds_name, team_b_odds_name)

            # Find team_1 in odds
            if team_a_mapped == team_1 and ml_team_a is not None:
                if best_odds_t1 is None or american_to_decimal(ml_team_a) > american_to_decimal(best_odds_t1):
                    best_odds_t1 = ml_team_a
                    best_bm_t1 = bm_name

            if team_b_mapped == team_1 and ml_team_b is not None:
                if best_odds_t1 is None or american_to_decimal(ml_team_b) > american_to_decimal(best_odds_t1):
                    best_odds_t1 = ml_team_b
                    best_bm_t1 = bm_name

        # Calculate EV for team_1 if odds found
        if best_odds_t1 is not None:
            ev_t1 = calculate_ev(ensemble_prob_t1, int(best_odds_t1))
            # Check if matches any profitable rule
            if matches_ml_rule(ensemble_prob_t1, ev_t1 * 100):
                ml_bets.append({
                    'type': 'ML',
                    'game_id': game_id,
                    'date': date,
                    'team': team_1,
                    'opponent': team_2,
                    'odds': int(best_odds_t1),
                    'decimal': american_to_decimal(best_odds_t1),
                    'win_prob': ensemble_prob_t1,
                    'ev': ev_t1,
                    'ev_percent': ev_t1 * 100,
                    'bookmaker': best_bm_t1
                })

        # Process Team 2
        # Find best ML odds for team_2
        best_odds_t2 = None
        best_bm_t2 = None

        for odds_rec in all_odds:
            bm_name = odds_rec.get('bookmaker')
            team_a_odds_name = odds_rec.get('home_team')
            team_b_odds_name = odds_rec.get('away_team')
            ml_team_a = odds_rec.get('ml_home')
            ml_team_b = odds_rec.get('ml_away')

            # Map odds team names
            team_a_mapped = team_a_odds_name
            team_b_mapped = team_b_odds_name
            if team_mappings:
                team_a_mapped = team_mappings.get(team_a_odds_name, team_a_odds_name)
                team_b_mapped = team_mappings.get(team_b_odds_name, team_b_odds_name)

            # Find team_2 in odds
            if team_a_mapped == team_2 and ml_team_a is not None:
                if best_odds_t2 is None or american_to_decimal(ml_team_a) > american_to_decimal(best_odds_t2):
                    best_odds_t2 = ml_team_a
                    best_bm_t2 = bm_name

            if team_b_mapped == team_2 and ml_team_b is not None:
                if best_odds_t2 is None or american_to_decimal(ml_team_b) > american_to_decimal(best_odds_t2):
                    best_odds_t2 = ml_team_b
                    best_bm_t2 = bm_name

        # Calculate EV for team_2 if odds found
        if best_odds_t2 is not None:
            ev_t2 = calculate_ev(ensemble_prob_t2, int(best_odds_t2))
            # Check if matches any profitable rule
            if matches_ml_rule(ensemble_prob_t2, ev_t2 * 100):
                ml_bets.append({
                    'type': 'ML',
                    'game_id': game_id,
                    'date': date,
                    'team': team_2,
                    'opponent': team_1,
                    'odds': int(best_odds_t2),
                    'decimal': american_to_decimal(best_odds_t2),
                    'win_prob': ensemble_prob_t2,
                    'ev': ev_t2,
                    'ev_percent': ev_t2 * 100,
                    'bookmaker': best_bm_t2
                })

    print(f"[+] Found {len(ml_bets)} bets meeting Clay's Optimal Rules\n")
    return ml_bets


def get_sportsbook_recommendation(confidence: float) -> tuple:
    """
    Get recommended sportsbook and bet type based on ensemble confidence.
    Returns (sportsbook, bet_type) or (None, None) if not in profitable range.

    Profitable ranges identified from ROI analysis:
    UNDER:
    - 0.16-0.21: MyBookie
    - 0.26-0.31: BetOnline
    - 0.36-0.41: MyBookie
    - 0.41-0.46: Bovada (PRIORITY - .36-.51 is main range)
    - 0.46-0.51: Bovada (PRIORITY - .36-.51 is main range)

    OVER:
    - 0.65-0.70: MyBookie
    """
    # Check UNDER ranges
    if 0.16 <= confidence < 0.21:
        return 'MyBookie.ag', 'UNDER'
    elif 0.26 <= confidence < 0.31:
        return 'BetOnline.ag', 'UNDER'
    elif 0.36 <= confidence < 0.41:
        return 'MyBookie.ag', 'UNDER'
    elif 0.41 <= confidence < 0.51:  # 0.41-0.46 and 0.46-0.51 - Bovada priority range
        return 'Bovada', 'UNDER'
    # Check OVER ranges
    elif 0.65 <= confidence < 0.70:
        return 'MyBookie.ag', 'OVER'

    return None, None


def get_ou_bets(ou_predictions_df: pl.DataFrame, difference_threshold: float = 2.3, allowed_bookmakers: set = None) -> list:
    """
    Extract OU bets using V3.0 comprehensive betting rules.

    Uses good_bet_score + ensemble_confidence + difference in various combinations.

    13 OVER rules + 13 UNDER rules = 26 total rules
    Overall ROI: +10.13% (from 1,091 profitable bets out of 4,936 games)
    """
    ou_bets = []

    if ou_predictions_df is None or len(ou_predictions_df) == 0:
        return ou_bets

    # Try to get allowed bookmakers from odds table
    if allowed_bookmakers is None:
        allowed_bookmakers = {'betonlineag', 'BetOnline.ag', 'Bovada', 'MyBookie.ag'}

    def matches_over_rule(good_bet_score: float, ensemble_confidence: float, difference: float) -> bool:
        """Check if parameters match any OVER betting rule (V3.0 - 13 rules)"""

        # 3D RULES (good_bet_score + ensemble_confidence + difference)
        # O1: 31 bets | 64.5% win | +23.79% ROI
        if 0.8 <= good_bet_score < 0.9 and 0.7 <= ensemble_confidence < 0.8 and 4 <= difference < 5:
            return True
        # O2: 45 bets | 55.6% win | +6.72% ROI
        if 0.8 <= good_bet_score < 0.9 and 0.6 <= ensemble_confidence < 0.7 and 2 <= difference < 3:
            return True
        # O3: 112 bets | 54.5% win | +4.48% ROI
        if 0.7 <= good_bet_score < 0.8 and 0.7 <= ensemble_confidence < 0.8 and 2 <= difference < 3:
            return True

        # 2D RULES - good_bet_score + difference
        # O4: 46 bets | 65.2% win | +25.27% ROI ⭐⭐
        if 0.80 <= good_bet_score < 0.85 and 3.5 <= difference < 4.0:
            return True
        # O5: 74 bets | 59.5% win | +13.89% ROI
        if 0.55 <= good_bet_score < 0.60 and 1.5 <= difference < 2.0:
            return True
        # O6: 44 bets | 59.1% win | +13.32% ROI
        if 0.50 <= good_bet_score < 0.55 and 1.5 <= difference < 2.0:
            return True
        # O7: 76 bets | 55.3% win | +5.81% ROI
        if 0.60 <= good_bet_score < 0.65 and 1.0 <= difference < 1.5:
            return True
        # O8: 96 bets | 54.2% win | +4.05% ROI
        if 0.65 <= good_bet_score < 0.70 and 1.5 <= difference < 2.0:
            return True
        # O9: 94 bets | 54.3% win | +4.01% ROI
        if 0.75 <= good_bet_score < 0.80 and 2.5 <= difference < 3.0:
            return True

        # 2D RULES - good_bet_score + ensemble_confidence
        # O10: 33 bets | 66.7% win | +27.48% ROI ⭐⭐⭐
        if 0.55 <= good_bet_score < 0.60 and 0.50 <= ensemble_confidence < 0.55:
            return True
        # O11: 44 bets | 63.6% win | +22.33% ROI ⭐
        if 0.80 <= good_bet_score < 0.85 and 0.65 <= ensemble_confidence < 0.70:
            return True
        # O12: 65 bets | 55.4% win | +6.39% ROI
        if 0.50 <= good_bet_score < 0.55 and 0.50 <= ensemble_confidence < 0.55:
            return True
        # O13: 114 bets | 55.3% win | +6.17% ROI
        if 0.65 <= good_bet_score < 0.70 and 0.60 <= ensemble_confidence < 0.65:
            return True

        return False

    def matches_under_rule(good_bet_score: float, ensemble_confidence: float, difference: float) -> bool:
        """Check if parameters match any UNDER betting rule (V3.0 - 13 rules)"""

        # 2D RULES - good_bet_score + difference
        # U1: 36 bets | 66.7% win | +27.80% ROI ⭐⭐⭐
        if 0.40 <= good_bet_score < 0.45 and -0.5 <= difference < 0.0:
            return True
        # U2: 48 bets | 64.6% win | +23.96% ROI ⭐⭐
        if 0.40 <= good_bet_score < 0.45 and -1.0 <= difference < 0.0:
            return True
        # U3: 47 bets | 62.8% win | +18.43% ROI ⭐
        if 0.25 <= good_bet_score < 0.30 and 0.0 <= difference < 0.5:
            return True
        # U4: 54 bets | 61.1% win | +17.21% ROI ⭐
        if 0.25 <= good_bet_score < 0.30 and 0.0 <= difference < 1.0:
            return True
        # U5: 44 bets | 61.4% win | +17.89% ROI ⭐
        if 0.00 <= good_bet_score < 0.05 and -4.0 <= difference < -3.0:
            return True
        # U6: 53 bets | 60.4% win | +15.68% ROI
        if 0.20 <= good_bet_score < 0.25 and -0.5 <= difference < 0.0:
            return True
        # U7: 41 bets | 56.1% win | +8.14% ROI
        if 0.15 <= good_bet_score < 0.20 and -1.0 <= difference < -0.5:
            return True
        # U8: 72 bets | 54.2% win | +4.00% ROI
        if 0.25 <= good_bet_score < 0.30 and -1.0 <= difference < -0.5:
            return True

        # 2D RULES - good_bet_score + ensemble_confidence
        # U9: 36 bets | 66.7% win | +27.82% ROI ⭐⭐⭐
        if 0.25 <= good_bet_score < 0.30 and 0.50 <= ensemble_confidence < 0.55:
            return True
        # U10: 50 bets | 64.0% win | +22.92% ROI ⭐⭐
        if 0.40 <= good_bet_score < 0.45 and 0.45 <= ensemble_confidence < 0.50:
            return True
        # U11: 54 bets | 61.1% win | +17.95% ROI ⭐
        if 0.15 <= good_bet_score < 0.20 and 0.40 <= ensemble_confidence < 0.45:
            return True
        # U12: 37 bets | 56.8% win | +9.24% ROI
        if 0.00 <= good_bet_score < 0.05 and 0.25 <= ensemble_confidence < 0.30:
            return True
        # U13: 65 bets | 56.9% win | +9.16% ROI
        if 0.20 <= good_bet_score < 0.25 and 0.45 <= ensemble_confidence < 0.50:
            return True

        return False

    try:
        conn = sqlconn.create_connection()
        if not conn:
            print("[-] Failed to connect to database for OU odds")
            return ou_bets

        matched_over = 0
        matched_under = 0
        no_odds = 0

        for row in ou_predictions_df.iter_rows(named=True):
            try:
                game_id = row.get('game_id')
                ensemble = row.get('ensemble_pred', row.get('prediction'))
                ensemble_confidence = row.get('ensemble_confidence')
                good_bets_confidence = row.get('good_bets_confidence')
                date = row.get('date')

                if game_id is None or ensemble is None:
                    continue

                # Skip if good_bets_confidence is missing (required for V3.0 rules)
                if good_bets_confidence is None:
                    continue

                # Get odds for this game
                query = """
                    SELECT over_point, over_odds, under_odds, bookmaker
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

                if not results:
                    no_odds += 1
                    continue

                # Use first bookmaker's O/U line for calculation
                ou_result = results[0]
                over_point = float(ou_result.get('over_point'))
                if over_point is None:
                    continue

                # Calculate difference = predicted_total - line
                difference = ensemble - over_point

                # Check OVER rules (V3.0 - uses good_bet_score, ensemble_confidence, difference)
                if matches_over_rule(good_bets_confidence, ensemble_confidence, difference):
                    matched_over += 1
                    ou_bets.append({
                        'type': 'OU',
                        'game_id': game_id,
                        'date': date,
                        'ensemble': round(ensemble, 2),
                        'over_point': round(over_point, 2),
                        'difference': round(difference, 2),
                        'confidence': round(ensemble_confidence, 4),
                        'good_bet_score': round(good_bets_confidence, 4),
                        'bookmaker': ou_result.get('bookmaker'),
                        'bet': 'OVER'
                    })

                # Check UNDER rules (V3.0 - uses good_bet_score, ensemble_confidence, difference)
                elif matches_under_rule(good_bets_confidence, ensemble_confidence, difference):
                    matched_under += 1
                    ou_bets.append({
                        'type': 'OU',
                        'game_id': game_id,
                        'date': date,
                        'ensemble': round(ensemble, 2),
                        'under_point': round(over_point, 2),
                        'difference': round(difference, 2),
                        'confidence': round(ensemble_confidence, 4),
                        'good_bet_score': round(good_bets_confidence, 4),
                        'bookmaker': ou_result.get('bookmaker'),
                        'bet': 'UNDER'
                    })

            except Exception as row_error:
                # Continue processing other rows even if one fails
                print(f"  [!] Error processing game: {row_error}")
                continue

        conn.close()

        # Debug output
        print(f"[+] Evaluated {len(ou_predictions_df)} games for OU bets")
        print(f"    OVER bets matched: {matched_over}")
        print(f"    UNDER bets matched: {matched_under}")
        print(f"    No odds available: {no_odds}")
        print()

    except Exception as e:
        print(f"[-] Error processing OU bets: {e}")
        import traceback
        traceback.print_exc()

    return ou_bets


def export_features_to_excel(features_df: pl.DataFrame, target_date_yyyymmdd: str) -> None:
    """Export all generated features to Excel file"""
    try:
        output_file = Path(__file__).parent / f"features_{target_date_yyyymmdd}.xlsx"
        features_df.write_excel(str(output_file))
        print(f"[+] Exported {len(features_df)} games with {len(features_df.columns)} features to {output_file}\n")
    except Exception as e:
        print(f"[-] Error exporting features: {e}\n")


def export_all_predictions(features_df: pl.DataFrame, lgb_model, xgb_model, ou_predictions_df: pl.DataFrame, target_date_yyyymmdd: str) -> None:
    """Export all predictions (ML and OU) regardless of EV/probability thresholds"""
    try:
        all_predictions = []

        # Get ML predictions for all games
        try:
            X, feature_cols = align_features_to_model(features_df)
            if X is not None:
                lgb_proba = lgb_model.predict(X)
                xgb_proba = xgb_model.predict_proba(X)[:, 1]
                ensemble_proba = 0.18 * lgb_proba + 0.82 * xgb_proba

                # Add all ML predictions
                for i, row in enumerate(features_df.iter_rows(named=True)):
                    game_id = row.get('game_id')
                    team_1 = row.get('team_1')
                    team_2 = row.get('team_2')
                    date = row.get('date')

                    prob_team_1 = ensemble_proba[i]
                    prob_team_2 = 1 - prob_team_1

                    all_predictions.append({
                        'type': 'ML',
                        'date': date,
                        'game_id': game_id,
                        'team_1': team_1,
                        'team_2': team_2,
                        'prob_team_1': round(prob_team_1, 4),
                        'prob_team_2': round(prob_team_2, 4),
                        'lgb_prob_team_1': round(lgb_proba[i], 4),
                        'xgb_prob_team_1': round(xgb_proba[i], 4),
                    })
        except Exception as e:
            print(f"[-] Error getting ML predictions: {e}")

        # Add all OU predictions
        if ou_predictions_df is not None:
            for row in ou_predictions_df.iter_rows(named=True):
                game_id = row.get('game_id')
                date = row.get('date')
                ensemble = row.get('ensemble_pred')
                ensemble_confidence = row.get('ensemble_confidence')
                good_bets_confidence = row.get('good_bets_confidence')

                all_predictions.append({
                    'type': 'OU',
                    'date': date,
                    'game_id': game_id,
                    'team_1': '',
                    'team_2': '',
                    'prob_team_1': '',
                    'prob_team_2': '',
                    'lgb_prob_team_1': '',
                    'xgb_prob_team_1': '',
                    'ensemble_pred': round(ensemble, 2) if ensemble else '',
                    'ensemble_confidence': round(ensemble_confidence, 4) if ensemble_confidence else '',
                    'good_bets_confidence': round(good_bets_confidence, 4) if good_bets_confidence else '',
                })

        if all_predictions:
            output_df = pl.DataFrame(all_predictions)
            output_file = Path(__file__).parent / f"all_predictions_{target_date_yyyymmdd}.xlsx"
            output_df.write_excel(str(output_file))
            print(f"[+] Exported {len(all_predictions)} all predictions (ML and OU) to {output_file}\n")
        else:
            print("[*] No predictions to export\n")

    except Exception as e:
        print(f"[-] Error exporting all predictions: {e}\n")


def export_all_games_with_odds(features_df: pl.DataFrame, lgb_model, xgb_model, target_date_yyyymmdd: str = None) -> None:
    """Export all games with probabilities and moneyline averages"""
    all_games = []

    # Get LGB predictions for all games
    try:
        X, feature_cols = align_features_to_model(features_df)
        if X is None:
            print("[-] Could not align features to LGB model\n")
            return

        # LGB returns probabilities as 1D array (probability of class 1)
        lgb_proba = lgb_model.predict(X)
    except Exception as e:
        print(f"[-] Error during LGB prediction: {e}\n")
        return

    # Get XGB predictions
    try:
        xgb_proba = xgb_model.predict_proba(X)[:, 1]  # Get probability of class 1
    except Exception as e:
        print(f"[-] Error during XGB prediction: {e}\n")
        return

    # Create ensemble predictions: 18% LGB + 82% XGB
    ensemble_proba = 0.18 * lgb_proba + 0.82 * xgb_proba

    # Process each game
    for i, row in enumerate(features_df.iter_rows(named=True)):
        game_id = row.get('game_id')
        team_1 = row.get('team_1')
        team_2 = row.get('team_2')
        date = row.get('date')
        ml_avg_team_1 = row.get('avg_ml_team_1')
        ml_avg_team_2 = row.get('avg_ml_team_2')

        prob_team_1 = ensemble_proba[i]
        prob_team_2 = 1 - prob_team_1

        all_games.append({
            'date': date,
            'game_id': game_id,
            'team_1': team_1,
            'team_2': team_2,
            'prob_team_1': round(prob_team_1, 4),
            'prob_team_2': round(prob_team_2, 4),
            'ml_avg_team_1': ml_avg_team_1,
            'ml_avg_team_2': ml_avg_team_2
        })

    if all_games:
        output_df = pl.DataFrame(all_games)
        # Sort by date and game_id
        output_df = output_df.sort(['date', 'game_id'])

        if target_date_yyyymmdd is None:
            target_date_yyyymmdd = get_todays_date_yyyymmdd()
        output_file = Path(__file__).parent / f"all_games_{target_date_yyyymmdd}.xlsx"
        output_df.write_excel(str(output_file))
        print(f"[+] Exported {len(all_games)} games with probabilities and ML averages to {output_file}\n")
    else:
        print("[*] No games to export\n")


def main(manual_date: str = None):
    print("\n")
    print("="*80)
    print("FUTURES REPORT - ML & OU BETTING RECOMMENDATIONS")
    print("="*80 + "\n")

    # Import ou_main to run data scraping first
    import ou_main as ou_module

    # Set the date to use (manual_date overrides today's date)
    if manual_date:
        # Validate date format (YYYY-MM-DD)
        try:
            datetime.strptime(manual_date, '%Y-%m-%d')
            target_date = manual_date
            target_date_yyyymmdd = manual_date.replace('-', '')
            print(f"[*] Running for manual date: {target_date}\n")
        except ValueError:
            print(f"[-] Invalid date format: {manual_date}. Expected YYYY-MM-DD")
            return
    else:
        target_date = get_todays_date()
        target_date_yyyymmdd = get_todays_date_yyyymmdd()

    # Step 0: Run ou_main.main() to scrape all data and get predictions
    print("STEP 0: Running OU pipeline to scrape data and make predictions")
    print("-"*80 + "\n")

    features_df, predictions_df = ou_module.main(target_date=target_date)

    if features_df is None or predictions_df is None:
        print("[-] OU pipeline failed")
        return

    print("\n[+] OU pipeline complete - data has been scraped and prepared")
    print(f"[DEBUG] features_df shape: {features_df.shape}")

    # Define allowed bookmakers
    allowed_bookmakers = {'BetOnline.ag', 'Bovada', 'MyBookie.ag', 'betonlineag'}

    # We already have todays_games_df and features_df from the OU pipeline above
    # Step 1: Prepare ML features from the OU features_df
    print("STEP 1: Preparing ML Features for Moneyline Model")
    print("-"*80 + "\n")

    # Use the features_df generated during OU pipeline for ML predictions
    # (it has all the game data and generated features already)
    print(f"[+] Using {features_df.shape[0]} games with {features_df.shape[1]} feature columns")

    # DEBUG: Print all columns to check for moneyline features
    print(f"\n[DEBUG] All columns in features_df:")
    all_cols = features_df.columns
    print(f"Total columns: {len(all_cols)}")
    for i, col in enumerate(all_cols, 1):
        print(f"  {i:3}. {col}")

    # Check if any moneyline-related columns exist
    ml_cols = [col for col in all_cols if 'ml_' in col.lower() or 'moneyline' in col.lower()]
    if ml_cols:
        print(f"\n[!] Found {len(ml_cols)} moneyline-related columns:")
        for col in ml_cols:
            print(f"    - {col}")
    else:
        print(f"\n[+] No moneyline-related columns found in features_df")

    # Drop rows with missing odds
    before = len(features_df)
    features_df = features_df.filter(
        pl.col('avg_ml_team_1').is_not_null() &
        pl.col('avg_ml_team_2').is_not_null()
    )
    dropped = before - len(features_df)
    if dropped > 0:
        print(f"[*] Dropped {dropped} games missing odds")

    # Keep original features_df for later export (after filtering out games with missing odds)
    features_df_original = features_df.clone()

    # Analyze nulls after dropping metadata
    feature_cols = identify_feature_columns(features_df)
    print(f"[*] Identified {len(feature_cols)} numeric features (excluding metadata)\n")

    # Check for nulls in actual feature columns
    feature_subset = features_df.select(feature_cols)
    null_counts = feature_subset.null_count().to_dicts()[0]
    null_cols = {col: count for col, count in null_counts.items() if count > 0}

    if null_cols:
        print("[!] Columns with null values (after dropping metadata):")
        null_pcts = [(col, count/len(features_df)*100) for col, count in null_cols.items()]
        null_pcts.sort(key=lambda x: x[1], reverse=True)
        print(f"    Total columns with nulls: {len(null_pcts)}\n")
        for col, pct in null_pcts[:15]:
            print(f"    {col:40} {pct:6.1f}%")
        if len(null_pcts) > 15:
            print(f"    ... and {len(null_pcts) - 15} more columns with nulls")
        print()
    else:
        print("[+] No null values in feature columns\n")

    # Export features to Excel
    print("STEP 1.5: Exporting All Generated Features to Excel")
    print("-"*80 + "\n")
    export_features_to_excel(features_df, target_date_yyyymmdd)

    # Load ML models (Ensemble: 18% LGB + 82% XGB)
    print("STEP 2: Loading Moneyline Models (Ensemble: 18% LGB + 82% XGB)")
    print("-"*80 + "\n")
    lgb_model, xgb_model = load_ml_models()
    if lgb_model is None or xgb_model is None:
        print("[-] Could not load models\n")
        return

    print("STEP 3: Getting ML Predictions (Ensemble: 18% LGB + 82% XGB)")
    print("-"*80 + "\n")

    # Step 3b: Extract OU bets from predictions_df using V3.0 comprehensive betting rules
    print("STEP 3b: Extracting OU Valid Bets (V3.0 Comprehensive Rules - 26 rules)")
    print("-"*80 + "\n")

    # Use the updated get_ou_bets function with V3.0 betting rules
    ou_bets = get_ou_bets(predictions_df, allowed_bookmakers=allowed_bookmakers)

    print(f"[+] Found {len(ou_bets)} OU bets meeting V3.0 Rules criteria (Expected ROI: +10.13%)\n")

    # Step 3c: Get ML bets with Clay's Optimal Rules
    print("STEP 3c: Extracting ML Valid Bets (Clay's Optimal Rules)")
    print("-"*80 + "\n")

    team_mappings = load_team_mappings()
    ml_bets = get_ml_bets(features_df, lgb_model, xgb_model, allowed_bookmakers, team_mappings=team_mappings)

    # Export ALL predictions (before filtering by EV/probability thresholds)
    print("\nSTEP 3d: Exporting All Predictions (ML and OU)")
    print("-"*80 + "\n")
    export_all_predictions(features_df, lgb_model, xgb_model, predictions_df, target_date_yyyymmdd)

    # Export valid bets to Excel with date in filename
    print("Exporting valid bets to Excel...")

    output_file = Path(__file__).parent / f"bets_{target_date_yyyymmdd}.xlsx"

    # Combine ML and OU bets
    all_bets = []

    # Add ML bets
    for bet in ml_bets:
        all_bets.append({
            'type': 'ML',
            'date': bet['date'],
            'game_id': bet['game_id'],
            'bet_on': bet['team'],
            'opponent': bet['opponent'],
            'odds': bet['odds'],
            'decimal': round(bet['decimal'], 3),
            'win_prob': round(bet['win_prob'], 4),
            'ev_percent': round(bet['ev_percent'], 2),
            'bookmaker': bet['bookmaker']
        })

    # Add OU bets
    for bet in ou_bets:
        point_key = 'over_point' if bet.get('bet') == 'OVER' else 'under_point'
        point_value = bet.get(point_key, bet.get('over_point', ''))

        all_bets.append({
            'type': 'OU',
            'date': bet['date'],
            'game_id': bet['game_id'],
            'bet_on': bet.get('bet', 'OVER'),
            'opponent': '',
            'odds': '',
            'decimal': '',
            'win_prob': '',
            'ev_percent': round(bet['difference'], 2),
            'bookmaker': bet['bookmaker'],
            'ensemble': round(bet['ensemble'], 2),
            'ou_point': round(point_value, 2) if point_value else '',
            'confidence': round(bet.get('confidence', 0), 4),
            'good_bet_score': round(bet.get('good_bet_score', 0), 4)
        })

    if all_bets:
        output_df = pl.DataFrame(all_bets)
        output_df.write_excel(str(output_file))
        print(f"[+] Exported {len(all_bets)} bets to {output_file}\n")
    else:
        print(f"[*] No valid bets found to export\n")

    # Done - exported to Excel
    print("\n" + "="*80)
    print("DONE")
    print("="*80 + "\n")

    if True:  # Print results
        # Print ML bets
        if ml_bets:
            print(f"MONEYLINE BETS (Clay's Optimal Rules, Bookmakers: {', '.join(sorted(allowed_bookmakers))})")
            print("-"*80 + "\n")

            # Sort by EV descending
            ml_bets_sorted = sorted(ml_bets, key=lambda x: x['ev_percent'], reverse=True)

            for bet in ml_bets_sorted:
                print(f"Date: {bet['date']}")
                print(f"Game ID: {bet['game_id']}")
                print(f"BET ON: {bet['team']}")
                print(f"  Odds: {bet['odds']:+d}  |  Decimal: {bet['decimal']:.3f}")
                print(f"  Win Prob: {bet['win_prob']*100:.2f}%")
                print(f"  EV: {bet['ev_percent']:.2f}%")
                print(f"  Bookmaker: {bet['bookmaker']}")
                print()

        # Print OU bets
        if ou_bets:
            print(f"\nOVER/UNDER BETS (V3.0 Comprehensive Rules - 26 rules)")
            print("-"*80 + "\n")

            # Sort by good_bet_score descending
            ou_bets_sorted = sorted(ou_bets, key=lambda x: x.get('good_bet_score', 0), reverse=True)

            for bet in ou_bets_sorted:
                point_key = 'over_point' if bet.get('bet') == 'OVER' else 'under_point'
                point_val = bet.get(point_key, bet.get('over_point', 'N/A'))
                print(f"Date: {bet['date']}")
                print(f"Game ID: {bet['game_id']}")
                print(f"  Ensemble: {bet['ensemble']}  |  Line: {point_val}")
                print(f"  Difference: {bet['difference']} (BET: {bet['bet']})")
                print(f"  Ensemble Confidence: {bet.get('confidence', 0):.4f}")
                print(f"  Good Bet Score: {bet.get('good_bet_score', 0):.4f}")
                print(f"  Bookmaker: {bet['bookmaker']}")
                print()

        # Summary
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80 + "\n")
        print(f"Total ML Bets (Clay's Optimal Rules): {len(ml_bets)}")
        print(f"Total OU Bets (V3.0 Comprehensive Rules): {len(ou_bets)}")
        print(f"Total Bets: {len(ml_bets) + len(ou_bets)}\n")

    else:
        print("No bets meet criteria")
        print("  ML: Clay's Optimal Rules (specific probability and EV ranges)")
        print("  OU: V3.0 Comprehensive Rules (26 rules using good_bet_score + ensemble_confidence + difference)\n")

    print("="*80 + "\n")

    # Export all games with their probabilities and moneyline averages
    print("\nSTEP 4: Exporting All Games with Probabilities and ML Averages")
    print("-"*80 + "\n")
    export_all_games_with_odds(features_df_original, lgb_model, xgb_model, target_date_yyyymmdd)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Run betting recommendations for a specific date')
    parser.add_argument('--date', type=str, default=None, help='Date to run for in YYYY-MM-DD format (default: today)')
    args = parser.parse_args()

    main(manual_date=args.date)
