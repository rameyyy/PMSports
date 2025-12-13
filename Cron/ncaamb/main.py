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
from scrapes.gamehistory import scrape_game_history
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
    df_pd[feature_cols] = df_pd[feature_cols].astype('float64').fillna(0)

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

            # Map odds team nasmes
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

    def matches_over_rule(good_bet_score: float, ensemble_confidence: float, difference: float) -> str:
        """Check if parameters match any OVER betting rule (V3.0 - 13 rules). Returns rule name or None."""

        # 3D RULES (good_bet_score + ensemble_confidence + difference)
        # O1: 31 bets | 64.5% win | +23.79% ROI
        if 0.8 <= good_bet_score < 0.9 and 0.7 <= ensemble_confidence < 0.8 and 4 <= difference < 5:
            return "O1"
        # O2: 45 bets | 55.6% win | +6.72% ROI
        if 0.8 <= good_bet_score < 0.9 and 0.6 <= ensemble_confidence < 0.7 and 2 <= difference < 3:
            return "O2"
        # O3: 112 bets | 54.5% win | +4.48% ROI
        if 0.7 <= good_bet_score < 0.8 and 0.7 <= ensemble_confidence < 0.8 and 2 <= difference < 3:
            return "O3"

        # 2D RULES - good_bet_score + difference
        # O4: 46 bets | 65.2% win | +25.27% ROI ⭐⭐
        if 0.80 <= good_bet_score < 0.85 and 3.5 <= difference < 4.0:
            return "O4"
        # O5: 74 bets | 59.5% win | +13.89% ROI
        if 0.55 <= good_bet_score < 0.60 and 1.5 <= difference < 2.0:
            return "O5"
        # O6: 44 bets | 59.1% win | +13.32% ROI
        if 0.50 <= good_bet_score < 0.55 and 1.5 <= difference < 2.0:
            return "O6"
        # O7: 76 bets | 55.3% win | +5.81% ROI
        if 0.60 <= good_bet_score < 0.65 and 1.0 <= difference < 1.5:
            return "O7"
        # O8: 96 bets | 54.2% win | +4.05% ROI
        if 0.65 <= good_bet_score < 0.70 and 1.5 <= difference < 2.0:
            return "O8"
        # O9: 94 bets | 54.3% win | +4.01% ROI
        if 0.75 <= good_bet_score < 0.80 and 2.5 <= difference < 3.0:
            return "O9"

        # 2D RULES - good_bet_score + ensemble_confidence
        # O10: 33 bets | 66.7% win | +27.48% ROI ⭐⭐⭐
        if 0.55 <= good_bet_score < 0.60 and 0.50 <= ensemble_confidence < 0.55:
            return "O10"
        # O11: 44 bets | 63.6% win | +22.33% ROI ⭐
        if 0.80 <= good_bet_score < 0.85 and 0.65 <= ensemble_confidence < 0.70:
            return "O11"
        # O12: 65 bets | 55.4% win | +6.39% ROI
        if 0.50 <= good_bet_score < 0.55 and 0.50 <= ensemble_confidence < 0.55:
            return "O12"
        # O13: 114 bets | 55.3% win | +6.17% ROI
        if 0.65 <= good_bet_score < 0.70 and 0.60 <= ensemble_confidence < 0.65:
            return "O13"

        return None

    def matches_under_rule(good_bet_score: float, ensemble_confidence: float, difference: float) -> str:
        """Check if parameters match any UNDER betting rule (V3.0 - 13 rules). Returns rule name or None."""

        # 2D RULES - good_bet_score + difference
        # U1: 36 bets | 66.7% win | +27.80% ROI ⭐⭐⭐
        if 0.40 <= good_bet_score < 0.45 and -0.5 <= difference < 0.0:
            return "U1"
        # U2: 48 bets | 64.6% win | +23.96% ROI ⭐⭐
        if 0.40 <= good_bet_score < 0.45 and -1.0 <= difference < 0.0:
            return "U2"
        # U3: 47 bets | 62.8% win | +18.43% ROI ⭐
        if 0.25 <= good_bet_score < 0.30 and 0.0 <= difference < 0.5:
            return "U3"
        # U4: 54 bets | 61.1% win | +17.21% ROI ⭐
        if 0.25 <= good_bet_score < 0.30 and 0.0 <= difference < 1.0:
            return "U4"
        # U5: 44 bets | 61.4% win | +17.89% ROI ⭐
        if 0.00 <= good_bet_score < 0.05 and -4.0 <= difference < -3.0:
            return "U5"
        # U6: 53 bets | 60.4% win | +15.68% ROI
        if 0.20 <= good_bet_score < 0.25 and -0.5 <= difference < 0.0:
            return "U6"
        # U7: 41 bets | 56.1% win | +8.14% ROI
        if 0.15 <= good_bet_score < 0.20 and -1.0 <= difference < -0.5:
            return "U7"
        # U8: 72 bets | 54.2% win | +4.00% ROI
        if 0.25 <= good_bet_score < 0.30 and -1.0 <= difference < -0.5:
            return "U8"

        # 2D RULES - good_bet_score + ensemble_confidence
        # U9: 36 bets | 66.7% win | +27.82% ROI ⭐⭐⭐
        if 0.25 <= good_bet_score < 0.30 and 0.50 <= ensemble_confidence < 0.55:
            return "U9"
        # U10: 50 bets | 64.0% win | +22.92% ROI ⭐⭐
        if 0.40 <= good_bet_score < 0.45 and 0.45 <= ensemble_confidence < 0.50:
            return "U10"
        # U11: 54 bets | 61.1% win | +17.95% ROI ⭐
        if 0.15 <= good_bet_score < 0.20 and 0.40 <= ensemble_confidence < 0.45:
            return "U11"
        # U12: 37 bets | 56.8% win | +9.24% ROI
        if 0.00 <= good_bet_score < 0.05 and 0.25 <= ensemble_confidence < 0.30:
            return "U12"
        # U13: 65 bets | 56.9% win | +9.16% ROI
        if 0.20 <= good_bet_score < 0.25 and 0.45 <= ensemble_confidence < 0.50:
            return "U13"

        return None

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
                over_rule = matches_over_rule(good_bets_confidence, ensemble_confidence, difference)
                if over_rule:
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
                        'bet': 'OVER',
                        'bet_rule': over_rule
                    })

                # Check UNDER rules (V3.0 - uses good_bet_score, ensemble_confidence, difference)
                else:
                    under_rule = matches_under_rule(good_bets_confidence, ensemble_confidence, difference)
                    if under_rule:
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
                            'bet': 'UNDER',
                            'bet_rule': under_rule
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


def calculate_implied_prob(american_odds: int) -> float:
    """Calculate implied probability from American odds"""
    if american_odds >= 0:
        return 100 / (american_odds + 100)
    else:
        return abs(american_odds) / (abs(american_odds) + 100)


def dvig_probabilities(prob_over: float, prob_under: float) -> tuple:
    """Remove vig from implied probabilities by normalizing"""
    total = prob_over + prob_under
    if total == 0:
        return 0.5, 0.5
    return prob_over / total, prob_under / total


def extract_season_from_game_id(game_id: str) -> int:
    """Extract season from game_id (YYYYMMDD format). 10/2024-04/2025 = 2025"""
    try:
        date_str = str(game_id)[:8]  # Get YYYYMMDD part
        year = int(date_str[:4])
        month = int(date_str[4:6])

        # If month is October (10) or later, it's the next season
        # e.g., October 2024 = season 2025
        if month >= 10:
            return year + 1
        else:
            return year
    except:
        return None


def insert_moneyline_bets(features_df: pl.DataFrame, lgb_proba, xgb_proba, ml_bets: list, odds_dict: dict, allowed_bookmakers: set, team_mappings: dict, target_date_yyyymmdd: str):
    """
    Insert all moneyline game predictions into database
    Stores every game with all probabilities, regardless of whether it matches betting rules
    """
    try:
        conn = sqlconn.create_connection()
        if not conn:
            print("[-] Failed to connect to database for moneyline insertion")
            return

        cursor = conn.cursor()

        # Create ensemble proba
        ensemble_proba = 0.18 * lgb_proba + 0.82 * xgb_proba

        # Create set of game_ids that matched betting rules
        bet_game_ids = set()
        for bet in ml_bets:
            key = (bet['game_id'], bet['team'])
            bet_game_ids.add(key)

        print(f"[*] Debug: odds_dict has {len(odds_dict)} game entries")
        print(f"[*] Debug: features_df has {len(features_df)} games")
        print(f"[*] Debug: ml_bets has {len(ml_bets)} bets")

        inserted = 0

        for i, game_row in enumerate(features_df.iter_rows(named=True)):
            game_id = game_row.get('game_id')
            team_1 = game_row.get('team_1')
            team_2 = game_row.get('team_2')
            date_str = game_row.get('date')

            if not game_id or not team_1 or not team_2:
                continue

            # Get probabilities for this game
            lgb_prob_t1 = float(lgb_proba[i])
            xgb_prob_t1 = float(xgb_proba[i])
            ensemble_prob_t1 = float(ensemble_proba[i])
            ensemble_prob_t2 = 1.0 - ensemble_prob_t1
            xgb_prob_t2 = 1.0 - xgb_prob_t1
            lgb_prob_t2 = 1.0 - lgb_prob_t1

            # Get odds for this game
            all_odds = odds_dict.get(game_id, [])

            # If odds_dict is empty or doesn't have this game, query database directly
            if not all_odds and len(odds_dict) == 0:
                try:
                    query = f"""
                        SELECT game_id, bookmaker, home_team, away_team, ml_home, ml_away
                        FROM odds
                        WHERE game_id = %s
                        ORDER BY bookmaker
                    """
                    results = sqlconn.fetch(conn, query, (game_id,))
                    for row in results:
                        all_odds.append({
                            'bookmaker': row['bookmaker'],
                            'home_team': row['home_team'],
                            'away_team': row['away_team'],
                            'ml_home': row['ml_home'],
                            'ml_away': row['ml_away']
                        })
                except:
                    pass  # If query fails, just continue with empty all_odds

            # Extract game_date and season from game_id
            game_date_str = f"{game_id[:4]}-{game_id[4:6]}-{game_id[6:8]}"  # YYYY-MM-DD format
            season = extract_season_from_game_id(game_id)

            # Get average moneyline odds
            avg_ml_t1 = game_row.get('avg_ml_team_1')
            avg_ml_t2 = game_row.get('avg_ml_team_2')

            # Calculate implied probabilities from average moneyline
            if avg_ml_t1 is not None and avg_ml_t2 is not None:
                impl_prob_t1_vig = calculate_implied_prob(int(avg_ml_t1))
                impl_prob_t2_vig = calculate_implied_prob(int(avg_ml_t2))
                impl_prob_t1_dvig, impl_prob_t2_dvig = dvig_probabilities(impl_prob_t1_vig, impl_prob_t2_vig)
            else:
                impl_prob_t1_vig = impl_prob_t2_vig = impl_prob_t1_dvig = impl_prob_t2_dvig = None

            # Determine which team is predicted to win (higher ensemble prob)
            team_predicted_to_win = team_1 if ensemble_prob_t1 >= 0.5 else team_2

            # Get odds for each team from all bookmakers and our allowed bookmakers
            best_odds_t1 = None
            best_book_t1 = None
            best_decimal_t1 = 0
            my_best_odds_t1 = None
            my_best_book_t1 = None

            best_odds_t2 = None
            best_book_t2 = None
            best_decimal_t2 = 0
            my_best_odds_t2 = None
            my_best_book_t2 = None

            for odds_rec in all_odds:
                bm_name = odds_rec.get('bookmaker')
                team_a_odds_name = odds_rec.get('home_team')
                team_b_odds_name = odds_rec.get('away_team')
                ml_team_a = odds_rec.get('ml_home')
                ml_team_b = odds_rec.get('ml_away')

                # Map odds team names using team_mappings
                team_a_mapped = team_mappings.get(team_a_odds_name, team_a_odds_name) if team_mappings else team_a_odds_name
                team_b_mapped = team_mappings.get(team_b_odds_name, team_b_odds_name) if team_mappings else team_b_odds_name

                # Track best odds overall
                if team_a_mapped == team_1 and ml_team_a is not None:
                    decimal_val = american_to_decimal(ml_team_a)
                    if decimal_val > best_decimal_t1:
                        best_odds_t1 = ml_team_a
                        best_book_t1 = bm_name
                        best_decimal_t1 = decimal_val

                    if bm_name in allowed_bookmakers:
                        my_decimal = american_to_decimal(ml_team_a)
                        if my_best_odds_t1 is None or my_decimal > american_to_decimal(my_best_odds_t1):
                            my_best_odds_t1 = ml_team_a
                            my_best_book_t1 = bm_name

                if team_b_mapped == team_1 and ml_team_b is not None:
                    decimal_val = american_to_decimal(ml_team_b)
                    if decimal_val > best_decimal_t1:
                        best_odds_t1 = ml_team_b
                        best_book_t1 = bm_name
                        best_decimal_t1 = decimal_val

                    if bm_name in allowed_bookmakers:
                        my_decimal = american_to_decimal(ml_team_b)
                        if my_best_odds_t1 is None or my_decimal > american_to_decimal(my_best_odds_t1):
                            my_best_odds_t1 = ml_team_b
                            my_best_book_t1 = bm_name

                # Track best odds for team 2
                if team_a_mapped == team_2 and ml_team_a is not None:
                    decimal_val = american_to_decimal(ml_team_a)
                    if decimal_val > best_decimal_t2:
                        best_odds_t2 = ml_team_a
                        best_book_t2 = bm_name
                        best_decimal_t2 = decimal_val

                    if bm_name in allowed_bookmakers:
                        my_decimal = american_to_decimal(ml_team_a)
                        if my_best_odds_t2 is None or my_decimal > american_to_decimal(my_best_odds_t2):
                            my_best_odds_t2 = ml_team_a
                            my_best_book_t2 = bm_name

                if team_b_mapped == team_2 and ml_team_b is not None:
                    decimal_val = american_to_decimal(ml_team_b)
                    if decimal_val > best_decimal_t2:
                        best_odds_t2 = ml_team_b
                        best_book_t2 = bm_name
                        best_decimal_t2 = decimal_val

                    if bm_name in allowed_bookmakers:
                        my_decimal = american_to_decimal(ml_team_b)
                        if my_best_odds_t2 is None or my_decimal > american_to_decimal(my_best_odds_t2):
                            my_best_odds_t2 = ml_team_b
                            my_best_book_t2 = bm_name

            # Calculate EVs
            best_ev_t1 = calculate_ev(ensemble_prob_t1, int(best_odds_t1)) * 100 if best_odds_t1 else None
            best_ev_t2 = calculate_ev(ensemble_prob_t2, int(best_odds_t2)) * 100 if best_odds_t2 else None
            my_best_ev_t1 = calculate_ev(ensemble_prob_t1, int(my_best_odds_t1)) * 100 if my_best_odds_t1 else None
            my_best_ev_t2 = calculate_ev(ensemble_prob_t2, int(my_best_odds_t2)) * 100 if my_best_odds_t2 else None

            # Determine betting rule, bet_on flag, and which team to bet on
            bet_rule = None
            bet_on = 0
            team_bet_on = None

            # Check if this bet matches ML rules (for team_1 or team_2)
            if (game_id, team_1) in bet_game_ids:
                bet_on = 1
                team_bet_on = team_1
                # Find the rule that matched for team_1
                for bet in ml_bets:
                    if bet['game_id'] == game_id and bet['team'] == team_1:
                        if 0.52 <= ensemble_prob_t1 < 0.57:
                            bet_rule = "PV"
                        elif 0.59 <= ensemble_prob_t1 < 0.60:
                            bet_rule = "HRT"
                        elif 0.74 <= ensemble_prob_t1 < 0.76:
                            bet_rule = "HG746"
                        elif 0.80 <= ensemble_prob_t1 < 0.84:
                            bet_rule = "HG804"
                        break
            elif (game_id, team_2) in bet_game_ids:
                bet_on = 1
                team_bet_on = team_2
                # Find the rule that matched for team_2
                for bet in ml_bets:
                    if bet['game_id'] == game_id and bet['team'] == team_2:
                        if 0.52 <= ensemble_prob_t2 < 0.57:
                            bet_rule = "PV"
                        elif 0.59 <= ensemble_prob_t2 < 0.60:
                            bet_rule = "HRT"
                        elif 0.74 <= ensemble_prob_t2 < 0.76:
                            bet_rule = "HG746"
                        elif 0.80 <= ensemble_prob_t2 < 0.84:
                            bet_rule = "HG804"
                        break

            # Insert for team_1
            if team_1 and game_id:
                insert_query = """
                    INSERT INTO moneyline (
                        game_id, team_1, team_2, team_predicted_to_win,
                        xgb_prob_team_1, xgb_prob_team_2,
                        gbm_prob_team_1, gbm_prob_team_2,
                        ensemble_prob_team_1, ensemble_prob_team_2,
                        best_ev_team_1, best_ev_team_2,
                        my_best_ev_team_1, my_best_ev_team_2,
                        best_book_team_1, best_book_odds_team_1,
                        best_book_team_2, best_book_odds_team_2,
                        my_best_book_team_1, my_best_book_odds_team_1,
                        my_best_book_team_2, my_best_book_odds_team_2,
                        implied_prob_team_1_with_vig, implied_prob_team_2_with_vig,
                        implied_prob_team_1_devigged, implied_prob_team_2_devigged,
                        bet_rule, bet_on, team_bet_on, game_date, season
                    ) VALUES (
                        %s, %s, %s, %s,
                        %s, %s, %s, %s,
                        %s, %s, %s, %s,
                        %s, %s, %s, %s,
                        %s, %s, %s, %s,
                        %s, %s, %s, %s,
                        %s, %s, %s, %s,
                        %s, %s, %s
                    )
                """

                values = (
                    game_id, team_1, team_2, team_predicted_to_win,
                    xgb_prob_t1, xgb_prob_t2,
                    lgb_prob_t1, lgb_prob_t2,
                    ensemble_prob_t1, ensemble_prob_t2,
                    best_ev_t1, best_ev_t2,
                    my_best_ev_t1, my_best_ev_t2,
                    best_book_t1, best_odds_t1,
                    best_book_t2, best_odds_t2,
                    my_best_book_t1, my_best_odds_t1,
                    my_best_book_t2, my_best_odds_t2,
                    impl_prob_t1_vig, impl_prob_t2_vig,
                    impl_prob_t1_dvig, impl_prob_t2_dvig,
                    bet_rule, bet_on, team_bet_on, game_date_str, season
                )

                try:
                    cursor.execute(insert_query, values)
                    inserted += 1
                except Exception as e:
                    print(f"  [!] Error inserting game {game_id}: {e}")

        conn.commit()
        cursor.close()
        conn.close()

        print(f"[+] Inserted {inserted} moneyline records into database\n")

    except Exception as e:
        print(f"[-] Error inserting moneyline bets: {e}\n")
        import traceback
        traceback.print_exc()


def get_yesterday_date():
    """Get yesterday's date in YYYY-MM-DD format"""
    from datetime import datetime, timedelta
    return (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')


def update_my_bankroll():
    """
    Update my_bankroll table for yesterday's bets using my_best_book_odds
    Dynamic wager sizing: bankroll * 0.85% per bet
    """
    from datetime import datetime, timedelta

    try:
        conn = sqlconn.create_connection()
        if not conn:
            print("[-] Failed to connect to database for my_bankroll update")
            return

        yesterday = get_yesterday_date()
        day_before = (datetime.strptime(yesterday, '%Y-%m-%d') - timedelta(days=1)).strftime('%Y-%m-%d')

        # Get previous day's bankroll to calculate wager size
        bankroll_query = "SELECT bankroll FROM my_bankroll WHERE date = %s"
        bankroll_result = sqlconn.fetch(conn, bankroll_query, (day_before,))

        if bankroll_result:
            previous_bankroll = float(bankroll_result[0].get('bankroll', 1800))
        else:
            previous_bankroll = 1800.0  # Default starting bankroll

        wager_per_bet = previous_bankroll * 0.0085  # 0.85% per bet

        # Get yesterday's OU bets with odds for profit recalculation
        ou_query = """
            SELECT bet_win_or_lose, bet_on_side,
                   my_best_book_odds_over, my_best_book_odds_under,
                   best_book_odds_over, best_book_odds_under,
                   COUNT(*) as count
            FROM overunder
            WHERE game_date = %s AND bet_on = 1
            GROUP BY bet_win_or_lose, bet_on_side, my_best_book_odds_over, my_best_book_odds_under,
                     best_book_odds_over, best_book_odds_under
        """
        ou_bets = sqlconn.fetch(conn, ou_query, (yesterday,))

        # Get yesterday's ML bets with odds for profit recalculation
        ml_query = """
            SELECT bet_win_or_lose,
                   my_best_book_odds_team_1, my_best_book_odds_team_2,
                   best_book_odds_team_1, best_book_odds_team_2,
                   COUNT(*) as count
            FROM moneyline
            WHERE game_date = %s AND bet_on = 1
            GROUP BY bet_win_or_lose, my_best_book_odds_team_1, my_best_book_odds_team_2,
                     best_book_odds_team_1, best_book_odds_team_2
        """
        ml_bets = sqlconn.fetch(conn, ml_query, (yesterday,))

        # Helper function to convert American odds to decimal
        def odds_to_decimal(american_odds):
            if american_odds is None:
                return 1.0
            odds_int = int(american_odds)
            if odds_int < 0:
                return (100 / abs(odds_int)) + 1
            else:
                return (odds_int / 100) + 1

        # Calculate totals
        ou_bets_count = sum(bet['count'] for bet in ou_bets) if ou_bets else 0
        ml_bets_count = sum(bet['count'] for bet in ml_bets) if ml_bets else 0
        bet_qty = ou_bets_count + ml_bets_count

        # Calculate OU stats
        ou_wins = 0
        ou_wagered = 0
        ou_profit = 0

        if ou_bets:
            for bet in ou_bets:
                count = bet['count']
                result = bet['bet_win_or_lose']

                if result == 'WIN':
                    ou_wins += count
                    # Determine which odds to use, fall back to best_book_odds if my_best_book_odds is NULL
                    if bet['bet_on_side'] == 'OVER':
                        odds = bet['my_best_book_odds_over'] if bet['my_best_book_odds_over'] is not None else bet.get('best_book_odds_over')
                    else:
                        odds = bet['my_best_book_odds_under'] if bet['my_best_book_odds_under'] is not None else bet.get('best_book_odds_under')

                    decimal = odds_to_decimal(odds)
                    for _ in range(count):
                        ou_wagered += wager_per_bet
                        ou_profit += round(wager_per_bet * (decimal - 1), 2)
                else:  # LOSE
                    for _ in range(count):
                        ou_wagered += wager_per_bet
                        ou_profit -= wager_per_bet

        # Calculate ML stats
        ml_wins = 0
        ml_wagered = 0
        ml_profit = 0

        if ml_bets:
            # Query with team info to use correct odds based on which team was bet on
            ml_detail_query = """
                SELECT bet_win_or_lose, team_1, team_2, team_predicted_to_win,
                       my_best_book_odds_team_1, my_best_book_odds_team_2,
                       best_book_odds_team_1, best_book_odds_team_2, COUNT(*) as count
                FROM moneyline
                WHERE game_date = %s AND bet_on = 1
                GROUP BY bet_win_or_lose, team_1, team_2, team_predicted_to_win,
                         my_best_book_odds_team_1, my_best_book_odds_team_2,
                         best_book_odds_team_1, best_book_odds_team_2
            """
            ml_bets_detail = sqlconn.fetch(conn, ml_detail_query, (yesterday,))

            if ml_bets_detail:
                for bet in ml_bets_detail:
                    count = bet['count']
                    result = bet['bet_win_or_lose']

                    if result == 'WIN':
                        ml_wins += count
                        # Use odds for the team that was predicted to win (and thus bet on)
                        # Fall back to best_book_odds if my_best_book_odds is NULL
                        if bet['team_predicted_to_win'] == bet['team_1']:
                            odds = bet['my_best_book_odds_team_1'] if bet['my_best_book_odds_team_1'] is not None else bet['best_book_odds_team_1']
                        else:
                            odds = bet['my_best_book_odds_team_2'] if bet['my_best_book_odds_team_2'] is not None else bet['best_book_odds_team_2']

                        decimal = odds_to_decimal(odds)
                        for _ in range(count):
                            ml_wagered += wager_per_bet
                            ml_profit += round(wager_per_bet * (decimal - 1), 2)
                    else:  # LOSE
                        for _ in range(count):
                            ml_wagered += wager_per_bet
                            ml_profit -= wager_per_bet

        # Calculate totals
        total_wagered = ou_wagered + ml_wagered
        net_profit_loss = ou_profit + ml_profit

        # Calculate ROI
        roi = (net_profit_loss / total_wagered * 100) if total_wagered > 0 else 0
        ml_roi = (ml_profit / ml_wagered * 100) if ml_wagered > 0 else 0
        ou_roi = (ou_profit / ou_wagered * 100) if ou_wagered > 0 else 0

        # Calculate new bankroll and cumulative profit
        current_bankroll = previous_bankroll + net_profit_loss

        # Get previous cumulative profit
        prev_cum_query = "SELECT cumulative_profit FROM my_bankroll WHERE date = %s"
        prev_cum_result = sqlconn.fetch(conn, prev_cum_query, (day_before,))
        if prev_cum_result and prev_cum_result[0].get('cumulative_profit') is not None:
            cumulative_profit = float(prev_cum_result[0]['cumulative_profit']) + net_profit_loss
        else:
            cumulative_profit = net_profit_loss

        # Update or insert my_bankroll
        cursor = conn.cursor()
        upsert_query = """
            INSERT INTO my_bankroll (
                date, bankroll, net_profit_loss, roi,
                bet_qty, ml_bets, ou_bets, spread_bets,
                total_wagered, ml_wagered, ou_wagered, spread_wagered,
                ml_net_profit_loss, ou_net_profit_loss, spread_net_profit_loss,
                ml_roi, ou_roi, spread_roi,
                ml_wins, ou_wins, spread_wins,
                cumulative_profit, season
            ) VALUES (
                %s, %s, %s, %s,
                %s, %s, %s, %s,
                %s, %s, %s, %s,
                %s, %s, %s,
                %s, %s, %s,
                %s, %s, %s,
                %s, %s
            )
            ON DUPLICATE KEY UPDATE
                bankroll = VALUES(bankroll),
                net_profit_loss = VALUES(net_profit_loss),
                roi = VALUES(roi),
                bet_qty = VALUES(bet_qty),
                ml_bets = VALUES(ml_bets),
                ou_bets = VALUES(ou_bets),
                total_wagered = VALUES(total_wagered),
                ml_wagered = VALUES(ml_wagered),
                ou_wagered = VALUES(ou_wagered),
                ml_net_profit_loss = VALUES(ml_net_profit_loss),
                ou_net_profit_loss = VALUES(ou_net_profit_loss),
                ml_roi = VALUES(ml_roi),
                ou_roi = VALUES(ou_roi),
                ml_wins = VALUES(ml_wins),
                ou_wins = VALUES(ou_wins),
                cumulative_profit = VALUES(cumulative_profit),
                updated_at = NOW()
        """

        season = int(yesterday[:4]) + 1 if int(yesterday[5:7]) >= 10 else int(yesterday[:4])

        cursor.execute(upsert_query, (
            yesterday, round(current_bankroll, 2), round(net_profit_loss, 2), round(roi, 2),
            bet_qty, ml_bets_count, ou_bets_count, 0,
            round(total_wagered, 2), round(ml_wagered, 2), round(ou_wagered, 2), 0,
            round(ml_profit, 2), round(ou_profit, 2), 0,
            round(ml_roi, 2), round(ou_roi, 2), 0,
            ml_wins, ou_wins, 0,
            round(cumulative_profit, 2), season
        ))
        conn.commit()
        cursor.close()
        print(f"[+] Updated my_bankroll for {yesterday} (wager: ${wager_per_bet:.2f}/bet from ${previous_bankroll:.2f} bankroll)\n")

        conn.close()

    except Exception as e:
        print(f"[-] Error updating my_bankroll: {e}")
        import traceback
        traceback.print_exc()


def cleanup_cancelled_games():
    """
    Delete games from OU and ML tables that still have NULL predictions
    (indicating they were probably cancelled) if they are at least 7 days old.
    Only runs after predictions should have been updated.
    """
    from datetime import datetime, timedelta

    try:
        conn = sqlconn.create_connection()
        if not conn:
            print("[-] Failed to connect to database for cleanup")
            return

        # Calculate cutoff date (7 days ago)
        cutoff_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')

        cursor = conn.cursor()

        # Delete OU games with NULL ensemble_pred that are 7+ days old
        ou_delete_query = """
            DELETE FROM overunder
            WHERE ensemble_pred IS NULL
            AND game_date <= %s
        """

        cursor.execute(ou_delete_query, (cutoff_date,))
        ou_deleted = cursor.rowcount

        # Delete ML games with NULL ensemble_prob_team_1 that are 7+ days old
        ml_delete_query = """
            DELETE FROM moneyline
            WHERE ensemble_prob_team_1 IS NULL
            AND game_date <= %s
        """

        cursor.execute(ml_delete_query, (cutoff_date,))
        ml_deleted = cursor.rowcount

        conn.commit()
        cursor.close()

        if ou_deleted > 0 or ml_deleted > 0:
            print(f"[+] Cleanup: Deleted {ou_deleted} OU games and {ml_deleted} ML games with missing predictions (7+ days old)\n")
        else:
            print(f"[+] Cleanup: No cancelled games found\n")

        conn.close()

    except Exception as e:
        print(f"[-] Error during cleanup: {e}")
        import traceback
        traceback.print_exc()


def update_bankroll():
    """
    Update bankroll table for yesterday's bets using best_book_odds
    Dynamic wager sizing: bankroll * 1% total pool distributed across all bets
    """
    from datetime import datetime, timedelta

    try:
        conn = sqlconn.create_connection()
        if not conn:
            print("[-] Failed to connect to database for bankroll update")
            return

        yesterday = get_yesterday_date()
        day_before = (datetime.strptime(yesterday, '%Y-%m-%d') - timedelta(days=1)).strftime('%Y-%m-%d')

        # Get previous day's bankroll to calculate wager size
        bankroll_query = "SELECT bankroll FROM bankroll WHERE date = %s"
        bankroll_result = sqlconn.fetch(conn, bankroll_query, (day_before,))

        if bankroll_result:
            previous_bankroll = float(bankroll_result[0].get('bankroll', 10000))
        else:
            previous_bankroll = 10000.0  # Default starting bankroll

        # Get yesterday's OU bets with odds for profit recalculation
        ou_query = """
            SELECT bet_win_or_lose, bet_on_side,
                   best_book_odds_over, best_book_odds_under,
                   COUNT(*) as count
            FROM overunder
            WHERE game_date = %s AND bet_on = 1
            GROUP BY bet_win_or_lose, bet_on_side, best_book_odds_over, best_book_odds_under
        """
        ou_bets = sqlconn.fetch(conn, ou_query, (yesterday,))

        # Get yesterday's ML bets with odds for profit recalculation
        ml_query = """
            SELECT bet_win_or_lose,
                   best_book_odds_team_1, best_book_odds_team_2,
                   COUNT(*) as count
            FROM moneyline
            WHERE game_date = %s AND bet_on = 1
            GROUP BY bet_win_or_lose, best_book_odds_team_1, best_book_odds_team_2
        """
        ml_bets = sqlconn.fetch(conn, ml_query, (yesterday,))

        # Helper function to convert American odds to decimal
        def odds_to_decimal(american_odds):
            if american_odds is None:
                return 1.0
            odds_int = int(american_odds)
            if odds_int < 0:
                return (100 / abs(odds_int)) + 1
            else:
                return (odds_int / 100) + 1

        # Calculate bet counts
        ou_bets_count = sum(bet['count'] for bet in ou_bets) if ou_bets else 0
        ml_bets_count = sum(bet['count'] for bet in ml_bets) if ml_bets else 0
        total_bet_count = ou_bets_count + ml_bets_count

        # Calculate wager per bet: 0.85% of previous bankroll per bet
        wager_per_bet = previous_bankroll * 0.0085

        # Calculate OU stats
        ou_wins = 0
        ou_wagered = 0
        ou_profit = 0

        if ou_bets:
            for bet in ou_bets:
                count = bet['count']
                result = bet['bet_win_or_lose']

                if result == 'WIN':
                    ou_wins += count
                    # Determine which odds to use
                    if bet['bet_on_side'] == 'OVER':
                        odds = bet['best_book_odds_over']
                    else:
                        odds = bet['best_book_odds_under']

                    decimal = odds_to_decimal(odds)
                    for _ in range(count):
                        ou_wagered += wager_per_bet
                        ou_profit += round(wager_per_bet * (decimal - 1), 2)
                else:  # LOSE
                    for _ in range(count):
                        ou_wagered += wager_per_bet
                        ou_profit -= wager_per_bet

        # Calculate ML stats
        ml_wins = 0
        ml_wagered = 0
        ml_profit = 0

        if ml_bets:
            ml_detail_query = """
                SELECT bet_win_or_lose, team_1, team_2, team_predicted_to_win,
                       best_book_odds_team_1, best_book_odds_team_2,
                       my_best_book_odds_team_1, my_best_book_odds_team_2, COUNT(*) as count
                FROM moneyline
                WHERE game_date = %s AND bet_on = 1
                GROUP BY bet_win_or_lose, team_1, team_2, team_predicted_to_win,
                         best_book_odds_team_1, best_book_odds_team_2,
                         my_best_book_odds_team_1, my_best_book_odds_team_2
            """
            ml_bets_detail = sqlconn.fetch(conn, ml_detail_query, (yesterday,))

            if ml_bets_detail:
                for bet in ml_bets_detail:
                    count = bet['count']
                    result = bet['bet_win_or_lose']

                    if result == 'WIN':
                        ml_wins += count
                        # Use odds for the team that was predicted to win (and thus bet on)
                        # Fall back to my_best_book_odds if best_book_odds is NULL
                        if bet['team_predicted_to_win'] == bet['team_1']:
                            odds = bet['best_book_odds_team_1'] if bet['best_book_odds_team_1'] is not None else bet['my_best_book_odds_team_1']
                        else:
                            odds = bet['best_book_odds_team_2'] if bet['best_book_odds_team_2'] is not None else bet['my_best_book_odds_team_2']

                        decimal = odds_to_decimal(odds)
                        for _ in range(count):
                            ml_wagered += wager_per_bet
                            ml_profit += round(wager_per_bet * (decimal - 1), 2)
                    else:  # LOSE
                        for _ in range(count):
                            ml_wagered += wager_per_bet
                            ml_profit -= wager_per_bet

        # Calculate totals
        total_wagered = ou_wagered + ml_wagered
        net_profit_loss = ou_profit + ml_profit

        # Calculate ROI
        roi = (net_profit_loss / total_wagered * 100) if total_wagered > 0 else 0
        ml_roi = (ml_profit / ml_wagered * 100) if ml_wagered > 0 else 0
        ou_roi = (ou_profit / ou_wagered * 100) if ou_wagered > 0 else 0

        # Calculate new bankroll and cumulative profit
        current_bankroll = previous_bankroll + net_profit_loss

        # Get previous cumulative profit
        prev_cum_query = "SELECT cumulative_profit FROM bankroll WHERE date = %s"
        prev_cum_result = sqlconn.fetch(conn, prev_cum_query, (day_before,))
        if prev_cum_result and prev_cum_result[0].get('cumulative_profit') is not None:
            cumulative_profit = float(prev_cum_result[0]['cumulative_profit']) + net_profit_loss
        else:
            cumulative_profit = net_profit_loss

        # Update or insert bankroll
        cursor = conn.cursor()
        upsert_query = """
            INSERT INTO bankroll (
                date, bankroll, net_profit_loss, roi,
                bet_qty, ml_bets, ou_bets, spread_bets,
                total_wagered, ml_wagered, ou_wagered, spread_wagered,
                ml_net_profit_loss, ou_net_profit_loss, spread_net_profit_loss,
                ml_roi, ou_roi, spread_roi,
                ml_wins, ou_wins, spread_wins,
                cumulative_profit, season
            ) VALUES (
                %s, %s, %s, %s,
                %s, %s, %s, %s,
                %s, %s, %s, %s,
                %s, %s, %s,
                %s, %s, %s,
                %s, %s, %s,
                %s, %s
            )
            ON DUPLICATE KEY UPDATE
                bankroll = VALUES(bankroll),
                net_profit_loss = VALUES(net_profit_loss),
                roi = VALUES(roi),
                bet_qty = VALUES(bet_qty),
                ml_bets = VALUES(ml_bets),
                ou_bets = VALUES(ou_bets),
                total_wagered = VALUES(total_wagered),
                ml_wagered = VALUES(ml_wagered),
                ou_wagered = VALUES(ou_wagered),
                ml_net_profit_loss = VALUES(ml_net_profit_loss),
                ou_net_profit_loss = VALUES(ou_net_profit_loss),
                ml_roi = VALUES(ml_roi),
                ou_roi = VALUES(ou_roi),
                ml_wins = VALUES(ml_wins),
                ou_wins = VALUES(ou_wins),
                cumulative_profit = VALUES(cumulative_profit),
                updated_at = NOW()
        """

        season = int(yesterday[:4]) + 1 if int(yesterday[5:7]) >= 10 else int(yesterday[:4])

        cursor.execute(upsert_query, (
            yesterday, round(current_bankroll, 2), round(net_profit_loss, 2), round(roi, 2),
            total_bet_count, ml_bets_count, ou_bets_count, 0,
            round(total_wagered, 2), round(ml_wagered, 2), round(ou_wagered, 2), 0,
            round(ml_profit, 2), round(ou_profit, 2), 0,
            round(ml_roi, 2), round(ou_roi, 2), 0,
            ml_wins, ou_wins, 0,
            round(cumulative_profit, 2), season
        ))
        conn.commit()
        cursor.close()
        print(f"[+] Updated bankroll for {yesterday} (wager: ${wager_per_bet:.2f}/bet = 1% of ${previous_bankroll:.2f})\n")

        conn.close()

    except Exception as e:
        print(f"[-] Error updating bankroll: {e}")
        import traceback
        traceback.print_exc()


def update_overunder_outcomes():
    """
    Update overunder table with actual game outcomes for any games where actual_total is NULL
    Scrapes game history to get scores and calculates outcomes
    """
    try:
        conn = sqlconn.create_connection()
        if not conn:
            print("[-] Failed to connect to database for OU outcome update")
            return 0

        cursor = conn.cursor()

        # Find all games in overunder table where actual_total is NULL (game hasn't finished)
        query = """
            SELECT ou.id, ou.game_id, ou.team_1, ou.team_2, ou.over_point, ou.bet_on_side, ou.season
            FROM overunder ou
            WHERE ou.actual_total IS NULL
        """
        results = sqlconn.fetch(conn, query)

        if not results:
            print("[+] All OU games have outcomes - nothing to update")
            return 0

        print(f"[*] Found {len(results)} OU games needing outcome updates")

        # Collect all unique teams that need scraping
        teams_to_scrape = {}
        for ou_row in results:
            team_1 = ou_row.get('team_1')
            season = ou_row.get('season')
            season_key = str(season)

            if season_key not in teams_to_scrape:
                teams_to_scrape[season_key] = set()
            teams_to_scrape[season_key].add(team_1)

        # Scrape game history for each unique team (once per team)
        print(f"[*] Scraping game history for unique teams...")
        game_histories_cache = {}

        for season_key, teams in teams_to_scrape.items():
            for team in sorted(teams):
                cache_key = f"{season_key}_{team}"
                try:
                    hist = scrape_game_history(season_key, team)
                    if hist is not None and len(hist) > 0:
                        game_histories_cache[cache_key] = hist
                        print(f"  [+] {team}: {len(hist)} games retrieved")
                    else:
                        game_histories_cache[cache_key] = None
                        print(f"  [-] {team}: No game history found")
                except Exception as e:
                    game_histories_cache[cache_key] = None
                    print(f"  [-] {team}: Error - {e}")

        print(f"[*] Processing {len(results)} games...")

        updated = 0
        for ou_row in results:
            ou_id = ou_row.get('id')
            game_id = ou_row.get('game_id')
            team_1 = ou_row.get('team_1')
            team_2 = ou_row.get('team_2')
            over_point = ou_row.get('over_point')
            bet_on_side = ou_row.get('bet_on_side')
            season = ou_row.get('season')

            # Get cached game history for team_1
            cache_key = f"{season}_{team_1}"
            game_history = game_histories_cache.get(cache_key)

            if game_history is None or len(game_history) == 0:
                continue

            # Find the game by matching opponent and game_id date
            game_date_str = f"{game_id[:4]}-{game_id[4:6]}-{game_id[6:8]}"

            # Find game in history that matches opponent and date
            matching_game = None
            for idx, row in game_history.iterrows():
                if row['opponent'] == team_2 and row['date'] == game_date_str:
                    matching_game = row
                    break

            if matching_game is None:
                continue

            team_1_score = int(matching_game['team_score'])
            team_2_score = int(matching_game['opp_score'])
            actual_total = team_1_score + team_2_score

            # Skip if over_point is missing
            if over_point is None:
                continue

            # Determine winning side (OVER or UNDER)
            winning_side = 'OVER' if actual_total > over_point else 'UNDER'

            # Calculate bet result if bet was placed
            bet_win_or_lose = None
            if bet_on_side and bet_on_side in ['OVER', 'UNDER']:
                bet_win_or_lose = 'WIN' if bet_on_side == winning_side else 'LOSE'

            # Calculate wager and profit if bet was placed
            wager = None
            profit = None
            if bet_on_side and bet_on_side in ['OVER', 'UNDER']:
                wager = 15.00
                if bet_win_or_lose == 'LOSE':
                    profit = -15.00
                elif bet_win_or_lose == 'WIN':
                    # Get the odds for the bet side
                    if bet_on_side == 'OVER':
                        odds = ou_row.get('my_best_book_odds_over') or ou_row.get('best_book_odds_over')
                    else:
                        odds = ou_row.get('my_best_book_odds_under') or ou_row.get('best_book_odds_under')

                    if odds is not None:
                        try:
                            odds_int = int(odds)
                            # Convert American odds to decimal
                            if odds_int < 0:
                                decimal = (100 / abs(odds_int)) + 1
                            else:
                                decimal = (odds_int / 100) + 1
                            profit = round(wager * (decimal - 1), 2)
                        except:
                            profit = None

            # Update the record
            update_query = """
                UPDATE overunder
                SET actual_total = %s, winning_side = %s, bet_win_or_lose = %s, wager = %s, profit = %s, updated_at = NOW()
                WHERE id = %s
            """
            try:
                cursor.execute(update_query, (actual_total, winning_side, bet_win_or_lose, wager, profit, ou_id))
                updated += 1
            except Exception as e:
                print(f"  [!] Error updating OU game {game_id}: {e}")

        conn.commit()
        cursor.close()
        print(f"[+] Updated {updated} OU game outcomes\n")
        return updated

    except Exception as e:
        print(f"[-] Error updating OU outcomes: {e}")
        import traceback
        traceback.print_exc()
        return 0
    finally:
        if conn:
            conn.close()


def update_moneyline_outcomes():
    """
    Update moneyline table with actual game outcomes for any games where winning_team is NULL
    Scrapes game history to get scores and determines winner
    """
    try:
        conn = sqlconn.create_connection()
        if not conn:
            print("[-] Failed to connect to database for ML outcome update")
            return 0

        cursor = conn.cursor()

        # Find all games in moneyline table where winning_team is NULL (game hasn't finished)
        query = """
            SELECT ml.id, ml.game_id, ml.team_1, ml.team_2, ml.team_predicted_to_win, ml.bet_on, ml.season
            FROM moneyline ml
            WHERE ml.winning_team IS NULL
        """
        results = sqlconn.fetch(conn, query)

        if not results:
            print("[+] All ML games have outcomes - nothing to update")
            return 0

        print(f"[*] Found {len(results)} ML games needing outcome updates")

        # Collect all unique teams that need scraping
        teams_to_scrape = {}
        for ml_row in results:
            team_1 = ml_row.get('team_1')
            season = ml_row.get('season')
            season_key = str(season)

            if season_key not in teams_to_scrape:
                teams_to_scrape[season_key] = set()
            teams_to_scrape[season_key].add(team_1)

        # Scrape game history for each unique team (once per team)
        print(f"[*] Scraping game history for unique teams...")
        game_histories_cache = {}

        for season_key, teams in teams_to_scrape.items():
            for team in sorted(teams):
                cache_key = f"{season_key}_{team}"
                try:
                    hist = scrape_game_history(season_key, team)
                    if hist is not None and len(hist) > 0:
                        game_histories_cache[cache_key] = hist
                        print(f"  [+] {team}: {len(hist)} games retrieved")
                    else:
                        game_histories_cache[cache_key] = None
                        print(f"  [-] {team}: No game history found")
                except Exception as e:
                    game_histories_cache[cache_key] = None
                    print(f"  [-] {team}: Error - {e}")

        print(f"[*] Processing {len(results)} games...")

        updated = 0
        for ml_row in results:
            ml_id = ml_row.get('id')
            game_id = ml_row.get('game_id')
            team_1 = ml_row.get('team_1')
            team_2 = ml_row.get('team_2')
            team_predicted_to_win = ml_row.get('team_predicted_to_win')
            bet_on = ml_row.get('bet_on')
            season = ml_row.get('season')

            # Get cached game history for team_1
            cache_key = f"{season}_{team_1}"
            game_history = game_histories_cache.get(cache_key)

            if game_history is None or len(game_history) == 0:
                continue

            # Find the game by matching opponent and game_id date
            game_date_str = f"{game_id[:4]}-{game_id[4:6]}-{game_id[6:8]}"

            # Find game in history that matches opponent and date
            matching_game = None
            for idx, row in game_history.iterrows():
                if row['opponent'] == team_2 and row['date'] == game_date_str:
                    matching_game = row
                    break

            if matching_game is None:
                continue

            team_1_score = int(matching_game['team_score'])
            team_2_score = int(matching_game['opp_score'])
            actual_total = team_1_score + team_2_score

            # Determine winning team
            winning_team = team_1 if team_1_score > team_2_score else team_2

            # Calculate bet result if bet was placed
            bet_win_or_lose = None
            wager = None
            profit = None
            if bet_on == 1:
                bet_win_or_lose = 'WIN' if team_predicted_to_win == winning_team else 'LOSE'
                wager = 15.00

                # Get the odds for the predicted team
                if team_predicted_to_win == team_1:
                    odds = ml_row.get('my_best_book_odds_team_1') or ml_row.get('best_book_odds_team_1')
                else:
                    odds = ml_row.get('my_best_book_odds_team_2') or ml_row.get('best_book_odds_team_2')

                # Calculate profit
                if bet_win_or_lose == 'LOSE':
                    profit = -15.00
                elif odds is not None:
                    try:
                        odds_int = int(odds)
                        # Convert American odds to decimal
                        if odds_int < 0:
                            decimal = (100 / abs(odds_int)) + 1
                        else:
                            decimal = (odds_int / 100) + 1
                        profit = round(wager * (decimal - 1), 2)
                    except:
                        profit = None

            # Update the record
            update_query = """
                UPDATE moneyline
                SET winning_team = %s, actual_score_team_1 = %s, actual_score_team_2 = %s, actual_total = %s, bet_win_or_lose = %s, wager = %s, profit = %s, updated_at = NOW()
                WHERE id = %s
            """
            try:
                cursor.execute(update_query, (winning_team, team_1_score, team_2_score, actual_total, bet_win_or_lose, wager, profit, ml_id))
                updated += 1
            except Exception as e:
                print(f"  [!] Error updating ML game {game_id}: {e}")

        conn.commit()
        cursor.close()
        print(f"[+] Updated {updated} ML game outcomes\n")
        return updated

    except Exception as e:
        print(f"[-] Error updating ML outcomes: {e}")
        import traceback
        traceback.print_exc()
        return 0
    finally:
        if conn:
            conn.close()


def insert_overunder_bets(ou_predictions_df: pl.DataFrame, ou_bets: list, allowed_bookmakers: set, target_date_yyyymmdd: str):
    """
    Insert all over/under game predictions into database
    Stores every game with all predictions, regardless of whether it matches betting rules
    """
    try:
        conn = sqlconn.create_connection()
        if not conn:
            print("[-] Failed to connect to database for OU insertion")
            return

        cursor = conn.cursor()

        print(f"[*] Debug: ou_predictions_df has {len(ou_predictions_df)} games")
        print(f"[*] Debug: ou_bets has {len(ou_bets)} bets")

        # Create set of game_ids that matched betting rules
        bet_game_ids = set()
        for bet in ou_bets:
            key = (bet['game_id'], bet['bet'])
            bet_game_ids.add(key)

        inserted = 0

        for i, game_row in enumerate(ou_predictions_df.iter_rows(named=True)):
            game_id = game_row.get('game_id')
            team_1 = game_row.get('team_1')
            team_2 = game_row.get('team_2')

            xgb_pred = game_row.get('xgb_pred')
            lgb_pred = game_row.get('lgb_pred')
            cb_pred = game_row.get('cb_pred')
            ensemble_pred = game_row.get('ensemble_pred')
            ensemble_confidence = game_row.get('ensemble_confidence')
            good_bets_confidence = game_row.get('good_bets_confidence')

            if not game_id or not team_1 or not team_2:
                continue

            # Extract game_date and season from game_id
            game_date_str = f"{game_id[:4]}-{game_id[4:6]}-{game_id[6:8]}"
            season = extract_season_from_game_id(game_id)

            # Query odds for this game from database
            all_odds = []
            try:
                query = f"""
                    SELECT game_id, bookmaker, over_point, over_odds, under_odds
                    FROM odds
                    WHERE game_id = %s
                    ORDER BY bookmaker
                """
                results = sqlconn.fetch(conn, query, (game_id,))
                all_odds = results if results else []
            except:
                pass  # Continue with empty odds if query fails

            # Get odds from first bookmaker
            over_point = None
            best_book_over = None
            best_book_odds_over = None
            best_book_under = None
            best_book_odds_under = None
            my_best_book_over = None
            my_best_book_odds_over = None
            my_best_book_under = None
            my_best_book_odds_under = None

            for odds_rec in all_odds:
                bm_name = odds_rec.get('bookmaker')
                over_pt = odds_rec.get('over_point')
                over_od = odds_rec.get('over_odds')
                under_od = odds_rec.get('under_odds')

                if over_point is None and over_pt is not None:
                    over_point = float(over_pt)

                # Track best odds (highest)
                if over_od is not None and (best_book_odds_over is None or over_od > best_book_odds_over):
                    best_book_odds_over = over_od
                    best_book_over = bm_name

                if under_od is not None and (best_book_odds_under is None or under_od > best_book_odds_under):
                    best_book_odds_under = under_od
                    best_book_under = bm_name

                # Track best from allowed bookmakers
                if bm_name in allowed_bookmakers:
                    if over_od is not None and (my_best_book_odds_over is None or over_od > my_best_book_odds_over):
                        my_best_book_odds_over = over_od
                        my_best_book_over = bm_name

                    if under_od is not None and (my_best_book_odds_under is None or under_od > my_best_book_odds_under):
                        my_best_book_odds_under = under_od
                        my_best_book_under = bm_name

            # Calculate difference if we have over_point
            difference = None
            if over_point is not None and ensemble_pred is not None:
                difference = float(ensemble_pred) - float(over_point)

            # Calculate implied probabilities from odds
            impl_prob_over_vig = None
            impl_prob_under_vig = None
            impl_prob_over_dvig = None
            impl_prob_under_dvig = None

            if best_book_odds_over is not None and best_book_odds_under is not None:
                impl_prob_over_vig = calculate_implied_prob(int(best_book_odds_over))
                impl_prob_under_vig = calculate_implied_prob(int(best_book_odds_under))
                impl_prob_over_dvig, impl_prob_under_dvig = dvig_probabilities(impl_prob_over_vig, impl_prob_under_vig)

            # Determine betting rule and bet_on flag
            bet_rule = None
            bet_on = 0
            bet_on_side = None

            # Check if this game matched any betting rule
            if (game_id, 'OVER') in bet_game_ids:
                bet_on = 1
                bet_on_side = 'OVER'
                # Find matching rule
                for bet in ou_bets:
                    if bet['game_id'] == game_id and bet['bet'] == 'OVER':
                        bet_rule = bet.get('bet_rule', None)
                        break
            elif (game_id, 'UNDER') in bet_game_ids:
                bet_on = 1
                bet_on_side = 'UNDER'
                # Find matching rule
                for bet in ou_bets:
                    if bet['game_id'] == game_id and bet['bet'] == 'UNDER':
                        bet_rule = bet.get('bet_rule', None)
                        break

            # Insert record
            insert_query = """
                INSERT INTO overunder (
                    game_id, team_1, team_2,
                    over_point,
                    xgb_pred, lgb_pred, cb_pred,
                    ensemble_pred, ensemble_confidence,
                    good_bets_confidence, difference,
                    best_book_over, best_book_odds_over,
                    best_book_under, best_book_odds_under,
                    my_best_book_over, my_best_book_odds_over,
                    my_best_book_under, my_best_book_odds_under,
                    implied_prob_over_with_vig, implied_prob_under_with_vig,
                    implied_prob_over_devigged, implied_prob_under_devigged,
                    bet_rule, bet_on, bet_on_side,
                    game_date, season
                ) VALUES (
                    %s, %s, %s,
                    %s,
                    %s, %s, %s,
                    %s, %s,
                    %s, %s,
                    %s, %s,
                    %s, %s,
                    %s, %s,
                    %s, %s,
                    %s, %s,
                    %s, %s,
                    %s, %s, %s,
                    %s, %s
                )
            """

            values = (
                game_id, team_1, team_2,
                over_point,
                xgb_pred, lgb_pred, cb_pred,
                ensemble_pred, ensemble_confidence,
                good_bets_confidence, difference,
                best_book_over, best_book_odds_over,
                best_book_under, best_book_odds_under,
                my_best_book_over, my_best_book_odds_over,
                my_best_book_under, my_best_book_odds_under,
                impl_prob_over_vig, impl_prob_under_vig,
                impl_prob_over_dvig, impl_prob_under_dvig,
                bet_rule, bet_on, bet_on_side,
                game_date_str, season
            )

            try:
                cursor.execute(insert_query, values)
                inserted += 1
            except Exception as e:
                print(f"  [!] Error inserting OU game {game_id}: {e}")

        conn.commit()
        cursor.close()
        conn.close()

        print(f"[+] Inserted {inserted} over/under records into database\n")

    except Exception as e:
        print(f"[-] Error inserting OU bets: {e}\n")
        import traceback
        traceback.print_exc()


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
            # Export disabled - only bets file will be exported
            # output_df = pl.DataFrame(all_predictions)
            # output_file = Path(__file__).parent / f"all_predictions_{target_date_yyyymmdd}.xlsx"
            # output_df.write_excel(str(output_file))
            # print(f"[+] Exported {len(all_predictions)} all predictions (ML and OU) to {output_file}\n")
            pass
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
        # Export disabled - only bets file will be exported
        # output_df = pl.DataFrame(all_games)
        # # Sort by date and game_id
        # output_df = output_df.sort(['date', 'game_id'])
        #
        # if target_date_yyyymmdd is None:
        #     target_date_yyyymmdd = get_todays_date_yyyymmdd()
        # output_file = Path(__file__).parent / f"all_games_{target_date_yyyymmdd}.xlsx"
        # output_df.write_excel(str(output_file))
        # print(f"[+] Exported {len(all_games)} games with probabilities and ML averages to {output_file}\n")
        pass
    else:
        print("[*] No games to export\n")


def main(manual_date: str = None, scrape_data: bool = True, push_predictions: bool = True):
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

    # Display configuration
    print(f"[*] Configuration:")
    print(f"    Scrape data: {scrape_data}")
    print(f"    Push predictions: {push_predictions}\n")

    # Step 0: Update outcomes for any completed games and bankroll tables
    print("STEP 0: Updating Game Outcomes and Bankroll Tracking")
    print("-"*80 + "\n")

    update_overunder_outcomes()
    update_moneyline_outcomes()

    # Update bankroll tables for yesterday's bets
    print("STEP 0.5: Updating Bankroll Tables")
    print("-"*80 + "\n")
    update_my_bankroll()
    update_bankroll()

    # Step 1: Run ou_main.main() to scrape all data and get predictions
    print("STEP 1: Running OU pipeline to scrape data and make predictions")
    print("-"*80 + "\n")

    features_df, predictions_df = ou_module.main(target_date=target_date, scrape_data=scrape_data)

    if features_df is None or predictions_df is None:
        print("[-] OU pipeline failed")
        return

    print("\n[+] OU pipeline complete - data has been scraped and prepared")
    print(f"[DEBUG] features_df shape: {features_df.shape}")

    # Define allowed bookmakers
    allowed_bookmakers = {'BetOnline.ag', 'Bovada', 'MyBookie.ag', 'betonlineag'}

    # We already have todays_games_df and features_df from the OU pipeline above
    # Step 2: Prepare ML features from the OU features_df
    print("STEP 2: Preparing ML Features for Moneyline Model")
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

    # Export features to Excel - DISABLED
    # print("STEP 2.5: Exporting All Generated Features to Excel")
    # print("-"*80 + "\n")
    # export_features_to_excel(features_df, target_date_yyyymmdd)

    # Load ML models (Ensemble: 18% LGB + 82% XGB)
    print("STEP 3: Loading Moneyline Models (Ensemble: 18% LGB + 82% XGB)")
    print("-"*80 + "\n")
    lgb_model, xgb_model = load_ml_models()
    if lgb_model is None or xgb_model is None:
        print("[-] Could not load models\n")
        return

    print("STEP 4: Getting ML Predictions (Ensemble: 18% LGB + 82% XGB)")
    print("-"*80 + "\n")

    # Step 4b: Extract OU bets from predictions_df using V3.0 comprehensive betting rules
    print("STEP 4b: Extracting OU Valid Bets (V3.0 Comprehensive Rules - 26 rules)")
    print("-"*80 + "\n")

    # Use the updated get_ou_bets function with V3.0 betting rules
    ou_bets = get_ou_bets(predictions_df, allowed_bookmakers=allowed_bookmakers)

    print(f"[+] Found {len(ou_bets)} OU bets meeting V3.0 Rules criteria (Expected ROI: +10.13%)\n")

    # Step 4b.5: Insert over/under data into database
    if push_predictions:
        print("STEP 4b.5: Inserting Over/Under Data into Database")
        print("-"*80 + "\n")

        insert_overunder_bets(predictions_df, ou_bets, allowed_bookmakers, target_date_yyyymmdd)
    else:
        print("STEP 4b.5: Skipping Over/Under Database Insert (--no-push flag)")
        print("-"*80 + "\n")

    # Step 4c: Get ML bets with Clay's Optimal Rules
    print("STEP 4c: Extracting ML Valid Bets (Clay's Optimal Rules)")
    print("-"*80 + "\n")

    team_mappings = load_team_mappings()
    ml_bets = get_ml_bets(features_df, lgb_model, xgb_model, allowed_bookmakers, team_mappings=team_mappings)

    # Step 4c.5: Insert moneyline data into database
    if push_predictions:
        print("\nSTEP 4c.5: Inserting Moneyline Data into Database")
        print("-"*80 + "\n")

        # Get LGB and XGB predictions for insertion
        X, feature_cols = align_features_to_model(features_df)
        if X is not None:
            lgb_proba_all = lgb_model.predict(X)
            xgb_proba_all = xgb_model.predict_proba(X)[:, 1]

            # Load odds for database insertion
            game_ids = features_df['game_id'].to_list()
            odds_dict_all = load_odds_for_games(game_ids)

            insert_moneyline_bets(features_df, lgb_proba_all, xgb_proba_all, ml_bets, odds_dict_all, allowed_bookmakers, team_mappings, target_date_yyyymmdd)
        else:
            print("[-] Could not align features for database insertion\n")
    else:
        print("\nSTEP 4c.5: Skipping Moneyline Database Insert (--no-push flag)")
        print("-"*80 + "\n")

    # Step 4c.6: Cleanup cancelled games (missing predictions after 7+ days)
    print("\nSTEP 4c.6: Cleaning Up Cancelled Games")
    print("-"*80 + "\n")
    cleanup_cancelled_games()

    # Export ALL predictions (before filtering by EV/probability thresholds)
    print("STEP 4d: Exporting All Predictions (ML and OU)")
    print("-"*80 + "\n")
    export_all_predictions(features_df, lgb_model, xgb_model, predictions_df, target_date_yyyymmdd)

    # Done
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
    print("\nSTEP 5: Exporting All Games with Probabilities and ML Averages")
    print("-"*80 + "\n")
    export_all_games_with_odds(features_df_original, lgb_model, xgb_model, target_date_yyyymmdd)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Run betting recommendations for a specific date')
    parser.add_argument('--date', type=str, default=None, help='Date to run for in YYYY-MM-DD format (default: today)')
    parser.add_argument('--no-scrape', action='store_true', help='Skip data scraping (default: scrape data)')
    parser.add_argument('--no-push', action='store_true', help='Skip pushing predictions to database (default: push predictions)')
    args = parser.parse_args()

    main(manual_date=args.date, scrape_data=not args.no_scrape, push_predictions=not args.no_push)
