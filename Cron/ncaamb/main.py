#!/usr/bin/env python3
"""
Futures Report: Combined ML and OU Betting Recommendations
Generates daily betting report with:
- Moneyline (ML) bets using LGB + Good Bets Model (Strategy 2 Variance-Optimized: EV >= 3%, Good Bets >= 0.40)
  Expected: 221 bets/season, 55.2% win rate, 299% ROI
- Over/Under (OU) bets where ensemble - over_point > 2.3
Only considers odds from: BetOnline.ag, Bovada, MyBookie.ag
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import polars as pl
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
    """Load LGB and Good Bets models for Strategy 2 implementation"""
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

    # Load Good Bets model
    gb_path = saved_dir / "good_bets_rf_model_final.pkl"
    if not gb_path.exists():
        print(f"[-] Good Bets model not found at {gb_path}\n")
        return lgb_model, None

    try:
        gb_model = pickle.load(open(gb_path, 'rb'))
        print(f"[+] Good Bets model loaded from {gb_path}\n")
    except Exception as e:
        print(f"[-] Error loading Good Bets model: {e}\n")
        return lgb_model, None

    return lgb_model, gb_model


def identify_feature_columns(df: pl.DataFrame) -> list:
    """Identify numeric feature columns (same as make_ou_predictions)"""
    metadata_cols = {
        'game_id', 'date', 'season', 'team_1', 'team_2', 'actual_total',
        'team_1_conference', 'team_2_conference', 'team_1_is_home', 'team_2_is_home',
        'location', 'team_1_score', 'team_2_score', 'total_score_outcome', 'team_1_winloss',
        'team_1_leaderboard', 'team_2_leaderboard', 'team_1_match_hist', 'team_2_match_hist',
        'team_1_hist_count', 'team_2_hist_count', 'start_time', 'game_odds',
        'avg_ml_home', 'avg_ml_away'  # Exclude odds from features
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
    # Replace any NaN values that might remain (e.g., from None values in training)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X


def load_expected_feature_columns() -> list:
    """Load the list of expected feature columns from file"""
    expected_file = Path(__file__).parent / "models" / "moneyline" / "saved" / "feature_columns.txt"

    if expected_file.exists():
        with open(expected_file, 'r') as f:
            return [line.strip() for line in f if line.strip()]

    return None


def align_features_to_model(df: pl.DataFrame, model: XGBClassifier) -> tuple:
    """
    Ensure dataframe has exactly the features the model expects.
    Returns (X array, feature_cols list) or (None, None) if alignment fails.
    """
    # Get expected feature list
    expected_features = load_expected_feature_columns()

    if expected_features is None:
        print("[-] Could not load expected feature columns")
        return None, None

    # Check which expected features are present in df
    df_cols = set(df.columns)
    missing_features = [f for f in expected_features if f not in df_cols]
    extra_features = [c for c in df.columns if c not in set(expected_features) and c not in {
        'game_id', 'date', 'season', 'team_1', 'team_2', 'actual_total',
        'team_1_conference', 'team_2_conference', 'team_1_is_home', 'team_2_is_home',
        'location', 'team_1_score', 'team_2_score', 'total_score_outcome', 'team_1_winloss',
        'team_1_leaderboard', 'team_2_leaderboard', 'team_1_match_hist', 'team_2_match_hist',
        'team_1_hist_count', 'team_2_hist_count', 'start_time', 'game_odds'
    }]

    # Report feature alignment status
    print(f"[*] Feature alignment check:")
    print(f"    Expected features: {len(expected_features)}")
    print(f"    Features present in data: {len([f for f in expected_features if f in df_cols])}")
    print(f"    Missing features: {len(missing_features)}")
    print(f"    Extra features (will be dropped): {len(extra_features)}\n")

    if missing_features:
        print(f"[!] Missing {len(missing_features)} expected features:")
        for f in missing_features[:5]:
            print(f"    - {f}")
        if len(missing_features) > 5:
            print(f"    ... and {len(missing_features) - 5} more")
        print(f"    These will be filled with 0s\n")

        # Add missing features with 0 values
        for feature in missing_features:
            df = df.with_columns(pl.lit(0.0).alias(feature))

    # Select features in EXACT order expected by model
    # This is critical - the order must match training order
    try:
        X = df.select(expected_features).fill_null(0).to_numpy()
        # Replace any NaN values that might remain (e.g., from None values in training)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    except Exception as e:
        print(f"[-] Error selecting features in order: {e}\n")
        return None, None

    print(f"[+] Feature matrix prepared:")
    print(f"    Shape: {X.shape}")
    print(f"    Columns in order: {len(expected_features)} (matching model training)")
    print(f"    Nulls filled with 0.0\n")

    return X, expected_features


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


def create_good_bets_features(df: pl.DataFrame, lgb_proba: np.ndarray,
                              xgb_proba: np.ndarray) -> pl.DataFrame:
    """
    Create good bets features in 2-row-per-game format using vectorized Polars operations.
    Returns DataFrame with features ready for good_bets model prediction.
    """
    # Add predictions to dataframe
    df = df.with_columns([
        pl.lit(xgb_proba[:, 1]).alias('xgb_prob_team_1'),
        pl.lit(lgb_proba).alias('lgb_prob_team_1'),
    ])

    # Fill missing odds/spread data with 0
    for col in ['avg_ml_team_1', 'avg_ml_team_2', 'avg_spread_pts_team_1', 'avg_spread_pts_team_2',
                'avg_spread_odds_team_1', 'avg_spread_odds_team_2', 'month', 'team_1_adjoe',
                'team_1_adjde', 'team_2_adjoe', 'team_2_adjde']:
        if col in df.columns:
            df = df.with_columns(pl.col(col).fill_null(0))
        else:
            df = df.with_columns(pl.lit(0).alias(col))

    # Calculate implied probabilities
    df = df.with_columns([
        pl.col('avg_ml_team_1').map_elements(
            lambda x: 0.5 if x == 0 else (100/(x+100) if x > 0 else abs(x)/(abs(x)+100)),
            return_dtype=pl.Float64
        ).alias('implied_prob_team_1'),
        pl.col('avg_ml_team_2').map_elements(
            lambda x: 0.5 if x == 0 else (100/(x+100) if x > 0 else abs(x)/(abs(x)+100)),
            return_dtype=pl.Float64
        ).alias('implied_prob_team_2'),
    ])

    # Calculate EV using LGB probabilities (strategy 2 uses LGB EV)
    df = df.with_columns([
        (pl.col('lgb_prob_team_1') * pl.when(pl.col('avg_ml_team_1') == 0).then(1.0)
            .when(pl.col('avg_ml_team_1') > 0).then(1 + (pl.col('avg_ml_team_1') / 100))
            .otherwise(1 + (100 / pl.col('avg_ml_team_1').abs())) - 1)
        .alias('ev_team_1'),
        ((1 - pl.col('lgb_prob_team_1')) * pl.when(pl.col('avg_ml_team_2') == 0).then(1.0)
            .when(pl.col('avg_ml_team_2') > 0).then(1 + (pl.col('avg_ml_team_2') / 100))
            .otherwise(1 + (100 / pl.col('avg_ml_team_2').abs())) - 1)
        .alias('ev_team_2'),
    ])

    # Calculate strength differentials
    df = df.with_columns([
        (pl.col('team_1_adjoe') - pl.col('team_2_adjoe')).abs().alias('strength_diff_1'),
        (pl.col('team_2_adjoe') - pl.col('team_1_adjoe')).abs().alias('strength_diff_2'),
    ])

    # Create team 1 perspective rows
    team_1_data = df.select([
        pl.col('game_id'),
        pl.col('team_1').alias('team'),
        pl.col('team_2').alias('opponent'),
        pl.col('date'),
        pl.col('team_1_is_home_game').alias('is_home'),
        pl.lit(1).alias('team_perspective'),  # team 1
        pl.col('xgb_prob_team_1'),
        pl.col('lgb_prob_team_1'),
        (pl.col('xgb_prob_team_1') - pl.col('lgb_prob_team_1')).abs().alias('model_disagreement'),
        pl.col('avg_ml_team_1').alias('moneyline_odds'),
        pl.col('implied_prob_team_1').alias('implied_prob'),
        pl.col('ev_team_1').alias('ev'),
        pl.col('avg_spread_pts_team_1').alias('spread_pts_self'),
        pl.col('avg_spread_pts_team_2').alias('spread_pts_opp'),
        pl.col('avg_spread_odds_team_1').alias('spread_odds_self'),
        pl.col('avg_spread_odds_team_2').alias('spread_odds_opp'),
        pl.col('month'),
        pl.col('strength_diff_1').alias('strength_differential'),
    ])

    # Create team 2 perspective rows
    team_2_data = df.select([
        pl.col('game_id'),
        pl.col('team_2').alias('team'),
        pl.col('team_1').alias('opponent'),
        pl.col('date'),
        (1 - pl.col('team_1_is_home_game')).alias('is_home'),
        pl.lit(2).alias('team_perspective'),  # team 2
        (1 - pl.col('xgb_prob_team_1')).alias('xgb_prob_team_1'),
        (1 - pl.col('lgb_prob_team_1')).alias('lgb_prob_team_1'),
        (pl.col('xgb_prob_team_1') - pl.col('lgb_prob_team_1')).abs().alias('model_disagreement'),
        pl.col('avg_ml_team_2').alias('moneyline_odds'),
        pl.col('implied_prob_team_2').alias('implied_prob'),
        pl.col('ev_team_2').alias('ev'),
        pl.col('avg_spread_pts_team_2').alias('spread_pts_self'),
        pl.col('avg_spread_pts_team_1').alias('spread_pts_opp'),
        pl.col('avg_spread_odds_team_2').alias('spread_odds_self'),
        pl.col('avg_spread_odds_team_1').alias('spread_odds_opp'),
        pl.col('month'),
        pl.col('strength_diff_2').alias('strength_differential'),
    ])

    # Combine both perspectives
    all_bets = pl.concat([team_1_data, team_2_data])

    return all_bets


def get_ml_bets(features_df: pl.DataFrame, lgb_model, gb_model, allowed_bookmakers: set, team_mappings: dict = None) -> list:
    """
    Get moneyline bets using Strategy 2: EV >= 5% AND Good Bets confidence >= 0.55
    Uses LightGBM predictions + Good Bets Random Forest confidence filtering.
    """
    ml_bets = []

    if lgb_model is None:
        print("[-] LGB model not loaded\n")
        return ml_bets

    if gb_model is None:
        print("[-] Good Bets model not loaded\n")
        return ml_bets

    print("STEP 2.5: Getting ML Predictions (Strategy 2)")
    print("-"*80 + "\n")

    # Get LGB predictions for all games
    try:
        X, feature_cols = align_features_to_model(features_df, lgb_model)
        if X is None:
            print("[-] Could not align features to LGB model\n")
            return ml_bets

        # LGB returns probabilities as 1D array (probability of class 1)
        lgb_proba = lgb_model.predict(X)
        print(f"[+] Made LGB predictions for {len(X)} games\n")
    except Exception as e:
        print(f"[-] Error during LGB prediction: {e}\n")
        return ml_bets

    # Get XGB predictions for good_bets features (need both for feature engineering)
    try:
        xgb_model = XGBClassifier()
        xgb_path = Path(__file__).parent / "models" / "moneyline" / "saved" / "xgboost_model.pkl"
        xgb_model.load_model(str(xgb_path))
        xgb_proba = xgb_model.predict_proba(X)
        print(f"[+] Made XGB predictions for {len(X)} games\n")
    except Exception as e:
        print(f"[-] Could not load XGB for good_bets features: {e}")
        print(f"    Continuing with LGB only...\n")
        # Create fake XGB proba from LGB proba
        xgb_proba = np.column_stack([1 - lgb_proba, lgb_proba])

    # Create good_bets features
    try:
        gb_features_df = create_good_bets_features(features_df, lgb_proba, xgb_proba)
        print(f"[+] Created {len(gb_features_df)} betting rows\n")
    except Exception as e:
        print(f"[-] Error creating good_bets features: {e}\n")
        return ml_bets

    # Get good_bets predictions
    try:
        # Extract features in the correct order
        feature_order = ['xgb_prob_team_1', 'lgb_prob_team_1', 'model_disagreement', 'moneyline_odds',
                        'implied_prob', 'ev', 'spread_pts_self', 'spread_pts_opp',
                        'spread_odds_self', 'spread_odds_opp', 'month', 'strength_differential']

        X_gb = gb_features_df.select(feature_order).to_numpy()
        gb_proba = gb_model.predict_proba(X_gb)[:, 1]  # Get probability of class 1 (good bet)

        gb_features_df = gb_features_df.with_columns(pl.lit(gb_proba).alias('gb_confidence'))
        print(f"[+] Got good_bets confidence scores\n")
    except Exception as e:
        print(f"[-] Error getting good_bets predictions: {e}\n")
        return ml_bets

    # Load odds
    game_ids = features_df['game_id'].to_list()
    odds_dict = load_odds_for_games(game_ids)

    # Strategy 2 (Optimized for Variance): Filter by EV >= 3% AND Good Bets confidence >= 0.40
    # This configuration provides 221 bets/season, 55.2% win rate, 299% ROI
    print(f"[*] Evaluating bets with Strategy 2 (Variance-Optimized) criteria:")
    print(f"    - EV >= 3.0%")
    print(f"    - Good Bets confidence >= 0.40\n")

    # Process each bet row
    for bet_row in gb_features_df.iter_rows(named=True):
        game_id = bet_row.get('game_id')
        team = bet_row.get('team')
        opponent = bet_row.get('opponent')
        date = bet_row.get('date')
        ev = bet_row.get('ev')
        gb_confidence = bet_row.get('gb_confidence')

        # Check Strategy 2 (Optimized) criteria
        if ev < 0.03:  # EV < 3%
            continue
        if gb_confidence < 0.40:  # Good Bets confidence < 40%
            continue

        # Get odds for this team
        all_odds = odds_dict.get(game_id, [])
        if not all_odds:
            continue

        # Find best odds from allowed bookmakers
        best_odds = None
        best_bm = None

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

            # Find our team in odds
            if team_a_mapped == team and ml_team_a is not None:
                if best_odds is None or american_to_decimal(ml_team_a) > american_to_decimal(best_odds):
                    best_odds = ml_team_a
                    best_bm = bm_name

            if team_b_mapped == team and ml_team_b is not None:
                if best_odds is None or american_to_decimal(ml_team_b) > american_to_decimal(best_odds):
                    best_odds = ml_team_b
                    best_bm = bm_name

        # Add bet if odds found
        if best_odds is not None:
            ml_bets.append({
                'type': 'ML',
                'game_id': game_id,
                'date': date,
                'team': team,
                'opponent': opponent,
                'odds': int(best_odds),
                'decimal': american_to_decimal(best_odds),
                'win_prob': bet_row.get('lgb_prob_team_1'),
                'ev': ev,
                'ev_percent': ev * 100,
                'gb_confidence': gb_confidence,
                'bookmaker': best_bm
            })

    print(f"[+] Found {len(ml_bets)} bets meeting Strategy 2 criteria\n")
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

    # Import ou_main to run data scraping first
    import ou_main as ou_module

    # Step 0: Run ou_main.main() to scrape all data and get predictions
    print("STEP 0: Running OU pipeline to scrape data and make predictions")
    print("-"*80 + "\n")

    features_df, predictions_df = ou_module.main()

    if features_df is None or predictions_df is None:
        print("[-] OU pipeline failed")
        return

    print("\n[+] OU pipeline complete - data has been scraped and prepared\n")

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

    # Analyze nulls after dropping metadata
    feature_cols = identify_feature_columns(features_df)
    print(f"[*] Using {len(feature_cols)} numeric features (excluding metadata)")
    print(f"[*] Expected features for model: 319\n")

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

    # Load ML models (Strategy 2: LGB + Good Bets)
    print("STEP 2: Loading Moneyline Models (Strategy 2)")
    print("-"*80 + "\n")
    lgb_model, gb_model = load_ml_models()
    if lgb_model is None or gb_model is None:
        print("[-] Could not load models\n")
        return

    print("STEP 3: Getting ML Predictions (Strategy 2)")
    print("-"*80 + "\n")

    # Step 3b: Extract OU bets from predictions_df (ensemble - over_point > 2.3)
    print("STEP 3b: Extracting OU Valid Bets (Difference > 2.3)")
    print("-"*80 + "\n")

    # Use the predictions_df generated in Step 0 to find valid OU bets
    ou_bets = []

    try:
        conn = ou_module.sqlconn.create_connection()
        if not conn:
            print("[-] Failed to connect to database for OU odds")
        else:
            for row in predictions_df.iter_rows(named=True):
                game_id = row.get('game_id')
                ensemble = row.get('ensemble_pred')
                date = row.get('date')

                if game_id is None or ensemble is None:
                    continue

                # Get O/U line from database
                query = """
                    SELECT over_point, bookmaker
                    FROM odds
                    WHERE game_id = %s
                    AND bookmaker IN (%s, %s, %s, %s)
                    ORDER BY bookmaker
                    LIMIT 1
                """

                results = ou_module.sqlconn.fetch(conn, query, (
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
                        difference = float(ensemble) - float(over_point)
                        if difference > 2.3:
                            ou_bets.append({
                                'type': 'OU',
                                'game_id': game_id,
                                'date': date,
                                'ensemble': round(float(ensemble), 2),
                                'over_point': round(float(over_point), 2),
                                'difference': round(difference, 2),
                                'bookmaker': bookmaker,
                                'bet': 'OVER'
                            })

            conn.close()

    except Exception as e:
        print(f"[-] Error extracting OU bets: {e}")

    print(f"[+] Found {len(ou_bets)} OU bets with difference > 2.3\n")

    # Step 3c: Get ML bets with Strategy 2 (EV >= 5% AND Good Bets >= 0.55)
    print("STEP 3c: Extracting ML Valid Bets (Strategy 2: EV >= 5%, Good Bets >= 0.55)")
    print("-"*80 + "\n")

    team_mappings = load_team_mappings()
    ml_bets = get_ml_bets(features_df, lgb_model, gb_model, allowed_bookmakers, team_mappings=team_mappings)

    # Export valid bets to Excel with date in filename
    print("\nExporting valid bets to Excel...")

    todays_date = get_todays_date_yyyymmdd()
    output_file = Path(__file__).parent / f"bets_{todays_date}.xlsx"

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
            'over_point': round(bet['over_point'], 2)
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
            print(f"MONEYLINE BETS (Strategy 2: EV >= 5%, Good Bets >= 0.55, Bookmakers: {', '.join(sorted(allowed_bookmakers))})")
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
                print(f"  Good Bets Confidence: {bet.get('gb_confidence', 'N/A'):.2%}" if isinstance(bet.get('gb_confidence'), float) else f"  Good Bets Confidence: {bet.get('gb_confidence', 'N/A')}")
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
        print(f"Total ML Bets (Strategy 2 Variance-Optimized: EV >= 3%, Good Bets >= 0.40): {len(ml_bets)}")
        print(f"Total OU Bets (Diff > 2.3): {len(ou_bets)}")
        print(f"Total Bets: {len(ml_bets) + len(ou_bets)}\n")

    else:
        print("No bets meet criteria")
        print("  ML: Strategy 2 (EV >= 3% AND Good Bets >= 0.40)")
        print("  OU: Difference > 2.3\n")

    print("="*80 + "\n")


if __name__ == "__main__":
    main()
