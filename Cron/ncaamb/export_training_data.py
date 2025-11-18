#!/usr/bin/env python3
"""
Export training data to CSV with one row per team
For each game: team, model_odds (win probability), best_available_odds, actual_winner
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


def american_to_decimal(american_odds: float) -> float:
    """Convert American odds to decimal odds"""
    american_odds = float(american_odds)
    if american_odds >= 0:
        return float((american_odds / 100) + 1)
    else:
        return float((100 / abs(american_odds)) + 1)


def get_best_odds(all_odds: list, team_position: str) -> float:
    """
    Get best odds for a team from all bookmakers
    team_position: 'home' or 'away'
    Returns american odds (as integer like -110, +250, etc)
    """
    if not all_odds:
        return None

    odds_key = 'ml_home' if team_position == 'home' else 'ml_away'
    valid_odds = [o[odds_key] for o in all_odds if o[odds_key] is not None]

    if not valid_odds:
        return None

    # Find best odds (highest positive or least negative)
    best_odds = max(valid_odds, key=lambda x: american_to_decimal(x))
    return best_odds


def export_training_data_csv(df: pl.DataFrame, predictions_proba: np.ndarray, odds_dict: dict, output_file: str):
    """
    Export training data to CSV with one row per team
    Columns: game_id, date, team, model_win_prob, best_odds, actual_winner, home_away
    """
    rows = []

    for idx, game in enumerate(df.iter_rows(named=True)):
        game_id = game.get('game_id')
        date = game.get('date')
        team_1 = game.get('team_1')
        team_2 = game.get('team_2')
        team_1_is_home = game.get('team_1_is_home_game')

        # Get model predictions
        team_1_win_prob = float(predictions_proba[idx, 1])
        team_2_win_prob = float(predictions_proba[idx, 0])

        # Get actual result
        actual_winner = team_1 if game['team_1_score'] > game['team_2_score'] else team_2

        # Get odds for this game
        all_odds = odds_dict.get(game_id, [])

        if not all_odds:
            continue

        # Determine home/away for team_1 and team_2
        if team_1_is_home == 1:
            team_1_odds = get_best_odds(all_odds, 'home')
            team_2_odds = get_best_odds(all_odds, 'away')
            team_1_home_away = 'HOME'
            team_2_home_away = 'AWAY'
        elif team_1_is_home == 0:
            team_1_odds = get_best_odds(all_odds, 'away')
            team_2_odds = get_best_odds(all_odds, 'home')
            team_1_home_away = 'AWAY'
            team_2_home_away = 'HOME'
        else:
            # Neutral
            team_1_odds = get_best_odds(all_odds, 'home')
            team_2_odds = get_best_odds(all_odds, 'away')
            team_1_home_away = 'NEUTRAL'
            team_2_home_away = 'NEUTRAL'

        # Add team_1
        if team_1_odds is not None:
            rows.append({
                'game_id': game_id,
                'date': date,
                'team': team_1,
                'home_away': team_1_home_away,
                'model_win_prob': f"{team_1_win_prob:.4f}",
                'best_odds': int(team_1_odds),
                'actual_winner': team_1 if actual_winner == team_1 else 'NO'
            })

        # Add team_2
        if team_2_odds is not None:
            rows.append({
                'game_id': game_id,
                'date': date,
                'team': team_2,
                'home_away': team_2_home_away,
                'model_win_prob': f"{team_2_win_prob:.4f}",
                'best_odds': int(team_2_odds),
                'actual_winner': team_2 if actual_winner == team_2 else 'NO'
            })

    # Write to CSV
    if rows:
        output_df = pl.DataFrame(rows)
        output_df.write_csv(output_file)
        print(f"[+] Exported {len(rows)} team records to {output_file}")
        print(f"[+] This represents {len(rows)//2} games\n")
        return len(rows)
    else:
        print("[-] No data to export")
        return 0


def main():
    print("\n")
    print("="*80)
    print("TRAINING DATA EXPORT - CSV FORMAT")
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

    # Train model
    print("STEP 2: Training Model")
    print("-"*80 + "\n")
    print(f"Training on {len(X_train)} samples...")
    # Optimized hyperparameters from Bayesian optimization
    model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.01,
        subsample=0.9980723455609309,
        colsample_bytree=1.0,
        min_child_weight=10,
        gamma=5,
        reg_alpha=1,
        reg_lambda=0,
        random_state=42,
        eval_metric='logloss',
        verbosity=0
    )

    model.fit(X_train, y_train)
    print(f"[+] Model training complete\n")

    # Load odds from database
    print("STEP 3: Loading Odds from Database")
    print("-"*80 + "\n")
    game_ids = train_df['game_id'].to_list()
    odds_dict = load_odds_for_games(game_ids)

    # Make predictions
    print("STEP 4: Making Predictions")
    print("-"*80 + "\n")
    pred_proba = model.predict_proba(X_train)
    print(f"Made predictions for {len(X_train)} games\n")

    # Export to CSV
    print("STEP 5: Exporting to CSV")
    print("-"*80 + "\n")
    output_file = Path(__file__).parent / "training_data_export.csv"
    num_records = export_training_data_csv(train_df, pred_proba, odds_dict, str(output_file))

    print("="*80)
    print("[SUCCESS] Export complete!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
