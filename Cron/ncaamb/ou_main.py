#!/usr/bin/env python3
"""
Main script for OU model predictions
Scrapes today's schedule and generates predictions
"""

import os
import sys
from datetime import datetime
import pandas as pd
import numpy as np

# Add current directory to path
ncaamb_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ncaamb_dir)

from scrapes.barttorvik_schedule import (
    scrape_barttorvik_schedule,
    format_date_to_mmddyy
)
from scrapes.push_todays_games import push_todays_games_to_db
from scrapes.gamehistory import scrape_game_history
from scrapes.leaderboard import scrape_barttorvik_csv
from load_player_stats import load_player_stats_to_db
from bookmaker.get_data import get_historical_odds, parse_odds_response
from bookmaker.map_game_id import (
    load_team_mappings,
    load_all_game_ids,
    update_odds_with_game_id,
    get_odds_without_game_id,
    format_game_id,
    map_team_name
)
from scrapes import sqlconn
from models.utils import fetch_games, fetch_teams, fetch_leaderboard
from models.build_flat_df import build_flat_df
from models.overunder.build_ou_features import build_ou_features
import polars as pl


def get_todays_date_mmddyy(target_date: str = None):
    """Get date in M/D/YY format (matching Barttorvik super_sked)

    Args:
        target_date: Date string in YYYY-MM-DD format. If None, uses today's date.

    Returns:
        Date in M/D/YY format
    """
    if target_date:
        try:
            date_obj = datetime.strptime(target_date, '%Y-%m-%d')
        except ValueError:
            print(f"[-] Invalid date format: {target_date}. Using today's date instead.")
            date_obj = datetime.now()
    else:
        date_obj = datetime.now()

    month = str(date_obj.month)
    day = str(date_obj.day)
    year = date_obj.strftime('%y')
    return f"{month}/{day}/{year}"


def get_todays_games(season: str = None, target_date: str = None):
    """
    Scrape Barttorvik super_sked schedule and filter for games happening on target date.

    Args:
        season: Season year (e.g., '2026'). If None, uses 2026.
        target_date: Date string in YYYY-MM-DD format. If None, uses today's date.

    Returns:
        DataFrame with games for the specified date or None if none found
    """
    if season is None:
        season = '2026'

    print("="*80)
    print(f"OU MODEL - SCRAPING SCHEDULE FOR SEASON {season}")
    print("="*80)

    # Get the date in M/D/YY format (what Barttorvik super_sked uses)
    target_mmddyy = get_todays_date_mmddyy(target_date)
    print(f"\nLooking for games on: {target_mmddyy}\n")

    # Scrape the schedule
    df = scrape_barttorvik_schedule(season)

    if df is None:
        print("Failed to fetch schedule")
        return None

    # Filter for target date games (convert to string to handle potential type issues)
    target_games = df[df['date'].astype(str) == target_mmddyy].copy()

    if len(target_games) == 0:
        print(f"No games found for {target_mmddyy}")
        return None

    print(f"Found {len(target_games)} games:\n")

    # Display target date games
    for idx, row in target_games.iterrows():
        t1_name = str(row['team1'])
        t2_name = str(row['team2'])
        t1_oe = row['t1oe'] if pd.notna(row['t1oe']) else 'N/A'
        t2_oe = row['t2oe'] if pd.notna(row['t2oe']) else 'N/A'
        prediction = str(row['prediction']) if pd.notna(row['prediction']) else 'N/A'

        print(f"  {t1_name:30} vs {t2_name:30}")
        print(f"    Prediction: {prediction} | T1 OE: {t1_oe} | T2 OE: {t2_oe}")

    return target_games


def fetch_and_push_odds_data():
    """
    Fetch today's odds from OddsAPI and push them to the odds table.
    Also maps game_ids to odds records.

    Returns:
        bool: True if successful, False otherwise
    """
    print("\n" + "-"*80)
    print("STEP 1.5: Fetching odds data from OddsAPI")
    print("-"*80 + "\n")

    # Get today's date in ISO format (9 AM EST)
    today = datetime.now()
    today_iso = f"{today.strftime('%Y-%m-%d')}T09:00:00Z"

    print(f"Fetching odds for {today_iso}...\n")

    # Fetch odds from OddsAPI
    odds_data = get_historical_odds(today_iso)

    if not odds_data or 'data' not in odds_data:
        print("  [-] Failed to fetch odds data from OddsAPI")
        return False

    # Parse the response
    odds_df = parse_odds_response(odds_data, today_iso)

    if odds_df.empty:
        print("  [-] No odds data returned from OddsAPI")
        return False

    print(f"  [+] Retrieved {len(odds_df)} odds records from {len(odds_df['bookmaker'].unique())} bookmakers\n")

    # Load team mappings and available game_ids for matching
    print("Loading team mappings and game_ids for matching...\n")
    mappings = load_team_mappings()
    available_game_ids = load_all_game_ids()

    if not mappings or not available_game_ids:
        print("  [!]  Warning: Could not load team mappings or game_ids, using raw odds data\n")
    else:
        # Map each odds record to find the correct game_id
        matched_count = 0
        unmatched_examples = []

        for idx, row in odds_df.iterrows():
            home_team = row.get('home_team')
            away_team = row.get('away_team')
            start_time = row.get('start_time')

            # Map team names using the mappings file
            mapped_home = map_team_name(home_team, mappings)
            mapped_away = map_team_name(away_team, mappings)

            # Parse start_time from ISO format (e.g., '2025-11-11T03:00:00Z')
            try:
                if isinstance(start_time, str):
                    # Remove 'Z' and parse ISO format
                    start_time_str = start_time.replace('Z', '').split('.')[0]  # Handle both formats
                    start_time_dt = datetime.fromisoformat(start_time_str)
                    start_time_mysql = start_time_dt.strftime('%Y-%m-%d %H:%M:%S')
                elif hasattr(start_time, 'strftime'):
                    start_time_mysql = start_time.strftime('%Y-%m-%d %H:%M:%S')
                else:
                    start_time_mysql = str(start_time)
            except Exception as e:
                print(f"  Error parsing start_time {start_time}: {e}")
                continue

            # Format game_ids (list of ±1 day variations)
            possible_game_ids = format_game_id(start_time_mysql, mapped_home, mapped_away)

            # Find first matching game_id
            matching_game_id = None
            for gid in possible_game_ids:
                if gid in available_game_ids:
                    matching_game_id = gid
                    break

            if matching_game_id:
                odds_df.at[idx, 'game_id'] = matching_game_id
                odds_df.at[idx, 'start_time'] = start_time_mysql
                matched_count += 1
            else:
                # Store example of unmatched record for debugging
                if len(unmatched_examples) < 3:
                    unmatched_examples.append({
                        'home_team': home_team,
                        'away_team': away_team,
                        'mapped_home': mapped_home,
                        'mapped_away': mapped_away,
                        'start_time': start_time_mysql,
                        'possible_game_ids': possible_game_ids
                    })

        print(f"  Matched {matched_count}/{len(odds_df)} odds records to game_ids\n")

        if unmatched_examples and matched_count == 0:
            print("  First few unmatched records:")
            for ex in unmatched_examples:
                print(f"    {ex['home_team']} vs {ex['away_team']}")
                print(f"      Mapped: {ex['mapped_home']} vs {ex['mapped_away']}")
                print(f"      Tried game_ids: {ex['possible_game_ids']}")
            print()

    # Convert all start_time values to MySQL format
    def convert_start_time(start_time):
        try:
            if isinstance(start_time, str):
                # Remove 'Z' and parse ISO format
                start_time_str = start_time.replace('Z', '').split('.')[0]
                start_time_dt = datetime.fromisoformat(start_time_str)
                return start_time_dt.strftime('%Y-%m-%d %H:%M:%S')
            elif hasattr(start_time, 'strftime'):
                return start_time.strftime('%Y-%m-%d %H:%M:%S')
            else:
                return str(start_time)
        except Exception:
            return None

    odds_df['start_time'] = odds_df['start_time'].apply(convert_start_time)

    # Map column names to match odds table schema and remove extra columns
    odds_df = odds_df.rename(columns={
        'h2h_home': 'ml_home',
        'h2h_away': 'ml_away'
    })

    # Keep only columns that exist in the odds table
    keep_columns = ['game_id', 'home_team', 'away_team', 'start_time', 'bookmaker',
                    'ml_home', 'ml_away', 'spread_home', 'spread_pts_home',
                    'spread_away', 'spread_pts_away', 'over_odds', 'under_odds',
                    'over_point', 'under_point']
    odds_df = odds_df[[col for col in keep_columns if col in odds_df.columns]]

    # Filter out records without game_id (didn't match to any game in database)
    records_before = len(odds_df)
    odds_df = odds_df[odds_df['game_id'].notna()]
    records_after = len(odds_df)
    skipped = records_before - records_after

    if skipped > 0:
        print(f"  Skipping {skipped} odds records without game_id match\n")

    if odds_df.empty:
        print("  [-] No matched odds records to push\n")
        return False

    # Push to database
    try:
        success = sqlconn.execute_query(df=odds_df, table_name='odds', if_exists='append')
        if success:
            print(f"  [+] Pushed {len(odds_df)} odds records to odds table\n")
            return True
        else:
            print(f"  [-] Failed to push odds records to database\n")
            return False
    except Exception as e:
        print(f"  [-] Error pushing odds to database: {e}\n")
        return False


def fetch_and_push_leaderboard(season: str = '2026'):
    """
    Fetch today's leaderboard from Barttorvik and push to database.

    Args:
        season: Season year (e.g., '2026')

    Returns:
        bool: True if successful, False otherwise
    """
    print("\n" + "-"*80)
    print("STEP 1.3: Fetching and pushing today's leaderboard")
    print("-"*80 + "\n")

    try:
        today = datetime.now()
        month = f"{today.month:02d}"
        day = f"{today.day:02d}"
        end_date = f"{month}{day}"

        print(f"Fetching leaderboard for {season} (end date: {month}/{day})...\n")
        leaderboard_df = scrape_barttorvik_csv(year=season, end_date=end_date)

        if leaderboard_df is not None and len(leaderboard_df) > 0:
            print(f"  [+] Retrieved leaderboard with {len(leaderboard_df)} teams\n")
            return True
        else:
            print(f"  [-] Failed to fetch leaderboard data\n")
            return False

    except Exception as e:
        print(f"  [-] Error fetching leaderboard: {e}\n")
        return False


def push_match_history(game_data: dict, season: str):
    """
    Push match history from game_data dictionary to ncaamb.games table.
    First inserts missing teams from opponent names to avoid FK constraint failures.

    Args:
        game_data: Dictionary with game histories already fetched
        season: Season year (e.g., '2026')

    Returns:
        int: Number of teams with history pushed
    """
    print("\n" + "-"*80)
    print("STEP 1.5: Pushing match history for today's teams")
    print("-"*80 + "\n")

    teams_pushed = set()
    total_games_pushed = 0

    # First pass: collect all unique teams from game histories (including opponents)
    all_teams_in_history = set()
    for game_id, data in game_data.items():
        if data.get('team_1_history') is not None:
            for idx, row in data['team_1_history'].iterrows():
                all_teams_in_history.add(row.get('team'))
                all_teams_in_history.add(row.get('opponent'))

        if data.get('team_2_history') is not None:
            for idx, row in data['team_2_history'].iterrows():
                all_teams_in_history.add(row.get('team'))
                all_teams_in_history.add(row.get('opponent'))

    # Insert missing teams into teams table
    if all_teams_in_history:
        try:
            conn = sqlconn.create_connection()
            if conn:
                cursor = conn.cursor()
                teams_inserted = 0

                for team in sorted(all_teams_in_history):
                    # Check if team already exists for this season
                    check_query = "SELECT team_name FROM teams WHERE season = %s AND team_name = %s LIMIT 1"
                    cursor.execute(check_query, (int(season), team))
                    result = cursor.fetchone()

                    if not result:
                        # Insert team with null fields (will be populated later if needed)
                        insert_query = "INSERT INTO teams (season, team_name, conference) VALUES (%s, %s, NULL)"
                        try:
                            cursor.execute(insert_query, (int(season), team))
                            teams_inserted += 1
                        except Exception as e:
                            # Silently ignore if team already exists (race condition)
                            pass

                conn.commit()
                cursor.close()
                conn.close()

                if teams_inserted > 0:
                    print(f"  [+] Inserted {teams_inserted} missing teams from match history\n")
        except Exception as e:
            print(f"  [!]  Warning: Could not insert teams for match history: {e}\n")

    # Extract game histories from game_data
    for game_id, data in game_data.items():
        # Push team_1 history
        if data.get('team_1_history') is not None and len(data['team_1_history']) > 0:
            team1 = data['team_1']
            if team1 not in teams_pushed:
                try:
                    sqlconn.push_to_games(data['team_1_history'], int(season), season)
                    teams_pushed.add(team1)
                    total_games_pushed += len(data['team_1_history'])
                    print(f"  [+] {team1}: {len(data['team_1_history'])} games pushed")
                except Exception as e:
                    print(f"  [-] {team1}: Error - {e}")

        # Push team_2 history
        if data.get('team_2_history') is not None and len(data['team_2_history']) > 0:
            team2 = data['team_2']
            if team2 not in teams_pushed:
                try:
                    sqlconn.push_to_games(data['team_2_history'], int(season), season)
                    teams_pushed.add(team2)
                    total_games_pushed += len(data['team_2_history'])
                    print(f"  [+] {team2}: {len(data['team_2_history'])} games pushed")
                except Exception as e:
                    print(f"  [-] {team2}: Error - {e}")

    print(f"\n  Pushed {total_games_pushed} total games from {len(teams_pushed)} teams\n")
    return len(teams_pushed)


def load_player_stats(season: str = '2026'):
    """
    Load player stats for the given season from Barttorvik API.

    Args:
        season: Season year (e.g., '2026')

    Returns:
        bool: True if successful, False otherwise
    """
    print("\n" + "-"*80)
    print("STEP 2: Loading player stats")
    print("-"*80 + "\n")

    try:
        print(f"Loading player stats for season {season}...")
        load_player_stats_to_db(year=season, season=int(season))
        print()
        return True
    except Exception as e:
        print(f"  [!]  Warning: Error loading player stats: {e}\n")
        # Don't fail completely, player stats are optional
        return True


def generate_features(flat_df: pl.DataFrame):
    """
    Generate O/U features from flat DataFrame.

    Args:
        flat_df: Polars DataFrame with flat game data

    Returns:
        Polars DataFrame with generated features or None if failed
    """
    print("\n" + "-"*80)
    print("STEP 4: Generating O/U features")
    print("-"*80 + "\n")

    try:
        print("Building features from flat DataFrame...")
        features_df = build_ou_features(flat_df)

        print(f"  [+] Generated {len(features_df)} games with {len(features_df.columns)} feature columns\n")

        # Show sample for validation
        print("Sample row for validation:")
        sample = features_df.select(['game_id', 'date', 'team_1', 'team_2', 'actual_total']).head(1)
        # Don't print the dataframe directly due to encoding issues, just show it's there
        if len(sample) > 0:
            row = sample.to_dicts()[0]
            print(f"  Game ID: {row['game_id']}, Date: {row['date']}, Teams: {row['team_1']} vs {row['team_2']}, Total: {row['actual_total']}")
        print()

        # Save to CSV
        print("Saving features to tempfeatures.csv...")
        features_df.write_csv('tempfeatures.csv')
        print(f"  [+] Features saved to tempfeatures.csv")

        # Check null percentages
        print("\nNull value analysis (top 20 columns by null %):")
        null_pct = []
        for col in features_df.columns:
            null_count = features_df[col].null_count()
            null_percent = (null_count / len(features_df)) * 100 if len(features_df) > 0 else 0
            if null_percent > 0:
                null_pct.append((col, null_percent))

        # Sort by null percentage descending
        null_pct.sort(key=lambda x: x[1], reverse=True)
        for col, pct in null_pct[:20]:
            print(f"  {col:40} {pct:6.1f}%")
        print()

        return features_df

    except Exception as e:
        print(f"  [-] Error generating features: {e}\n")
        import traceback
        traceback.print_exc()
        return None


def make_ou_predictions(features_df: pl.DataFrame):
    """
    Load trained models and make O/U predictions using ensemble.
    Also loads Good Bets model for confidence-based bet recommendations.

    Args:
        features_df: Polars DataFrame with generated features

    Returns:
        DataFrame with predictions or None if failed
    """
    print("\n" + "-"*80)
    print("STEP 5: Making O/U predictions with ensemble")
    print("-"*80 + "\n")

    try:
        import json
        from pathlib import Path
        from xgboost import XGBRegressor
        from lightgbm import LGBMRegressor
        from catboost import CatBoostRegressor
        import pickle

        # Load models from saved directory
        models_dir = Path(__file__).parent / "models" / "overunder" / "saved"

        print(f"Loading trained models from {models_dir}...\n")

        # Load XGBoost model (JSON format)
        xgb_model = XGBRegressor()
        xgb_model.load_model(str(models_dir / "xgboost_model.pkl"))
        print("  [+] Loaded XGBoost model")

        # Load LightGBM model (JSON format)
        from lightgbm import Booster
        lgb_booster = Booster(model_file=str(models_dir / "lightgbm_model.pkl"))
        print("  [+] Loaded LightGBM model")

        # Load CatBoost model (JSON format)
        cb_model = CatBoostRegressor(verbose=False)
        cb_model.load_model(str(models_dir / "catboost_model.pkl"))
        print("  [+] Loaded CatBoost model")

        # Load Good Bets model
        good_bets_path = models_dir / "ou_good_bets_final.pkl"
        if good_bets_path.exists():
            with open(good_bets_path, 'rb') as f:
                good_bets_model = pickle.load(f)
            print("  [+] Loaded Good Bets model\n")
        else:
            print(f"  [!] Good Bets model not found at {good_bets_path}")
            good_bets_model = None
            print()

        # Convert Polars to Pandas for sklearn compatibility
        print("Converting features to Pandas for prediction...")
        features_pd = features_df.to_pandas()

        # Get feature columns (all except metadata/object columns)
        # Reference columns from features2024.csv: game_id, date, team_1, team_2, team_1_score, team_2_score, actual_total
        # Then all numeric features
        metadata_cols = {'game_id', 'date', 'season', 'team_1', 'team_2', 'actual_total',
                         'team_1_conference', 'team_2_conference', 'team_1_is_home', 'team_2_is_home',
                         'location', 'team_1_score', 'team_2_score', 'total_score_outcome', 'team_1_winloss',
                         'team_1_leaderboard', 'team_2_leaderboard', 'team_1_match_hist', 'team_2_match_hist',
                         'team_1_hist_count', 'team_2_hist_count', 'start_time', 'game_odds'}

        # First, convert all non-metadata object columns to numeric (they may be strings from build_ou_features)
        print("Converting string columns to numeric...")
        for col in features_pd.columns:
            if col not in metadata_cols and features_pd[col].dtype == 'object':
                try:
                    features_pd[col] = pd.to_numeric(features_pd[col], errors='coerce')
                except:
                    pass

        # Get numeric columns only (excluding metadata)
        import numpy as np
        feature_cols = [col for col in features_pd.columns
                       if col not in metadata_cols
                       and np.issubdtype(features_pd[col].dtype, np.number)]

        print(f"  [+] Using {len(feature_cols)} numeric feature columns\n")

        # Get feature data
        X = features_pd[feature_cols].copy()

        # Check for NaN values and fill them with 0
        null_cols = X.columns[X.isnull().any()].tolist()
        if null_cols:
            print(f"  [!]  {len(null_cols)} columns have NaN values - filling with 0")
            print(f"     Examples: {null_cols[:5]}{'...' if len(null_cols) > 5 else ''}\n")
            X = X.fillna(0)

        print(f"Feature matrix shape: {X.shape}")
        print(f"Models were trained on: (n_samples, 321)\n")

        # Make predictions with each model
        print("Making predictions...\n")
        xgb_preds = xgb_model.predict(X)
        print(f"  [+] XGBoost predictions shape: {xgb_preds.shape}")

        lgb_preds = lgb_booster.predict(X.values)
        print(f"  [+] LightGBM predictions shape: {lgb_preds.shape}")

        cb_preds = cb_model.predict(X)
        print(f"  [+] CatBoost predictions shape: {cb_preds.shape}\n")

        # Ensemble: equal weights (can be optimized with BEST_OPTIMIZATIONS weights)
        # BEST_OPTIMIZATIONS: 44.1% XGB, 46.6% LGB, 9.3% CatBoost
        ensemble_preds = (0.441 * xgb_preds + 0.466 * lgb_preds + 0.093 * cb_preds)

        # Calculate ensemble confidence (sigmoid of difference from O/U line)
        # Using betonline_ou_line as reference for confidence calculation
        betonline_ou_line = features_pd['betonline_ou_line'].fillna(0).values
        ensemble_confidence = 1.0 / (1.0 + np.exp(-(ensemble_preds - betonline_ou_line) / 3.0))
        ensemble_confidence = np.clip(ensemble_confidence, 0.01, 0.99)

        print("="*80)
        print("ENSEMBLE PREDICTIONS (Total Points O/U)")
        print("="*80 + "\n")

        # Display predictions for each game
        for idx, row in features_pd.iterrows():
            game_id = row['game_id']
            team_1 = row['team_1']
            team_2 = row['team_2']
            actual = row['actual_total'] if pd.notna(row['actual_total']) else 'N/A'

            xgb_pred = xgb_preds[idx]
            lgb_pred = lgb_preds[idx]
            cb_pred = cb_preds[idx]
            ensemble_pred = ensemble_preds[idx]
            confidence = ensemble_confidence[idx]

            print(f"{team_1} vs {team_2}")
            print(f"  Game ID: {game_id}")
            print(f"  XGBoost:  {xgb_pred:.1f}")
            print(f"  LightGBM: {lgb_pred:.1f}")
            print(f"  CatBoost: {cb_pred:.1f}")
            print(f"  Ensemble: {ensemble_pred:.1f}")
            print(f"  Confidence: {confidence:.2f}")
            if actual != 'N/A':
                print(f"  Actual:   {actual:.1f}")
            print()

        # Make Good Bets predictions if model loaded
        good_bets_probs = None
        if good_bets_model is not None:
            print("\n[+] Making Good Bets predictions...\n")
            try:
                # Build Good Bets input features from ensemble predictions
                betonline_ou_line_vals = features_pd['betonline_ou_line'].fillna(0).values
                avg_ou_line_vals = features_pd.get('avg_ou_line', pd.Series([0]*len(features_pd))).fillna(0).values
                ou_line_variance_vals = features_pd.get('ou_line_variance', pd.Series([0]*len(features_pd))).fillna(0).values

                # Calculate confidence scores for each model
                xgb_confidence_over = 1.0 / (1.0 + np.exp(-(xgb_preds - betonline_ou_line_vals) / 3.0))
                lgb_confidence_over = 1.0 / (1.0 + np.exp(-(lgb_preds - betonline_ou_line_vals) / 3.0))
                cat_confidence_over = 1.0 / (1.0 + np.exp(-(cb_preds - betonline_ou_line_vals) / 3.0))

                # Clip to valid range
                xgb_confidence_over = np.clip(xgb_confidence_over, 0.01, 0.99)
                lgb_confidence_over = np.clip(lgb_confidence_over, 0.01, 0.99)
                cat_confidence_over = np.clip(cat_confidence_over, 0.01, 0.99)

                # Ensemble confidence
                ensemble_confidence_over = (
                    0.441 * xgb_confidence_over +
                    0.466 * lgb_confidence_over +
                    0.093 * cat_confidence_over
                )

                # Model std dev
                model_std_dev = np.std([xgb_confidence_over, lgb_confidence_over, cat_confidence_over], axis=0)

                # Build feature matrix for Good Bets model
                good_bets_features = np.column_stack([
                    xgb_confidence_over,
                    lgb_confidence_over,
                    cat_confidence_over,
                    ensemble_confidence_over,
                    model_std_dev,
                    betonline_ou_line_vals,
                    avg_ou_line_vals,
                    ou_line_variance_vals
                ])

                good_bets_probs = good_bets_model.predict_proba(good_bets_features)[:, 1]
                print(f"[+] Generated Good Bets confidence scores for {len(good_bets_probs)} games\n")
            except Exception as e:
                print(f"[-] Error generating Good Bets predictions: {e}\n")
                import traceback
                traceback.print_exc()
                good_bets_probs = None

        # Calculate ensemble_confidence_over if not already done (for Good Bets predictions)
        if good_bets_probs is None:
            # If Good Bets model failed, still calculate ensemble confidence for output
            betonline_ou_line_vals = features_pd['betonline_ou_line'].fillna(0).values

            xgb_confidence_over = 1.0 / (1.0 + np.exp(-(xgb_preds - betonline_ou_line_vals) / 3.0))
            lgb_confidence_over = 1.0 / (1.0 + np.exp(-(lgb_preds - betonline_ou_line_vals) / 3.0))
            cat_confidence_over = 1.0 / (1.0 + np.exp(-(cb_preds - betonline_ou_line_vals) / 3.0))

            xgb_confidence_over = np.clip(xgb_confidence_over, 0.01, 0.99)
            lgb_confidence_over = np.clip(lgb_confidence_over, 0.01, 0.99)
            cat_confidence_over = np.clip(cat_confidence_over, 0.01, 0.99)

            ensemble_confidence_over = (
                0.441 * xgb_confidence_over +
                0.466 * lgb_confidence_over +
                0.093 * cat_confidence_over
            )
        else:
            # Extract ensemble_confidence_over from the features we just built
            ensemble_confidence_over = ensemble_confidence_over if 'ensemble_confidence_over' in locals() else ensemble_confidence

        # Add predictions to dataframe
        cols_to_add = [
            pl.Series("xgb_pred", xgb_preds),
            pl.Series("lgb_pred", lgb_preds),
            pl.Series("cb_pred", cb_preds),
            pl.Series("ensemble_pred", ensemble_preds),
            pl.Series("ensemble_confidence", ensemble_confidence_over)
        ]

        if good_bets_probs is not None:
            cols_to_add.append(pl.Series("good_bets_confidence", good_bets_probs))

        features_df = features_df.with_columns(cols_to_add)

        print("="*80 + "\n")

        return features_df

    except Exception as e:
        print(f"  [-] Error making predictions: {e}\n")
        import traceback
        traceback.print_exc()
        return None


def build_todays_games_df(season: str = '2026', target_date: str = None):
    """
    Build flat DataFrame with games for target date and all their data.
    Queries games table for target date, fetches odds, and builds flat dataset.

    Args:
        season: Season year (e.g., '2026')
        target_date: Date string in YYYY-MM-DD format. If None, uses today's date.

    Returns:
        Polars DataFrame with target date's games data or None if failed
    """
    print("\n" + "-"*80)
    print("STEP 3: Building flat DataFrame for games")
    print("-"*80 + "\n")

    # Get target date in YYYY-MM-DD format
    if target_date:
        try:
            datetime.strptime(target_date, '%Y-%m-%d')
            target_date_str = target_date
        except ValueError:
            print(f"[-] Invalid date format: {target_date}. Using today's date instead.")
            target_date_str = datetime.now().strftime('%Y-%m-%d')
    else:
        target_date_str = datetime.now().strftime('%Y-%m-%d')

    print(f"Building flat DataFrame for games on {target_date_str}...\n")

    try:
        # Load data
        print("Loading games, teams, and leaderboard data...")
        games_df = fetch_games(season=int(season))
        teams_df = fetch_teams(season=int(season))
        leaderboard_df = fetch_leaderboard()
        print(f"  [+] Loaded {len(games_df)} games, {len(teams_df)} teams\n")

        # Fetch odds for all games first
        print("Fetching odds for all games...")
        game_ids = games_df['game_id'].to_list()

        odds_dict = {}
        if game_ids:
            conn = sqlconn.create_connection()
            if conn:
                placeholders = ','.join(['%s'] * len(game_ids))
                query = f"""
                    SELECT game_id, bookmaker, ml_home, ml_away, spread_home, spread_pts_home,
                           spread_away, spread_pts_away, over_odds, under_odds, over_point, under_point, start_time
                    FROM odds
                    WHERE game_id IN ({placeholders})
                    ORDER BY game_id, bookmaker, start_time
                """
                results = sqlconn.fetch(conn, query, tuple(game_ids))
                conn.close()

                # Group results by game_id
                for row in results:
                    gid = row['game_id']
                    if gid not in odds_dict:
                        odds_dict[gid] = []
                    odds_dict[gid].append(row)

                print(f"  [+] Retrieved odds for {len(odds_dict)} games\n")
            else:
                print("  [!] Warning: Could not connect to database for odds\n")
        else:
            print("  [*] No games found for today, skipping odds fetch\n")

        # Build flat dataset for target date
        print(f"Building flat DataFrame for {target_date_str}...")
        flat_df = build_flat_df(
            season=int(season),
            target_date=target_date_str,
            games_df=games_df,
            teams_df=teams_df,
            leaderboard_df=leaderboard_df,
            player_stats_df=None,
            odds_dict=odds_dict,
            filter_incomplete_data=False,  # Set to False to keep games with nulls
            min_match_history=1,
            require_leaderboard=False  # Allow games with missing leaderboard
        )

        if flat_df.is_empty():
            print(f"  [-] No games found for {target_date_str}")
            return None

        print(f"  [+] Built flat dataset with {len(flat_df)} games\n")

        # Add odds as a column (for feature engineering)
        print("Adding odds column...")
        game_ids_in_range = flat_df['game_id'].to_list()
        odds_list = [odds_dict.get(game_id, []) for game_id in game_ids_in_range]
        flat_df = flat_df.with_columns(pl.Series("game_odds", odds_list, strict=False))
        print(f"  [+] DataFrame shape: {flat_df.shape}\n")

        return flat_df

    except Exception as e:
        print(f"  [-] Error building games DataFrame: {e}\n")
        import traceback
        traceback.print_exc()
        return None


def get_game_data_for_games(games_df: pd.DataFrame, season: str):
    """
    Fetch game history and leaderboard data for each team in today's games.

    Args:
        games_df: DataFrame with today's games
        season: Season year (e.g., '2026')

    Returns:
        Dictionary with game_id as key and game data as value
    """
    game_data = {}

    print("\n" + "-"*80)
    print("STEP 2: Fetching game history and leaderboard data")
    print("-"*80 + "\n")

    # Get all unique teams from today's games
    teams = set()
    for idx, row in games_df.iterrows():
        teams.add(str(row['team1']).strip())
        teams.add(str(row['team2']).strip())

    total_teams = len(teams)
    print(f"Fetching data for {total_teams} unique teams...\n")

    # Fetch game history for each team
    game_histories = {}
    for team in sorted(teams):
        try:
            hist = scrape_game_history(year=season, team=team)
            if hist is not None:
                game_histories[team] = hist
                print(f"  [+] {team}: {len(hist)} games retrieved")
            else:
                print(f"  [-] {team}: No game history found")
                game_histories[team] = None
        except Exception as e:
            print(f"  [-] {team}: Error - {e}")
            game_histories[team] = None

    # Create game data dictionary with game histories
    print(f"\nBuilding game data for {len(games_df)} games...\n")
    for idx, row in games_df.iterrows():
        sched_team1 = str(row['team1']).strip()
        sched_team2 = str(row['team2']).strip()

        # Assign team_1 and team_2 based on ALPHABETICAL order (not schedule order)
        teams_sorted = sorted([sched_team1, sched_team2])
        team_1 = teams_sorted[0]
        team_2 = teams_sorted[1]

        game_id = f"{row['date'].replace('/', '')}_{team_1}_{team_2}"

        game_data[game_id] = {
            'team_1': team_1,
            'team_2': team_2,
            'date': row['date'],
            'team_1_history': game_histories.get(team_1),
            'team_2_history': game_histories.get(team_2),
        }

        t1_hist_count = len(game_histories.get(team_1, [])) if game_histories.get(team_1) is not None else 0
        t2_hist_count = len(game_histories.get(team_2, [])) if game_histories.get(team_2) is not None else 0

        print(f"  {team_1:25} vs {team_2:25} | Histories: {t1_hist_count}/{t2_hist_count}")

    return game_data


def main(target_date: str = None):
    print("\n")
    features_df = None
    predictions_df = None

    todays_games = get_todays_games(target_date=target_date)

    if todays_games is not None:
        print(f"\n{len(todays_games)} games to process")

        # Step 1: Push games to database
        print("\n" + "-"*80)
        print("STEP 1: Pushing games to database")
        print("-"*80 + "\n")
        SCRAPEDATA = False
        if SCRAPEDATA:
            success = push_todays_games_to_db(todays_games)
            game_data = get_game_data_for_games(todays_games, season='2026')
        else:
            success=True

        if success:
            if SCRAPEDATA:
                # Step 1.3: Fetch and push leaderboard
                fetch_and_push_leaderboard(season='2026')

                # # Step 1.5: Push match history for today's teams
                push_match_history(game_data, season='2026')

                # # Step 2: Load player stats
                load_player_stats(season='2026')

                # # Step 1.5b: Fetch and push odds data
                odds_success = fetch_and_push_odds_data()
            else:
                odds_success = True
            if odds_success:
                # Step 3: Build flat DataFrame for target date
                todays_games_df = build_todays_games_df(season='2026', target_date=target_date)

                if todays_games_df is not None:
                    # Step 4: Generate features
                    features_df = generate_features(todays_games_df)

                    if features_df is not None:
                        # Step 5: Make O/U predictions
                        predictions_df = make_ou_predictions(features_df)

                        if predictions_df is not None:
                            print("[+] Prediction pipeline complete!")
                        else:
                            print("\n[!]  Warning: Could not make predictions")
                    else:
                        print("\n[!]  Warning: Could not generate features")
                else:
                    print("\n[!]  Warning: Could not build flat DataFrame for today's games")
            else:
                print("\nWarning: Error fetching odds data, but continuing with other steps")
                # Still proceed with game data fetching even if odds failed
                # game_data = get_game_data_for_games(todays_games, season='2026')
        else:
            print("\nError pushing games to database")
    else:
        print("\nNo games today. Check back later!")

    print("\n" + "="*80 + "\n")
    
    return features_df, predictions_df


if __name__ == "__main__":
    main()
