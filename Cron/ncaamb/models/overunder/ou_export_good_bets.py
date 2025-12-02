#!/usr/bin/env python3
"""
Export 2025 O/U good bets - filtered high-confidence recommendations
Uses pre-trained ensemble + Random Forest good bets classifier
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from lightgbm import Booster
from catboost import CatBoostRegressor
from datetime import datetime
import pickle

# Add parent directory to path
ncaamb_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ncaamb_dir))


def load_features_by_year(years: list):
    """Load feature files for specified years"""
    features_dir = ncaamb_dir
    all_features = []

    print(f"Loading features for years: {years}")
    for year in years:
        features_file = features_dir / f"features{year}.csv"
        if features_file.exists():
            print(f"  Loading features{year}.csv...")
            try:
                df = pd.read_csv(features_file)
                print(f"    [OK] Loaded {len(df)} games")
                all_features.append(df)
            except Exception as e:
                print(f"    [ERR] Error loading {year}: {e}")

    if not all_features:
        return None

    combined_df = pd.concat(all_features, ignore_index=True)

    # Convert numeric-looking columns to float
    for col in combined_df.columns:
        if any(x in col.lower() for x in ['odds', 'line', 'spread', 'total', 'score', 'pace', 'efg', 'adj', 'avg']):
            combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')

    print(f"[OK] Combined: {len(combined_df)} total games\n")
    return combined_df


def load_trained_models():
    """Load pre-trained XGB, LGB, CatBoost models"""
    saved_dir = Path(__file__).parent / "saved"

    try:
        xgb_model = XGBRegressor()
        xgb_model.load_model(str(saved_dir / "xgboost_model.pkl"))

        lgb_booster = Booster(model_file=str(saved_dir / "lightgbm_model.pkl"))

        cat_model = CatBoostRegressor(verbose=False)
        cat_model.load_model(str(saved_dir / "catboost_model.pkl"))

        print("[OK] Loaded pre-trained ensemble models\n")
        return xgb_model, lgb_booster, cat_model
    except Exception as e:
        print(f"[ERR] Could not load pre-trained models: {e}\n")
        return None, None, None


def get_feature_columns(df: pd.DataFrame) -> list:
    """Identify numeric feature columns - match ou_main.py logic"""
    metadata_cols = {
        'game_id', 'date', 'season', 'team_1', 'team_2',
        'actual_total', 'team_1_conference', 'team_2_conference',
        'team_1_is_home', 'team_2_is_home', 'location',
        'team_1_score', 'team_2_score', 'total_score_outcome',
        'team_1_winloss', 'team_1_leaderboard', 'team_2_leaderboard',
        'team_1_match_hist', 'team_2_match_hist',
        'team_1_hist_count', 'team_2_hist_count', 'start_time', 'game_odds'
    }

    feature_cols = [col for col in df.columns
                   if col not in metadata_cols
                   and np.issubdtype(df[col].dtype, np.number)]
    return feature_cols


def american_to_decimal(american_odds):
    """Convert American odds to decimal"""
    american_odds = float(american_odds)
    if american_odds >= 0:
        return (american_odds / 100) + 1
    else:
        return (100 / abs(american_odds)) + 1


def get_best_odds(row, bet_type):
    """Get best odds across all sportsbooks"""
    if bet_type == 'Over':
        odds_list = [
            (row.get('betonline_over_odds'), 'Betonline'),
            (row.get('bovada_over_odds'), 'Bovada'),
            (row.get('mybookie_over_odds'), 'MyBookie'),
        ]
    else:  # Under
        odds_list = [
            (row.get('betonline_under_odds'), 'Betonline'),
            (row.get('bovada_under_odds'), 'Bovada'),
            (row.get('mybookie_under_odds'), 'MyBookie'),
        ]

    best_odds = None
    best_book = None
    best_decimal = 0

    for odds, book in odds_list:
        if odds and not np.isnan(odds):
            decimal = american_to_decimal(odds)
            if decimal > best_decimal:
                best_decimal = decimal
                best_odds = float(odds)
                best_book = book

    return best_odds, best_book


def main():
    print("\n" + "="*80)
    print("O/U GOOD BETS EXPORT - 2025")
    print("Using pre-trained ensemble + Random Forest classifier")
    print("="*80 + "\n")

    # Load test data
    print("STEP 1: Loading Test Data (2025)")
    print("-"*80 + "\n")
    test_df = load_features_by_year(['2025'])
    if test_df is None:
        return

    # Load models
    print("STEP 2: Loading Pre-trained Models")
    print("-"*80 + "\n")
    xgb_model, lgb_booster, cat_model = load_trained_models()
    if xgb_model is None:
        return

    # Load good bets RF model
    saved_dir = Path(__file__).parent / "saved"
    good_bets_path = saved_dir / "ou_good_bets_final.pkl"
    if good_bets_path.exists():
        with open(good_bets_path, 'rb') as f:
            good_bets_model = pickle.load(f)
        print("[OK] Loaded Good Bets model\n")
    else:
        print(f"[!] Good Bets model not found at {good_bets_path}")
        print("Run train_good_bets_final.py first\n")
        return

    # Get feature columns
    feature_cols = get_feature_columns(test_df)
    print(f"Using {len(feature_cols)} feature columns\n")

    # Prepare features
    X_test = test_df[feature_cols].copy().fillna(0).values

    # Make predictions
    print("STEP 3: Making Ensemble Predictions")
    print("-"*80 + "\n")
    xgb_preds = xgb_model.predict(X_test)
    lgb_preds = lgb_booster.predict(X_test)
    cat_preds = cat_model.predict(X_test)

    # Ensemble: 44.1% XGB + 46.6% LGB + 9.3% CatBoost
    ensemble_preds = (0.441 * xgb_preds + 0.466 * lgb_preds + 0.093 * cat_preds)
    print(f"  [+] Made {len(ensemble_preds)} predictions\n")

    # Create good bets features
    print("STEP 4: Creating Good Bets Features")
    print("-"*80 + "\n")

    betonline_ou_line = test_df['betonline_ou_line'].fillna(0).values
    avg_ou_line = test_df.get('avg_ou_line', pd.Series([0]*len(test_df))).fillna(0).values
    ou_line_variance = test_df.get('ou_line_variance', pd.Series([0]*len(test_df))).fillna(0).values

    # Calculate confidence scores for each model
    xgb_confidence_over = 1.0 / (1.0 + np.exp(-(xgb_preds - betonline_ou_line) / 3.0))
    lgb_confidence_over = 1.0 / (1.0 + np.exp(-(lgb_preds - betonline_ou_line) / 3.0))
    cat_confidence_over = 1.0 / (1.0 + np.exp(-(cat_preds - betonline_ou_line) / 3.0))

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

    # Model disagreement
    model_std_dev = np.std([xgb_confidence_over, lgb_confidence_over, cat_confidence_over], axis=0)

    # Build feature matrix for Good Bets model
    good_bets_features = np.column_stack([
        xgb_confidence_over,
        lgb_confidence_over,
        cat_confidence_over,
        ensemble_confidence_over,
        model_std_dev,
        betonline_ou_line,
        avg_ou_line,
        ou_line_variance
    ])

    # Get good bets probabilities
    good_bets_probs = good_bets_model.predict_proba(good_bets_features)[:, 1]
    print(f"  [+] Generated good bets confidence scores\n")

    # Create betting records
    print("STEP 5: Creating Betting Records")
    print("-"*80 + "\n")

    all_bets = []

    for idx, row in test_df.iterrows():
        game_id = row.get('game_id', '')
        date = row.get('date', '')
        team_1 = row.get('team_1', '')
        team_2 = row.get('team_2', '')
        actual_total = row.get('actual_total')

        # Get O/U lines from all sportsbooks
        betonline_line = row.get('betonline_ou_line')
        bovada_line = row.get('bovada_ou_line')
        mybookie_line = row.get('mybookie_ou_line')

        # Skip if no O/U data
        if all(pd.isna(x) for x in [betonline_line, bovada_line, mybookie_line]):
            continue

        predicted_total = ensemble_preds[idx]
        good_bet_score = good_bets_probs[idx]

        # Only export if good bet score is high enough
        if good_bet_score < 0.50:  # Filter for good bets above 50% confidence
            continue

        confidence = ensemble_confidence_over[idx]

        # Get best odds for Over and Under
        odds_over, book_over = get_best_odds(row, 'Over')
        odds_under, book_under = get_best_odds(row, 'Under')

        # Skip if missing odds
        if odds_over is None or odds_under is None:
            continue

        decimal_over = american_to_decimal(odds_over)
        decimal_under = american_to_decimal(odds_under)

        profit_over = round(10 * (decimal_over - 1), 2)
        profit_under = round(10 * (decimal_under - 1), 2)

        all_bets.append({
            'game_id': game_id,
            'date': date,
            'team_1': team_1,
            'team_2': team_2,
            'betonline_line': round(betonline_line, 1) if not pd.isna(betonline_line) else None,
            'bovada_line': round(bovada_line, 1) if not pd.isna(bovada_line) else None,
            'mybookie_line': round(mybookie_line, 1) if not pd.isna(mybookie_line) else None,
            'predicted_total': round(predicted_total, 1),
            'actual_total': round(actual_total, 1) if not pd.isna(actual_total) else None,
            'confidence': round(confidence, 4),
            'good_bet_score': round(good_bet_score, 4),
            'over_best_book': book_over,
            'over_odds_american': round(odds_over, 0),
            'over_odds_decimal': round(decimal_over, 3),
            'over_potential_profit': profit_over,
            'under_best_book': book_under,
            'under_odds_american': round(odds_under, 0),
            'under_odds_decimal': round(decimal_under, 3),
            'under_potential_profit': profit_under,
        })

    if not all_bets:
        print("No good bets found with score >= 0.50")
        return

    # Export
    print("STEP 6: Exporting to Excel and CSV")
    print("-"*80 + "\n")

    df_bets = pd.DataFrame(all_bets)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    saved_dir = Path(__file__).parent / "saved"

    # Excel export
    excel_filename = saved_dir / f'ou_good_bets_2025_{timestamp}.xlsx'
    with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
        df_bets.to_excel(writer, sheet_name='Good_Bets', index=False)

        from openpyxl.styles import Font, PatternFill, Alignment
        workbook = writer.book
        worksheet = writer.sheets['Good_Bets']

        # Header formatting
        for cell in worksheet[1]:
            cell.font = Font(bold=True, color="FFFFFF")
            cell.fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
            cell.alignment = Alignment(horizontal="center", vertical="center")

        # Adjust column widths
        for col, width in [('A', 18), ('B', 12), ('C', 20), ('D', 20), ('E', 12),
                          ('F', 12), ('G', 12), ('H', 15), ('I', 15), ('J', 12),
                          ('K', 12), ('L', 12), ('M', 12), ('N', 14), ('O', 14)]:
            worksheet.column_dimensions[col].width = width

    print(f"[+] Excel saved: {excel_filename}")

    # CSV export
    csv_filename = saved_dir / f'ou_good_bets_2025_{timestamp}.csv'
    df_bets.to_csv(csv_filename, index=False)
    print(f"[+] CSV saved: {csv_filename}")

    # Summary
    print(f"\n[SUMMARY]")
    print(f"  Good bets found: {len(all_bets)}")
    print(f"  Avg confidence: {df_bets['confidence'].mean():.4f}")
    print(f"  Avg good bet score: {df_bets['good_bet_score'].mean():.4f}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
