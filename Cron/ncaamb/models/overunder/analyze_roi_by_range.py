#!/usr/bin/env python3
"""
Analyze ROI by confidence range and ensemble difference for all 3 sportsbooks
Shows correlation between ensemble difference and ROI
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
ncaamb_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ncaamb_dir))


def load_features_by_year(years: list):
    """Load feature files for specified years"""
    all_features = []
    print(f"Loading features for years: {years}")

    for year in years:
        features_file = ncaamb_dir / f"features{year}.csv"
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

    # Convert numeric columns
    for col in combined_df.columns:
        if any(x in col.lower() for x in ['odds', 'line', 'spread', 'total', 'score', 'pace', 'efg', 'adj', 'avg']):
            combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')

    print(f"[OK] Combined: {len(combined_df)} total games\n")
    return combined_df


def create_ou_target_variable(df, ou_line_col):
    """Create binary target variable for over/under betting"""
    df_filtered = df.dropna(subset=['actual_total', ou_line_col]).copy()

    before = len(df)
    after = len(df_filtered)
    print(f"Filtered for {ou_line_col}: removed {before - after}, kept {after}")

    df_filtered['ou_target'] = (df_filtered['actual_total'] > df_filtered[ou_line_col]).astype(int)

    over_count = (df_filtered['ou_target'] == 1).sum()
    under_count = (df_filtered['ou_target'] == 0).sum()

    print(f"Target distribution:")
    print(f"  Over hits (1):  {over_count}")
    print(f"  Under hits (0): {under_count}\n")

    return df_filtered


def load_trained_models():
    """Load pre-trained XGB, LGB, CatBoost models"""
    saved_dir = Path(__file__).parent / "saved"

    try:
        xgb_model = xgb.XGBRegressor()
        xgb_model.load_model(str(saved_dir / "xgboost_model.pkl"))
        lgb_model = lgb.Booster(model_file=str(saved_dir / "lightgbm_model.pkl"))
        cat_model = CatBoostRegressor(verbose=False)
        cat_model.load_model(str(saved_dir / "catboost_model.pkl"), format='cbm')
        print("[OK] Loaded pre-trained ensemble models\n")
        return xgb_model, lgb_model, cat_model
    except Exception as e:
        print(f"[ERR] Could not load pre-trained models: {e}\n")
        return None, None, None


def get_predictions_all_models(df, xgb_model, lgb_model, cat_model):
    """Get predictions from all 3 models"""
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    exclude = {'actual_total', 'team_1_score', 'team_2_score', 'game_id', 'date', 'team_1', 'team_2', 'ou_target'}
    feature_cols = [c for c in numeric_cols if c not in exclude]

    X = df[feature_cols].fillna(0)

    xgb_preds = xgb_model.predict(X)
    lgb_preds = lgb_model.predict(X, num_iteration=lgb_model.best_iteration)
    cat_preds = cat_model.predict(X)

    return xgb_preds, lgb_preds, cat_preds


def create_ou_good_bets_data(df, xgb_preds, lgb_preds, cat_preds, ou_line_col):
    """Create good bets features"""
    df_filtered = df.dropna(subset=['actual_total', ou_line_col]).copy()
    df_filtered['xgb_point_pred'] = xgb_preds[:len(df_filtered)]
    df_filtered['lgb_point_pred'] = lgb_preds[:len(df_filtered)]
    df_filtered['cat_point_pred'] = cat_preds[:len(df_filtered)]

    # Create confidence scores
    df_filtered['xgb_confidence_over'] = 1.0 / (1.0 + np.exp(-(df_filtered['xgb_point_pred'] - df_filtered[ou_line_col]) / 3.0))
    df_filtered['lgb_confidence_over'] = 1.0 / (1.0 + np.exp(-(df_filtered['lgb_point_pred'] - df_filtered[ou_line_col]) / 3.0))
    df_filtered['cat_confidence_over'] = 1.0 / (1.0 + np.exp(-(df_filtered['cat_point_pred'] - df_filtered[ou_line_col]) / 3.0))

    for col in ['xgb_confidence_over', 'lgb_confidence_over', 'cat_confidence_over']:
        df_filtered[col] = df_filtered[col].clip(0.01, 0.99)

    # Ensemble confidence
    df_filtered['ensemble_confidence_over'] = (
        0.441 * df_filtered['xgb_confidence_over'] +
        0.466 * df_filtered['lgb_confidence_over'] +
        0.093 * df_filtered['cat_confidence_over']
    )

    # Ensemble point prediction
    df_filtered['ensemble_point_pred'] = (
        0.441 * df_filtered['xgb_point_pred'] +
        0.466 * df_filtered['lgb_point_pred'] +
        0.093 * df_filtered['cat_point_pred']
    )

    # Model std dev
    df_filtered['model_std_dev'] = np.std([
        df_filtered['xgb_confidence_over'].values,
        df_filtered['lgb_confidence_over'].values,
        df_filtered['cat_confidence_over'].values
    ], axis=0)

    # Build features
    feature_cols = [
        'xgb_confidence_over', 'lgb_confidence_over', 'cat_confidence_over',
        'ensemble_confidence_over', 'model_std_dev',
        ou_line_col, 'avg_ou_line', 'ou_line_variance'
    ]

    feature_cols = [c for c in feature_cols if c in df_filtered.columns]
    X = df_filtered[feature_cols].fillna(0)
    y = df_filtered['ou_target'].values

    return X.values, y, df_filtered, feature_cols


def calculate_bet_roi(wins, total_bets, stake=10.0):
    """Calculate ROI for -110 odds"""
    if total_bets == 0:
        return 0, 0
    profit_if_win = stake * (100 / 110)  # $9.09 for $10 stake
    loss_if_lose = stake
    total_profit = wins * profit_if_win - (total_bets - wins) * loss_if_lose
    roi = (total_profit / (total_bets * stake)) * 100
    return roi, total_profit


def analyze_roi_by_range(ensemble_confidence, y_actual, ou_line, ensemble_point_pred):
    """Analyze ROI by 0.05 increment ranges and show ensemble difference correlation"""
    results = []

    # Define ranges with 0.05 increments
    ranges = []

    # UNDER ranges: 0.01-0.05, 0.05-0.10, ..., 0.45-0.50
    for lower in np.arange(0.01, 0.50, 0.05):
        upper = round(lower + 0.05, 2)
        ranges.append((lower, upper, "UNDER"))

    # OVER ranges: 0.50-0.55, 0.55-0.60, ..., 0.95-1.00
    for lower in np.arange(0.50, 0.95, 0.05):
        upper = round(lower + 0.05, 2)
        ranges.append((lower, upper, "OVER"))

    for lower, upper, bet_type in ranges:
        mask = (ensemble_confidence >= lower) & (ensemble_confidence < upper)
        total_bets = mask.sum()

        if bet_type == "OVER":
            wins = np.sum((y_actual == 1) & mask)
        else:  # UNDER
            wins = np.sum((y_actual == 0) & mask)

        if total_bets > 0:
            win_rate = (wins / total_bets) * 100
            roi, total_profit = calculate_bet_roi(wins, total_bets)

            # Calculate average ensemble point difference (prediction - line)
            avg_ensemble_diff = (ensemble_point_pred[mask] - ou_line[mask]).mean()
            min_ensemble_diff = (ensemble_point_pred[mask] - ou_line[mask]).min()
            max_ensemble_diff = (ensemble_point_pred[mask] - ou_line[mask]).max()
        else:
            win_rate = 0
            roi = 0
            total_profit = 0
            avg_ensemble_diff = 0
            min_ensemble_diff = 0
            max_ensemble_diff = 0

        results.append({
            'Range': f"{lower:.2f}-{upper:.2f}",
            'Bet Type': bet_type,
            'Bets': int(total_bets),
            'Wins': int(wins),
            'Win Rate %': f"{win_rate:.1f}%",
            'ROI %': f"{roi:.1f}%",
            'Avg Diff': f"{avg_ensemble_diff:.2f}",
            'Min Diff': f"{min_ensemble_diff:.2f}",
            'Max Diff': f"{max_ensemble_diff:.2f}",
            'Total Profit': f"${total_profit:.2f}"
        })

    return pd.DataFrame(results)


def main():
    print("="*120)
    print("ROI ANALYSIS BY CONFIDENCE RANGE AND ENSEMBLE DIFFERENCE - ALL SPORTSBOOKS")
    print("="*120 + "\n")

    # Load ensemble models
    print("[STEP 0] Loading pre-trained ensemble models")
    xgb_model, lgb_model, cat_model = load_trained_models()
    if not all([xgb_model, lgb_model, cat_model]):
        print("[-] Could not load models, exiting")
        return

    # Load test data (2025)
    print("[STEP 1] Loading test data (2025)")
    test_df = load_features_by_year(['2025'])

    print("[STEP 2] Getting ensemble predictions for test data")
    xgb_preds_test, lgb_preds_test, cat_preds_test = get_predictions_all_models(test_df, xgb_model, lgb_model, cat_model)

    # Dictionary to store results for each sportsbook
    all_results = {}
    sportsbooks = [
        ('BetOnline', 'betonline_ou_line'),
        ('MyBookie', 'mybookie_ou_line'),
        ('Bovada', 'bovada_ou_line'),
    ]

    for sportsbook_name, ou_line_col in sportsbooks:
        print(f"\n{'='*120}")
        print(f"ANALYZING {sportsbook_name.upper()}")
        print(f"{'='*120}\n")

        # Create target for this sportsbook
        print(f"[TEST] Creating target for {sportsbook_name}")
        test_df_sb = create_ou_target_variable(test_df, ou_line_col)

        print(f"[TEST] Creating good bets features for {sportsbook_name}")
        X_bets_test, y_bets_test, test_df_full, _ = create_ou_good_bets_data(
            test_df_sb, xgb_preds_test, lgb_preds_test, cat_preds_test, ou_line_col
        )

        # Get ensemble confidence values, ensemble point predictions, and O/U lines
        ensemble_confidence = test_df_full['ensemble_confidence_over'].values
        ensemble_point_pred = test_df_full['ensemble_point_pred'].values
        ou_lines = test_df_full[ou_line_col].values

        # Analyze ranges
        print(f"[ANALYSIS] Analyzing ranges for {sportsbook_name}")
        ranges_df = analyze_roi_by_range(ensemble_confidence, y_bets_test, ou_lines, ensemble_point_pred)
        print(f"[OK] Analyzed {len(ranges_df)} ranges\n")

        all_results[sportsbook_name] = ranges_df

    # Export all to Excel
    output_file = ncaamb_dir / "ou_roi_by_range_analysis.xlsx"
    print(f"\n[EXPORT] Exporting analysis to Excel: {output_file}")

    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        for sportsbook_name, data in all_results.items():
            sheet_name = f"{sportsbook_name}"
            data.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"[OK] Export complete\n")

    # Print summary
    print("="*120)
    print("ROI ANALYSIS BY RANGE - ALL SPORTSBOOKS")
    print("="*120 + "\n")

    for sportsbook_name, data in all_results.items():
        print(f"\n{sportsbook_name.upper()}")
        print("-"*120)
        print(data.to_string(index=False))
        print()

    # Find best ranges for each sportsbook
    print("\n" + "="*120)
    print("TOP 5 RANGES BY ROI - EACH SPORTSBOOK")
    print("="*120 + "\n")

    for sportsbook_name, data in all_results.items():
        print(f"\n{sportsbook_name}:")
        data_copy = data.copy()
        data_copy['ROI_numeric'] = data_copy['ROI %'].str.rstrip('%').astype(float)
        top5 = data_copy.nlargest(5, 'ROI_numeric')[['Range', 'Bet Type', 'Bets', 'Win Rate %', 'Avg Diff', 'ROI %']]
        for idx, row in top5.iterrows():
            print(f"  {row['Range']} {row['Bet Type']:5} | Bets: {row['Bets']:3} | Win: {row['Win Rate %']:6} | Avg Diff: {row['Avg Diff']:6} | ROI: {row['ROI %']:6}")


if __name__ == "__main__":
    main()
