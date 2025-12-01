#!/usr/bin/env python3
"""
Analyze OU Good Bets Model with Custom Thresholds and Ranges
Creates two sheets:
1. THRESHOLDS: For each threshold (0.01-0.99), simulate betting all games >= threshold (for OVER) or <= threshold (for UNDER)
2. RANGES: For confidence ranges (0.51-0.53, 0.531-0.551, etc.), simulate bets in that range
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
ncaamb_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ncaamb_dir))

# Import directly from analyze_good_bets instead
sys.path.insert(0, str(Path(__file__).parent))


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


def create_ou_target_variable(df):
    """Create binary target variable for over/under betting"""
    df_filtered = df.dropna(subset=['actual_total', 'betonline_ou_line'])

    before = len(df)
    after = len(df_filtered)
    print(f"Filtered for betonline O/U line: removed {before - after}, kept {after}")

    df_with_target = df_filtered.copy()
    df_with_target['ou_target'] = (df_with_target['actual_total'] > df_with_target['betonline_ou_line']).astype(int)

    over_count = (df_with_target['ou_target'] == 1).sum()
    under_count = (df_with_target['ou_target'] == 0).sum()

    print(f"Target distribution:")
    print(f"  Over hits (1):  {over_count}")
    print(f"  Under hits (0): {under_count}\n")

    return df_with_target


def load_trained_models():
    """Load pre-trained XGB, LGB, CatBoost models"""
    saved_dir = Path(__file__).parent / "saved"

    try:
        xgb_model = xgb.XGBRegressor()
        xgb_model.load_model(str(saved_dir / "xgboost_model.pkl"))
        lgb_model = lgb.Booster(model_file=str(saved_dir / "lightgbm_model.pkl"))
        cat_model = CatBoostRegressor()
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


def create_ou_good_bets_data(df, xgb_preds, lgb_preds, cat_preds):
    """Create good bets features"""
    df_filtered = df.dropna(subset=['actual_total', 'betonline_ou_line']).copy()
    df_filtered['xgb_point_pred'] = xgb_preds[:len(df_filtered)]
    df_filtered['lgb_point_pred'] = lgb_preds[:len(df_filtered)]
    df_filtered['cat_point_pred'] = cat_preds[:len(df_filtered)]

    # Create confidence scores
    df_filtered['xgb_confidence_over'] = 1.0 / (1.0 + np.exp(-(df_filtered['xgb_point_pred'] - df_filtered['betonline_ou_line']) / 3.0))
    df_filtered['lgb_confidence_over'] = 1.0 / (1.0 + np.exp(-(df_filtered['lgb_point_pred'] - df_filtered['betonline_ou_line']) / 3.0))
    df_filtered['cat_confidence_over'] = 1.0 / (1.0 + np.exp(-(df_filtered['cat_point_pred'] - df_filtered['betonline_ou_line']) / 3.0))

    for col in ['xgb_confidence_over', 'lgb_confidence_over', 'cat_confidence_over']:
        df_filtered[col] = df_filtered[col].clip(0.01, 0.99)

    # Ensemble confidence
    df_filtered['ensemble_confidence_over'] = (
        0.441 * df_filtered['xgb_confidence_over'] +
        0.466 * df_filtered['lgb_confidence_over'] +
        0.093 * df_filtered['cat_confidence_over']
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
        'betonline_ou_line', 'avg_ou_line', 'ou_line_variance'
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


def analyze_thresholds(ensemble_confidence, y_bets_test):
    """Analyze betting at each individual threshold from 0.01 to 0.99"""
    results = []

    for threshold in np.arange(0.01, 1.00, 0.01):
        threshold_rounded = round(threshold, 2)

        # OVER bets: confidence >= threshold (for thresholds > 0.5)
        if threshold_rounded > 0.5:
            mask = ensemble_confidence >= threshold_rounded
            wins = np.sum((y_bets_test == 1) & mask)
            total_bets = mask.sum()
            bet_type = "OVER"
        # UNDER bets: confidence <= threshold (for thresholds < 0.5)
        elif threshold_rounded < 0.5:
            mask = ensemble_confidence <= threshold_rounded
            wins = np.sum((y_bets_test == 0) & mask)
            total_bets = mask.sum()
            bet_type = "UNDER"
        # Skip 0.5 exactly
        else:
            continue

        if total_bets > 0:
            win_rate = (wins / total_bets) * 100
            roi, total_profit = calculate_bet_roi(wins, total_bets)
        else:
            win_rate = 0
            roi = 0
            total_profit = 0

        results.append({
            'Threshold': f"{threshold_rounded:.2f}",
            'Bet Type': bet_type,
            'Bet Quantity': int(total_bets),
            'Wins': int(wins),
            'Losses': int(total_bets - wins),
            'Win Rate %': f"{win_rate:.1f}%",
            'Total Profit': f"${total_profit:.2f}",
            'ROI %': f"{roi:.1f}%"
        })

    return pd.DataFrame(results)


def analyze_ranges(ensemble_confidence, y_bets_test):
    """Analyze betting in confidence ranges"""
    results = []

    # Define ranges
    ranges = [
        (0.01, 0.10, "UNDER"),
        (0.10, 0.20, "UNDER"),
        (0.20, 0.30, "UNDER"),
        (0.30, 0.40, "UNDER"),
        (0.40, 0.50, "UNDER"),
        (0.51, 0.53, "OVER"),
        (0.53, 0.55, "OVER"),
        (0.55, 0.57, "OVER"),
        (0.57, 0.59, "OVER"),
        (0.59, 0.61, "OVER"),
        (0.61, 0.63, "OVER"),
        (0.63, 0.65, "OVER"),
        (0.65, 0.70, "OVER"),
        (0.70, 0.75, "OVER"),
        (0.75, 0.80, "OVER"),
        (0.80, 0.85, "OVER"),
        (0.85, 0.90, "OVER"),
        (0.90, 0.95, "OVER"),
        (0.95, 1.00, "OVER"),
    ]

    for lower, upper, bet_type in ranges:
        mask = (ensemble_confidence >= lower) & (ensemble_confidence < upper)
        total_bets = mask.sum()

        if bet_type == "OVER":
            wins = np.sum((y_bets_test == 1) & mask)
        else:  # UNDER
            wins = np.sum((y_bets_test == 0) & mask)

        if total_bets > 0:
            win_rate = (wins / total_bets) * 100
            roi, total_profit = calculate_bet_roi(wins, total_bets)
        else:
            win_rate = 0
            roi = 0
            total_profit = 0

        results.append({
            'Range': f"{lower:.2f}-{upper:.2f}",
            'Bet Type': bet_type,
            'Bet Quantity': int(total_bets),
            'Wins': int(wins),
            'Losses': int(total_bets - wins),
            'Win Rate %': f"{win_rate:.1f}%",
            'Total Profit': f"${total_profit:.2f}",
            'ROI %': f"{roi:.1f}%"
        })

    return pd.DataFrame(results)


def main():
    print("="*100)
    print("THRESHOLD AND RANGE ANALYSIS FOR OU GOOD BETS MODEL")
    print("="*100 + "\n")

    # Load models
    print("[STEP 0] Loading pre-trained ensemble models")
    xgb_model, lgb_model, cat_model = load_trained_models()
    if not all([xgb_model, lgb_model, cat_model]):
        print("[-] Could not load models, exiting")
        return

    # Load and process data (same as analyze_good_bets.py)
    print("[STEP 1] Loading training data (2021-2024)")
    train_df = load_features_by_year(['2021', '2022', '2023', '2024'])
    train_df = create_ou_target_variable(train_df)

    print("[STEP 2] Getting ensemble predictions for training data")
    xgb_preds_train, lgb_preds_train, cat_preds_train = get_predictions_all_models(train_df, xgb_model, lgb_model, cat_model)

    print("[STEP 3] Creating good bets features for training data")
    X_bets_train, y_bets_train, train_df_full, _ = create_ou_good_bets_data(
        train_df, xgb_preds_train, lgb_preds_train, cat_preds_train
    )

    print("[STEP 4] Training Random Forest")
    rf_model = RandomForestClassifier(
        n_estimators=178,
        max_depth=5,
        min_samples_split=48,
        min_samples_leaf=44,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_bets_train, y_bets_train)
    print(f"[OK] Training complete\n")

    # Load and process test data
    print("[STEP 5] Loading test data (2025)")
    test_df = load_features_by_year(['2025'])
    test_df = create_ou_target_variable(test_df)

    print("[STEP 6] Getting ensemble predictions for test data")
    xgb_preds_test, lgb_preds_test, cat_preds_test = get_predictions_all_models(test_df, xgb_model, lgb_model, cat_model)

    print("[STEP 7] Creating good bets features for test data")
    X_bets_test, y_bets_test, test_df_full, _ = create_ou_good_bets_data(
        test_df, xgb_preds_test, lgb_preds_test, cat_preds_test
    )

    # Get ensemble confidence values
    ensemble_confidence = test_df_full['ensemble_confidence_over'].values

    # Analyze thresholds and ranges
    print("\n[STEP 8] Analyzing individual thresholds")
    thresholds_df = analyze_thresholds(ensemble_confidence, y_bets_test)
    print(f"[OK] Analyzed {len(thresholds_df)} thresholds")

    print("\n[STEP 9] Analyzing confidence ranges")
    ranges_df = analyze_ranges(ensemble_confidence, y_bets_test)
    print(f"[OK] Analyzed {len(ranges_df)} ranges")

    # Export to Excel
    output_file = Path(__file__).parent.parent.parent / "ou_threshold_analysis.xlsx"
    print(f"\n[STEP 10] Exporting to Excel: {output_file}")

    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        thresholds_df.to_excel(writer, sheet_name='THRESHOLDS', index=False)
        ranges_df.to_excel(writer, sheet_name='RANGES', index=False)

    print(f"[OK] Export complete\n")

    # Summary
    print("="*100)
    print("ANALYSIS COMPLETE")
    print("="*100)
    print(f"\nThresholds sheet: {len(thresholds_df)} rows (0.01 to 0.99)")
    print(f"Ranges sheet: {len(ranges_df)} rows (custom confidence ranges)")
    print(f"\nOutput file: {output_file}")
    print("\nTop 5 most profitable thresholds:")
    thresholds_sorted = thresholds_df.copy()
    thresholds_sorted['ROI'] = thresholds_sorted['ROI %'].str.rstrip('%').astype(float)
    print(thresholds_sorted.nlargest(5, 'ROI')[['Threshold', 'Bet Type', 'Bet Quantity', 'Win Rate %', 'ROI %']])

    print("\nTop 5 most profitable ranges:")
    ranges_sorted = ranges_df.copy()
    ranges_sorted['ROI'] = ranges_sorted['ROI %'].str.rstrip('%').astype(float)
    print(ranges_sorted.nlargest(5, 'ROI')[['Range', 'Bet Type', 'Bet Quantity', 'Win Rate %', 'ROI %']])


if __name__ == "__main__":
    main()
