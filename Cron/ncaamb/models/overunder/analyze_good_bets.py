#!/usr/bin/env python3
"""
Analyze O/U Good Bets Model Performance
Train on 2021-2024, test on 2025 with optimized parameters
Show profitability by confidence threshold buckets at $10 stake with -110 odds
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
from sklearn.metrics import accuracy_score, confusion_matrix
import json
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
ncaamb_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ncaamb_dir))


def load_features_by_year(years: list):
    """Load feature files for specified years - returns Pandas DataFrame"""
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
        else:
            print(f"    [ERR] File not found: {features_file}")

    if not all_features:
        return None

    combined_df = pd.concat(all_features, ignore_index=True)

    # Convert numeric-looking columns to float
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


def get_predictions_all_models(df, xgb_model, lgb_model, cat_model) -> tuple:
    """Get predictions from all 3 models"""
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    exclude = {'actual_total', 'team_1_score', 'team_2_score', 'game_id', 'date', 'team_1', 'team_2', 'ou_target'}
    feature_cols = [c for c in numeric_cols if c not in exclude]

    X = df[feature_cols].fillna(0).values

    xgb_preds = xgb_model.predict(X)
    lgb_preds = lgb_model.predict(X)
    cat_preds = cat_model.predict(X)

    return xgb_preds, lgb_preds, cat_preds


def create_ou_good_bets_data(df, xgb_preds: np.ndarray,
                             lgb_preds: np.ndarray, cat_preds: np.ndarray):
    """Create good bets training data with all intermediate values"""
    df_filtered = df.dropna(subset=['ou_target']).copy()

    # Add predictions
    df_filtered['xgb_point_pred'] = xgb_preds
    df_filtered['lgb_point_pred'] = lgb_preds
    df_filtered['cat_point_pred'] = cat_preds

    # Convert to confidence
    df_filtered['xgb_confidence_over'] = 1.0 / (1.0 + np.exp(-(df_filtered['xgb_point_pred'] - df_filtered['betonline_ou_line']) / 3.0))
    df_filtered['lgb_confidence_over'] = 1.0 / (1.0 + np.exp(-(df_filtered['lgb_point_pred'] - df_filtered['betonline_ou_line']) / 3.0))
    df_filtered['cat_confidence_over'] = 1.0 / (1.0 + np.exp(-(df_filtered['cat_point_pred'] - df_filtered['betonline_ou_line']) / 3.0))

    for col in ['xgb_confidence_over', 'lgb_confidence_over', 'cat_confidence_over']:
        df_filtered[col] = df_filtered[col].clip(0.0, 1.0)

    # Ensemble confidence
    df_filtered['ensemble_confidence_over'] = (
        0.441 * df_filtered['xgb_confidence_over'] +
        0.466 * df_filtered['lgb_confidence_over'] +
        0.093 * df_filtered['cat_confidence_over']
    )

    # Model disagreement
    confs = np.array([
        df_filtered['xgb_confidence_over'].values,
        df_filtered['lgb_confidence_over'].values,
        df_filtered['cat_confidence_over'].values
    ])
    df_filtered['model_std_dev'] = np.std(confs, axis=0)

    # Fill missing odds data
    for col in ['betonline_ou_line', 'avg_ou_line', 'ou_line_variance',
                'avg_over_odds', 'avg_under_odds',
                'betonline_over_odds', 'betonline_under_odds']:
        if col not in df_filtered.columns:
            df_filtered[col] = 0.0
        else:
            df_filtered[col] = df_filtered[col].fillna(0.0)

    # Select feature columns
    feature_cols = [
        'xgb_confidence_over', 'lgb_confidence_over', 'cat_confidence_over',
        'ensemble_confidence_over', 'model_std_dev',
        'betonline_ou_line', 'avg_ou_line', 'ou_line_variance',
        'avg_over_odds', 'avg_under_odds',
        'betonline_over_odds', 'betonline_under_odds',
    ]

    selected_cols = [col for col in feature_cols if col in df_filtered.columns]
    X_bets = df_filtered[selected_cols].values
    y_bets = df_filtered['ou_target'].values

    return X_bets, y_bets, df_filtered, selected_cols


def calculate_bucket_stats(y_true, y_pred_proba, ensemble_confidence, stake=10.0):
    """
    Calculate stats for each confidence bucket
    stake: amount staked per bet (default $10)

    For -110 odds:
    - Win: profit = stake * (100/110) = stake * 0.909
    - Loss: loss = -stake
    """
    buckets = {}

    # Create buckets at 0.05 intervals
    for threshold in np.arange(0.0, 1.05, 0.05):
        threshold_rounded = round(threshold, 2)

        # Get bets in this bucket and next bucket (range)
        if threshold_rounded >= 1.0:
            continue
        next_threshold = min(threshold_rounded + 0.05, 1.0)

        mask = (ensemble_confidence >= threshold_rounded) & (ensemble_confidence < next_threshold)

        if mask.sum() == 0:
            continue

        y_true_bucket = y_true[mask]
        y_pred_bucket = y_pred_proba[mask]

        wins = np.sum(y_pred_bucket == y_true_bucket)
        losses = np.sum(y_pred_bucket != y_true_bucket)
        total = wins + losses
        win_pct = wins / total if total > 0 else 0

        # Calculate profit/loss at $10 stake with -110 odds
        # For over bets (confidence > 0.5): -110 odds
        # For under bets (confidence < 0.5): -110 odds
        profit_if_win = stake * (100 / 110)  # $9.09 for $10 stake
        loss_if_lose = -stake  # -$10 for $10 stake

        total_profit = wins * profit_if_win + losses * loss_if_lose
        roi = (total_profit / (total * stake)) if total > 0 else 0

        buckets[threshold_rounded] = {
            'confidence_range': f"{threshold_rounded:.2f}-{next_threshold:.2f}",
            'bet_count': total,
            'wins': int(wins),
            'losses': int(losses),
            'win_pct': win_pct,
            'stake': stake,
            'profit_per_win': profit_if_win,
            'loss_per_lose': loss_if_lose,
            'total_profit': total_profit,
            'roi_pct': roi * 100
        }

    return buckets


def main():
    print("\n")
    print("="*100)
    print("O/U GOOD BETS MODEL - THRESHOLD ANALYSIS (2025 TEST DATA)")
    print("Profitability by confidence buckets at $10 stake with -110 odds")
    print("="*100 + "\n")

    # Load optimized parameters
    params_file = Path(__file__).parent / "saved" / "ou_good_bets_best_params.json"
    if not params_file.exists():
        print(f"[ERR] Optimized parameters not found at {params_file}")
        print("Run optimize_good_bets.py first")
        return

    with open(params_file, 'r') as f:
        best_params = json.load(f)

    print("STEP 1: Loading Optimized Hyperparameters")
    print("-"*100 + "\n")
    print("Loaded hyperparameters:")
    for key, val in best_params.items():
        print(f"  {key}: {val}")
    print()

    # Load training data (2021-2024)
    print("STEP 2: Loading Training Data (2021-2024)")
    print("-"*100 + "\n")
    train_df = load_features_by_year(['2021', '2022', '2023', '2024'])

    if train_df is None or len(train_df) == 0:
        print("Failed to load training features")
        return

    train_df = create_ou_target_variable(train_df)

    if len(train_df) == 0:
        print("No training games with betonline O/U data")
        return

    # Load pre-trained ensemble models
    print("STEP 3: Loading Pre-trained Ensemble Models")
    print("-"*100 + "\n")
    xgb_model, lgb_model, cat_model = load_trained_models()

    if xgb_model is None or lgb_model is None or cat_model is None:
        print("Could not load pre-trained models")
        return

    # Get predictions on training data
    print("STEP 4: Generating Ensemble Predictions (Training Data)")
    print("-"*100 + "\n")
    xgb_preds_train, lgb_preds_train, cat_preds_train = get_predictions_all_models(
        train_df, xgb_model, lgb_model, cat_model
    )

    # Create good bets data
    print("STEP 5: Building Good Bets Features (Training Data)")
    print("-"*100 + "\n")
    X_bets_train, y_bets_train, train_df_full, feature_cols = create_ou_good_bets_data(
        train_df, xgb_preds_train, lgb_preds_train, cat_preds_train
    )

    # Train Random Forest with optimized parameters
    print("STEP 6: Training Random Forest with Optimized Parameters")
    print("-"*100 + "\n")
    print(f"Training with parameters: {best_params}")

    rf_model = RandomForestClassifier(
        **best_params,
        random_state=42,
        n_jobs=-1,
        verbose=0
    )

    rf_model.fit(X_bets_train, y_bets_train)
    print("[OK] Training complete\n")

    # Load test data (2025)
    print("STEP 7: Loading Test Data (2025)")
    print("-"*100 + "\n")

    test_df = load_features_by_year(['2025'])

    if test_df is None or len(test_df) == 0:
        print("No test data (2025) available")
        return

    test_df = create_ou_target_variable(test_df)

    if len(test_df) == 0:
        print("No test games with betonline O/U data")
        return

    # Get predictions on test data
    print("STEP 8: Generating Ensemble Predictions (Test Data)")
    print("-"*100 + "\n")
    xgb_preds_test, lgb_preds_test, cat_preds_test = get_predictions_all_models(
        test_df, xgb_model, lgb_model, cat_model
    )

    # Create good bets data for test set
    print("STEP 9: Building Good Bets Features (Test Data)")
    print("-"*100 + "\n")
    X_bets_test, y_bets_test, test_df_full, _ = create_ou_good_bets_data(
        test_df, xgb_preds_test, lgb_preds_test, cat_preds_test
    )

    # Get predictions on test data
    print("STEP 10: Getting Model Predictions on Test Data")
    print("-"*100 + "\n")
    y_pred_test = rf_model.predict(X_bets_test)
    y_pred_proba_test = rf_model.predict_proba(X_bets_test)[:, 1]

    test_acc = accuracy_score(y_bets_test, y_pred_test)
    print(f"Test Accuracy: {test_acc:.4f}\n")

    # Get ensemble confidence values for bucketing
    ensemble_confidence = test_df_full['ensemble_confidence_over'].values

    # Calculate bucket statistics
    print("STEP 11: Analyzing Profitability by Confidence Threshold")
    print("-"*100 + "\n")

    buckets = calculate_bucket_stats(y_bets_test, y_pred_test, ensemble_confidence, stake=10.0)

    # Print results
    print("\n" + "="*100)
    print("PROFITABILITY ANALYSIS - $10 STAKE PER BET WITH -110 ODDS")
    print("="*100 + "\n")

    print(f"{'Confidence Range':<20} {'Bets':<8} {'Wins':<8} {'Loss':<8} {'Win %':<10} {'Total P/L':<12} {'ROI %':<10}")
    print("-"*100)

    total_bets = 0
    total_profit = 0
    total_wins = 0
    total_losses = 0

    for threshold in sorted(buckets.keys()):
        data = buckets[threshold]
        total_bets += data['bet_count']
        total_profit += data['total_profit']
        total_wins += data['wins']
        total_losses += data['losses']

        print(f"{data['confidence_range']:<20} {data['bet_count']:<8} {data['wins']:<8} {data['losses']:<8} "
              f"{data['win_pct']*100:>8.1f}% ${data['total_profit']:>10.2f} {data['roi_pct']:>8.1f}%")

    print("-"*100)
    overall_win_pct = total_wins / total_bets if total_bets > 0 else 0
    overall_roi = (total_profit / (total_bets * 10.0)) * 100 if total_bets > 0 else 0
    print(f"{'TOTAL':<20} {total_bets:<8} {total_wins:<8} {total_losses:<8} "
          f"{overall_win_pct*100:>8.1f}% ${total_profit:>10.2f} {overall_roi:>8.1f}%")

    print("\n" + "="*100)
    print("SUMMARY STATISTICS")
    print("="*100 + "\n")

    print(f"Total test games analyzed:  {len(y_bets_test)}")
    print(f"Total bets made (by model): {total_bets}")
    print(f"Total wins:                 {total_wins}")
    print(f"Total losses:               {total_losses}")
    print(f"Overall win rate:           {overall_win_pct*100:.1f}%")
    print(f"Total profit/loss:          ${total_profit:.2f}")
    print(f"Overall ROI:                {overall_roi:.1f}%")
    print(f"Average bet size:           $10.00")
    print(f"Break-even win rate:        52.4% (for -110 odds)")
    print()

    # Analysis by over/under predictions
    print("="*100)
    print("BREAKDOWN BY PREDICTED OUTCOME")
    print("="*100 + "\n")

    over_mask = ensemble_confidence > 0.5
    under_mask = ensemble_confidence <= 0.5

    if over_mask.sum() > 0:
        over_wins = np.sum((y_pred_test == 1) & over_mask)
        over_total = over_mask.sum()
        over_pct = over_wins / over_total
        over_profit = over_wins * 9.09 - (over_total - over_wins) * 10.0
        print(f"OVER BETS (confidence > 0.50):")
        print(f"  Bet count:  {int(over_total)}")
        print(f"  Wins:       {int(over_wins)}")
        print(f"  Win rate:   {over_pct*100:.1f}%")
        print(f"  Total P/L:  ${over_profit:.2f}")
        print(f"  ROI:        {(over_profit / (over_total * 10.0)) * 100:.1f}%")
        print()

    if under_mask.sum() > 0:
        under_wins = np.sum((y_pred_test == 0) & under_mask)
        under_total = under_mask.sum()
        under_pct = under_wins / under_total
        under_profit = under_wins * 9.09 - (under_total - under_wins) * 10.0
        print(f"UNDER BETS (confidence <= 0.50):")
        print(f"  Bet count:  {int(under_total)}")
        print(f"  Wins:       {int(under_wins)}")
        print(f"  Win rate:   {under_pct*100:.1f}%")
        print(f"  Total P/L:  ${under_profit:.2f}")
        print(f"  ROI:        {(under_profit / (under_total * 10.0)) * 100:.1f}%")
        print()

    print("="*100 + "\n")


if __name__ == "__main__":
    main()
