#!/usr/bin/env python3
"""
EV Comparison Analysis
Compare EV calculated from LGB model vs Good Bets model
Analyze ROI correlation with both EV sources
"""

import os
import sys
from pathlib import Path
import polars as pl
import numpy as np
import lightgbm as lgb
from xgboost import XGBClassifier
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

# Add parent directory to path
ncaamb_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ncaamb_dir))


def load_features_by_year(years: list) -> pl.DataFrame:
    """Load feature files from specified years"""
    features_dir = ncaamb_dir
    all_features = []

    print(f"Loading features for years: {years}")
    for year in years:
        features_file = features_dir / f"features{year}.csv"
        if features_file.exists():
            print(f"  Loading features{year}.csv...")
            try:
                df = pl.read_csv(features_file)
                print(f"    [OK] Loaded {len(df)} games")
                all_features.append(df)
            except Exception as e:
                print(f"    [ERR] Error loading {year}: {e}")
        else:
            print(f"    [ERR] File not found: {features_file}")

    if not all_features:
        return None

    # Handle schema mismatches
    float_cols_to_fix = [
        'betonline_ml_team_1', 'betonline_ml_team_2',
        'betonline_spread_odds_team_1', 'betonline_spread_odds_team_2',
        'betonline_spread_pts_team_1', 'betonline_spread_pts_team_2',
        'fanduel_spread_odds_team_1', 'fanduel_spread_odds_team_2',
        'fanduel_spread_pts_team_1', 'fanduel_spread_pts_team_2',
        'mybookie_ml_team_1', 'mybookie_ml_team_2',
        'mybookie_spread_odds_team_1', 'mybookie_spread_odds_team_2',
        'mybookie_spread_pts_team_1', 'mybookie_spread_pts_team_2'
    ]

    all_features_fixed = []
    for df in all_features:
        for col in float_cols_to_fix:
            if col in df.columns:
                df = df.with_columns(pl.col(col).cast(pl.Float64, strict=False))
        all_features_fixed.append(df)

    combined_df = pl.concat(all_features_fixed)
    print(f"[OK] Combined: {len(combined_df)} total games\n")
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
    print(f"Filtered low-quality games: removed {removed}, kept {after}\n")
    return df


def create_target_variable(df: pl.DataFrame) -> pl.DataFrame:
    """Create binary target variable for moneyline"""
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

    print(f"Created target for {len(df_with_scores)} games\n")
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
        'start_time', 'game_odds', 'ml_target'
    }

    feature_cols = []
    for col in df.columns:
        if col not in metadata_cols:
            dtype = df[col].dtype
            if dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]:
                feature_cols.append(col)

    return feature_cols


def prepare_training_data(df: pl.DataFrame, feature_cols: list) -> tuple:
    """Prepare X and y"""
    X = df.select(feature_cols).fill_null(0)
    y = df.select('ml_target')
    return X, y


def american_to_decimal(american_odds):
    """Convert American odds to decimal odds"""
    if american_odds == 0:
        return 1.0
    if american_odds > 0:
        return 1 + (american_odds / 100)
    else:
        return 1 + (100 / abs(american_odds))


def calculate_ev(prob, american_odds):
    """Calculate EV using probability"""
    if american_odds == 0:
        return 0
    decimal_odds = american_to_decimal(american_odds)
    ev = (prob * decimal_odds) - 1
    return ev


def calculate_bet_roi(prob, american_odds, actual_result, bet_size=10):
    """Calculate ROI for a single bet"""
    decimal_odds = american_to_decimal(american_odds)

    if actual_result == 1:  # Bet won
        payout = bet_size * decimal_odds
        profit = payout - bet_size
    else:  # Bet lost
        profit = -bet_size

    roi = (profit / bet_size) * 100
    return profit, roi


def main():
    print("\n")
    print("="*80)
    print("EV COMPARISON ANALYSIS")
    print("LGB Model EV vs Good Bets Model EV")
    print("="*80 + "\n")

    # Load training data (2021-2024)
    print("STEP 1: Loading Training Data (2021-2024)")
    print("-"*80 + "\n")
    train_df = load_features_by_year(['2021', '2022', '2023', '2024'])

    if train_df is None:
        print("Failed to load training features")
        return

    train_df = filter_low_quality_games(train_df, min_data_quality=0.5)
    train_df = create_target_variable(train_df)
    feature_cols = identify_feature_columns(train_df)

    X_train, y_train = prepare_training_data(train_df, feature_cols)
    X_train_np = X_train.to_numpy()
    y_train_np = y_train.to_numpy().ravel()

    # Train base models
    print("STEP 2: Training Base Models")
    print("-"*80 + "\n")

    print("Training XGBoost...")
    xgb_model = XGBClassifier(
        n_estimators=128,
        max_depth=3,
        learning_rate=0.01,
        subsample=0.82,
        colsample_bytree=1.0,
        random_state=42,
        eval_metric='logloss',
        verbosity=0
    )
    xgb_model.fit(X_train_np, y_train_np)
    print("[OK] XGBoost training complete\n")

    print("Training LightGBM...")
    train_data = lgb.Dataset(X_train_np, label=y_train_np, feature_name=feature_cols)

    lgb_params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'num_leaves': 10,
        'learning_rate': 0.011905546738777037,
        'feature_fraction': 0.5,
        'bagging_fraction': 0.6902301680678105,
        'bagging_freq': 5,
        'min_data_in_leaf': 100,
        'lambda_l1': 5,
        'lambda_l2': 5,
        'verbose': -1
    }

    lgb_model = lgb.train(
        lgb_params,
        train_data,
        num_boost_round=200,
        callbacks=[lgb.log_evaluation(period=0)]
    )
    print("[OK] LightGBM training complete\n")

    # Load test data
    print("STEP 3: Loading Test Data (2025)")
    print("-"*80 + "\n")

    test_df = load_features_by_year(['2025'])

    if test_df is None:
        print("Failed to load test features")
        return

    test_df = filter_low_quality_games(test_df, min_data_quality=0.5)
    test_df = create_target_variable(test_df)

    if len(test_df) == 0:
        print("No test games with results available")
        return

    X_test, y_test = prepare_training_data(test_df, feature_cols)
    X_test_np = X_test.to_numpy()
    y_test_np = y_test.to_numpy().ravel()

    print(f"Test data shape: X={X_test.shape}\n")

    # Get predictions
    print("STEP 4: Generating Base Model Predictions")
    print("-"*80 + "\n")

    xgb_proba_test = xgb_model.predict_proba(X_test_np)
    lgb_proba_test = lgb_model.predict(X_test_np)

    print("[OK] Predictions generated\n")

    # Extract odds data from test set
    print("STEP 5: Extracting Odds Data")
    print("-"*80 + "\n")

    test_df_filtered = test_df.filter(
        pl.col('team_1_score').is_not_null() &
        pl.col('team_2_score').is_not_null()
    )

    avg_ml_team_1 = test_df_filtered.select('avg_ml_team_1').fill_null(0).to_numpy().ravel()
    avg_ml_team_2 = test_df_filtered.select('avg_ml_team_2').fill_null(0).to_numpy().ravel()

    print(f"[OK] Extracted odds for {len(avg_ml_team_1)} games\n")

    # Calculate EV from XGB and LGB for both team perspectives
    print("STEP 6: Calculating EV from Different Models")
    print("-"*80 + "\n")

    # For team 1 perspective
    xgb_probs_team1 = xgb_proba_test[: len(avg_ml_team_1), 1]
    lgb_probs_team1 = lgb_proba_test[: len(avg_ml_team_1)]

    ev_from_xgb_team1 = np.array([calculate_ev(prob, odds) for prob, odds in zip(xgb_probs_team1, avg_ml_team_1)])
    ev_from_lgb_team1 = np.array([calculate_ev(prob, odds) for prob, odds in zip(lgb_probs_team1, avg_ml_team_1)])

    # For team 2 perspective
    xgb_probs_team2 = 1 - xgb_probs_team1
    lgb_probs_team2 = 1 - lgb_probs_team1

    ev_from_xgb_team2 = np.array([calculate_ev(prob, odds) for prob, odds in zip(xgb_probs_team2, avg_ml_team_2)])
    ev_from_lgb_team2 = np.array([calculate_ev(prob, odds) for prob, odds in zip(lgb_probs_team2, avg_ml_team_2)])

    # Combine both perspectives
    all_ev_xgb = np.concatenate([ev_from_xgb_team1, ev_from_xgb_team2])
    all_ev_lgb = np.concatenate([ev_from_lgb_team1, ev_from_lgb_team2])
    all_odds = np.concatenate([avg_ml_team_1, avg_ml_team_2])
    all_results = np.concatenate([y_test_np[:len(avg_ml_team_1)], 1 - y_test_np[:len(avg_ml_team_1)]])

    print(f"[OK] EV calculated for {len(all_ev_lgb)} betting rows\n")

    # Load good bets model to get good bets EV
    print("STEP 7: Loading Good Bets Model")
    print("-"*80 + "\n")

    model_path = Path(__file__).parent / "saved" / "good_bets_rf_model_final.pkl"
    with open(model_path, 'rb') as f:
        good_bets_model = pickle.load(f)
    print(f"[OK] Good bets model loaded\n")

    # Create good bets features for EV calculation
    print("STEP 8: Creating Good Bets Features")
    print("-"*80 + "\n")

    # For simplicity, we'll create a basic feature set for good bets model
    # Good bets model needs: xgb_prob, lgb_prob, model_disagreement, moneyline_odds, implied_prob, ev, spread, month, strength

    # For team 1 perspective
    xgb_prob_t1 = xgb_probs_team1
    lgb_prob_t1 = lgb_probs_team1
    disagreement = np.abs(xgb_probs_team1 - lgb_probs_team1)
    ml_odds_t1 = avg_ml_team_1
    implied_prob_t1 = np.array([calculate_ev(0.5, odds) + 1 for odds in ml_odds_t1])  # Placeholder

    # Create feature matrix for team 1
    features_t1 = np.column_stack([
        xgb_prob_t1, lgb_prob_t1, disagreement, ml_odds_t1,
        implied_prob_t1, ev_from_xgb_team1,  # ev column
        np.zeros_like(ml_odds_t1),  # spread_pts_self (dummy)
        np.zeros_like(ml_odds_t1),  # spread_pts_opp (dummy)
        np.zeros_like(ml_odds_t1),  # spread_odds_self (dummy)
        np.zeros_like(ml_odds_t1),  # spread_odds_opp (dummy)
        np.ones_like(ml_odds_t1) * 1,  # month (dummy)
        np.zeros_like(ml_odds_t1),  # strength_differential (dummy)
    ])

    # For team 2 perspective
    xgb_prob_t2 = xgb_probs_team2
    lgb_prob_t2 = lgb_probs_team2
    disagreement_t2 = disagreement
    ml_odds_t2 = avg_ml_team_2
    implied_prob_t2 = np.array([calculate_ev(0.5, odds) + 1 for odds in ml_odds_t2])

    features_t2 = np.column_stack([
        xgb_prob_t2, lgb_prob_t2, disagreement_t2, ml_odds_t2,
        implied_prob_t2, ev_from_xgb_team2,  # ev column
        np.zeros_like(ml_odds_t2),  # spread_pts_self (dummy)
        np.zeros_like(ml_odds_t2),  # spread_pts_opp (dummy)
        np.zeros_like(ml_odds_t2),  # spread_odds_self (dummy)
        np.zeros_like(ml_odds_t2),  # spread_odds_opp (dummy)
        np.ones_like(ml_odds_t2) * 1,  # month (dummy)
        np.zeros_like(ml_odds_t2),  # strength_differential (dummy)
    ])

    all_features = np.vstack([features_t1, features_t2])

    # Get good bets probabilities
    good_bet_probs = good_bets_model.predict_proba(all_features)[:, 1]

    # Calculate EV from good bets model probabilities
    ev_from_good_bets = np.array([calculate_ev(prob, odds) for prob, odds in zip(good_bet_probs, all_odds)])

    print(f"[OK] Good bets EV calculated\n")

    # Calculate ROI for all bets
    print("STEP 9: Calculating ROI")
    print("-"*80 + "\n")

    rois = np.array([calculate_bet_roi(0.5, odds, result)[1] for odds, result in zip(all_odds, all_results)])

    print(f"[OK] ROI calculated for all bets\n")

    # Correlation analysis
    print("STEP 10: Correlation Analysis")
    print("-"*80 + "\n")

    corr_xgb_ev_roi = pearsonr(all_ev_xgb, rois)[0]
    corr_lgb_ev_roi = pearsonr(all_ev_lgb, rois)[0]
    corr_gb_ev_roi = pearsonr(ev_from_good_bets, rois)[0]

    print(f"Correlation with ROI:")
    print(f"  XGB EV ↔ ROI:        {corr_xgb_ev_roi:.4f}")
    print(f"  LGB EV ↔ ROI:        {corr_lgb_ev_roi:.4f}")
    print(f"  Good Bets EV ↔ ROI:  {corr_gb_ev_roi:.4f}\n")

    # Create visualizations
    print("STEP 11: Creating Visualizations")
    print("-"*80 + "\n")

    output_dir = Path(__file__).parent / "saved" / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set style
    sns.set_style("whitegrid")

    # 1. EV comparison scatter plots
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle('EV vs ROI: XGB vs LGB vs Good Bets Model', fontsize=16, fontweight='bold')

    # XGB EV vs ROI
    axes[0].scatter(all_ev_xgb, rois, alpha=0.3, s=10, c=all_results, cmap='RdYlGn')
    axes[0].set_xlabel('XGB Model EV', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('ROI (%)', fontsize=12, fontweight='bold')
    axes[0].set_title(f'XGB EV vs ROI\nCorr: {corr_xgb_ev_roi:.4f}')
    axes[0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    axes[0].axvline(x=0, color='r', linestyle='--', alpha=0.5)
    axes[0].grid(True, alpha=0.3)

    # LGB EV vs ROI
    axes[1].scatter(all_ev_lgb, rois, alpha=0.3, s=10, c=all_results, cmap='RdYlGn')
    axes[1].set_xlabel('LGB Model EV', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('ROI (%)', fontsize=12, fontweight='bold')
    axes[1].set_title(f'LGB EV vs ROI\nCorr: {corr_lgb_ev_roi:.4f}')
    axes[1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    axes[1].axvline(x=0, color='r', linestyle='--', alpha=0.5)
    axes[1].grid(True, alpha=0.3)

    # Good Bets EV vs ROI
    axes[2].scatter(ev_from_good_bets, rois, alpha=0.3, s=10, c=all_results, cmap='RdYlGn')
    axes[2].set_xlabel('Good Bets Model EV', fontsize=12, fontweight='bold')
    axes[2].set_ylabel('ROI (%)', fontsize=12, fontweight='bold')
    axes[2].set_title(f'Good Bets EV vs ROI\nCorr: {corr_gb_ev_roi:.4f}')
    axes[2].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    axes[2].axvline(x=0, color='r', linestyle='--', alpha=0.5)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'ev_comparison_scatter.png', dpi=150, bbox_inches='tight')
    print(f"[OK] Saved: ev_comparison_scatter.png")
    plt.close()

    # 2. EV correlation heatmap
    fig, ax = plt.subplots(figsize=(10, 8))

    ev_comparison = np.array([all_ev_xgb, all_ev_lgb, ev_from_good_bets, rois])
    corr_ev_matrix = np.corrcoef(ev_comparison)

    sns.heatmap(corr_ev_matrix, annot=True, fmt='.4f', cmap='coolwarm', center=0,
                xticklabels=['XGB EV', 'LGB EV', 'Good Bets EV', 'ROI'],
                yticklabels=['XGB EV', 'LGB EV', 'Good Bets EV', 'ROI'],
                ax=ax, vmin=-1, vmax=1, square=True, cbar_kws={'label': 'Correlation'})

    plt.title('EV Comparison Correlation Matrix', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'ev_comparison_correlation.png', dpi=150, bbox_inches='tight')
    print(f"[OK] Saved: ev_comparison_correlation.png")
    plt.close()

    # 3. Binned analysis for LGB EV
    fig = plt.figure(figsize=(18, 6))
    gs = fig.add_gridspec(1, 2, hspace=0.3, wspace=0.3)

    fig.suptitle('EV Bin Analysis: LGB vs Good Bets', fontsize=16, fontweight='bold')

    # LGB EV bins
    lgb_bins = np.linspace(all_ev_lgb.min(), all_ev_lgb.max(), 11)
    lgb_binned = np.digitize(all_ev_lgb, lgb_bins)
    lgb_bin_means = [rois[lgb_binned == i].mean() if np.sum(lgb_binned == i) > 0 else 0 for i in range(1, len(lgb_bins))]
    lgb_bin_counts = [np.sum(lgb_binned == i) for i in range(1, len(lgb_bins))]
    lgb_bin_centers = (lgb_bins[:-1] + lgb_bins[1:]) / 2

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.bar(lgb_bin_centers, lgb_bin_means, width=(lgb_bins[1]-lgb_bins[0])*0.8, alpha=0.7, color='coral', edgecolor='black')
    ax0.axhline(y=0, color='r', linestyle='--', alpha=0.5, linewidth=2)
    ax0.set_xlabel('LGB Model EV', fontsize=12, fontweight='bold')
    ax0.set_ylabel('Average ROI (%)', fontsize=12, fontweight='bold')
    ax0.set_title('LGB EV Bins vs Avg ROI', fontsize=13, fontweight='bold')
    ax0.grid(axis='y', alpha=0.3)

    max_roi_lgb = max(lgb_bin_means) if lgb_bin_means else 0
    min_roi_lgb = min(lgb_bin_means) if lgb_bin_means else 0
    y_margin_lgb = (max_roi_lgb - min_roi_lgb) * 0.25
    ax0.set_ylim(min_roi_lgb - y_margin_lgb, max_roi_lgb + y_margin_lgb)

    for x, y, cnt in zip(lgb_bin_centers, lgb_bin_means, lgb_bin_counts):
        if max_roi_lgb - min_roi_lgb > 0:
            offset = (max_roi_lgb - min_roi_lgb) * 0.05 if y >= 0 else -(max_roi_lgb - min_roi_lgb) * 0.08
        else:
            offset = 5 if y >= 0 else -10
        ax0.text(x, y + offset, f'n={cnt}', ha='center', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8, edgecolor='black'))

    # Good Bets EV bins
    gb_bins = np.linspace(ev_from_good_bets.min(), ev_from_good_bets.max(), 11)
    gb_binned = np.digitize(ev_from_good_bets, gb_bins)
    gb_bin_means = [rois[gb_binned == i].mean() if np.sum(gb_binned == i) > 0 else 0 for i in range(1, len(gb_bins))]
    gb_bin_counts = [np.sum(gb_binned == i) for i in range(1, len(gb_bins))]
    gb_bin_centers = (gb_bins[:-1] + gb_bins[1:]) / 2

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.bar(gb_bin_centers, gb_bin_means, width=(gb_bins[1]-gb_bins[0])*0.8, alpha=0.7, color='seagreen', edgecolor='black')
    ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5, linewidth=2)
    ax1.set_xlabel('Good Bets Model EV', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Average ROI (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Good Bets EV Bins vs Avg ROI', fontsize=13, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)

    max_roi_gb = max(gb_bin_means) if gb_bin_means else 0
    min_roi_gb = min(gb_bin_means) if gb_bin_means else 0
    y_margin_gb = (max_roi_gb - min_roi_gb) * 0.25
    ax1.set_ylim(min_roi_gb - y_margin_gb, max_roi_gb + y_margin_gb)

    for x, y, cnt in zip(gb_bin_centers, gb_bin_means, gb_bin_counts):
        if max_roi_gb - min_roi_gb > 0:
            offset = (max_roi_gb - min_roi_gb) * 0.05 if y >= 0 else -(max_roi_gb - min_roi_gb) * 0.08
        else:
            offset = 5 if y >= 0 else -10
        ax1.text(x, y + offset, f'n={cnt}', ha='center', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='yellow', alpha=0.8, edgecolor='black', linewidth=1.5))

    plt.savefig(output_dir / 'ev_bins_comparison.png', dpi=150, bbox_inches='tight')
    print(f"[OK] Saved: ev_bins_comparison.png")
    plt.close()

    # Save report
    print("\nSTEP 12: Saving Report")
    print("-"*80 + "\n")

    report_file = output_dir.parent / "ev_comparison_report.txt"
    with open(report_file, 'w') as f:
        f.write("EV COMPARISON ANALYSIS REPORT\n")
        f.write("="*80 + "\n\n")

        f.write("CORRELATION WITH ROI:\n")
        f.write("-"*80 + "\n")
        f.write(f"XGB Model EV vs ROI:        {corr_xgb_ev_roi:.4f}\n")
        f.write(f"LGB Model EV vs ROI:        {corr_lgb_ev_roi:.4f}\n")
        f.write(f"Good Bets Model EV vs ROI:  {corr_gb_ev_roi:.4f}\n\n")

        f.write("KEY FINDINGS:\n")
        f.write("-"*80 + "\n")

        best_model = max(
            [('XGB', corr_xgb_ev_roi), ('LGB', corr_lgb_ev_roi), ('Good Bets', corr_gb_ev_roi)],
            key=lambda x: abs(x[1])
        )

        f.write(f"Best EV Predictor of ROI:    {best_model[0]} (correlation: {best_model[1]:.4f})\n\n")

        f.write(f"Difference (LGB - XGB):      {corr_lgb_ev_roi - corr_xgb_ev_roi:.4f}\n")
        f.write(f"Difference (Good Bets - LGB): {corr_gb_ev_roi - corr_lgb_ev_roi:.4f}\n\n")

        f.write("STATISTICS:\n")
        f.write("-"*80 + "\n")
        f.write(f"Total Bets Analyzed:      {len(all_results)}\n")
        f.write(f"Win Rate:                 {np.sum(all_results)/len(all_results)*100:.2f}%\n")
        f.write(f"Average ROI:              {np.mean(rois):.2f}%\n\n")

        f.write("EV STATISTICS:\n")
        f.write("-"*80 + "\n")
        f.write(f"XGB EV Mean:              {np.mean(all_ev_xgb):.4f}\n")
        f.write(f"XGB EV Std Dev:           {np.std(all_ev_xgb):.4f}\n\n")

        f.write(f"LGB EV Mean:              {np.mean(all_ev_lgb):.4f}\n")
        f.write(f"LGB EV Std Dev:           {np.std(all_ev_lgb):.4f}\n\n")

        f.write(f"Good Bets EV Mean:        {np.mean(ev_from_good_bets):.4f}\n")
        f.write(f"Good Bets EV Std Dev:     {np.std(ev_from_good_bets):.4f}\n")

    print(f"[OK] Report saved to {report_file}\n")

    print("="*80)
    print("[OK] EV comparison analysis complete!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
