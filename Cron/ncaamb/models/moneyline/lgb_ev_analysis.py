#!/usr/bin/env python3
"""
LGB EV Analysis
Analyze ROI vs EV calculated from LGB model probabilities
"""

import os
import sys
from pathlib import Path
import polars as pl
import numpy as np
import lightgbm as lgb
from xgboost import XGBClassifier
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


def calculate_bet_roi(american_odds, actual_result, bet_size=10):
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
    print("LGB EV ANALYSIS")
    print("ROI vs EV calculated from LGB model probabilities")
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

    # Train LGB model
    print("STEP 2: Training LightGBM Model")
    print("-"*80 + "\n")

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

    # Get LGB predictions
    print("STEP 4: Generating LGB Predictions")
    print("-"*80 + "\n")

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

    # Calculate EV from LGB for both team perspectives
    print("STEP 6: Calculating EV from LGB Probabilities")
    print("-"*80 + "\n")

    # For team 1 perspective
    lgb_probs_team1 = lgb_proba_test[: len(avg_ml_team_1)]
    ev_from_lgb_team1 = np.array([calculate_ev(prob, odds) for prob, odds in zip(lgb_probs_team1, avg_ml_team_1)])

    # For team 2 perspective
    lgb_probs_team2 = 1 - lgb_probs_team1
    ev_from_lgb_team2 = np.array([calculate_ev(prob, odds) for prob, odds in zip(lgb_probs_team2, avg_ml_team_2)])

    # Combine both perspectives
    all_ev_lgb = np.concatenate([ev_from_lgb_team1, ev_from_lgb_team2])
    all_odds = np.concatenate([avg_ml_team_1, avg_ml_team_2])
    all_results = np.concatenate([y_test_np[:len(avg_ml_team_1)], 1 - y_test_np[:len(avg_ml_team_1)]])

    print(f"[OK] EV calculated for {len(all_ev_lgb)} betting rows\n")

    # Calculate ROI for all bets
    print("STEP 7: Calculating ROI")
    print("-"*80 + "\n")

    rois = np.array([calculate_bet_roi(odds, result)[1] for odds, result in zip(all_odds, all_results)])

    print(f"[OK] ROI calculated for all bets\n")

    # Correlation analysis
    print("STEP 8: Correlation Analysis")
    print("-"*80 + "\n")

    corr_lgb_ev_roi = pearsonr(all_ev_lgb, rois)[0]

    print(f"LGB EV <-> ROI Correlation: {corr_lgb_ev_roi:.4f}\n")

    # Create visualizations
    print("STEP 9: Creating Visualizations")
    print("-"*80 + "\n")

    output_dir = Path(__file__).parent / "saved" / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)

    sns.set_style("whitegrid")

    # 1. LGB EV vs ROI scatter
    fig, ax = plt.subplots(figsize=(14, 8))

    scatter = ax.scatter(all_ev_lgb, rois, alpha=0.4, s=20, c=all_results, cmap='RdYlGn', edgecolors='black', linewidth=0.5)
    ax.set_xlabel('LGB Model EV', fontsize=14, fontweight='bold')
    ax.set_ylabel('ROI (%)', fontsize=14, fontweight='bold')
    ax.set_title(f'LGB EV vs ROI (Correlation: {corr_lgb_ev_roi:.4f})', fontsize=16, fontweight='bold')
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5, linewidth=2)
    ax.axvline(x=0, color='r', linestyle='--', alpha=0.5, linewidth=2)
    ax.grid(True, alpha=0.3)

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Actual Result (Green=Win, Red=Loss)', fontsize=12)

    plt.tight_layout()
    plt.savefig(output_dir / 'lgb_ev_vs_roi_scatter.png', dpi=150, bbox_inches='tight')
    print(f"[OK] Saved: lgb_ev_vs_roi_scatter.png")
    plt.close()

    # 2. LGB EV bins analysis
    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(1, 1, hspace=0.3, wspace=0.3)

    fig.suptitle('LGB EV Bins vs Average ROI', fontsize=16, fontweight='bold')

    # LGB EV bins
    lgb_bins = np.linspace(all_ev_lgb.min(), all_ev_lgb.max(), 11)
    lgb_binned = np.digitize(all_ev_lgb, lgb_bins)
    lgb_bin_means = [rois[lgb_binned == i].mean() if np.sum(lgb_binned == i) > 0 else 0 for i in range(1, len(lgb_bins))]
    lgb_bin_counts = [np.sum(lgb_binned == i) for i in range(1, len(lgb_bins))]
    lgb_bin_centers = (lgb_bins[:-1] + lgb_bins[1:]) / 2

    ax0 = fig.add_subplot(gs[0, 0])
    bars = ax0.bar(lgb_bin_centers, lgb_bin_means, width=(lgb_bins[1]-lgb_bins[0])*0.8, alpha=0.7, color='steelblue', edgecolor='black', linewidth=1.5)
    ax0.axhline(y=0, color='r', linestyle='--', alpha=0.5, linewidth=2)
    ax0.set_xlabel('LGB Model EV', fontsize=13, fontweight='bold')
    ax0.set_ylabel('Average ROI (%)', fontsize=13, fontweight='bold')
    ax0.set_title('LGB EV Bins vs Avg ROI', fontsize=14, fontweight='bold')
    ax0.grid(axis='y', alpha=0.3)

    max_roi_lgb = max(lgb_bin_means) if lgb_bin_means else 0
    min_roi_lgb = min(lgb_bin_means) if lgb_bin_means else 0
    y_margin_lgb = (max_roi_lgb - min_roi_lgb) * 0.25
    ax0.set_ylim(min_roi_lgb - y_margin_lgb, max_roi_lgb + y_margin_lgb)

    # Add count labels
    for x, y, cnt in zip(lgb_bin_centers, lgb_bin_means, lgb_bin_counts):
        if max_roi_lgb - min_roi_lgb > 0:
            offset = (max_roi_lgb - min_roi_lgb) * 0.05 if y >= 0 else -(max_roi_lgb - min_roi_lgb) * 0.08
        else:
            offset = 5 if y >= 0 else -10
        ax0.text(x, y + offset, f'n={cnt}', ha='center', fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='yellow', alpha=0.9, edgecolor='black', linewidth=1.5))

    plt.savefig(output_dir / 'lgb_ev_bins_analysis.png', dpi=150, bbox_inches='tight')
    print(f"[OK] Saved: lgb_ev_bins_analysis.png")
    plt.close()

    # 3. Hexbin density plot
    fig, ax = plt.subplots(figsize=(12, 8))

    hexbin = ax.hexbin(all_ev_lgb, rois, gridsize=25, cmap='YlOrRd', mincnt=1, edgecolors='black', linewidths=0.2)
    ax.set_xlabel('LGB Model EV', fontsize=13, fontweight='bold')
    ax.set_ylabel('ROI (%)', fontsize=13, fontweight='bold')
    ax.set_title('LGB EV vs ROI - Density Heatmap', fontsize=14, fontweight='bold')
    ax.axhline(y=0, color='b', linestyle='--', alpha=0.5, linewidth=2)
    ax.axvline(x=0, color='b', linestyle='--', alpha=0.5, linewidth=2)

    cbar = plt.colorbar(hexbin, ax=ax)
    cbar.set_label('Bet Count', fontsize=12)

    plt.tight_layout()
    plt.savefig(output_dir / 'lgb_ev_roi_density.png', dpi=150, bbox_inches='tight')
    print(f"[OK] Saved: lgb_ev_roi_density.png")
    plt.close()

    # Save report
    print("\nSTEP 10: Saving Report")
    print("-"*80 + "\n")

    report_file = output_dir.parent / "lgb_ev_analysis_report.txt"
    with open(report_file, 'w') as f:
        f.write("LGB EV ANALYSIS REPORT\n")
        f.write("="*80 + "\n\n")

        f.write("CORRELATION ANALYSIS:\n")
        f.write("-"*80 + "\n")
        f.write(f"LGB EV vs ROI Correlation:  {corr_lgb_ev_roi:.4f}\n\n")

        f.write("INTERPRETATION:\n")
        f.write("-"*80 + "\n")
        if corr_lgb_ev_roi > 0.5:
            f.write("Strong positive correlation - EV is a good predictor of ROI\n")
        elif corr_lgb_ev_roi > 0.3:
            f.write("Moderate positive correlation - EV has some predictive power\n")
        else:
            f.write("Weak correlation - EV alone has limited predictive power\n")
        f.write("\n")

        f.write("STATISTICS:\n")
        f.write("-"*80 + "\n")
        f.write(f"Total Bets Analyzed:      {len(all_results)}\n")
        f.write(f"Total Wins:               {int(np.sum(all_results))}\n")
        f.write(f"Win Rate:                 {np.sum(all_results)/len(all_results)*100:.2f}%\n")
        f.write(f"Average ROI:              {np.mean(rois):.2f}%\n")
        f.write(f"Median ROI:               {np.median(rois):.2f}%\n")
        f.write(f"Std Dev ROI:              {np.std(rois):.2f}%\n\n")

        f.write("EV STATISTICS:\n")
        f.write("-"*80 + "\n")
        f.write(f"LGB EV Mean:              {np.mean(all_ev_lgb):.4f}\n")
        f.write(f"LGB EV Median:            {np.median(all_ev_lgb):.4f}\n")
        f.write(f"LGB EV Std Dev:           {np.std(all_ev_lgb):.4f}\n")
        f.write(f"LGB EV Min:               {np.min(all_ev_lgb):.4f}\n")
        f.write(f"LGB EV Max:               {np.max(all_ev_lgb):.4f}\n\n")

        f.write("BIN ANALYSIS:\n")
        f.write("-"*80 + "\n")
        f.write(f"{'EV Bin':<20} {'Count':<10} {'Avg ROI':<15}\n")
        for center, mean, cnt in zip(lgb_bin_centers, lgb_bin_means, lgb_bin_counts):
            f.write(f"{center:7.4f}          {cnt:<10} {mean:<15.2f}%\n")

    print(f"[OK] Report saved to {report_file}\n")

    print("="*80)
    print("[OK] LGB EV analysis complete!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
