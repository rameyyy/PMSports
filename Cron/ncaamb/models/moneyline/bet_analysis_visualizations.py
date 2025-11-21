#!/usr/bin/env python3
"""
Bet Analysis Visualizations
Correlation analysis and visual insights
ROI vs Model Probabilities, EV, and other factors
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
from scipy.stats import pearsonr, spearmanr

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


def create_good_bets_data(df: pl.DataFrame, y: np.ndarray,
                         xgb_proba: np.ndarray, lgb_proba: np.ndarray) -> tuple:
    """Create good bets training data"""
    # Add model predictions to dataframe
    df = df.with_columns([
        pl.lit(xgb_proba[:, 1]).alias('xgb_prob_team_1'),
        pl.lit(lgb_proba).alias('lgb_prob_team_1'),
        pl.lit(y).alias('game_result'),
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

    # Calculate EV
    df = df.with_columns([
        (pl.col('xgb_prob_team_1') * pl.when(pl.col('avg_ml_team_1') == 0).then(1.0)
            .when(pl.col('avg_ml_team_1') > 0).then(1 + (pl.col('avg_ml_team_1') / 100))
            .otherwise(1 + (100 / pl.col('avg_ml_team_1').abs())) - 1)
        .alias('ev_team_1'),
        ((1 - pl.col('xgb_prob_team_1')) * pl.when(pl.col('avg_ml_team_2') == 0).then(1.0)
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
        (pl.col('game_result') == 1).cast(pl.Int32).alias('target'),
    ])

    # Create team 2 perspective rows
    team_2_data = df.select([
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
        (pl.col('game_result') == 0).cast(pl.Int32).alias('target'),
    ])

    # Combine both perspectives
    all_bets = pl.concat([team_1_data, team_2_data])

    # Convert to numpy for sklearn
    feature_cols = ['xgb_prob_team_1', 'lgb_prob_team_1', 'model_disagreement', 'moneyline_odds',
                   'implied_prob', 'ev', 'spread_pts_self', 'spread_pts_opp',
                   'spread_odds_self', 'spread_odds_opp', 'month', 'strength_differential']

    X_bets = all_bets.select(feature_cols).to_numpy()
    y_bets = all_bets.select('target').to_numpy().ravel()
    moneyline_odds = all_bets.select('moneyline_odds').to_numpy().ravel()

    return X_bets, y_bets, moneyline_odds, all_bets


def american_to_decimal(american_odds):
    """Convert American odds to decimal odds"""
    if american_odds == 0:
        return 1.0
    if american_odds > 0:
        return 1 + (american_odds / 100)
    else:
        return 1 + (100 / abs(american_odds))


def calculate_ev(good_bet_prob, american_odds):
    """Calculate EV using good bets model probability"""
    if american_odds == 0:
        return 0
    decimal_odds = american_to_decimal(american_odds)
    ev = (good_bet_prob * decimal_odds) - 1
    return ev


def calculate_bet_roi(good_bet_prob, american_odds, actual_result, bet_size=10):
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
    print("BET ANALYSIS VISUALIZATIONS")
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

    # Get predictions
    print("STEP 4: Generating Base Model Predictions")
    print("-"*80 + "\n")

    xgb_proba_test = xgb_model.predict_proba(X_test_np)
    lgb_proba_test = lgb_model.predict(X_test_np)

    print("[OK] Predictions generated\n")

    # Create good bets data
    print("STEP 5: Creating Good Bets Test Data")
    print("-"*80 + "\n")

    X_bets_test, y_bets_test, moneyline_odds, bets_df = create_good_bets_data(
        test_df, y_test_np, xgb_proba_test, lgb_proba_test
    )

    # Load good bets model
    print("STEP 6: Loading Good Bets Model")
    print("-"*80 + "\n")

    model_path = Path(__file__).parent / "saved" / "good_bets_rf_model_final.pkl"
    with open(model_path, 'rb') as f:
        good_bets_model = pickle.load(f)
    print(f"[OK] Model loaded\n")

    # Get good bets probabilities
    print("STEP 7: Getting Good Bets Predictions")
    print("-"*80 + "\n")

    good_bet_probs = good_bets_model.predict_proba(X_bets_test)[:, 1]
    print(f"[OK] Good bets predictions generated\n")

    # Calculate EV and ROI
    print("STEP 8: Calculating EV and ROI")
    print("-"*80 + "\n")

    evs = np.array([calculate_ev(prob, odds) for prob, odds in zip(good_bet_probs, moneyline_odds)])

    profits_and_rois = [
        calculate_bet_roi(prob, odds, result)
        for prob, odds, result in zip(good_bet_probs, moneyline_odds, y_bets_test)
    ]
    profits = np.array([p[0] for p in profits_and_rois])
    rois = np.array([p[1] for p in profits_and_rois])

    xgb_probs = bets_df.select('xgb_prob_team_1').to_numpy().ravel()
    lgb_probs = bets_df.select('lgb_prob_team_1').to_numpy().ravel()

    print(f"[OK] EV and ROI calculated\n")

    # Correlation analysis
    print("STEP 9: Correlation Analysis")
    print("-"*80 + "\n")

    corr_data = {
        'XGB Prob': xgb_probs,
        'LGB Prob': lgb_probs,
        'Good Bet Prob': good_bet_probs,
        'EV': evs,
        'ROI': rois,
        'Actual Result': y_bets_test
    }

    corr_matrix = np.corrcoef([
        xgb_probs, lgb_probs, good_bet_probs, evs, rois, y_bets_test.astype(float)
    ])

    print("Correlation Matrix:\n")
    print(f"{'Variable':<20} {'XGB':<12} {'LGB':<12} {'Good Bet':<12} {'EV':<12} {'ROI':<12} {'Actual':<12}")
    print("-" * 95)

    labels = ['XGB Prob', 'LGB Prob', 'Good Bet Prob', 'EV', 'ROI', 'Actual Result']
    for i, label in enumerate(labels):
        print(f"{label:<20}", end='')
        for j in range(len(labels)):
            print(f"{corr_matrix[i][j]:<12.4f}", end='')
        print()

    # Create visualizations
    print("\n" + "STEP 10: Creating Visualizations")
    print("-"*80 + "\n")

    output_dir = Path(__file__).parent / "saved" / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (15, 12)

    # 1. Scatter plots: Model Probs vs ROI
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Model Probabilities vs ROI', fontsize=16, fontweight='bold')

    # XGB Prob vs ROI
    axes[0, 0].scatter(xgb_probs, rois, alpha=0.3, s=10, c=y_bets_test, cmap='RdYlGn')
    axes[0, 0].set_xlabel('XGB Probability')
    axes[0, 0].set_ylabel('ROI (%)')
    axes[0, 0].set_title('XGB Probability vs ROI')
    axes[0, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    corr_xgb_roi = pearsonr(xgb_probs, rois)[0]
    axes[0, 0].text(0.05, 0.95, f'Correlation: {corr_xgb_roi:.4f}',
                    transform=axes[0, 0].transAxes, va='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # LGB Prob vs ROI
    axes[0, 1].scatter(lgb_probs, rois, alpha=0.3, s=10, c=y_bets_test, cmap='RdYlGn')
    axes[0, 1].set_xlabel('LGB Probability')
    axes[0, 1].set_ylabel('ROI (%)')
    axes[0, 1].set_title('LGB Probability vs ROI')
    axes[0, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    corr_lgb_roi = pearsonr(lgb_probs, rois)[0]
    axes[0, 1].text(0.05, 0.95, f'Correlation: {corr_lgb_roi:.4f}',
                    transform=axes[0, 1].transAxes, va='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Good Bet Prob vs ROI
    axes[1, 0].scatter(good_bet_probs, rois, alpha=0.3, s=10, c=y_bets_test, cmap='RdYlGn')
    axes[1, 0].set_xlabel('Good Bet Model Probability')
    axes[1, 0].set_ylabel('ROI (%)')
    axes[1, 0].set_title('Good Bet Model Probability vs ROI')
    axes[1, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    corr_gb_roi = pearsonr(good_bet_probs, rois)[0]
    axes[1, 0].text(0.05, 0.95, f'Correlation: {corr_gb_roi:.4f}',
                    transform=axes[1, 0].transAxes, va='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # EV vs ROI
    axes[1, 1].scatter(evs, rois, alpha=0.3, s=10, c=y_bets_test, cmap='RdYlGn')
    axes[1, 1].set_xlabel('Expected Value (EV)')
    axes[1, 1].set_ylabel('ROI (%)')
    axes[1, 1].set_title('EV vs ROI')
    axes[1, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    axes[1, 1].axvline(x=0, color='r', linestyle='--', alpha=0.5)
    corr_ev_roi = pearsonr(evs, rois)[0]
    axes[1, 1].text(0.05, 0.95, f'Correlation: {corr_ev_roi:.4f}',
                    transform=axes[1, 1].transAxes, va='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_dir / 'model_probs_vs_roi.png', dpi=150, bbox_inches='tight')
    print(f"[OK] Saved: model_probs_vs_roi.png")
    plt.close()

    # 2. Hexbin plots for density
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Density Analysis: Model Probs vs ROI', fontsize=16, fontweight='bold')

    axes[0, 0].hexbin(xgb_probs, rois, gridsize=30, cmap='YlOrRd', mincnt=1)
    axes[0, 0].set_xlabel('XGB Probability')
    axes[0, 0].set_ylabel('ROI (%)')
    axes[0, 0].set_title('XGB Probability vs ROI (Density)')

    axes[0, 1].hexbin(lgb_probs, rois, gridsize=30, cmap='YlOrRd', mincnt=1)
    axes[0, 1].set_xlabel('LGB Probability')
    axes[0, 1].set_ylabel('ROI (%)')
    axes[0, 1].set_title('LGB Probability vs ROI (Density)')

    axes[1, 0].hexbin(good_bet_probs, rois, gridsize=30, cmap='YlOrRd', mincnt=1)
    axes[1, 0].set_xlabel('Good Bet Model Probability')
    axes[1, 0].set_ylabel('ROI (%)')
    axes[1, 0].set_title('Good Bet Model Probability vs ROI (Density)')

    axes[1, 1].hexbin(evs, rois, gridsize=30, cmap='YlOrRd', mincnt=1)
    axes[1, 1].set_xlabel('Expected Value (EV)')
    axes[1, 1].set_ylabel('ROI (%)')
    axes[1, 1].set_title('EV vs ROI (Density)')

    plt.tight_layout()
    plt.savefig(output_dir / 'density_analysis.png', dpi=150, bbox_inches='tight')
    print(f"[OK] Saved: density_analysis.png")
    plt.close()

    # 3. Correlation heatmap
    fig, ax = plt.subplots(figsize=(10, 8))

    corr_vars = np.array([
        xgb_probs, lgb_probs, good_bet_probs, evs, rois, y_bets_test.astype(float)
    ])
    corr_full = np.corrcoef(corr_vars)

    sns.heatmap(corr_full, annot=True, fmt='.3f', cmap='coolwarm', center=0,
                xticklabels=['XGB', 'LGB', 'Good Bet', 'EV', 'ROI', 'Actual'],
                yticklabels=['XGB', 'LGB', 'Good Bet', 'EV', 'ROI', 'Actual'],
                ax=ax, vmin=-1, vmax=1, square=True, cbar_kws={'label': 'Correlation'})

    plt.title('Correlation Matrix - All Variables', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'correlation_heatmap.png', dpi=150, bbox_inches='tight')
    print(f"[OK] Saved: correlation_heatmap.png")
    plt.close()

    # 4. Binned analysis - improved with better spacing
    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)

    fig.suptitle('ROI by Probability Bins', fontsize=18, fontweight='bold', y=0.995)

    # XGB bins
    xgb_bins = np.linspace(0, 1, 11)
    xgb_binned = np.digitize(xgb_probs, xgb_bins)
    xgb_bin_means = [rois[xgb_binned == i].mean() if np.sum(xgb_binned == i) > 0 else 0 for i in range(1, len(xgb_bins))]
    xgb_bin_counts = [np.sum(xgb_binned == i) for i in range(1, len(xgb_bins))]
    xgb_bin_centers = (xgb_bins[:-1] + xgb_bins[1:]) / 2

    ax0 = fig.add_subplot(gs[0, 0])
    bars0 = ax0.bar(xgb_bin_centers, xgb_bin_means, width=0.08, alpha=0.7, color='steelblue', edgecolor='black')
    ax0.axhline(y=0, color='r', linestyle='--', alpha=0.5, linewidth=2)
    ax0.set_xlabel('XGB Probability', fontsize=12, fontweight='bold')
    ax0.set_ylabel('Average ROI (%)', fontsize=12, fontweight='bold')
    ax0.set_title('XGB Probability Bins vs Avg ROI', fontsize=13, fontweight='bold')
    ax0.set_ylim(-150, 300)
    ax0.grid(axis='y', alpha=0.3)

    # Add count labels with background
    for x, y, cnt in zip(xgb_bin_centers, xgb_bin_means, xgb_bin_counts):
        offset = 15 if y >= 0 else -25
        ax0.text(x, y + offset, f'n={cnt}', ha='center', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

    # LGB bins
    lgb_bins = np.linspace(0, 1, 11)
    lgb_binned = np.digitize(lgb_probs, lgb_bins)
    lgb_bin_means = [rois[lgb_binned == i].mean() if np.sum(lgb_binned == i) > 0 else 0 for i in range(1, len(lgb_bins))]
    lgb_bin_counts = [np.sum(lgb_binned == i) for i in range(1, len(lgb_bins))]
    lgb_bin_centers = (lgb_bins[:-1] + lgb_bins[1:]) / 2

    ax1 = fig.add_subplot(gs[0, 1])
    bars1 = ax1.bar(lgb_bin_centers, lgb_bin_means, width=0.08, alpha=0.7, color='coral', edgecolor='black')
    ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5, linewidth=2)
    ax1.set_xlabel('LGB Probability', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Average ROI (%)', fontsize=12, fontweight='bold')
    ax1.set_title('LGB Probability Bins vs Avg ROI', fontsize=13, fontweight='bold')
    ax1.set_ylim(-150, 300)
    ax1.grid(axis='y', alpha=0.3)

    for x, y, cnt in zip(lgb_bin_centers, lgb_bin_means, lgb_bin_counts):
        offset = 15 if y >= 0 else -25
        ax1.text(x, y + offset, f'n={cnt}', ha='center', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

    # Good Bet bins
    gb_bins = np.linspace(0, 1, 11)
    gb_binned = np.digitize(good_bet_probs, gb_bins)
    gb_bin_means = [rois[gb_binned == i].mean() if np.sum(gb_binned == i) > 0 else 0 for i in range(1, len(gb_bins))]
    gb_bin_counts = [np.sum(gb_binned == i) for i in range(1, len(gb_bins))]
    gb_bin_centers = (gb_bins[:-1] + gb_bins[1:]) / 2

    ax2 = fig.add_subplot(gs[1, 0])
    bars2 = ax2.bar(gb_bin_centers, gb_bin_means, width=0.08, alpha=0.7, color='seagreen', edgecolor='black')
    ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5, linewidth=2)
    ax2.set_xlabel('Good Bet Model Probability', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Average ROI (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Good Bet Probability Bins vs Avg ROI', fontsize=13, fontweight='bold')
    ax2.set_ylim(-150, 300)
    ax2.grid(axis='y', alpha=0.3)

    for x, y, cnt in zip(gb_bin_centers, gb_bin_means, gb_bin_counts):
        offset = 15 if y >= 0 else -25
        ax2.text(x, y + offset, f'n={cnt}', ha='center', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

    # EV bins - with larger text and dynamic y-axis
    ev_bins = np.linspace(evs.min(), evs.max(), 11)
    ev_binned = np.digitize(evs, ev_bins)
    ev_bin_means = [rois[ev_binned == i].mean() if np.sum(ev_binned == i) > 0 else 0 for i in range(1, len(ev_bins))]
    ev_bin_counts = [np.sum(ev_binned == i) for i in range(1, len(ev_bins))]
    ev_bin_centers = (ev_bins[:-1] + ev_bins[1:]) / 2

    ax3 = fig.add_subplot(gs[1, 1])
    bars3 = ax3.bar(ev_bin_centers, ev_bin_means, width=(ev_bins[1]-ev_bins[0])*0.8, alpha=0.7, color='purple', edgecolor='black')
    ax3.axhline(y=0, color='r', linestyle='--', alpha=0.5, linewidth=2)
    ax3.set_xlabel('Expected Value (EV)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Average ROI (%)', fontsize=12, fontweight='bold')
    ax3.set_title('EV Bins vs Avg ROI', fontsize=13, fontweight='bold')

    # Dynamic y-axis based on actual data
    max_roi_ev = max(ev_bin_means) if ev_bin_means else 0
    min_roi_ev = min(ev_bin_means) if ev_bin_means else 0
    y_margin = (max_roi_ev - min_roi_ev) * 0.25
    ax3.set_ylim(min_roi_ev - y_margin, max_roi_ev + y_margin)
    ax3.grid(axis='y', alpha=0.3)

    # EV bins with larger, more visible labels - positioned inside or outside bars
    for x, y, cnt in zip(ev_bin_centers, ev_bin_means, ev_bin_counts):
        # Position label outside bar, with smart positioning
        if y >= 0:
            label_y = y + (max_roi_ev - min_roi_ev) * 0.05
        else:
            label_y = y - (max_roi_ev - min_roi_ev) * 0.08

        ax3.text(x, label_y, f'n={cnt}', ha='center', fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='yellow', alpha=0.9, edgecolor='black', linewidth=1.5))

    plt.savefig(output_dir / 'binned_roi_analysis.png', dpi=200, bbox_inches='tight')
    print(f"[OK] Saved: binned_roi_analysis.png")
    plt.close()

    # 5. Distribution plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Distribution Analysis', fontsize=16, fontweight='bold')

    axes[0, 0].hist(xgb_probs, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    axes[0, 0].set_xlabel('XGB Probability')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('XGB Probability Distribution')

    axes[0, 1].hist(lgb_probs, bins=50, alpha=0.7, color='coral', edgecolor='black')
    axes[0, 1].set_xlabel('LGB Probability')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('LGB Probability Distribution')

    axes[0, 2].hist(good_bet_probs, bins=50, alpha=0.7, color='seagreen', edgecolor='black')
    axes[0, 2].set_xlabel('Good Bet Model Probability')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].set_title('Good Bet Probability Distribution')

    axes[1, 0].hist(evs, bins=50, alpha=0.7, color='purple', edgecolor='black')
    axes[1, 0].axvline(x=0, color='r', linestyle='--', alpha=0.5)
    axes[1, 0].set_xlabel('Expected Value (EV)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('EV Distribution')

    axes[1, 1].hist(rois, bins=50, alpha=0.7, color='gold', edgecolor='black')
    axes[1, 1].axvline(x=0, color='r', linestyle='--', alpha=0.5)
    axes[1, 1].set_xlabel('ROI (%)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('ROI Distribution')

    # Win/Loss pie chart
    wins = np.sum(y_bets_test)
    losses = len(y_bets_test) - wins
    axes[1, 2].pie([wins, losses], labels=['Wins', 'Losses'], autopct='%1.1f%%',
                   colors=['green', 'red'], startangle=90)
    axes[1, 2].set_title(f'Win/Loss Distribution (n={len(y_bets_test)})')

    plt.tight_layout()
    plt.savefig(output_dir / 'distributions.png', dpi=150, bbox_inches='tight')
    print(f"[OK] Saved: distributions.png")
    plt.close()

    # Save correlation analysis report
    print("\n" + "STEP 11: Saving Analysis Report")
    print("-"*80 + "\n")

    report_file = output_dir.parent / "analysis_report.txt"
    with open(report_file, 'w') as f:
        f.write("BET ANALYSIS - CORRELATION REPORT\n")
        f.write("="*80 + "\n\n")

        f.write("CORRELATION WITH ROI:\n")
        f.write("-"*80 + "\n")
        f.write(f"XGB Probability vs ROI:        {corr_xgb_roi:.4f}\n")
        f.write(f"LGB Probability vs ROI:        {corr_lgb_roi:.4f}\n")
        f.write(f"Good Bet Probability vs ROI:   {corr_gb_roi:.4f}\n")
        f.write(f"EV vs ROI:                     {corr_ev_roi:.4f}\n\n")

        f.write("INTERPRETATION:\n")
        f.write("-"*80 + "\n")
        if abs(corr_gb_roi) > abs(corr_xgb_roi) and abs(corr_gb_roi) > abs(corr_lgb_roi):
            f.write("Good Bet Model shows STRONGEST correlation with ROI\n")
        if corr_ev_roi < 0.3 and corr_ev_roi > 0:
            f.write("EV has weak positive correlation with ROI (expected)\n")
        f.write("\n")

        f.write("KEY STATISTICS:\n")
        f.write("-"*80 + "\n")
        f.write(f"Total Bets:        {len(y_bets_test)}\n")
        f.write(f"Total Wins:        {int(wins)}\n")
        f.write(f"Win Rate:          {wins/len(y_bets_test)*100:.2f}%\n")
        f.write(f"Total ROI:         {np.sum(profits)/len(y_bets_test)*100:.2f}%\n")
        f.write(f"Average ROI/Bet:   {np.mean(rois):.2f}%\n")
        f.write(f"Median ROI/Bet:    {np.median(rois):.2f}%\n")
        f.write(f"Std Dev ROI:       {np.std(rois):.2f}%\n\n")

        f.write("FULL CORRELATION MATRIX:\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Variable':<20} {'XGB':<12} {'LGB':<12} {'Good Bet':<12} {'EV':<12} {'ROI':<12} {'Actual':<12}\n")
        for i, label in enumerate(labels):
            f.write(f"{label:<20}", )
            for j in range(len(labels)):
                f.write(f"{corr_matrix[i][j]:<12.4f}")
            f.write("\n")

    print(f"[OK] Analysis report saved to {report_file}\n")

    print("="*80)
    print("[OK] Analysis complete! Check 'visualizations' folder for charts")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
