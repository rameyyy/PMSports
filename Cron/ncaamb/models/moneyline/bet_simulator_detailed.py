#!/usr/bin/env python3
"""
Detailed Bet Simulator - 1% EV Buckets
Shows ROI for each EV range from 0% to 100%
"""

import os
import sys
from pathlib import Path
import polars as pl
import numpy as np
import lightgbm as lgb
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
import pickle

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

    print(f"Created target for {len(df_with_scores)} games")
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

    print(f"Created {len(X_bets)} betting rows ({len(df)} games * 2 perspectives)\n")

    return X_bets, y_bets, moneyline_odds


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


def simulate_bets(good_bet_probs, american_odds, actual_results, bet_size=10):
    """Simulate betting with $10 per bet"""
    profit = 0
    wins = 0
    losses = 0

    for prob, odds, result in zip(good_bet_probs, american_odds, actual_results):
        decimal_odds = american_to_decimal(odds)

        if result == 1:  # Bet won
            payout = bet_size * decimal_odds
            profit += payout - bet_size
            wins += 1
        else:  # Bet lost
            profit -= bet_size
            losses += 1

    total_bets = wins + losses
    win_rate = wins / total_bets if total_bets > 0 else 0
    roi = (profit / (total_bets * bet_size)) * 100 if total_bets > 0 else 0

    return profit, wins, losses, roi, win_rate


def main():
    print("\n")
    print("="*80)
    print("DETAILED BET SIMULATOR - 1% EV BUCKETS")
    print("Train on 2021-2024, Test on 2025 with $10 bets")
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

    print(f"Training data shape: X={X_train.shape}\n")

    # Train base models
    print("STEP 2: Training Base Models (XGBoost + LightGBM)")
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

    # Create good bets data
    print("STEP 5: Creating Good Bets Test Data")
    print("-"*80 + "\n")

    X_bets_test, y_bets_test, moneyline_odds = create_good_bets_data(
        test_df, y_test_np, xgb_proba_test, lgb_proba_test
    )

    # Load good bets model
    print("STEP 6: Loading Good Bets Model")
    print("-"*80 + "\n")

    model_path = Path(__file__).parent / "saved" / "good_bets_rf_model_final.pkl"
    with open(model_path, 'rb') as f:
        good_bets_model = pickle.load(f)
    print(f"[OK] Model loaded from {model_path}\n")

    # Get good bets probabilities
    print("STEP 7: Getting Good Bets Predictions")
    print("-"*80 + "\n")

    good_bet_probs = good_bets_model.predict_proba(X_bets_test)[:, 1]
    print(f"[OK] Good bets predictions generated\n")

    # Calculate EV for each bet
    evs = np.array([calculate_ev(prob, odds) for prob, odds in zip(good_bet_probs, moneyline_odds)])

    # Create results dataframe for analysis
    print("STEP 8: Analyzing Bet Performance by 1% EV Buckets (0% to 100%)")
    print("-"*80 + "\n")

    # Define 1% EV buckets from 0% to 100%
    ev_ranges = []
    for i in range(0, 100):
        min_ev = i / 100.0
        max_ev = (i + 1) / 100.0
        label = f"EV {i}% to {i+1}%"
        ev_ranges.append((min_ev, max_ev, label))

    print(f"{'EV Range':<20} {'Bets':<8} {'Wins':<8} {'Losses':<8} {'Win %':<10} {'Profit':<12} {'ROI':<10}")
    print("-" * 95)

    summary_data = []

    for min_ev, max_ev, label in ev_ranges:
        mask = (evs >= min_ev) & (evs < max_ev)

        if np.sum(mask) == 0:
            continue

        range_results = y_bets_test[mask]
        range_odds = moneyline_odds[mask]

        profit, wins, losses, roi, win_rate = simulate_bets(
            good_bet_probs[mask],
            range_odds,
            range_results,
            bet_size=10
        )

        num_bets = wins + losses
        win_pct = (wins / num_bets * 100) if num_bets > 0 else 0

        print(f"{label:<20} {num_bets:<8} {wins:<8} {losses:<8} {win_pct:<10.2f}% ${profit:<11.2f} {roi:<10.2f}%")

        summary_data.append({
            'ev_range': label,
            'min_ev': min_ev,
            'num_bets': num_bets,
            'wins': wins,
            'losses': losses,
            'win_rate': win_pct,
            'profit': profit,
            'roi': roi
        })

    print("-" * 95)

    # Find cumulative performance
    print("\n" + "="*80)
    print("CUMULATIVE ANALYSIS")
    print("="*80 + "\n")

    cumulative_results = []
    cumulative_profit = 0
    cumulative_wins = 0
    cumulative_bets = 0

    # Analyze from 0% onwards
    print("Cumulative from EV >= X%:\n")
    print(f"{'Min EV':<15} {'Bets':<8} {'Wins':<8} {'Win %':<10} {'Cumulative Profit':<18} {'Cumulative ROI':<15}")
    print("-" * 95)

    for threshold_ev in range(0, 101, 5):
        threshold = threshold_ev / 100.0
        mask = evs >= threshold

        if np.sum(mask) == 0:
            continue

        range_results = y_bets_test[mask]
        range_odds = moneyline_odds[mask]
        range_probs = good_bet_probs[mask]

        profit, wins, losses, roi, win_rate = simulate_bets(
            range_probs,
            range_odds,
            range_results,
            bet_size=10
        )

        num_bets = wins + losses
        win_pct = (wins / num_bets * 100) if num_bets > 0 else 0

        print(f">= {threshold_ev}%          {num_bets:<8} {wins:<8} {win_pct:<10.2f}% ${profit:<17.2f} {roi:<15.2f}%")

    # Save results
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80 + "\n")

    results_file = Path(__file__).parent / "saved" / "bet_simulation_detailed_results.txt"
    with open(results_file, 'w') as f:
        f.write("DETAILED BET SIMULATION RESULTS - 2025 TEST DATA\n")
        f.write("Models trained on 2021-2024, tested on 2025\n")
        f.write("$10 per bet, decimal odds conversion\n")
        f.write("1% EV Buckets (0% to 100%)\n\n")

        f.write("="*95 + "\n")
        f.write("PERFORMANCE BY 1% EV BUCKET\n")
        f.write("="*95 + "\n\n")

        f.write(f"{'EV Range':<20} {'Bets':<8} {'Wins':<8} {'Losses':<8} {'Win %':<10} {'Profit':<12} {'ROI':<10}\n")
        f.write("-" * 95 + "\n")

        for data in summary_data:
            f.write(f"{data['ev_range']:<20} {data['num_bets']:<8} {data['wins']:<8} {data['losses']:<8} {data['win_rate']:<10.2f}% ${data['profit']:<11.2f} {data['roi']:<10.2f}%\n")

        f.write("-" * 95 + "\n\n")

        f.write("="*95 + "\n")
        f.write("CUMULATIVE ANALYSIS (EV >= X%)\n")
        f.write("="*95 + "\n\n")

        f.write(f"{'Min EV':<15} {'Bets':<8} {'Wins':<8} {'Win %':<10} {'Cumulative Profit':<18} {'Cumulative ROI':<15}\n")
        f.write("-" * 95 + "\n")

        for threshold_ev in range(0, 101, 5):
            threshold = threshold_ev / 100.0
            mask = evs >= threshold

            if np.sum(mask) == 0:
                continue

            range_results = y_bets_test[mask]
            range_odds = moneyline_odds[mask]
            range_probs = good_bet_probs[mask]

            profit, wins, losses, roi, win_rate = simulate_bets(
                range_probs,
                range_odds,
                range_results,
                bet_size=10
            )

            num_bets = wins + losses
            win_pct = (wins / num_bets * 100) if num_bets > 0 else 0

            f.write(f">= {threshold_ev}%          {num_bets:<8} {wins:<8} {win_pct:<10.2f}% ${profit:<17.2f} {roi:<15.2f}%\n")

    print(f"[OK] Results saved to {results_file}\n")

    print("="*80)
    print("[OK] Detailed bet simulation complete!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
