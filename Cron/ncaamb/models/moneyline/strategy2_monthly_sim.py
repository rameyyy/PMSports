#!/usr/bin/env python3
"""
Strategy 2 Monthly Simulation
Shows bets per month and ROI for each month in 2025 test set
"""

import pickle
import polars as pl
import numpy as np
from pathlib import Path
from xgboost import XGBClassifier

# Load the test data (2025)
test_file = Path("../../features2025.csv")
if not test_file.exists():
    print(f"[-] Test file not found: {test_file}")
    exit(1)

print("Loading 2025 test data...")
test_df = pl.read_csv(test_file)
print(f"[+] Loaded {len(test_df)} games\n")

# Load saved models
lgb_model = pickle.load(open("saved/lightgbm_model_final.pkl", "rb"))
gb_model = pickle.load(open("saved/good_bets_rf_model_final.pkl", "rb"))
xgb_model_path = Path("saved/xgboost_model.pkl")

print("[+] Models loaded\n")

# Load expected feature columns
feature_file = Path("saved/feature_columns.txt")
if feature_file.exists():
    with open(feature_file, 'r') as f:
        feature_cols = [line.strip() for line in f if line.strip()]
    print(f"[+] Loaded {len(feature_cols)} expected features\n")
else:
    print("[-] Feature columns file not found")
    exit(1)

# Prepare features - ensure correct order and alignment
df_cols = set(test_df.columns)
missing_features = [f for f in feature_cols if f not in df_cols]
if missing_features:
    print(f"[!] Missing {len(missing_features)} features, adding as 0")
    for feature in missing_features:
        test_df = test_df.with_columns(pl.lit(0.0).alias(feature))

X_test = test_df.select(feature_cols).fill_null(0).to_numpy()
X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

print(f"Getting predictions for {len(X_test)} games...")
lgb_proba = lgb_model.predict(X_test)

xgb_model = XGBClassifier()
xgb_model.load_model(str(xgb_model_path))
xgb_proba = xgb_model.predict_proba(X_test)

print("[+] Got predictions\n")

# Add predictions
test_df = test_df.with_columns([
    pl.lit(xgb_proba[:, 1]).alias('xgb_prob'),
    pl.lit(lgb_proba).alias('lgb_prob'),
])

# Fill nulls
for col in ['avg_ml_team_1', 'avg_ml_team_2', 'month', 'team_1_adjoe', 'team_1_adjde', 'team_2_adjoe', 'team_2_adjde']:
    if col in test_df.columns:
        test_df = test_df.with_columns(pl.col(col).fill_null(0))
    else:
        test_df = test_df.with_columns(pl.lit(0).alias(col))

# Calculate implied probs and EV
test_df = test_df.with_columns([
    pl.col('avg_ml_team_1').map_elements(
        lambda x: 0.5 if x == 0 else (100/(x+100) if x > 0 else abs(x)/(abs(x)+100)),
        return_dtype=pl.Float64
    ).alias('implied_prob_team_1'),
    pl.col('avg_ml_team_2').map_elements(
        lambda x: 0.5 if x == 0 else (100/(x+100) if x > 0 else abs(x)/(abs(x)+100)),
        return_dtype=pl.Float64
    ).alias('implied_prob_team_2'),
])

# EV from LGB
test_df = test_df.with_columns([
    (pl.col('lgb_prob') * pl.when(pl.col('avg_ml_team_1') == 0).then(1.0)
        .when(pl.col('avg_ml_team_1') > 0).then(1 + (pl.col('avg_ml_team_1') / 100))
        .otherwise(1 + (100 / pl.col('avg_ml_team_1').abs())) - 1)
    .alias('ev_team_1'),
    ((1 - pl.col('lgb_prob')) * pl.when(pl.col('avg_ml_team_2') == 0).then(1.0)
        .when(pl.col('avg_ml_team_2') > 0).then(1 + (pl.col('avg_ml_team_2') / 100))
        .otherwise(1 + (100 / pl.col('avg_ml_team_2').abs())) - 1)
    .alias('ev_team_2'),
])

# Create good_bets features
test_df = test_df.with_columns([
    (pl.col('xgb_prob') - pl.col('lgb_prob')).abs().alias('model_disagreement'),
    (pl.col('team_1_adjoe') - pl.col('team_2_adjoe')).abs().alias('strength_diff_1'),
    (pl.col('team_2_adjoe') - pl.col('team_1_adjoe')).abs().alias('strength_diff_2'),
])

# Team 1 perspective
team_1_bets = test_df.select([
    pl.col('xgb_prob'),
    pl.col('lgb_prob'),
    pl.col('model_disagreement'),
    pl.col('avg_ml_team_1').alias('moneyline_odds'),
    pl.col('implied_prob_team_1').alias('implied_prob'),
    pl.col('ev_team_1').alias('ev'),
    pl.col('month'),
    pl.col('strength_diff_1').alias('strength_differential'),
    pl.lit(0).alias('spread_pts_self'),
    pl.lit(0).alias('spread_pts_opp'),
    pl.lit(0).alias('spread_odds_self'),
    pl.lit(0).alias('spread_odds_opp'),
    (pl.col('team_1_score') > pl.col('team_2_score')).cast(pl.Int32).alias('target'),
    pl.col('game_id'),
    pl.col('team_1').alias('team'),
    pl.col('team_2').alias('opponent'),
    pl.col('date'),
])

# Team 2 perspective
team_2_bets = test_df.select([
    (1 - pl.col('xgb_prob')).alias('xgb_prob'),
    (1 - pl.col('lgb_prob')).alias('lgb_prob'),
    pl.col('model_disagreement'),
    pl.col('avg_ml_team_2').alias('moneyline_odds'),
    pl.col('implied_prob_team_2').alias('implied_prob'),
    pl.col('ev_team_2').alias('ev'),
    pl.col('month'),
    pl.col('strength_diff_2').alias('strength_differential'),
    pl.lit(0).alias('spread_pts_self'),
    pl.lit(0).alias('spread_pts_opp'),
    pl.lit(0).alias('spread_odds_self'),
    pl.lit(0).alias('spread_odds_opp'),
    (pl.col('team_2_score') > pl.col('team_1_score')).cast(pl.Int32).alias('target'),
    pl.col('game_id'),
    pl.col('team_2').alias('team'),
    pl.col('team_1').alias('opponent'),
    pl.col('date'),
])

all_bets = pl.concat([team_1_bets, team_2_bets])

# Get good_bets predictions
print("Getting Good Bets predictions...")
feature_order = ['xgb_prob', 'lgb_prob', 'model_disagreement', 'moneyline_odds',
                'implied_prob', 'ev', 'spread_pts_self', 'spread_pts_opp',
                'spread_odds_self', 'spread_odds_opp', 'month', 'strength_differential']

X_gb = all_bets.select(feature_order).to_numpy()
gb_proba = gb_model.predict_proba(X_gb)[:, 1]

all_bets = all_bets.with_columns(pl.lit(gb_proba).alias('gb_confidence'))
print("[+] Good Bets predictions complete\n")

print("=" * 90)
print("STRATEGY 2 SIMULATION - MONTHLY BREAKDOWN (2025 Test Set)")
print("Criteria: EV >= 3%, Good Bets Confidence >= 0.40, $10 per bet")
print("=" * 90 + "\n")

# Filter by Strategy 2 criteria (Variance-Optimized)
strategy2_bets = all_bets.filter(
    (pl.col('ev') >= 0.03) &
    (pl.col('gb_confidence') >= 0.40)
)

print(f"Total bets meeting Strategy 2 criteria: {len(strategy2_bets)}\n")

# Extract month from date - always do this to ensure correct month values
strategy2_bets = strategy2_bets.with_columns(
    pl.col('date').str.strptime(pl.Date, "%Y-%m-%d").dt.month().alias('month_from_date')
)

# Analyze by month
months = {11: "November", 12: "December", 1: "January", 2: "February", 3: "March"}
month_stats = {}

for month_num, month_name in months.items():
    month_bets = strategy2_bets.filter(pl.col('month_from_date') == month_num)

    if len(month_bets) == 0:
        month_stats[month_num] = {
            'name': month_name,
            'count': 0,
            'wins': 0,
            'losses': 0,
            'win_rate': 0,
            'roi': 0,
            'profit': 0,
            'wagered': 0
        }
        continue

    # Calculate outcomes
    wins = month_bets.filter(pl.col('target') == 1).height
    losses = month_bets.filter(pl.col('target') == 0).height
    win_rate = wins / len(month_bets) * 100 if len(month_bets) > 0 else 0

    # Calculate ROI (assuming $10 per bet)
    stake_per_bet = 10
    total_wagered = len(month_bets) * stake_per_bet

    # Profit calculation: sum of EV across all bets
    total_ev_profit = month_bets.select(pl.col('ev')).to_numpy().flatten()
    total_profit = np.sum(total_ev_profit * stake_per_bet)

    roi = (total_profit / total_wagered * 100) if total_wagered > 0 else 0

    month_stats[month_num] = {
        'name': month_name,
        'count': len(month_bets),
        'wins': wins,
        'losses': losses,
        'win_rate': win_rate,
        'roi': roi,
        'profit': total_profit,
        'wagered': total_wagered
    }

# Print results
print("MONTHLY BREAKDOWN:")
print("-" * 90)
print(f"{'Month':<12} {'Bets':<8} {'Wins':<8} {'Losses':<8} {'Win%':<10} {'Wagered':<12} {'ROI':<10} {'Profit':<12}")
print("-" * 90)

total_bets = 0
total_profit = 0
total_wagered = 0
total_wins = 0

for month_num in sorted(month_stats.keys()):
    stats = month_stats[month_num]
    if stats['count'] > 0:
        print(f"{stats['name']:<12} {stats['count']:<8} {stats['wins']:<8} {stats['losses']:<8} "
              f"{stats['win_rate']:>8.1f}% ${stats['wagered']:>10.2f} {stats['roi']:>8.1f}% ${stats['profit']:>10.2f}")
        total_bets += stats['count']
        total_profit += stats['profit']
        total_wagered += stats['wagered']
        total_wins += stats['wins']
    else:
        pass  # Skip empty months in output

print("-" * 90)
if total_bets > 0:
    overall_roi = (total_profit / total_wagered * 100) if total_wagered > 0 else 0
    overall_win_rate = (total_wins / total_bets * 100) if total_bets > 0 else 0
    print(f"{'TOTAL':<12} {total_bets:<8} {total_wins:<8} {total_bets - total_wins:<8} "
          f"{overall_win_rate:>8.1f}% ${total_wagered:>10.2f} {overall_roi:>8.1f}% ${total_profit:>10.2f}")
else:
    print(f"No bets met Strategy 2 criteria")

print("\n" + "=" * 90)
