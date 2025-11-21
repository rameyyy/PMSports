#!/usr/bin/env python3
"""
Strategy 2 Threshold Analysis
Explores different Good Bets confidence thresholds to find optimal balance
between bet quantity and ROI
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

# Prepare features
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

# Extract month from date
all_bets = all_bets.with_columns(
    pl.col('date').str.strptime(pl.Date, "%Y-%m-%d").dt.month().alias('month_from_date')
)

# Test different thresholds
print("=" * 100)
print("STRATEGY 2 THRESHOLD ANALYSIS - Finding Optimal Balance")
print("=" * 100 + "\n")

# EV thresholds to test
ev_thresholds = [0.03, 0.04, 0.05, 0.06, 0.07, 0.08]
gb_thresholds = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]

# Store results
results = []

for ev_thresh in ev_thresholds:
    for gb_thresh in gb_thresholds:
        # Filter bets
        filtered_bets = all_bets.filter(
            (pl.col('ev') >= ev_thresh) &
            (pl.col('gb_confidence') >= gb_thresh)
        )

        if len(filtered_bets) == 0:
            continue

        # Calculate stats
        wins = filtered_bets.filter(pl.col('target') == 1).height
        losses = filtered_bets.filter(pl.col('target') == 0).height
        win_rate = (wins / len(filtered_bets) * 100) if len(filtered_bets) > 0 else 0

        stake_per_bet = 10
        total_wagered = len(filtered_bets) * stake_per_bet
        total_ev_profit = filtered_bets.select(pl.col('ev')).to_numpy().flatten()
        total_profit = np.sum(total_ev_profit * stake_per_bet)
        roi = (total_profit / total_wagered * 100) if total_wagered > 0 else 0

        results.append({
            'ev_thresh': ev_thresh,
            'gb_thresh': gb_thresh,
            'bets': len(filtered_bets),
            'wins': wins,
            'losses': losses,
            'win_rate': win_rate,
            'roi': roi,
            'profit': total_profit,
            'wagered': total_wagered
        })

# Sort by ROI descending
results_sorted = sorted(results, key=lambda x: x['roi'], reverse=True)

print("\nTOP 20 CONFIGURATIONS BY ROI:")
print("-" * 100)
print(f"{'EV%':<8} {'GB Conf':<10} {'Bets':<8} {'Wins':<8} {'Win%':<10} {'Wagered':<12} {'ROI':<10} {'Profit':<12}")
print("-" * 100)

for i, r in enumerate(results_sorted[:20]):
    print(f"{r['ev_thresh']*100:>6.0f}% {r['gb_thresh']:>8.2f} {r['bets']:<8} {r['wins']:<8} "
          f"{r['win_rate']:>8.1f}% ${r['wagered']:>10.2f} {r['roi']:>8.1f}% ${r['profit']:>10.2f}")

print("\n" + "=" * 100)
print("TOP 20 CONFIGURATIONS BY BET COUNT (>= 100 bets):")
print("-" * 100)
print(f"{'EV%':<8} {'GB Conf':<10} {'Bets':<8} {'Wins':<8} {'Win%':<10} {'Wagered':<12} {'ROI':<10} {'Profit':<12}")
print("-" * 100)

high_volume = [r for r in results if r['bets'] >= 100]
high_volume_sorted = sorted(high_volume, key=lambda x: x['roi'], reverse=True)

for i, r in enumerate(high_volume_sorted[:20]):
    print(f"{r['ev_thresh']*100:>6.0f}% {r['gb_thresh']:>8.2f} {r['bets']:<8} {r['wins']:<8} "
          f"{r['win_rate']:>8.1f}% ${r['wagered']:>10.2f} {r['roi']:>8.1f}% ${r['profit']:>10.2f}")

print("\n" + "=" * 100)
print("TOP 20 CONFIGURATIONS BY BET COUNT (>= 50 bets):")
print("-" * 100)
print(f"{'EV%':<8} {'GB Conf':<10} {'Bets':<8} {'Wins':<8} {'Win%':<10} {'Wagered':<12} {'ROI':<10} {'Profit':<12}")
print("-" * 100)

medium_volume = [r for r in results if r['bets'] >= 50]
medium_volume_sorted = sorted(medium_volume, key=lambda x: x['roi'], reverse=True)

for i, r in enumerate(medium_volume_sorted[:20]):
    print(f"{r['ev_thresh']*100:>6.0f}% {r['gb_thresh']:>8.2f} {r['bets']:<8} {r['wins']:<8} "
          f"{r['win_rate']:>8.1f}% ${r['wagered']:>10.2f} {r['roi']:>8.1f}% ${r['profit']:>10.2f}")

print("\n" + "=" * 100)
print("RECOMMENDATIONS:")
print("=" * 100 + "\n")

# Find recommendations based on different criteria
best_roi = results_sorted[0]
print(f"BEST ROI: EV >= {best_roi['ev_thresh']*100:.0f}%, GB >= {best_roi['gb_thresh']:.2f}")
print(f"  Result: {best_roi['bets']} bets, {best_roi['win_rate']:.1f}% win rate, {best_roi['roi']:.1f}% ROI\n")

best_balanced = None
best_score = -999
for r in results:
    if r['bets'] >= 50:  # Minimum bet count
        # Score: balance ROI and bet count
        score = (r['roi'] * 0.6) + (min(r['bets'], 300) / 300 * 40)
        if score > best_score:
            best_score = score
            best_balanced = r

if best_balanced:
    print(f"BEST BALANCED (50+ bets): EV >= {best_balanced['ev_thresh']*100:.0f}%, GB >= {best_balanced['gb_thresh']:.2f}")
    print(f"  Result: {best_balanced['bets']} bets, {best_balanced['win_rate']:.1f}% win rate, {best_balanced['roi']:.1f}% ROI\n")

best_volume = None
best_volume_score = -999
for r in results:
    if r['bets'] >= 100:  # High volume
        # Prefer high volume with decent ROI
        score = (r['roi'] * 0.4) + (min(r['bets'], 500) / 500 * 60)
        if score > best_volume_score:
            best_volume_score = score
            best_volume = r

if best_volume:
    print(f"BEST HIGH VOLUME (100+ bets): EV >= {best_volume['ev_thresh']*100:.0f}%, GB >= {best_volume['gb_thresh']:.2f}")
    print(f"  Result: {best_volume['bets']} bets, {best_volume['win_rate']:.1f}% win rate, {best_volume['roi']:.1f}% ROI\n")

print("=" * 100)
