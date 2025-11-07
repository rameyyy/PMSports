#!/usr/bin/env python3
"""
Betting Simulation - 3-Model Ensemble (XGBoost + LightGBM + CatBoost)

This script:
1. Loads features from multiple years
2. Trains 3-model ensemble with optimal weights
3. Gets predictions on test set
4. Joins predictions with betting odds
5. Simulates bets at different thresholds
6. Shows profitability
"""
import polars as pl
import os
import sys
import numpy as np

# Add models directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'models'))

from models.ensemble3_model import Ensemble3Model


def apply_data_quality_filters(df, min_bookmakers=2):
    """Apply data quality filters"""
    df = df.filter(
        pl.col('team_1_adjoe').is_not_null() &
        pl.col('team_1_adjde').is_not_null() &
        pl.col('team_2_adjoe').is_not_null() &
        pl.col('team_2_adjde').is_not_null()
    )
    df = df.filter(pl.col('avg_ou_line').is_not_null())
    df = df.filter(pl.col('num_books_with_ou') >= min_bookmakers)
    return df


def remove_rows_with_too_many_nulls(df, null_threshold=0.2):
    """Remove rows where null percentage exceeds threshold"""
    metadata_cols = {'game_id', 'date', 'team_1', 'team_2'}
    feature_cols = [c for c in df.columns if c not in metadata_cols]

    null_counts = df.select(feature_cols).select([
        pl.sum_horizontal(pl.all().is_null()).alias('null_count')
    ])

    total_feature_cols = len(feature_cols)
    null_pct = null_counts['null_count'] / total_feature_cols if total_feature_cols > 0 else 0
    valid_rows = null_pct <= null_threshold
    return df.filter(valid_rows)


def load_features():
    """Load and concatenate features from multiple years"""
    print("\n1. Loading features from multiple years...")
    dfs = []

    for year in [2021, 2022, 2023, 2024, 2025]:
        filename = f"features{year}.csv"
        try:
            df = pl.read_csv(filename)
            dfs.append(df)
            print(f"   Loaded {filename}: {len(df)} games")
        except FileNotFoundError:
            pass

    try:
        df = pl.read_csv("features.csv")
        dfs.append(df)
        print(f"   Loaded features.csv: {len(df)} games")
    except FileNotFoundError:
        pass

    if not dfs:
        raise FileNotFoundError("No feature files found")

    features_df = pl.concat(dfs)
    print(f"   Total combined: {len(features_df)} games")
    return features_df


def american_to_decimal(american_odds):
    """Convert American odds to decimal odds"""
    if american_odds > 0:
        return (american_odds / 100) + 1
    else:
        return (100 / abs(american_odds)) + 1


def calculate_profit(decimal_odds, won):
    """Calculate profit from a $10 bet"""
    bet_amount = 10
    if won:
        return (bet_amount * decimal_odds) - bet_amount
    else:
        return -bet_amount


def main():
    print("="*80)
    print("BETTING SIMULATION - 3-MODEL ENSEMBLE (XGBoost + LightGBM + CatBoost)")
    print("="*80)

    # Load features
    features_df = load_features()

    # SAVE the betting odds data NOW (before any filtering)
    print("\n2. Extracting betting odds and lines data...")
    # Only use MyBookie.ag and Bovada
    bookmakers = ['MyBookie.ag', 'Bovada']
    betting_cols = ['game_id', 'avg_ou_line']

    for bookie in bookmakers:
        betting_cols.append(f"{bookie}_ou_line")
        betting_cols.append(f"{bookie}_over_odds")
        betting_cols.append(f"{bookie}_under_odds")

    betting_odds_df = features_df.select(betting_cols)
    print(f"   Saved betting data: MyBookie.ag and Bovada only")

    # Apply data quality filters
    print("\n3. Applying data quality filters...")
    features_df = apply_data_quality_filters(features_df, min_bookmakers=2)
    print(f"   Games after filters: {len(features_df)}")

    # Remove rows with too many nulls
    features_df = remove_rows_with_too_many_nulls(features_df, null_threshold=0.2)
    print(f"   Games after null filtering: {len(features_df)}")

    # Ensure actual_total is not null
    features_df = features_df.filter(pl.col('actual_total').is_not_null())
    print(f"   Games with valid target: {len(features_df)}")

    # Train 3-model ensemble
    print("\n4. Training 3-model ensemble...")
    ensemble = Ensemble3Model(xgb_weight=0.441, lgb_weight=0.466, cat_weight=0.093)

    optimized_xgb_params = {
        'learning_rate': 0.1356325569317646,
        'max_depth': 3,
        'n_estimators': 264,
        'min_child_weight': 10,
        'subsample': 0.9059553897628048,
        'colsample_bytree': 0.8651858536173023,
        'reg_alpha': 1.7596003894852836,
        'reg_lambda': 0.0687329597968497,
    }

    optimized_lgb_params = {
        'learning_rate': 0.1133837674716694,
        'max_depth': 3,
        'num_leaves': 49,
        'min_child_samples': 12,
        'subsample': 0.7991800060529038,
        'colsample_bytree': 0.8152595898952936,
        'reg_alpha': 0.8915908456370663,
        'reg_lambda': 0.2613802136226955,
    }

    optimized_cat_params = {
        'learning_rate': 0.076195,
        'depth': 3,
        'iterations': 238,
        'l2_leaf_reg': 9.63,
        'subsample': 0.859,
        'colsample_bylevel': 0.963,
    }

    metrics = ensemble.train(
        features_df,
        test_size=0.2,
        xgb_params=optimized_xgb_params,
        lgb_params=optimized_lgb_params,
        cat_params=optimized_cat_params,
    )

    print(f"   XGBoost Test MAE: {metrics['xgb_test_mae']:.4f}")
    print(f"   LightGBM Test MAE: {metrics['lgb_test_mae']:.4f}")
    print(f"   CatBoost Test MAE: {metrics['cat_test_mae']:.4f}")
    print(f"   3-Model Ensemble Test MAE: {metrics['ensemble_test_mae']:.4f}")
    print(f"   Test games: {metrics['n_test']}")

    # Make predictions on all data
    print("\n5. Making predictions...")
    predictions = ensemble.predict(features_df)

    # Get test set indices (chronological split)
    n_samples = len(features_df)
    n_test = int(n_samples * 0.2)
    n_train = n_samples - n_test
    test_indices = list(range(n_train, n_samples))

    print(f"   Test set size: {len(test_indices)}")

    # JOIN predictions with betting odds
    print("\n6. Joining predictions with betting odds...")
    pred_df = pl.DataFrame({
        'game_id': [predictions['game_id'][i] for i in test_indices],
        'date': [predictions['date'][i] for i in test_indices],
        'team_1': [predictions['team_1'][i] for i in test_indices],
        'team_2': [predictions['team_2'][i] for i in test_indices],
        'predicted_total': [predictions['predicted_total'][i] for i in test_indices],
        'actual_total': [predictions['actual_total'][i] for i in test_indices],
    })

    # Join with betting odds
    pred_df = pred_df.join(betting_odds_df, on='game_id', how='left')
    print(f"   Joined {len(pred_df)} predictions with odds")

    # Define betting thresholds (OVER BETS ONLY: +2.0 to +3.5 in 0.1 increments)
    thresholds = [round(x, 1) for x in np.arange(2.0, 3.6, 0.1)]

    print(f"\n7. Simulating bets at different OVER thresholds...")
    print("   Thresholds (+2.0 to +3.5): OVER bets only when prediction > line + threshold")
    print("   Bet amount: $10 per bet\n")

    results = []

    # For threshold +2.3, we'll collect detailed bet data
    threshold_2_3_bets = []

    for threshold in thresholds:
        bets_placed = 0
        wins = 0
        losses = 0
        total_profit = 0

        for row in pred_df.iter_rows(named=True):
            predicted = row['predicted_total']
            actual = row['actual_total']

            # Find best available line (use avg_ou_line as baseline)
            best_line = row['avg_ou_line']
            if best_line is None or np.isnan(best_line):
                continue

            # Find BEST (highest) over_odds between MyBookie.ag and Bovada only
            best_over_odds = None
            best_under_odds = None
            best_bookie = None

            for bookie in bookmakers:  # MyBookie.ag and Bovada only
                over_odds_col = f"{bookie}_over_odds"
                under_odds_col = f"{bookie}_under_odds"

                if over_odds_col in row and row[over_odds_col] is not None and not np.isnan(row[over_odds_col]):
                    odds = float(row[over_odds_col])
                    # For American odds, higher (less negative) is better
                    if best_over_odds is None or odds > best_over_odds:
                        best_over_odds = odds
                        best_bookie = bookie
                        if under_odds_col in row and row[under_odds_col] is not None:
                            best_under_odds = float(row[under_odds_col])

            # Skip if no valid odds from MyBookie or Bovada
            if best_over_odds is None:
                continue

            diff_from_line = predicted - best_line

            # BET OVER if prediction > line + threshold
            if diff_from_line >= threshold:
                bets_placed += 1

                # Use best over odds from data (American odds format)
                american_odds = float(best_over_odds)
                decimal_odds = american_to_decimal(american_odds)

                # Bet WINS if actual total is GREATER than the line
                won = actual > best_line

                profit = calculate_profit(decimal_odds, won)
                total_profit += profit

                if won:
                    wins += 1
                else:
                    losses += 1

                # Save detailed data for threshold +2.3
                if threshold == 2.3:
                    threshold_2_3_bets.append({
                        'game_id': row['game_id'],
                        'date': row['date'],
                        'team_1': row['team_1'],
                        'team_2': row['team_2'],
                        'prediction': predicted,
                        'actual_total': actual,
                        'sportsbook': best_bookie,
                        'bet_type': 'OVER',
                        'line': best_line,
                        'over_odds': best_over_odds,
                        'under_odds': best_under_odds if best_under_odds else -110,
                        'won': won,
                        'profit': profit,
                    })


        # Calculate results
        win_rate = (wins / bets_placed * 100) if bets_placed > 0 else 0
        roi = (total_profit / (bets_placed * 10) * 100) if bets_placed > 0 else 0

        results.append({
            'threshold': threshold,
            'bets_placed': bets_placed,
            'wins': wins,
            'losses': losses,
            'win_rate': win_rate,
            'total_profit': total_profit,
            'avg_profit_per_bet': total_profit / bets_placed if bets_placed > 0 else 0,
            'roi': roi,
        })

    # Display results
    print("="*80)
    print("BETTING SIMULATION RESULTS - 3-MODEL ENSEMBLE")
    print("="*80)

    print("\n" + "OVER BETS (prediction > line + threshold)")
    print(" "*5 + f"{'Threshold':<12} {'Bets':<8} {'Wins':<8} {'Win %':<10} {'Total $':<12} {'Avg/Bet':<12} {'ROI %':<10}")
    print("="*95)

    for result in results:
        marker = " <- SELECTED" if result['threshold'] == 2.3 else ""
        print(f"   +{result['threshold']:<10} {result['bets_placed']:<8} {result['wins']:<8} {result['win_rate']:<10.1f} ${result['total_profit']:<11.2f} ${result['avg_profit_per_bet']:<11.2f} {result['roi']:<10.1f}{marker}")

    # Save threshold +2.3 detailed results to CSV
    if threshold_2_3_bets:
        import pandas as pd
        import os
        df_threshold_2_3 = pd.DataFrame(threshold_2_3_bets)
        csv_path = "ensemble3_threshold_2_3_detailed_bets.csv"
        # Remove if exists to avoid permission issues
        if os.path.exists(csv_path):
            try:
                os.remove(csv_path)
            except:
                pass
        df_threshold_2_3.to_csv(csv_path, index=False)
        print(f"\nSaved {len(threshold_2_3_bets)} threshold +2.3 bets to ensemble3_threshold_2_3_detailed_bets.csv")

    # Analysis
    print(f"\n" + "="*80)
    print(f"ANALYSIS")
    print(f"="*80)

    if any(r['bets_placed'] > 0 for r in results):
        best_threshold = max([r for r in results if r['bets_placed'] > 0], key=lambda x: x['total_profit'])
        print(f"\nMost Profitable Threshold: ±{best_threshold['threshold']}")
        print(f"   Bets: {best_threshold['bets_placed']}")
        print(f"   Profit: ${best_threshold['total_profit']:.2f}")
        print(f"   ROI: {best_threshold['roi']:.1f}%")

        total_bets = sum(r['bets_placed'] for r in results)
        total_profit_all = sum(r['total_profit'] for r in results)
        print(f"\nTotal across all thresholds:")
        print(f"   Total bets placed: {total_bets}")
        print(f"   Total profit: ${total_profit_all:.2f}")

        print(f"\n" + "="*80)
        if total_profit_all > 0:
            print(f"✅ 3-Model Ensemble betting strategy is PROFITABLE: +${total_profit_all:.2f}")
        else:
            print(f"❌ 3-Model Ensemble betting strategy is NOT PROFITABLE: -${abs(total_profit_all):.2f}")
    else:
        print("\n❌ No bets placed at any threshold")


if __name__ == "__main__":
    main()
