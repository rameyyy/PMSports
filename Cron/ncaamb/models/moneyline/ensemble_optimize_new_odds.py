#!/usr/bin/env python3
"""
Train LightGBM and XGBoost, get predictions, and find optimal ensemble weighting
Tests all weights from 0-100 (LGB to XGB) in 0.5% increments
"""

import polars as pl
import lightgbm as lgb
import xgboost as xgb
import numpy as np
import os
from sklearn.metrics import accuracy_score, roc_auc_score
import pandas as pd

def load_features_fast(years):
    """Load features with minimal overhead"""
    dfs = []
    features_dir = os.path.join(os.path.dirname(__file__), '..', '..')

    first_loaded = False
    reference_schema = None

    for year in years:
        file = os.path.join(features_dir, f"features{year}.csv")
        if not os.path.exists(file):
            continue
        print(f"[+] Loading {file}...")
        df = pl.read_csv(file)

        # Standardize schema on first load
        if not first_loaded:
            for col in df.columns:
                if df[col].dtype in [pl.Float32, pl.Float64]:
                    df = df.with_columns(pl.col(col).cast(pl.Float64))
                elif df[col].dtype in [pl.Int32, pl.Int64]:
                    df = df.with_columns(pl.col(col).cast(pl.Int64))
            reference_schema = df.schema
            first_loaded = True
        else:
            # Cast to match reference schema
            for col in df.columns:
                if col in reference_schema:
                    target_dtype = reference_schema[col]
                    if df[col].dtype != target_dtype:
                        try:
                            df = df.with_columns(pl.col(col).cast(target_dtype))
                        except:
                            pass

        dfs.append(df)

    combined = pl.concat(dfs, how='diagonal')
    return combined

def create_target_variable(df):
    df = df.with_columns([
        pl.when(
            (pl.col('team_1_score').is_not_null()) &
            (pl.col('team_2_score').is_not_null())
        ).then(
            pl.when(pl.col('team_1_score') > pl.col('team_2_score')).then(1).otherwise(0)
        ).otherwise(None).alias('ml_target')
    ])
    return df

def preprocess_features(df):
    odds_cols = [
        'betonline_ml_team_1', 'betonline_ml_team_2',
        'betonline_spread_odds_team_1', 'betonline_spread_odds_team_2',
        'betonline_spread_pts_team_1', 'betonline_spread_pts_team_2',
        'fanduel_spread_odds_team_1', 'fanduel_spread_odds_team_2',
        'fanduel_spread_pts_team_1', 'fanduel_spread_pts_team_2',
        'mybookie_ml_team_1', 'mybookie_ml_team_2',
        'mybookie_spread_odds_team_1', 'mybookie_spread_odds_team_2',
        'mybookie_spread_pts_team_1', 'mybookie_spread_pts_team_2',
    ]
    for col in odds_cols:
        if col in df.columns:
            df = df.with_columns(pl.col(col).cast(pl.Float64))
    return df

def filter_quality_games(df):
    initial_count = len(df)
    df = df.filter(
        (pl.col('team_1_data_quality') >= 0.5) &
        (pl.col('team_2_data_quality') >= 0.5)
    )
    df = df.filter(pl.col('ml_target').is_not_null())
    df = df.filter(
        (pl.col('avg_ml_team_1').is_not_null()) &
        (pl.col('avg_ml_team_2').is_not_null())
    )
    removed = initial_count - len(df)
    print(f"[+] Filtered: {initial_count} -> {len(df)} ({removed} removed)")
    return df

def get_feature_columns(df):
    metadata_cols = {
        'game_id', 'date', 'season', 'team_1', 'team_2',
        'team_1_score', 'team_2_score', 'actual_total',
        'team_1_conference', 'team_2_conference',
        'team_1_is_home', 'team_2_is_home', 'location',
        'total_score_outcome', 'team_1_winloss',
        'team_1_leaderboard', 'team_2_leaderboard',
        'team_1_match_hist', 'team_2_match_hist',
        'team_1_hist_count', 'team_2_hist_count',
        'start_time', 'game_odds', 'ml_target',
        'team_1_data_quality', 'team_2_data_quality'
    }
    feature_cols = []
    for col in df.columns:
        if col not in metadata_cols:
            if df[col].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]:
                feature_cols.append(col)
    return feature_cols

def prepare_training_data(df, feature_cols):
    df = df.with_columns([
        pl.col(col).fill_null(0) for col in feature_cols
    ])
    X = df.select(feature_cols).to_numpy()
    y = df.select('ml_target').to_numpy().ravel()
    return X, y

def main():
    print("="*80)
    print("ENSEMBLE OPTIMIZATION - LIGHTGBM + XGBOOST")
    print("="*80)

    # Load data
    print("\n[STEP 1] Loading training data (2021-2024)...")
    train_df = load_features_fast(['2021', '2022', '2023', '2024'])

    print("[STEP 2] Loading test data (2025)...")
    test_df = load_features_fast(['2025'])

    # Prepare
    print("[STEP 3] Processing data...")
    train_df = create_target_variable(train_df)
    test_df = create_target_variable(test_df)
    train_df = preprocess_features(train_df)
    test_df = preprocess_features(test_df)

    print("[STEP 4] Filtering quality games...")
    train_df = filter_quality_games(train_df)
    test_df = filter_quality_games(test_df)

    feature_cols = get_feature_columns(train_df)
    print(f"[+] Features: {len(feature_cols)}")

    print("[STEP 5] Preparing data...")
    X_train, y_train = prepare_training_data(train_df, feature_cols)
    X_test, y_test = prepare_training_data(test_df, feature_cols)

    print(f"[+] Train: {X_train.shape[0]} games, {X_train.shape[1]} features")
    print(f"[+] Test: {X_test.shape[0]} games")

    # Train LightGBM
    print("\n[STEP 6] Training LightGBM...")
    lgb_params = {
        'num_leaves': 10,
        'learning_rate': 0.011905546738777037,
        'feature_fraction': 0.5,
        'bagging_fraction': 0.6902301680678105,
        'min_data_in_leaf': 100,
        'lambda_l1': 5,
        'lambda_l2': 5,
        'bagging_freq': 5,
        'objective': 'binary',
        'metric': 'binary_logloss',
        'verbose': -1,
        'seed': 42,
    }

    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
    lgb_model = lgb.train(
        lgb_params,
        train_data,
        num_boost_round=200,
        valid_sets=[train_data, test_data],
        callbacks=[lgb.log_evaluation(period=100), lgb.early_stopping(50)],
    )

    # Train XGBoost
    print("[STEP 7] Training XGBoost with optimized params...")
    xgb_params = {
        'max_depth': 5,
        'learning_rate': 0.186544,
        'subsample': 0.9937,
        'colsample_bytree': 0.4005,
        'min_child_weight': 15,
        'gamma': 9.7163,
        'lambda': 10.1068,
        'n_estimators': 79,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'random_state': 42,
        'verbosity': 0,
    }

    xgb_model = xgb.XGBClassifier(**xgb_params)
    xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    # Get predictions
    print("\n[STEP 8] Getting predictions on test set...")
    lgb_pred = lgb_model.predict(X_test)
    xgb_pred = xgb_model.predict_proba(X_test)[:, 1]

    print(f"[+] LGB predictions: min={lgb_pred.min():.4f}, max={lgb_pred.max():.4f}, mean={lgb_pred.mean():.4f}")
    print(f"[+] XGB predictions: min={xgb_pred.min():.4f}, max={xgb_pred.max():.4f}, mean={xgb_pred.mean():.4f}")

    # Find optimal weighting
    print("\n[STEP 9] Finding optimal ensemble weighting...")
    print("(Testing all weights from LGB:100/XGB:0 to LGB:0/XGB:100)\n")

    results = []
    best_acc = -1
    best_weight_lgb = 0
    best_weight_xgb = 0

    # Test weights from 0 to 100 in 0.5% increments
    weights = np.arange(0, 101, 0.5)

    for lgb_weight in weights:
        xgb_weight = 100 - lgb_weight

        # Normalize to sum to 1
        lgb_w_norm = lgb_weight / 100.0
        xgb_w_norm = xgb_weight / 100.0

        # Ensemble predictions
        ensemble_pred = (lgb_pred * lgb_w_norm) + (xgb_pred * xgb_w_norm)

        # Binary predictions
        ensemble_binary = (ensemble_pred > 0.5).astype(int)

        # Accuracy
        acc = accuracy_score(y_test, ensemble_binary)

        results.append({
            'lgb_weight': lgb_weight,
            'xgb_weight': xgb_weight,
            'accuracy': acc,
        })

        if acc > best_acc:
            best_acc = acc
            best_weight_lgb = lgb_weight
            best_weight_xgb = xgb_weight

        if lgb_weight % 10 == 0:  # Print every 10%
            print(f"[LGB {lgb_weight:5.1f}% / XGB {xgb_weight:5.1f}%] Accuracy: {acc:.4f}")

    # Convert to dataframe for easy analysis
    results_df = pd.DataFrame(results)

    # Get top 10 weightings
    print("\n" + "="*80)
    print("TOP 10 ENSEMBLE WEIGHTINGS")
    print("="*80)
    top_10 = results_df.nlargest(10, 'accuracy')

    for idx, row in top_10.iterrows():
        print(f"[{int(idx)+1:2d}] LGB {row['lgb_weight']:6.1f}% / XGB {row['xgb_weight']:6.1f}% => Accuracy: {row['accuracy']:.6f}")

    # Best result
    print("\n" + "="*80)
    print("OPTIMAL ENSEMBLE WEIGHTING")
    print("="*80)
    print(f"\nBest Weighting:")
    print(f"  LightGBM: {best_weight_lgb:.1f}%")
    print(f"  XGBoost:  {best_weight_xgb:.1f}%")
    print(f"  Test Accuracy: {best_acc:.6f}")

    # Individual model accuracy
    lgb_binary = (lgb_pred > 0.5).astype(int)
    xgb_binary = (xgb_pred > 0.5).astype(int)
    lgb_acc = accuracy_score(y_test, lgb_binary)
    xgb_acc = accuracy_score(y_test, xgb_binary)

    print(f"\nIndividual Model Accuracy:")
    print(f"  LightGBM only: {lgb_acc:.6f}")
    print(f"  XGBoost only:  {xgb_acc:.6f}")
    print(f"  Ensemble:      {best_acc:.6f}")
    print(f"  Improvement:   {best_acc - max(lgb_acc, xgb_acc):.6f}")

    print("\n" + "="*80)

    # Save results
    results_df.to_csv('ensemble_weighting_results.csv', index=False)
    print(f"[+] Full results saved to ensemble_weighting_results.csv")

if __name__ == "__main__":
    main()
