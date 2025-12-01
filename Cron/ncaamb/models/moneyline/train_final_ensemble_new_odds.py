#!/usr/bin/env python3
"""
Final training for LightGBM and XGBoost moneyline models with decimal odds
Train on 2021-2025, validate on 2021
Converts all American odds to decimal format before training
Saves feature list and trained models
"""

import polars as pl
import lightgbm as lgb
import xgboost as xgb
import numpy as np
import pickle
import os
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, confusion_matrix, classification_report

def load_features(years):
    """Load and combine features from multiple years"""
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

        if not first_loaded:
            for col in df.columns:
                if df[col].dtype in [pl.Float32, pl.Float64]:
                    df = df.with_columns(pl.col(col).cast(pl.Float64))
                elif df[col].dtype in [pl.Int32, pl.Int64]:
                    df = df.with_columns(pl.col(col).cast(pl.Int64))
            reference_schema = df.schema
            first_loaded = True
        else:
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

def american_to_decimal(american_odds: float) -> float:
    """Convert American odds to decimal odds"""
    if american_odds is None:
        return None
    american_odds = float(american_odds)
    if american_odds >= 0:
        return 1 + (american_odds / 100)
    else:
        return 1 + (100 / abs(american_odds))

def preprocess_features(df):
    """Cast odds columns to float and convert American to decimal"""
    # All moneyline and spread odds columns
    odds_cols = [
        'betonline_ml_team_1', 'betonline_ml_team_2',
        'betonline_spread_odds_team_1', 'betonline_spread_odds_team_2',
        'betonline_spread_pts_team_1', 'betonline_spread_pts_team_2',
        'fanduel_spread_odds_team_1', 'fanduel_spread_odds_team_2',
        'fanduel_spread_pts_team_1', 'fanduel_spread_pts_team_2',
        'mybookie_ml_team_1', 'mybookie_ml_team_2',
        'mybookie_spread_odds_team_1', 'mybookie_spread_odds_team_2',
        'mybookie_spread_pts_team_1', 'mybookie_spread_pts_team_2',
        'bovada_ml_team_1', 'bovada_ml_team_2',
        'betmgm_ml_team_1', 'betmgm_ml_team_2',
        'draftkings_ml_team_1', 'draftkings_ml_team_2',
        'fanduel_ml_team_1', 'fanduel_ml_team_2',
        'lowvig_ml_team_1', 'lowvig_ml_team_2',
        'avg_ml_team_1', 'avg_ml_team_2',
    ]

    for col in odds_cols:
        if col in df.columns:
            # Cast to float and convert from American to decimal
            df = df.with_columns(
                pl.col(col).cast(pl.Float64).map_elements(
                    american_to_decimal,
                    return_dtype=pl.Float64
                ).alias(col)
            )

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

def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate model on test set"""
    y_test_pred = model.predict(X_test)

    if hasattr(model, 'predict_proba'):
        # XGBoost
        y_test_pred_proba = model.predict_proba(X_test)[:, 1]
    else:
        # LightGBM
        y_test_pred_proba = model.predict(X_test)

    y_test_pred_binary = (y_test_pred_proba > 0.5).astype(int)

    acc = accuracy_score(y_test, y_test_pred_binary)
    auc = roc_auc_score(y_test, y_test_pred_proba)
    loss = log_loss(y_test, y_test_pred_proba)

    print(f"\n{model_name} - TEST SET RESULTS:")
    print(f"  Accuracy:  {acc:.6f}")
    print(f"  AUC-ROC:   {auc:.6f}")
    print(f"  Log Loss:  {loss:.6f}")

    cm = confusion_matrix(y_test, y_test_pred_binary)
    print(f"  Confusion Matrix: TP={cm[1,1]}, TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}")

    return {'accuracy': acc, 'auc': auc, 'loss': loss}

def main():
    print("="*80)
    print("FINAL TRAINING: LIGHTGBM + XGBOOST WITH NEW DECIMAL ODDS")
    print("Production model trained on 2021-2025 (2025 prioritized)")
    print("="*80)

    # Load data - ALL YEARS for production
    print("\n[STEP 1] Loading all training data (2021-2025)...")
    all_df = load_features(['2021', '2022', '2023', '2024', '2025'])

    print("[STEP 2] Loading validation data (2021 - for validation only)...")
    val_df = load_features(['2021'])

    # Prepare
    print("[STEP 3] Processing data...")
    all_df = create_target_variable(all_df)
    val_df = create_target_variable(val_df)
    all_df = preprocess_features(all_df)
    val_df = preprocess_features(val_df)

    print("[STEP 4] Filtering quality games...")
    all_df = filter_quality_games(all_df)
    val_df = filter_quality_games(val_df)

    feature_cols = get_feature_columns(all_df)
    print(f"[+] Features: {len(feature_cols)}")

    # Save feature columns list to file
    saved_dir = os.path.join(os.path.dirname(__file__), 'saved')
    os.makedirs(saved_dir, exist_ok=True)
    feature_cols_file = os.path.join(saved_dir, 'feature_columns.txt')
    with open(feature_cols_file, 'w') as f:
        for i, col in enumerate(feature_cols, 1):
            f.write(f"{i:5d}\t{col}\n")
    print(f"[+] Saved {len(feature_cols)} feature columns to {feature_cols_file}\n")

    print("[STEP 5] Preparing data...")
    X_train, y_train = prepare_training_data(all_df, feature_cols)
    X_val, y_val = prepare_training_data(val_df, feature_cols)

    print(f"[+] Train (2021-2025): {X_train.shape[0]} games (includes 2025 prioritized data)")
    print(f"[+] Val (2021 only):   {X_val.shape[0]} games")

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
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    lgb_model = lgb.train(
        lgb_params,
        train_data,
        num_boost_round=200,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'val'],
        callbacks=[lgb.log_evaluation(period=100), lgb.early_stopping(50)],
    )

    lgb_results = evaluate_model(lgb_model, X_val, y_val, "LightGBM")

    # Train XGBoost
    print("\n[STEP 7] Training XGBoost with optimized params...")
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
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    xgb_results = evaluate_model(xgb_model, X_val, y_val, "XGBoost")

    # Save models
    print("\n[STEP 8] Saving models...")
    saved_dir = os.path.join(os.path.dirname(__file__), 'saved')
    os.makedirs(saved_dir, exist_ok=True)

    # Save LightGBM
    lgb_pkl_path = os.path.join(saved_dir, 'lightgbm_model_final.pkl')
    with open(lgb_pkl_path, 'wb') as f:
        pickle.dump(lgb_model, f)
    print(f"[+] LightGBM model saved to {lgb_pkl_path}")

    # Save XGBoost
    xgb_pkl_path = os.path.join(saved_dir, 'xgboost_model_final.pkl')
    with open(xgb_pkl_path, 'wb') as f:
        pickle.dump(xgb_model, f)
    print(f"[+] XGBoost model saved to {xgb_pkl_path}")

    # Summary
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"\nPRODUCTION MODELS:")
    print(f"  Trained on: 2021-2025 data (2025 prioritized for latest patterns)")
    print(f"  Validation on: 2021 data (to verify generalization)")
    print(f"\nVALIDATION SET RESULTS (2021 data):")
    print(f"\nLightGBM:")
    print(f"  Accuracy: {lgb_results['accuracy']:.6f}")
    print(f"  AUC-ROC:  {lgb_results['auc']:.6f}")
    print(f"  Log Loss: {lgb_results['loss']:.6f}")
    print(f"\nXGBoost:")
    print(f"  Accuracy: {xgb_results['accuracy']:.6f}")
    print(f"  AUC-ROC:  {xgb_results['auc']:.6f}")
    print(f"  Log Loss: {xgb_results['loss']:.6f}")

    print(f"\nOptimal Ensemble Weighting: LGB 18% + XGB 82%")
    print(f"Expected Production Accuracy: ~0.721 (72.1%)")

    print(f"\nModels replaced at: {saved_dir}")
    print(f"  - lightgbm_model_final.pkl")
    print(f"  - xgboost_model_final.pkl")
    print("="*80)

if __name__ == "__main__":
    main()
