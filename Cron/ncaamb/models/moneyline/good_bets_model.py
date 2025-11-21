#!/usr/bin/env python3
"""
Good Bets Model - Random Forest
Trains on 2021-2024 predictions, tests on 2025
Predicts if a specific bet (team perspective) will win or lose
Uses XGBoost + LightGBM predictions + odds + spread data
"""

import os
import sys
from pathlib import Path
import polars as pl
import numpy as np
import lightgbm as lgb
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
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
    """
    Create good bets training data using vectorized Polars operations
    Creates 2 rows per game (one per team perspective)

    Returns:
        Tuple of (X_bets, y_bets) with 2 rows per game
    """
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

    # Calculate implied probabilities using vectorized ops
    def implied_prob_udf(odds):
        return pl.when(odds == 0).then(0.5)\
            .when(odds > 0).then(100 / (odds + 100))\
            .otherwise(pl.col(odds).abs() / (pl.col(odds).abs() + 100))

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

    # Calculate EV using vectorized ops
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

    print(f"Created {len(X_bets)} betting rows ({len(df)} games × 2 perspectives)")
    print(f"  Good bets (1): {np.sum(y_bets)}")
    print(f"  Bad bets (0):  {len(y_bets) - np.sum(y_bets)}\n")

    return X_bets, y_bets


def main():
    print("\n")
    print("="*80)
    print("GOOD BETS MODEL - RANDOM FOREST")
    print("Training on 2021-2024, Testing on 2025")
    print("="*80 + "\n")

    # Load training data (2021-2024)
    print("STEP 1: Loading Training Data (2021-2024)")
    print("-"*80 + "\n")
    train_df = load_features_by_year(['2021', '2022', '2023', '2024'])

    if train_df is None or len(train_df) == 0:
        print("Failed to load training features")
        return

    train_df = filter_low_quality_games(train_df, min_data_quality=0.5)
    train_df = create_target_variable(train_df)
    feature_cols = identify_feature_columns(train_df)

    X_train, y_train = prepare_training_data(train_df, feature_cols)
    X_train_np = X_train.to_numpy()
    y_train_np = y_train.to_numpy().ravel()

    print(f"Training data shape: X={X_train.shape}, y={y_train.shape}\n")

    # Train fresh XGBoost and LightGBM models (in-memory only)
    print("STEP 2: Training Fresh XGBoost Model (2021-2024 only)")
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

    print("STEP 3: Training Fresh LightGBM Model (2021-2024 only)")
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

    # Get predictions from both models
    print("STEP 4: Generating Predictions from Base Models")
    print("-"*80 + "\n")

    xgb_proba_train = xgb_model.predict_proba(X_train_np)
    lgb_proba_train = lgb_model.predict(X_train_np)

    print(f"[OK] XGBoost predictions shape: {xgb_proba_train.shape}")
    print(f"[OK] LightGBM predictions shape: {lgb_proba_train.shape}\n")

    # Create good bets training data
    print("STEP 5: Creating Good Bets Training Data")
    print("-"*80 + "\n")

    X_bets_train, y_bets_train = create_good_bets_data(
        train_df, y_train_np, xgb_proba_train, lgb_proba_train
    )

    # Train Random Forest
    print("STEP 6: Training Random Forest Good Bets Model")
    print("-"*80 + "\n")

    print("Training Random Forest on good bets data...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=50,
        min_samples_leaf=20,
        random_state=42,
        n_jobs=-1,
        verbose=0
    )

    rf_model.fit(X_bets_train, y_bets_train)
    print("[OK] Random Forest training complete\n")

    # Evaluate on training data
    print("STEP 7: Evaluating Good Bets Model")
    print("-"*80 + "\n")

    y_pred_train = rf_model.predict(X_bets_train)
    y_proba_train = rf_model.predict_proba(X_bets_train)[:, 1]

    accuracy = accuracy_score(y_bets_train, y_pred_train)
    precision = precision_score(y_bets_train, y_pred_train, zero_division=0)
    recall = recall_score(y_bets_train, y_pred_train, zero_division=0)
    f1 = f1_score(y_bets_train, y_pred_train, zero_division=0)
    auc = roc_auc_score(y_bets_train, y_proba_train)
    tn, fp, fn, tp = confusion_matrix(y_bets_train, y_pred_train).ravel()

    print(f"{'='*80}")
    print(f"GOOD BETS MODEL EVALUATION - Training Data (2021-2024)")
    print(f"{'='*80}\n")

    print(f"Accuracy:  {accuracy:.3f}")
    print(f"Precision: {precision:.3f} (when we say 'good bet', how often correct?)")
    print(f"Recall:    {recall:.3f} (of all good bets, how many do we catch?)")
    print(f"F1 Score:  {f1:.3f}")
    print(f"AUC:       {auc:.3f}\n")

    print(f"Confusion Matrix:")
    print(f"  True Negatives:  {tn:5d} (Bad bets correctly identified)")
    print(f"  False Positives: {fp:5d} (Bad bets marked as good)")
    print(f"  False Negatives: {fn:5d} (Good bets missed)")
    print(f"  True Positives:  {tp:5d} (Good bets correctly identified)\n")

    # Feature importance
    print("Feature Importance for Good Bet Decisions:")
    feature_names = [
        'xgb_prob', 'lgb_prob', 'model_disagreement',
        'moneyline_odds', 'implied_prob_moneyline', 'ev_moneyline',
        'spread_pts_team_1', 'spread_pts_team_2',
        'spread_odds_team_1', 'spread_odds_team_2',
        'month', 'strength_differential'
    ]

    importances = rf_model.feature_importances_
    feature_importance = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)

    for i, (feat, importance) in enumerate(feature_importance, 1):
        print(f"  {i:2}. {feat:30} {importance:.4f}")

    print()

    # Load test data
    print("STEP 8: Testing on 2025 Data")
    print("-"*80 + "\n")

    test_df = load_features_by_year(['2025'])

    if test_df is None or len(test_df) == 0:
        print("No test data (2025) available")
        print("\n" + "="*80)
        print("[OK] Training complete!")
        print("="*80 + "\n")
        return

    test_df = filter_low_quality_games(test_df, min_data_quality=0.5)
    test_df = create_target_variable(test_df)

    if len(test_df) == 0:
        print("No test games with results available")
        return

    X_test, y_test = prepare_training_data(test_df, feature_cols)
    X_test_np = X_test.to_numpy()
    y_test_np = y_test.to_numpy().ravel()

    # Get predictions from base models on test set
    xgb_proba_test = xgb_model.predict_proba(X_test_np)
    lgb_proba_test = lgb_model.predict(X_test_np)

    # Create good bets data for test set
    X_bets_test, y_bets_test = create_good_bets_data(
        test_df, y_test_np, xgb_proba_test, lgb_proba_test
    )

    # Evaluate on test set
    y_pred_test = rf_model.predict(X_bets_test)
    y_proba_test = rf_model.predict_proba(X_bets_test)[:, 1]

    accuracy_test = accuracy_score(y_bets_test, y_pred_test)
    precision_test = precision_score(y_bets_test, y_pred_test, zero_division=0)
    recall_test = recall_score(y_bets_test, y_pred_test, zero_division=0)
    f1_test = f1_score(y_bets_test, y_pred_test, zero_division=0)
    auc_test = roc_auc_score(y_bets_test, y_proba_test)
    tn_t, fp_t, fn_t, tp_t = confusion_matrix(y_bets_test, y_pred_test).ravel()

    print(f"{'='*80}")
    print(f"GOOD BETS MODEL EVALUATION - Test Data (2025)")
    print(f"{'='*80}\n")

    print(f"Accuracy:  {accuracy_test:.3f}")
    print(f"Precision: {precision_test:.3f}")
    print(f"Recall:    {recall_test:.3f}")
    print(f"F1 Score:  {f1_test:.3f}")
    print(f"AUC:       {auc_test:.3f}\n")

    print(f"Confusion Matrix:")
    print(f"  True Negatives:  {tn_t:5d}")
    print(f"  False Positives: {fp_t:5d}")
    print(f"  False Negatives: {fn_t:5d}")
    print(f"  True Positives:  {tp_t:5d}\n")

    # Summary
    print("\n" + "="*80)
    print("SUMMARY - TRAIN vs TEST")
    print("="*80 + "\n")

    print(f"Train Accuracy:  {accuracy:.3f}")
    print(f"Test Accuracy:   {accuracy_test:.3f}")
    print(f"Difference:      {accuracy - accuracy_test:.3f}\n")

    print(f"Train AUC:       {auc:.3f}")
    print(f"Test AUC:        {auc_test:.3f}\n")

    # Save model
    model_save_path = Path(__file__).parent / "saved" / "good_bets_rf_model.pkl"
    model_save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_save_path, 'wb') as f:
        pickle.dump(rf_model, f)
    print(f"[OK] Good bets model saved to {model_save_path}\n")

    print("="*80)
    print("[OK] Good bets model training and evaluation complete!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
