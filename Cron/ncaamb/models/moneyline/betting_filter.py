#!/usr/bin/env python3
"""
Betting Filter Model - Random Forest
Trains fresh XGBoost + LightGBM on 2021-2024 (in-memory only)
Uses predictions to train Random Forest betting filter
No saved models are modified - avoids data leakage
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


def get_implied_probability(odds: float) -> float:
    """Convert American odds to implied probability"""
    if odds > 0:  # Underdog
        return 100 / (odds + 100)
    else:  # Favorite
        return abs(odds) / (abs(odds) + 100)


def create_betting_data(df: pl.DataFrame, X: pl.DataFrame, y: np.ndarray,
                       xgb_proba: np.ndarray, lgb_proba: np.ndarray,
                       feature_cols: list) -> tuple:
    """
    Create betting filter training data (2 rows per game - one per team perspective)

    Args:
        df: Original dataframe with game info and odds
        X: Feature matrix
        y: Target (1=team_1 wins, 0=team_2 wins)
        xgb_proba: XGBoost predictions (n_samples, 2) - [prob_team_2, prob_team_1]
        lgb_proba: LightGBM predictions (n_samples,) - prob_team_1
        feature_cols: List of feature column names

    Returns:
        Tuple of (X_betting, y_betting) with 2 rows per game
    """
    betting_features = []
    betting_targets = []

    n_games = len(df)

    for i in range(n_games):
        # Get predictions and game info
        xgb_prob_team_1 = xgb_proba[i, 1]  # Probability team_1 wins from XGBoost
        lgb_prob_team_1 = lgb_proba[i]     # Probability team_1 wins from LightGBM
        game_result = y[i]                  # Actual result (1=team_1 wins, 0=team_2 wins)

        # Extract odds (team_1 = home, team_2 = away in most cases)
        avg_ml_team_1 = df[i]['avg_ml_team_1'] if 'avg_ml_team_1' in df.columns else 0
        avg_ml_team_2 = df[i]['avg_ml_team_2'] if 'avg_ml_team_2' in df.columns else 0
        month = df[i]['month'] if 'month' in df.columns else 11

        # Safe float conversion
        try:
            avg_ml_team_1 = float(avg_ml_team_1) if avg_ml_team_1 is not None else 0
            avg_ml_team_2 = float(avg_ml_team_2) if avg_ml_team_2 is not None else 0
        except:
            avg_ml_team_1 = 0
            avg_ml_team_2 = 0

        implied_prob_team_1 = get_implied_probability(avg_ml_team_1) if avg_ml_team_1 != 0 else 0.5
        implied_prob_team_2 = get_implied_probability(avg_ml_team_2) if avg_ml_team_2 != 0 else 0.5

        # Calculate EV (expected value)
        ev_team_1 = (xgb_prob_team_1 + lgb_prob_team_1) / 2 * (1 + abs(avg_ml_team_1) / 100) - 1 if avg_ml_team_1 != 0 else 0
        ev_team_2 = (2 - (xgb_prob_team_1 + lgb_prob_team_1) / 2) * (1 + abs(avg_ml_team_2) / 100) - 1 if avg_ml_team_2 != 0 else 0

        # Extract team strength features
        team_1_strength = df[i]['team_1_adjoe'] if 'team_1_adjoe' in df.columns else 0
        team_1_defense = df[i]['team_1_adjde'] if 'team_1_adjde' in df.columns else 0
        team_2_strength = df[i]['team_2_adjoe'] if 'team_2_adjoe' in df.columns else 0
        team_2_defense = df[i]['team_2_adjde'] if 'team_2_adjde' in df.columns else 0

        try:
            team_1_strength = float(team_1_strength) if team_1_strength is not None else 0
            team_1_defense = float(team_1_defense) if team_1_defense is not None else 0
            team_2_strength = float(team_2_strength) if team_2_strength is not None else 0
            team_2_defense = float(team_2_defense) if team_2_defense is not None else 0
        except:
            team_1_strength = team_1_defense = team_2_strength = team_2_defense = 0

        # ROW 1: Team 1 perspective
        # Is betting Team 1 a good bet?
        team_1_betting_features = [
            xgb_prob_team_1,                          # XGBoost prob
            lgb_prob_team_1,                          # LightGBM prob
            abs(xgb_prob_team_1 - lgb_prob_team_1),  # Model disagreement (confidence inverse)
            implied_prob_team_1,                      # What odds say
            ev_team_1,                                # EV if bet Team 1
            avg_ml_team_1,                            # Odds for Team 1
            month,                                    # Month (seasonality)
            team_1_strength,                          # Team 1 offensive strength
            team_1_defense,                           # Team 1 defensive strength
            abs(team_1_strength - team_2_strength),  # Strength differential
        ]

        # Target: 1 if betting Team 1 would have won (game_result == 1)
        team_1_betting_target = 1 if game_result == 1 else 0

        betting_features.append(team_1_betting_features)
        betting_targets.append(team_1_betting_target)

        # ROW 2: Team 2 perspective
        # Is betting Team 2 a good bet?
        team_2_betting_features = [
            1 - xgb_prob_team_1,                      # XGBoost prob for Team 2
            1 - lgb_prob_team_1,                      # LightGBM prob for Team 2
            abs(xgb_prob_team_1 - lgb_prob_team_1),  # Model disagreement
            implied_prob_team_2,                      # What odds say
            ev_team_2,                                # EV if bet Team 2
            avg_ml_team_2,                            # Odds for Team 2
            month,                                    # Month
            team_2_strength,                          # Team 2 offensive strength
            team_2_defense,                           # Team 2 defensive strength
            abs(team_2_strength - team_1_strength),  # Strength differential
        ]

        # Target: 1 if betting Team 2 would have won (game_result == 0)
        team_2_betting_target = 1 if game_result == 0 else 0

        betting_features.append(team_2_betting_features)
        betting_targets.append(team_2_betting_target)

    X_betting = np.array(betting_features)
    y_betting = np.array(betting_targets)

    print(f"Created {len(X_betting)} betting rows ({n_games} games × 2 perspectives)")
    print(f"  Good bets (1): {np.sum(y_betting)}")
    print(f"  Bad bets (0):  {len(y_betting) - np.sum(y_betting)}\n")

    return X_betting, y_betting


def main():
    print("\n")
    print("="*80)
    print("BETTING FILTER MODEL - RANDOM FOREST")
    print("Training fresh base models on 2021-2024 (no saved models modified)")
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

    # Train fresh XGBoost and LightGBM models (in-memory only, not saved)
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
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
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

    # Create betting filter training data
    print("STEP 5: Creating Betting Filter Training Data")
    print("-"*80 + "\n")

    X_betting_train, y_betting_train = create_betting_data(
        train_df, X_train, y_train_np, xgb_proba_train, lgb_proba_train, feature_cols
    )

    # Train Random Forest
    print("STEP 6: Training Random Forest Betting Filter")
    print("-"*80 + "\n")

    print("Training Random Forest on betting data...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=50,
        min_samples_leaf=20,
        random_state=42,
        n_jobs=-1,
        verbose=0
    )

    rf_model.fit(X_betting_train, y_betting_train)
    print("[OK] Random Forest training complete\n")

    # Evaluate on training data
    print("STEP 7: Evaluating Betting Filter")
    print("-"*80 + "\n")

    y_pred_train = rf_model.predict(X_betting_train)
    y_proba_train = rf_model.predict_proba(X_betting_train)[:, 1]

    accuracy = accuracy_score(y_betting_train, y_pred_train)
    precision = precision_score(y_betting_train, y_pred_train, zero_division=0)
    recall = recall_score(y_betting_train, y_pred_train, zero_division=0)
    f1 = f1_score(y_betting_train, y_pred_train, zero_division=0)
    auc = roc_auc_score(y_betting_train, y_proba_train)
    tn, fp, fn, tp = confusion_matrix(y_betting_train, y_pred_train).ravel()

    print(f"{'='*80}")
    print(f"BETTING FILTER EVALUATION - Training Data (2021-2024)")
    print(f"{'='*80}\n")

    print(f"Accuracy:  {accuracy:.3f}")
    print(f"Precision: {precision:.3f} (when we say 'good bet', how often correct?)")
    print(f"Recall:    {recall:.3f} (of all good bets, how many do we catch?)")
    print(f"F1 Score:  {f1:.3f}")
    print(f"AUC:       {auc:.3f}\n")

    print(f"Confusion Matrix:")
    print(f"  True Negatives:  {tn:5d} (Bad bets correctly identified as bad)")
    print(f"  False Positives: {fp:5d} (Bad bets incorrectly marked as good)")
    print(f"  False Negatives: {fn:5d} (Good bets missed)")
    print(f"  True Positives:  {tp:5d} (Good bets correctly identified)\n")

    # Feature importance
    print("Feature Importance for Betting Decisions:")
    feature_names = [
        'xgb_prob', 'lgb_prob', 'model_disagreement', 'implied_prob',
        'ev', 'moneyline_odds', 'month', 'team_strength', 'team_defense',
        'strength_differential'
    ]

    importances = rf_model.feature_importances_
    feature_importance = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)

    for i, (feat, importance) in enumerate(feature_importance, 1):
        print(f"  {i:2}. {feat:25} {importance:.4f}")

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

    # Create betting data for test set
    X_betting_test, y_betting_test = create_betting_data(
        test_df, X_test, y_test_np, xgb_proba_test, lgb_proba_test, feature_cols
    )

    # Evaluate on test set
    y_pred_test = rf_model.predict(X_betting_test)
    y_proba_test = rf_model.predict_proba(X_betting_test)[:, 1]

    accuracy_test = accuracy_score(y_betting_test, y_pred_test)
    precision_test = precision_score(y_betting_test, y_pred_test, zero_division=0)
    recall_test = recall_score(y_betting_test, y_pred_test, zero_division=0)
    f1_test = f1_score(y_betting_test, y_pred_test, zero_division=0)
    auc_test = roc_auc_score(y_betting_test, y_proba_test)
    tn_t, fp_t, fn_t, tp_t = confusion_matrix(y_betting_test, y_pred_test).ravel()

    print(f"{'='*80}")
    print(f"BETTING FILTER EVALUATION - Test Data (2025)")
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
    model_save_path = Path(__file__).parent / "betting_filter_rf_model.pkl"
    with open(model_save_path, 'wb') as f:
        pickle.dump(rf_model, f)
    print(f"[OK] Betting filter model saved to {model_save_path}\n")

    print("="*80)
    print("[OK] Betting filter training and evaluation complete!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
