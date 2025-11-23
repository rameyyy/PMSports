#!/usr/bin/env python3
"""
Good Bets Model for Over/Under - Random Forest
Trains on 2021-2024 predictions, tests on 2025
Uses ensemble predictions (XGB, LGB, CatBoost) + odds

Workflow:
1. Load features for 2021-2024 (training) and 2025 (test)
2. Load pre-trained XGB, LGB, CatBoost models and get predictions
3. Create good bets features (model predictions + odds)
4. Train Random Forest to classify "good bets"
5. Evaluate on 2025 test data

Key Logic:
  - Point predictions converted to confidence via sigmoid
  - 1 row per game (not 2 like moneyline)
  - Target: 1 if actual_total > betonline_ou_line (over hit), 0 if under
"""

import os
import sys
from pathlib import Path
import polars as pl
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
ncaamb_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ncaamb_dir))

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score


def load_features_by_year(years: list):
    """Load feature files for specified years - returns Pandas DataFrame"""
    import pandas as pd

    features_dir = ncaamb_dir
    all_features = []

    print(f"Loading features for years: {years}")
    for year in years:
        features_file = features_dir / f"features{year}.csv"
        if features_file.exists():
            print(f"  Loading features{year}.csv...")
            try:
                # Load directly with pandas to avoid Polars schema issues
                df = pd.read_csv(features_file)
                print(f"    [OK] Loaded {len(df)} games")
                all_features.append(df)
            except Exception as e:
                print(f"    [ERR] Error loading {year}: {e}")
        else:
            print(f"    [ERR] File not found: {features_file}")

    if not all_features:
        return None

    # Concatenate all years
    combined_df = pd.concat(all_features, ignore_index=True)

    # Convert numeric-looking columns to float
    for col in combined_df.columns:
        if any(x in col.lower() for x in ['odds', 'line', 'spread', 'total', 'score', 'pace', 'efg', 'adj', 'avg']):
            combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')

    print(f"[OK] Combined: {len(combined_df)} total games\n")
    return combined_df


def create_ou_target_variable(df):
    """
    Create binary target variable for over/under betting
    Target = 1 if actual_total > betonline_ou_line (over hit)
    Target = 0 if actual_total < betonline_ou_line (under hit)
    Works with Pandas DataFrames
    """
    import pandas as pd

    # Filter: must have actual total and betonline line
    df_filtered = df.dropna(subset=['actual_total', 'betonline_ou_line'])

    before = len(df)
    after = len(df_filtered)
    print(f"Filtered for betonline O/U line: removed {before - after}, kept {after}")

    # Create target: 1 if over, 0 if under
    df_with_target = df_filtered.copy()
    df_with_target['ou_target'] = (df_with_target['actual_total'] > df_with_target['betonline_ou_line']).astype(int)

    over_count = (df_with_target['ou_target'] == 1).sum()
    under_count = (df_with_target['ou_target'] == 0).sum()

    print(f"Target distribution:")
    print(f"  Over hits (1):  {over_count}")
    print(f"  Under hits (0): {under_count}\n")

    return df_with_target


def identify_feature_columns(df: pl.DataFrame) -> list:
    """Identify numeric feature columns for O/U model"""
    metadata_cols = {
        'game_id', 'date', 'season', 'team_1', 'team_2',
        'team_1_score', 'team_2_score', 'actual_total',
        'team_1_conference', 'team_2_conference',
        'team_1_is_home', 'team_2_is_home', 'location',
        'start_time', 'game_odds', 'ou_target',
        'betonline_ou_line', 'ou_line_variance'
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
    y = df.select('actual_total')
    return X, y


def load_trained_models():
    """Load pre-trained XGB, LGB, CatBoost models"""
    import xgboost as xgb
    import lightgbm as lgb
    from catboost import CatBoostRegressor

    saved_dir = Path(__file__).parent / "saved"

    try:
        # Load XGBoost using native loader
        xgb_model = xgb.XGBRegressor()
        xgb_model.load_model(str(saved_dir / "xgboost_model.pkl"))

        # Load LightGBM using Booster
        lgb_model = lgb.Booster(model_file=str(saved_dir / "lightgbm_model.pkl"))

        # Load CatBoost using native loader (CBM format)
        cat_model = CatBoostRegressor()
        cat_model.load_model(str(saved_dir / "catboost_model.pkl"), format='cbm')

        print("[OK] Loaded pre-trained ensemble models\n")
        return xgb_model, lgb_model, cat_model
    except Exception as e:
        print(f"[ERR] Could not load pre-trained models: {e}\n")
        import traceback
        traceback.print_exc()
        return None, None, None


def get_predictions_all_models(df, xgb_model, lgb_model, cat_model) -> tuple:
    """Get predictions from all 3 models - works with Pandas"""
    import pandas as pd

    # Get numeric columns that exist in df
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    # Exclude target, metadata, and any variables we added - match ou_model.py exactly
    exclude = {'actual_total', 'team_1_score', 'team_2_score', 'game_id', 'date', 'team_1', 'team_2', 'ou_target'}
    feature_cols = [c for c in numeric_cols if c not in exclude]

    X = df[feature_cols].fillna(0).values

    xgb_preds = xgb_model.predict(X)
    lgb_preds = lgb_model.predict(X)
    cat_preds = cat_model.predict(X)

    print(f"[OK] Generated predictions from all 3 models\n")
    return xgb_preds, lgb_preds, cat_preds


def create_ou_good_bets_data(df, xgb_preds: np.ndarray,
                             lgb_preds: np.ndarray, cat_preds: np.ndarray) -> tuple:
    """
    Create good bets training data for O/U (1 row per game) using Pandas

    Features:
      - xgb_confidence_over, lgb_confidence_over, cat_confidence_over
      - ensemble_confidence_over (weighted: 0.441 XGB, 0.466 LGB, 0.093 Cat)
      - model_std_dev (disagreement between models)
      - betonline_ou_line, avg_ou_line, ou_line_variance
      - avg_over_odds, avg_under_odds, betonline_over_odds, betonline_under_odds
    """
    import numpy as np

    # Filter to rows with ou_target
    df_filtered = df.dropna(subset=['ou_target']).copy()

    # Add predictions
    df_filtered['xgb_point_pred'] = xgb_preds
    df_filtered['lgb_point_pred'] = lgb_preds
    df_filtered['cat_point_pred'] = cat_preds

    # Convert point predictions to confidence using sigmoid
    # P(over) = 1 / (1 + exp(-(pred - line) / 3))
    df_filtered['xgb_confidence_over'] = 1.0 / (1.0 + np.exp(-(df_filtered['xgb_point_pred'] - df_filtered['betonline_ou_line']) / 3.0))
    df_filtered['lgb_confidence_over'] = 1.0 / (1.0 + np.exp(-(df_filtered['lgb_point_pred'] - df_filtered['betonline_ou_line']) / 3.0))
    df_filtered['cat_confidence_over'] = 1.0 / (1.0 + np.exp(-(df_filtered['cat_point_pred'] - df_filtered['betonline_ou_line']) / 3.0))

    # Clip to [0, 1]
    for col in ['xgb_confidence_over', 'lgb_confidence_over', 'cat_confidence_over']:
        df_filtered[col] = df_filtered[col].clip(0.0, 1.0)

    # Ensemble confidence (weighted average)
    df_filtered['ensemble_confidence_over'] = (
        0.441 * df_filtered['xgb_confidence_over'] +
        0.466 * df_filtered['lgb_confidence_over'] +
        0.093 * df_filtered['cat_confidence_over']
    )

    # Model disagreement: std dev of 3 confidences
    confs = np.array([
        df_filtered['xgb_confidence_over'].values,
        df_filtered['lgb_confidence_over'].values,
        df_filtered['cat_confidence_over'].values
    ])
    df_filtered['model_std_dev'] = np.std(confs, axis=0)

    # Fill missing odds data
    for col in ['betonline_ou_line', 'avg_ou_line', 'ou_line_variance',
                'avg_over_odds', 'avg_under_odds',
                'betonline_over_odds', 'betonline_under_odds']:
        if col not in df_filtered.columns:
            df_filtered[col] = 0.0
        else:
            df_filtered[col] = df_filtered[col].fillna(0.0)

    # Select feature columns
    feature_cols = [
        'xgb_confidence_over', 'lgb_confidence_over', 'cat_confidence_over',
        'ensemble_confidence_over', 'model_std_dev',
        'betonline_ou_line', 'avg_ou_line', 'ou_line_variance',
        'avg_over_odds', 'avg_under_odds',
        'betonline_over_odds', 'betonline_under_odds',
    ]

    selected_cols = [col for col in feature_cols if col in df_filtered.columns]
    X_bets = df_filtered[selected_cols].values
    y_bets = df_filtered['ou_target'].values

    print(f"Created {len(X_bets)} betting rows ({len(df)} games)")
    print(f"  Over hits (1): {int(np.sum(y_bets))}")
    print(f"  Under hits (0): {int(len(y_bets) - np.sum(y_bets))}\n")

    return X_bets, y_bets, selected_cols


def main():
    print("\n")
    print("="*80)
    print("OVER/UNDER GOOD BETS MODEL - RANDOM FOREST")
    print("Using pre-trained XGB/LGB/CatBoost ensemble models")
    print("Training on 2021-2024, Testing on 2025")
    print("="*80 + "\n")

    # Load training data (2021-2024)
    print("STEP 1: Loading Training Data (2021-2024)")
    print("-"*80 + "\n")
    train_df = load_features_by_year(['2021', '2022', '2023', '2024'])

    if train_df is None or len(train_df) == 0:
        print("Failed to load training features")
        return

    # Create target variable
    print("STEP 2: Creating Target Variable")
    print("-"*80 + "\n")
    train_df = create_ou_target_variable(train_df)

    if len(train_df) == 0:
        print("No training games with betonline O/U data")
        return

    # Load pre-trained models
    print("STEP 3: Loading Pre-trained Ensemble Models")
    print("-"*80 + "\n")
    xgb_model, lgb_model, cat_model = load_trained_models()

    if xgb_model is None or lgb_model is None or cat_model is None:
        print("Could not load pre-trained models")
        return

    # Get predictions on training data
    print("STEP 4: Getting Predictions from Ensemble (Training Data)")
    print("-"*80 + "\n")
    xgb_preds_train, lgb_preds_train, cat_preds_train = get_predictions_all_models(
        train_df, xgb_model, lgb_model, cat_model
    )

    # Create good bets data
    print("STEP 6: Creating Good Bets Training Data")
    print("-"*80 + "\n")
    X_bets_train, y_bets_train, gb_feature_cols = create_ou_good_bets_data(
        train_df, xgb_preds_train, lgb_preds_train, cat_preds_train
    )

    # Train Random Forest
    print("STEP 7: Training Random Forest Good Bets Model")
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
    print("STEP 8: Evaluating Good Bets Model on Training Data")
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
    print(f"O/U GOOD BETS MODEL EVALUATION - Training Data (2021-2024)")
    print(f"{'='*80}\n")

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f} (when we say 'good bet', how often correct?)")
    print(f"Recall:    {recall:.4f} (of all good bets, how many do we catch?)")
    print(f"F1 Score:  {f1:.4f}")
    print(f"AUC:       {auc:.4f}\n")

    print(f"Confusion Matrix:")
    print(f"  True Negatives:  {tn:5d} (Bad bets correctly identified)")
    print(f"  False Positives: {fp:5d} (Bad bets marked as good)")
    print(f"  False Negatives: {fn:5d} (Good bets missed)")
    print(f"  True Positives:  {tp:5d} (Good bets correctly identified)\n")

    # Feature importance
    print("Feature Importance for Good O/U Bet Decisions:")
    importances = rf_model.feature_importances_
    feature_importance = sorted(zip(gb_feature_cols, importances), key=lambda x: x[1], reverse=True)

    for i, (feat, importance) in enumerate(feature_importance, 1):
        print(f"  {i:2}. {feat:30} {importance:.4f}")

    print()

    # Load test data
    print("STEP 9: Testing on 2025 Data")
    print("-"*80 + "\n")

    test_df = load_features_by_year(['2025'])

    if test_df is None or len(test_df) == 0:
        print("No test data (2025) available")
        print("\n" + "="*80)
        print("[OK] Training complete!")
        print("="*80 + "\n")

        # Save model anyway
        model_save_path = Path(__file__).parent / "saved" / "ou_good_bets_rf_model.pkl"
        model_save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(model_save_path, 'wb') as f:
            pickle.dump(rf_model, f)
        print(f"[OK] O/U good bets model saved to {model_save_path}\n")
        return

    test_df = create_ou_target_variable(test_df)

    if len(test_df) == 0:
        print("No test games with betonline O/U data")
        return

    # Get predictions on test set
    print("STEP 10: Getting Predictions from Ensemble (Test Data)")
    print("-"*80 + "\n")
    xgb_preds_test, lgb_preds_test, cat_preds_test = get_predictions_all_models(
        test_df, xgb_model, lgb_model, cat_model
    )

    # Create good bets data for test set
    print("STEP 11: Creating Good Bets Test Data")
    print("-"*80 + "\n")
    X_bets_test, y_bets_test, _ = create_ou_good_bets_data(
        test_df, xgb_preds_test, lgb_preds_test, cat_preds_test
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
    print(f"O/U GOOD BETS MODEL EVALUATION - Test Data (2025)")
    print(f"{'='*80}\n")

    print(f"Accuracy:  {accuracy_test:.4f}")
    print(f"Precision: {precision_test:.4f}")
    print(f"Recall:    {recall_test:.4f}")
    print(f"F1 Score:  {f1_test:.4f}")
    print(f"AUC:       {auc_test:.4f}\n")

    print(f"Confusion Matrix:")
    print(f"  True Negatives:  {tn_t:5d}")
    print(f"  False Positives: {fp_t:5d}")
    print(f"  False Negatives: {fn_t:5d}")
    print(f"  True Positives:  {tp_t:5d}\n")

    # Summary
    print("\n" + "="*80)
    print("SUMMARY - TRAIN vs TEST")
    print("="*80 + "\n")

    print(f"Train Accuracy:  {accuracy:.4f}")
    print(f"Test Accuracy:   {accuracy_test:.4f}")
    print(f"Difference:      {accuracy - accuracy_test:.4f}\n")

    print(f"Train AUC:       {auc:.4f}")
    print(f"Test AUC:        {auc_test:.4f}\n")

    print(f"Training samples:  {len(X_bets_train)}")
    print(f"Test samples:      {len(X_bets_test)}\n")

    # Save model
    model_save_path = Path(__file__).parent / "saved" / "ou_good_bets_rf_model.pkl"
    model_save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_save_path, 'wb') as f:
        pickle.dump(rf_model, f)
    print(f"[OK] O/U good bets model saved to {model_save_path}\n")

    print("="*80)
    print("[OK] O/U good bets model training and evaluation complete!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
