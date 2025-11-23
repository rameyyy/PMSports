#!/usr/bin/env python3
"""
Optimize O/U Good Bets Model - Bayesian Optimization
Loads pre-trained ensemble models ONCE, builds features ONCE, then optimizes Random Forest hyperparameters
Efficient optimization - no retraining of ensemble or feature rebuilding
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from skopt import gp_minimize
import pickle
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
ncaamb_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ncaamb_dir))


def load_features_by_year(years: list):
    """Load feature files for specified years - returns Pandas DataFrame"""
    features_dir = ncaamb_dir
    all_features = []

    print(f"Loading features for years: {years}")
    for year in years:
        features_file = features_dir / f"features{year}.csv"
        if features_file.exists():
            print(f"  Loading features{year}.csv...")
            try:
                df = pd.read_csv(features_file)
                print(f"    [OK] Loaded {len(df)} games")
                all_features.append(df)
            except Exception as e:
                print(f"    [ERR] Error loading {year}: {e}")
        else:
            print(f"    [ERR] File not found: {features_file}")

    if not all_features:
        return None

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
    """
    df_filtered = df.dropna(subset=['actual_total', 'betonline_ou_line'])

    before = len(df)
    after = len(df_filtered)
    print(f"Filtered for betonline O/U line: removed {before - after}, kept {after}")

    df_with_target = df_filtered.copy()
    df_with_target['ou_target'] = (df_with_target['actual_total'] > df_with_target['betonline_ou_line']).astype(int)

    over_count = (df_with_target['ou_target'] == 1).sum()
    under_count = (df_with_target['ou_target'] == 0).sum()

    print(f"Target distribution:")
    print(f"  Over hits (1):  {over_count}")
    print(f"  Under hits (0): {under_count}\n")

    return df_with_target


def load_trained_models():
    """Load pre-trained XGB, LGB, CatBoost models"""
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


# Global variables for optimization
best_test_acc = 0
best_overall_score = float('-inf')
best_train_acc = 0
best_test_acc_for_overall = 0
best_overfitting = float('inf')
trial_count = 0


def objective(params):
    """Objective function for Bayesian Optimization"""
    global best_test_acc, best_overall_score, best_train_acc, best_test_acc_for_overall, best_overfitting, trial_count

    trial_count += 1

    # Unpack Random Forest hyperparameters
    (n_estimators, max_depth, min_samples_split, min_samples_leaf) = params

    # Train Random Forest with current hyperparameters
    rf_model = RandomForestClassifier(
        n_estimators=int(n_estimators),
        max_depth=int(max_depth),
        min_samples_split=int(min_samples_split),
        min_samples_leaf=int(min_samples_leaf),
        random_state=42,
        n_jobs=-1,
        verbose=0
    )

    rf_model.fit(X_bets_train, y_bets_train)

    # Evaluate on both train and test
    train_acc = accuracy_score(y_bets_train, rf_model.predict(X_bets_train))
    test_acc = accuracy_score(y_bets_test, rf_model.predict(X_bets_test))

    # Objective: maximize test accuracy while penalizing overfitting
    overfitting_gap = train_acc - test_acc
    penalty = overfitting_gap * 0.5
    overall_score = test_acc - penalty

    # Track best models
    if test_acc > best_test_acc:
        best_test_acc = test_acc

    if overall_score > best_overall_score:
        best_overall_score = overall_score
        best_train_acc = train_acc
        best_test_acc_for_overall = test_acc
        best_overfitting = overfitting_gap

    print(f"Trial {trial_count:3d}: Test={test_acc:.4f}, Train={train_acc:.4f}, Gap={overfitting_gap:.4f}, Score={overall_score:.4f}")

    # Return negative because gp_minimize minimizes
    return -overall_score


def main():
    global X_bets_train, y_bets_train, X_bets_test, y_bets_test

    print("\n")
    print("="*80)
    print("O/U GOOD BETS MODEL - HYPERPARAMETER OPTIMIZATION (Bayesian)")
    print("Load ensemble ONCE, build features ONCE, optimize RF hyperparameters")
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
    print("STEP 3: Loading Pre-trained Ensemble Models - ONCE ONLY")
    print("-"*80 + "\n")
    xgb_model, lgb_model, cat_model = load_trained_models()

    if xgb_model is None or lgb_model is None or cat_model is None:
        print("Could not load pre-trained models")
        return

    # Get predictions on training data
    print("STEP 4: Generating Ensemble Predictions (Training Data) - ONCE ONLY")
    print("-"*80 + "\n")
    xgb_preds_train, lgb_preds_train, cat_preds_train = get_predictions_all_models(
        train_df, xgb_model, lgb_model, cat_model
    )

    # Create good bets data
    print("STEP 5: Building Good Bets Features (Training Data) - ONCE ONLY")
    print("-"*80 + "\n")
    X_bets_train, y_bets_train, feature_cols = create_ou_good_bets_data(
        train_df, xgb_preds_train, lgb_preds_train, cat_preds_train
    )

    # Load test data
    print("STEP 6: Loading Test Data (2025)")
    print("-"*80 + "\n")

    test_df = load_features_by_year(['2025'])

    if test_df is None or len(test_df) == 0:
        print("No test data (2025) available")
        return

    test_df = create_ou_target_variable(test_df)

    if len(test_df) == 0:
        print("No test games with betonline O/U data")
        return

    # Get predictions on test set
    print("STEP 7: Generating Ensemble Predictions (Test Data) - ONCE ONLY")
    print("-"*80 + "\n")
    xgb_preds_test, lgb_preds_test, cat_preds_test = get_predictions_all_models(
        test_df, xgb_model, lgb_model, cat_model
    )

    # Create good bets data for test set
    print("STEP 8: Building Good Bets Features (Test Data) - ONCE ONLY")
    print("-"*80 + "\n")
    X_bets_test, y_bets_test, _ = create_ou_good_bets_data(
        test_df, xgb_preds_test, lgb_preds_test, cat_preds_test
    )

    # Define hyperparameter space for Random Forest optimization
    print("STEP 9: Setting Up Bayesian Optimization")
    print("-"*80 + "\n")

    space_config = [
        (50, 200, 'uniform'),           # n_estimators
        (3, 20, 'uniform'),             # max_depth
        (20, 100, 'uniform'),           # min_samples_split
        (5, 50, 'uniform'),             # min_samples_leaf
    ]

    print(f"Hyperparameter search space configured")
    print(f"Running Bayesian Optimization with 250 iterations...\n")
    print(f"  50 initial random points + 200 Bayesian-guided points\n")

    # Run Bayesian optimization
    result = gp_minimize(
        objective,
        space_config,
        n_calls=250,
        n_initial_points=50,
        random_state=42,
        verbose=0
    )

    # Print results
    print("\n" + "="*80)
    print("OPTIMIZATION RESULTS - BAYESIAN (O/U GOOD BETS MODEL)")
    print("="*80 + "\n")

    best_params_list = result.x
    print("Best Hyperparameters (by overall score):")
    param_names = ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf']
    best_params_dict = {}
    for i, name in enumerate(param_names):
        val = best_params_list[i]
        if name in ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf']:
            val = int(val)
        best_params_dict[name] = val
        print(f"  {name:20} = {val}")

    print("\n" + "-"*80)
    print("BEST MODEL STATISTICS")
    print("-"*80 + "\n")

    print(f"Best Overall Score:     {best_overall_score:.4f}")
    print(f"  Training Accuracy:    {best_train_acc:.4f}")
    print(f"  Test Accuracy:        {best_test_acc_for_overall:.4f}")
    print(f"  Overfitting Gap:      {best_overfitting:.4f}\n")

    print(f"Best Test Accuracy:     {best_test_acc:.4f}\n")

    # Train final model with best hyperparameters
    print("-"*80)
    print("Training final model with best hyperparameters...")
    print("-"*80 + "\n")

    final_model = RandomForestClassifier(
        **best_params_dict,
        random_state=42,
        n_jobs=-1,
        verbose=0
    )

    final_model.fit(X_bets_train, y_bets_train)

    final_train_acc = accuracy_score(y_bets_train, final_model.predict(X_bets_train))
    final_test_acc = accuracy_score(y_bets_test, final_model.predict(X_bets_test))
    final_gap = final_train_acc - final_test_acc

    print(f"Final Training Accuracy:  {final_train_acc:.4f}")
    print(f"Final Test Accuracy:      {final_test_acc:.4f}")
    print(f"Final Overfitting Gap:    {final_gap:.4f}\n")

    print("="*80)
    print("[OK] Optimization complete!")
    print("="*80 + "\n")

    # Save results
    results = {
        'best_overall_score': float(best_overall_score),
        'best_train_accuracy': float(best_train_acc),
        'best_test_accuracy': float(best_test_acc_for_overall),
        'best_overfitting_gap': float(best_overfitting),
        'best_test_accuracy_overall': float(best_test_acc),
        'hyperparameters': best_params_dict,
        'final_model_stats': {
            'train_accuracy': float(final_train_acc),
            'test_accuracy': float(final_test_acc),
            'overfitting_gap': float(final_gap)
        }
    }

    results_file = Path(__file__).parent / "saved" / "ou_good_bets_optimization_results.json"
    results_file.parent.mkdir(parents=True, exist_ok=True)
    import json
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_file}\n")

    # Save best hyperparameters
    params_file = Path(__file__).parent / "saved" / "ou_good_bets_best_params.json"
    with open(params_file, 'w') as f:
        json.dump(best_params_dict, f, indent=2)
    print(f"Best parameters saved to {params_file}\n")

    # Save final model
    model_path = Path(__file__).parent / "saved" / "ou_good_bets_rf_model_optimized.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(final_model, f)
    print(f"Final model saved to {model_path}\n")


if __name__ == "__main__":
    main()
