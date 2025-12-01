#!/usr/bin/env python3
"""
Bayesian Optimization for XGBoost moneyline model with new decimal odds features
Uses scikit-optimize to find best hyperparameters
Train on 2021-2024, evaluate on 2025
"""

import polars as pl
import xgboost as xgb
import numpy as np
import json
import os
from datetime import datetime
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
import warnings
warnings.filterwarnings('ignore')

# Global counters
iteration = 0
best_score = -np.inf
best_params = None

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
        df = pl.read_csv(file, infer_schema_length=1000)

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
    """Create binary target: 1 if team_1 wins, 0 if team_2 wins"""
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
    """Preprocess features"""
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
    """Filter low-quality games"""
    df = df.filter(
        (pl.col('team_1_data_quality') >= 0.5) &
        (pl.col('team_2_data_quality') >= 0.5)
    )
    df = df.filter(pl.col('ml_target').is_not_null())
    df = df.filter(
        (pl.col('avg_ml_team_1').is_not_null()) &
        (pl.col('avg_ml_team_2').is_not_null())
    )
    return df

def get_feature_columns(df):
    """Get numeric feature columns"""
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
    """Prepare X and y"""
    df = df.with_columns([
        pl.col(col).fill_null(0) for col in feature_cols
    ])
    X = df.select(feature_cols).to_numpy()
    y = df.select('ml_target').to_numpy().ravel()
    return X, y

# ============================================================================
# MAIN OPTIMIZATION
# ============================================================================

def main():
    print("="*80)
    print("BAYESIAN OPTIMIZATION FOR XGBOOST MONEYLINE MODEL")
    print("="*80)

    # Load and prepare data
    print("\n[STEP 1] Loading data...")
    train_df = load_features(['2021', '2022', '2023', '2024'])
    test_df = load_features(['2025'])

    print("[STEP 2] Creating target variables...")
    train_df = create_target_variable(train_df)
    test_df = create_target_variable(test_df)

    print("[STEP 3] Preprocessing...")
    train_df = preprocess_features(train_df)
    test_df = preprocess_features(test_df)

    print("[STEP 4] Filtering quality games...")
    train_df = filter_quality_games(train_df)
    test_df = filter_quality_games(test_df)

    print("[STEP 5] Getting feature columns...")
    feature_cols = get_feature_columns(train_df)

    print("[STEP 6] Preparing data...")
    X_train, y_train = prepare_training_data(train_df, feature_cols)
    X_test, y_test = prepare_training_data(test_df, feature_cols)

    print(f"[+] Train: {X_train.shape[0]} games, {X_train.shape[1]} features")
    print(f"[+] Test: {X_test.shape[0]} games, {X_test.shape[1]} features")

    # Define search space
    print("\n[STEP 7] Defining search space...")
    space = [
        Integer(3, 10, name='max_depth'),
        Real(0.001, 0.3, prior='log-uniform', name='learning_rate'),
        Real(0.3, 1.0, name='subsample'),
        Real(0.3, 1.0, name='colsample_bytree'),
        Integer(1, 20, name='min_child_weight'),
        Real(0, 10, name='gamma'),
        Real(0, 10, name='lambda_l1'),
        Real(0, 10, name='lambda_l2'),
        Integer(50, 500, name='n_estimators'),
    ]

    # Define objective function
    @use_named_args(space)
    def objective(max_depth, learning_rate, subsample, colsample_bytree,
                  min_child_weight, gamma, lambda_l1, lambda_l2, n_estimators):
        global iteration, best_score, best_params

        iteration += 1

        params = {
            'max_depth': int(max_depth),
            'learning_rate': learning_rate,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'min_child_weight': int(min_child_weight),
            'gamma': gamma,
            'lambda': lambda_l1 + lambda_l2,
            'n_estimators': int(n_estimators),
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'random_state': 42,
            'verbosity': 0,
        }

        try:
            # Train model
            model = xgb.XGBClassifier(**params)
            model.fit(X_train, y_train, verbose=False)

            # Evaluate
            y_test_pred = model.predict_proba(X_test)[:, 1]
            test_auc = roc_auc_score(y_test, y_test_pred)
            test_loss = log_loss(y_test, y_test_pred)

            # Objective: maximize AUC, but penalize high log loss
            score = test_auc - (test_loss * 0.1)

            if score > best_score:
                best_score = score
                best_params = params.copy()
                print(f"\n[Iteration {iteration}] NEW BEST! AUC: {test_auc:.4f}, Loss: {test_loss:.4f}, Score: {score:.4f}")
                print(f"  Params: depth={max_depth}, lr={learning_rate:.4f}, ss={subsample:.2f}, "
                      f"csb={colsample_bytree:.2f}, mcw={min_child_weight}, gamma={gamma:.2f}, "
                      f"n_est={n_estimators}")
            else:
                if iteration % 10 == 0:
                    print(f"[Iteration {iteration}] AUC: {test_auc:.4f}, Loss: {test_loss:.4f}, Score: {score:.4f}")

            # Return negative because we're minimizing
            return -score

        except Exception as e:
            print(f"[Iteration {iteration}] ERROR: {str(e)}")
            return 0.0

    # Run optimization
    print("\n[STEP 8] Running Bayesian Optimization...")
    print("(This will take several minutes...)\n")

    result = gp_minimize(
        objective,
        space,
        base_estimator='GP',
        acq_func='EI',
        n_calls=100,
        n_initial_points=20,
        random_state=42,
        verbose=0,
    )

    # Save results
    print("\n[STEP 9] Saving results...")

    results = {
        'best_score': float(best_score),
        'best_params': {
            'max_depth': int(best_params['max_depth']),
            'learning_rate': float(best_params['learning_rate']),
            'subsample': float(best_params['subsample']),
            'colsample_bytree': float(best_params['colsample_bytree']),
            'min_child_weight': int(best_params['min_child_weight']),
            'gamma': float(best_params['gamma']),
            'lambda': float(best_params['lambda']),
            'n_estimators': int(best_params['n_estimators']),
        },
        'iterations': iteration,
        'timestamp': datetime.now().isoformat(),
    }

    with open('xgb_optimization_results_new_odds.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"[+] Results saved to xgb_optimization_results_new_odds.json")

    # Final report
    print("\n" + "="*80)
    print("OPTIMIZATION COMPLETE!")
    print("="*80)
    print(f"\nBest Score: {best_score:.4f}")
    print(f"Iterations: {iteration}")
    print(f"\nBest Hyperparameters:")
    print(f"  max_depth: {results['best_params']['max_depth']}")
    print(f"  learning_rate: {results['best_params']['learning_rate']:.6f}")
    print(f"  subsample: {results['best_params']['subsample']:.4f}")
    print(f"  colsample_bytree: {results['best_params']['colsample_bytree']:.4f}")
    print(f"  min_child_weight: {results['best_params']['min_child_weight']}")
    print(f"  gamma: {results['best_params']['gamma']:.4f}")
    print(f"  lambda: {results['best_params']['lambda']:.4f}")
    print(f"  n_estimators: {results['best_params']['n_estimators']}")
    print("="*80)

if __name__ == "__main__":
    main()
