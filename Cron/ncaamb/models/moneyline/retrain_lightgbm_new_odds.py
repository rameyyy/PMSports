#!/usr/bin/env python3
"""
Retrain moneyline LightGBM model with new decimal odds features
Train on features2021-2024.csv, test on features2025.csv
Uses best hyperparameters found from Bayesian optimization
"""

import polars as pl
import lightgbm as lgb
import numpy as np
import json
import os
from datetime import datetime
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, confusion_matrix, classification_report

def load_features(years):
    """Load and combine features from multiple years"""
    dfs = []
    # Features are in the parent directory (ncaamb root)
    features_dir = os.path.join(os.path.dirname(__file__), '..', '..')

    # Load first file to get schema
    first_loaded = False
    reference_schema = None

    for year in years:
        file = os.path.join(features_dir, f"features{year}.csv")
        if not os.path.exists(file):
            print(f"[!] File not found: {file}")
            continue
        print(f"[+] Loading {file}...")
        df = pl.read_csv(file, infer_schema_length=1000)

        # On first load, standardize schema
        if not first_loaded:
            # Standardize all numeric columns
            for col in df.columns:
                if df[col].dtype in [pl.Float32, pl.Float64]:
                    df = df.with_columns(pl.col(col).cast(pl.Float64))
                elif df[col].dtype in [pl.Int32, pl.Int64]:
                    df = df.with_columns(pl.col(col).cast(pl.Int64))
                elif df[col].dtype == pl.Utf8:
                    # Keep string columns as-is
                    pass
            reference_schema = df.schema
            first_loaded = True
        else:
            # Cast subsequent dataframes to match reference schema
            for col in df.columns:
                if col in reference_schema:
                    target_dtype = reference_schema[col]
                    if df[col].dtype != target_dtype:
                        try:
                            df = df.with_columns(pl.col(col).cast(target_dtype))
                        except:
                            print(f"  [!] Could not cast {col} to {target_dtype}, keeping as-is")

        dfs.append(df)

    if not dfs:
        raise ValueError("No feature files found")

    combined = pl.concat(dfs, how='diagonal')
    print(f"[+] Combined {len(dfs)} files, total rows: {len(combined)}")
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
    """Preprocess features for model training"""
    # Cast betting odds columns to Float64
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
    """Filter out low-quality games"""
    initial_count = len(df)

    # Filter by data quality
    df = df.filter(
        (pl.col('team_1_data_quality') >= 0.5) &
        (pl.col('team_2_data_quality') >= 0.5)
    )

    # Filter games without target
    df = df.filter(pl.col('ml_target').is_not_null())

    # Filter games without moneyline odds
    df = df.filter(
        (pl.col('avg_ml_team_1').is_not_null()) &
        (pl.col('avg_ml_team_2').is_not_null())
    )

    removed = initial_count - len(df)
    print(f"[+] Filtered quality games: {initial_count} -> {len(df)} ({removed} removed)")

    return df

def get_feature_columns(df):
    """Get all numeric feature columns (excluding metadata)"""
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
        if col in metadata_cols:
            continue
        if df[col].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]:
            feature_cols.append(col)

    print(f"[+] Found {len(feature_cols)} feature columns")
    return feature_cols

def prepare_training_data(df, feature_cols):
    """Prepare X and y for training"""
    # Fill NaN values
    df = df.with_columns([
        pl.col(col).fill_null(0) for col in feature_cols
    ])

    X = df.select(feature_cols).to_numpy()
    y = df.select('ml_target').to_numpy().ravel()

    print(f"[+] X shape: {X.shape}, y shape: {y.shape}")
    print(f"[+] Target distribution: {np.bincount(y.astype(int))}")

    return X, y, feature_cols

def get_best_params():
    """Return best hyperparameters from optimization"""
    return {
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

def train_model(X_train, y_train, X_test, y_test, params):
    """Train LightGBM model"""
    print("\n[+] Training LightGBM model...")
    print(f"[+] Hyperparameters: {json.dumps({k: v for k, v in params.items() if k != 'verbose'}, indent=2)}")

    # Create datasets
    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

    # Train
    model = lgb.train(
        params,
        train_data,
        num_boost_round=200,
        valid_sets=[train_data, test_data],
        valid_names=['train', 'test'],
        callbacks=[
            lgb.log_evaluation(period=50),
            lgb.early_stopping(stopping_rounds=50),
        ]
    )

    return model

def evaluate_model(model, X_train, y_train, X_test, y_test):
    """Evaluate model on train and test sets"""
    print("\n[+] Evaluating model...")

    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    y_train_pred_binary = (y_train_pred > 0.5).astype(int)
    y_test_pred_binary = (y_test_pred > 0.5).astype(int)

    # Metrics
    train_acc = accuracy_score(y_train, y_train_pred_binary)
    test_acc = accuracy_score(y_test, y_test_pred_binary)
    train_auc = roc_auc_score(y_train, y_train_pred)
    test_auc = roc_auc_score(y_test, y_test_pred)
    train_loss = log_loss(y_train, y_train_pred)
    test_loss = log_loss(y_test, y_test_pred)

    print(f"\nTRAINING METRICS:")
    print(f"  Accuracy: {train_acc:.4f}")
    print(f"  AUC-ROC:  {train_auc:.4f}")
    print(f"  Log Loss: {train_loss:.4f}")

    print(f"\nTEST METRICS:")
    print(f"  Accuracy: {test_acc:.4f}")
    print(f"  AUC-ROC:  {test_auc:.4f}")
    print(f"  Log Loss: {test_loss:.4f}")

    print(f"\nOVERFITTING GAP:")
    print(f"  Accuracy Gap: {train_acc - test_acc:.4f}")
    print(f"  AUC Gap:      {train_auc - test_auc:.4f}")

    # Confusion matrix
    print(f"\nTEST SET CONFUSION MATRIX:")
    cm = confusion_matrix(y_test, y_test_pred_binary)
    print(f"  True Negatives:  {cm[0,0]}")
    print(f"  False Positives: {cm[0,1]}")
    print(f"  False Negatives: {cm[1,0]}")
    print(f"  True Positives:  {cm[1,1]}")

    print(f"\nTEST SET CLASSIFICATION REPORT:")
    print(classification_report(y_test, y_test_pred_binary, target_names=['Team 2 Wins', 'Team 1 Wins']))

    return {
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'train_auc': train_auc,
        'test_auc': test_auc,
        'train_loss': train_loss,
        'test_loss': test_loss,
        'overfitting_gap': train_acc - test_acc,
    }

def save_model(model, feature_cols):
    """Save model and metadata"""
    print("\n[+] Saving model...")

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_dir = 'saved'
    os.makedirs(model_dir, exist_ok=True)

    # Save model
    model_file = os.path.join(model_dir, f'moneyline_lgb_new_odds_{timestamp}.txt')
    model.save_model(model_file)
    print(f"[+] Model saved to {model_file}")

    # Save feature columns
    features_file = os.path.join(model_dir, f'moneyline_features_{timestamp}.json')
    with open(features_file, 'w') as f:
        json.dump({'feature_columns': feature_cols}, f)
    print(f"[+] Features saved to {features_file}")

    return model_file

def main():
    print("="*80)
    print("RETRAINING MONEYLINE LIGHTGBM WITH NEW DECIMAL ODDS FEATURES")
    print("="*80)

    # Load data
    print("\n[STEP 1] Loading training data (2021-2024)...")
    train_df = load_features(['2021', '2022', '2023', '2024'])

    print("\n[STEP 2] Loading test data (2025)...")
    test_df = load_features(['2025'])

    # Create target
    print("\n[STEP 3] Creating target variable...")
    train_df = create_target_variable(train_df)
    test_df = create_target_variable(test_df)

    # Preprocess
    print("\n[STEP 4] Preprocessing features...")
    train_df = preprocess_features(train_df)
    test_df = preprocess_features(test_df)

    # Filter quality
    print("\n[STEP 5] Filtering low-quality games...")
    train_df = filter_quality_games(train_df)
    test_df = filter_quality_games(test_df)

    # Get features
    print("\n[STEP 6] Identifying feature columns...")
    feature_cols = get_feature_columns(train_df)

    # Prepare data
    print("\n[STEP 7] Preparing training/test data...")
    X_train, y_train, _ = prepare_training_data(train_df, feature_cols)
    X_test, y_test, _ = prepare_training_data(test_df, feature_cols)

    # Train
    print("\n[STEP 8] Training model with best hyperparameters...")
    params = get_best_params()
    model = train_model(X_train, y_train, X_test, y_test, params)

    # Evaluate
    print("\n[STEP 9] Evaluating model...")
    metrics = evaluate_model(model, X_train, y_train, X_test, y_test)

    # Save
    print("\n[STEP 10] Saving model...")
    model_file = save_model(model, feature_cols)

    # Feature importance
    print("\n[STEP 11] Top 20 feature importances...")
    importance = model.feature_importance(importance_type='gain')
    top_indices = np.argsort(importance)[-20:][::-1]
    for rank, idx in enumerate(top_indices, 1):
        print(f"  {rank:2d}. {feature_cols[idx]:40s} : {importance[idx]:10.2f}")

    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print(f"Test Accuracy: {metrics['test_accuracy']:.4f}")
    print(f"Test AUC-ROC:  {metrics['test_auc']:.4f}")
    print(f"Model saved to: {model_file}")
    print("="*80)

if __name__ == "__main__":
    main()
