#!/usr/bin/env python3
"""
Grid search to find optimal XGBoost parameters for O/U prediction
Tests 25 different parameter combinations and ranks them
"""
import polars as pl
import xgboost as xgb
import numpy as np
import sys
sys.path.insert(0, 'models')

# Load features
print("Loading features...")
features_df = pl.read_csv('ou_features.csv')

# Prepare features
from models.ou_model import OUModel
model_temp = OUModel()
X, feature_cols, game_info = model_temp.prepare_features(features_df)
y = np.array(game_info['actual_total'])

# Remove NaN targets
valid_idx = ~np.isnan(y)
X = X[valid_idx]
y = y[valid_idx]

# Train/test split
n_samples = len(X)
n_test = int(n_samples * 0.2)
n_train = n_samples - n_test

X_train, X_test = X[:n_train], X[n_test:]
y_train, y_test = y[:n_train], y[n_test:]

print(f"Training set: {len(X_train)} games")
print(f"Test set: {len(X_test)} games")
print()

# Define parameter grid - 25 different combinations
param_grids = [
    # Current best
    {
        'learning_rate': 0.02, 'max_depth': 3, 'min_child_weight': 10,
        'subsample': 0.5, 'colsample_bytree': 0.5, 'reg_alpha': 5.0, 'reg_lambda': 5.0,
        'n_estimators': 200, 'name': 'CURRENT_BEST'
    },
    # Less aggressive regularization
    {
        'learning_rate': 0.03, 'max_depth': 4, 'min_child_weight': 8,
        'subsample': 0.6, 'colsample_bytree': 0.6, 'reg_alpha': 3.0, 'reg_lambda': 3.0,
        'n_estimators': 200, 'name': 'LESS_AGGRESSIVE'
    },
    # More aggressive regularization
    {
        'learning_rate': 0.01, 'max_depth': 2, 'min_child_weight': 15,
        'subsample': 0.4, 'colsample_bytree': 0.4, 'reg_alpha': 10.0, 'reg_lambda': 10.0,
        'n_estimators': 300, 'name': 'MORE_AGGRESSIVE'
    },
    # Deep trees with light regularization
    {
        'learning_rate': 0.05, 'max_depth': 6, 'min_child_weight': 3,
        'subsample': 0.7, 'colsample_bytree': 0.7, 'reg_alpha': 1.0, 'reg_lambda': 1.0,
        'n_estimators': 100, 'name': 'DEEP_LIGHT'
    },
    # Shallow trees with heavy regularization
    {
        'learning_rate': 0.01, 'max_depth': 2, 'min_child_weight': 20,
        'subsample': 0.3, 'colsample_bytree': 0.3, 'reg_alpha': 15.0, 'reg_lambda': 15.0,
        'n_estimators': 300, 'name': 'SHALLOW_HEAVY'
    },
    # Balanced shallow-medium
    {
        'learning_rate': 0.02, 'max_depth': 4, 'min_child_weight': 7,
        'subsample': 0.55, 'colsample_bytree': 0.55, 'reg_alpha': 4.0, 'reg_lambda': 4.0,
        'n_estimators': 250, 'name': 'BALANCED_1'
    },
    # Medium learning rate with medium depth
    {
        'learning_rate': 0.04, 'max_depth': 4, 'min_child_weight': 5,
        'subsample': 0.65, 'colsample_bytree': 0.65, 'reg_alpha': 2.0, 'reg_lambda': 2.0,
        'n_estimators': 150, 'name': 'MEDIUM_1'
    },
    # Very slow learning, shallow
    {
        'learning_rate': 0.01, 'max_depth': 3, 'min_child_weight': 12,
        'subsample': 0.45, 'colsample_bytree': 0.45, 'reg_alpha': 8.0, 'reg_lambda': 8.0,
        'n_estimators': 400, 'name': 'VERY_SLOW'
    },
    # Fast learning, deep
    {
        'learning_rate': 0.1, 'max_depth': 5, 'min_child_weight': 2,
        'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_alpha': 0.5, 'reg_lambda': 0.5,
        'n_estimators': 50, 'name': 'FAST_DEEP'
    },
    # Asymmetric regularization (high L1, low L2)
    {
        'learning_rate': 0.02, 'max_depth': 3, 'min_child_weight': 10,
        'subsample': 0.5, 'colsample_bytree': 0.5, 'reg_alpha': 10.0, 'reg_lambda': 1.0,
        'n_estimators': 200, 'name': 'HIGH_L1_LOW_L2'
    },
    # Asymmetric regularization (low L1, high L2)
    {
        'learning_rate': 0.02, 'max_depth': 3, 'min_child_weight': 10,
        'subsample': 0.5, 'colsample_bytree': 0.5, 'reg_alpha': 1.0, 'reg_lambda': 10.0,
        'n_estimators': 200, 'name': 'LOW_L1_HIGH_L2'
    },
    # Mid-range balanced
    {
        'learning_rate': 0.03, 'max_depth': 3, 'min_child_weight': 8,
        'subsample': 0.55, 'colsample_bytree': 0.55, 'reg_alpha': 4.0, 'reg_lambda': 4.0,
        'n_estimators': 200, 'name': 'MID_BALANCED'
    },
    # High feature subsampling
    {
        'learning_rate': 0.02, 'max_depth': 3, 'min_child_weight': 10,
        'subsample': 0.5, 'colsample_bytree': 0.3, 'reg_alpha': 5.0, 'reg_lambda': 5.0,
        'n_estimators': 250, 'name': 'HIGH_FEATURE_SUB'
    },
    # Low feature subsampling
    {
        'learning_rate': 0.02, 'max_depth': 3, 'min_child_weight': 10,
        'subsample': 0.5, 'colsample_bytree': 0.7, 'reg_alpha': 5.0, 'reg_lambda': 5.0,
        'n_estimators': 200, 'name': 'LOW_FEATURE_SUB'
    },
    # Low data subsampling
    {
        'learning_rate': 0.02, 'max_depth': 3, 'min_child_weight': 10,
        'subsample': 0.3, 'colsample_bytree': 0.5, 'reg_alpha': 5.0, 'reg_lambda': 5.0,
        'n_estimators': 300, 'name': 'LOW_DATA_SUB'
    },
    # High data subsampling
    {
        'learning_rate': 0.02, 'max_depth': 3, 'min_child_weight': 10,
        'subsample': 0.7, 'colsample_bytree': 0.5, 'reg_alpha': 5.0, 'reg_lambda': 5.0,
        'n_estimators': 150, 'name': 'HIGH_DATA_SUB'
    },
    # Depth 4 focus
    {
        'learning_rate': 0.025, 'max_depth': 4, 'min_child_weight': 6,
        'subsample': 0.6, 'colsample_bytree': 0.6, 'reg_alpha': 3.0, 'reg_lambda': 3.0,
        'n_estimators': 200, 'name': 'DEPTH_4_FOCUS'
    },
    # Depth 5 focus
    {
        'learning_rate': 0.03, 'max_depth': 5, 'min_child_weight': 4,
        'subsample': 0.65, 'colsample_bytree': 0.65, 'reg_alpha': 2.0, 'reg_lambda': 2.0,
        'n_estimators': 150, 'name': 'DEPTH_5_FOCUS'
    },
    # Conservative (safe)
    {
        'learning_rate': 0.015, 'max_depth': 3, 'min_child_weight': 15,
        'subsample': 0.4, 'colsample_bytree': 0.4, 'reg_alpha': 7.0, 'reg_lambda': 7.0,
        'n_estimators': 300, 'name': 'CONSERVATIVE'
    },
    # Aggressive but controlled
    {
        'learning_rate': 0.04, 'max_depth': 4, 'min_child_weight': 4,
        'subsample': 0.65, 'colsample_bytree': 0.65, 'reg_alpha': 1.5, 'reg_lambda': 1.5,
        'n_estimators': 150, 'name': 'CONTROLLED_AGG'
    },
    # Min child weight focus (high)
    {
        'learning_rate': 0.02, 'max_depth': 3, 'min_child_weight': 20,
        'subsample': 0.5, 'colsample_bytree': 0.5, 'reg_alpha': 5.0, 'reg_lambda': 5.0,
        'n_estimators': 250, 'name': 'HIGH_MCW'
    },
    # Min child weight focus (low)
    {
        'learning_rate': 0.02, 'max_depth': 3, 'min_child_weight': 3,
        'subsample': 0.5, 'colsample_bytree': 0.5, 'reg_alpha': 5.0, 'reg_lambda': 5.0,
        'n_estimators': 200, 'name': 'LOW_MCW'
    },
    # High estimators
    {
        'learning_rate': 0.01, 'max_depth': 3, 'min_child_weight': 10,
        'subsample': 0.5, 'colsample_bytree': 0.5, 'reg_alpha': 5.0, 'reg_lambda': 5.0,
        'n_estimators': 500, 'name': 'HIGH_ESTIMATORS'
    },
    # Low estimators
    {
        'learning_rate': 0.03, 'max_depth': 3, 'min_child_weight': 10,
        'subsample': 0.5, 'colsample_bytree': 0.5, 'reg_alpha': 5.0, 'reg_lambda': 5.0,
        'n_estimators': 100, 'name': 'LOW_ESTIMATORS'
    },
    # Tuned hybrid
    {
        'learning_rate': 0.025, 'max_depth': 3, 'min_child_weight': 9,
        'subsample': 0.52, 'colsample_bytree': 0.52, 'reg_alpha': 4.5, 'reg_lambda': 4.5,
        'n_estimators': 220, 'name': 'HYBRID_TUNED'
    },
]

results = []

print("="*120)
print("TESTING 25 PARAMETER COMBINATIONS")
print("="*120)
print()

for idx, params in enumerate(param_grids, 1):
    name = params.pop('name')

    print(f"[{idx:2d}/25] Testing {name:<25}", end=' ... ', flush=True)

    try:
        # Train model
        model = xgb.XGBRegressor(
            objective='reg:squarederror',
            random_state=42,
            tree_method='hist',
            **params
        )

        model.fit(X_train, y_train, verbose=False)

        # Predictions
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

        # Metrics
        train_mae = float(np.mean(np.abs(y_train - train_pred)))
        test_mae = float(np.mean(np.abs(y_test - test_pred)))
        train_rmse = float(np.sqrt(np.mean((y_train - train_pred) ** 2)))
        test_rmse = float(np.sqrt(np.mean((y_test - test_pred) ** 2)))

        mae_gap = test_mae - train_mae

        # Within ranges
        test_errors = np.abs(y_test - test_pred)
        within_5 = (test_errors <= 5).sum() / len(test_errors) * 100
        within_10 = (test_errors <= 10).sum() / len(test_errors) * 100

        results.append({
            'name': name,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'mae_gap': mae_gap,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'within_5': within_5,
            'within_10': within_10,
            'params': params
        })

        print(f"MAE: {train_mae:.2f}|{test_mae:.2f} (gap: {mae_gap:.2f})")

    except Exception as e:
        print(f"ERROR: {str(e)[:50]}")

print()
print("="*120)
print("RESULTS RANKED BY TEST MAE (Best on Top)")
print("="*120)
print()

# Sort by test_mae
results_sorted = sorted(results, key=lambda x: x['test_mae'])

print(f"{'Rank':<5} {'Name':<25} {'Train MAE':<12} {'Test MAE':<12} {'Gap':<12} {'±5 pts':<10} {'±10 pts':<10}")
print("-" * 120)

for idx, r in enumerate(results_sorted[:10], 1):
    print(f"{idx:<5} {r['name']:<25} {r['train_mae']:<12.2f} {r['test_mae']:<12.2f} {r['mae_gap']:<12.2f} {r['within_5']:<10.1f} {r['within_10']:<10.1f}")

print()
print("="*120)
print("TOP 3 MODELS DETAILED COMPARISON")
print("="*120)
print()

top_3 = results_sorted[:3]

for rank, r in enumerate(top_3, 1):
    print(f"#{rank}: {r['name']}")
    print(f"  Parameters:")
    for k, v in r['params'].items():
        print(f"    {k:<20} = {v}")
    print(f"  Metrics:")
    print(f"    Train MAE:  {r['train_mae']:.2f} pts")
    print(f"    Test MAE:   {r['test_mae']:.2f} pts")
    print(f"    MAE Gap:    {r['mae_gap']:.2f} pts (overfitting measure)")
    print(f"    Train RMSE: {r['train_rmse']:.2f} pts")
    print(f"    Test RMSE:  {r['test_rmse']:.2f} pts")
    print(f"    ±5 pts:     {r['within_5']:.1f}%")
    print(f"    ±10 pts:    {r['within_10']:.1f}%")
    print()

# Save best model
print("="*120)
print("SAVING BEST MODEL")
print("="*120)
print()

best = results_sorted[0]
print(f"Best model: {best['name']}")
print(f"Test MAE: {best['test_mae']:.2f} pts")
print(f"Saving parameters for future use...")
print()

# Train best model on full dataset and save
best_params = best['params'].copy()
best_params['objective'] = 'reg:squarederror'
best_params['random_state'] = 42
best_params['tree_method'] = 'hist'

best_model = xgb.XGBRegressor(**best_params)
best_model.fit(X_train, y_train, verbose=False)

import pickle
with open('ou_model_best.pkl', 'wb') as f:
    pickle.dump({
        'model': best_model,
        'feature_names': feature_cols,
        'best_params': best_params,
        'best_name': best['name']
    }, f)

print(f"Best model saved to: ou_model_best.pkl")
print(f"Best parameters: {best['name']}")
