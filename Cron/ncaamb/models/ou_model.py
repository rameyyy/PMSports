#!/usr/bin/env python3
"""
Train and save all 3 OU models (XGBoost, LightGBM, CatBoost) on full dataset
Loads all featuresYYYY.csv files, combines them, and trains using optimized parameters

Models saved to: models/overunder/saved/
- xgboost_model.pkl
- lightgbm_model.pkl
- catboost_model.pkl
"""

import os
import sys
import numpy as np
import polars as pl
from pathlib import Path
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

# Get ncaamb directory (parent of models)
models_dir = os.path.dirname(os.path.abspath(__file__))
ncaamb_dir = os.path.dirname(models_dir)
sys.path.insert(0, models_dir)

from overunder.ou_model import OUModel
from overunder.lgb_model import LGBModel
from overunder.cat_model import CatModel


# ============================================================================
# OPTIMIZED HYPERPARAMETERS (from BEST_OPTIMIZATIONS.txt)
# ============================================================================

OPTIMIZED_XGB_PARAMS = {
    'learning_rate': 0.1356325569317646,
    'max_depth': 3,
    'n_estimators': 264,
    'min_child_weight': 10,
    'subsample': 0.9059553897628048,
    'colsample_bytree': 0.8651858536173023,
    'reg_alpha': 1.7596003894852836,
    'reg_lambda': 0.0687329597968497,
}

OPTIMIZED_LGB_PARAMS = {
    'learning_rate': 0.1133837674716694,
    'max_depth': 3,
    'num_leaves': 49,
    'min_child_samples': 12,
    'subsample': 0.7991800060529038,
    'colsample_bytree': 0.8152595898952936,
    'reg_alpha': 0.8915908456370663,
    'reg_lambda': 0.2613802136226955,
}

OPTIMIZED_CAT_PARAMS = {
    'learning_rate': 0.076195,
    'depth': 3,
    'iterations': 238,
    'l2_leaf_reg': 9.63,
    'subsample': 0.859,
    'colsample_bylevel': 0.963,
}


def load_all_features() -> pl.DataFrame:
    """Load and combine all featuresYYYY.csv files"""
    features_dir = ncaamb_dir
    features_files = sorted(Path(features_dir).glob('features*.csv'))

    if not features_files:
        raise FileNotFoundError(f"No features*.csv files found in {features_dir}")

    print(f"Found {len(features_files)} features files:")
    for f in features_files:
        print(f"  - {f.name}")

    # Load and combine all files
    dataframes = []
    for file_path in features_files:
        print(f"Loading {file_path.name}...")
        df = pl.read_csv(file_path)
        dataframes.append(df)
        print(f"  -> {len(df)} rows")

    # Concatenate all dataframes
    combined_df = pl.concat(dataframes)
    print(f"\nTotal combined: {len(combined_df)} rows")

    return combined_df


def train_xgboost(features_df: pl.DataFrame, save_path: str) -> None:
    """Train XGBoost model on full dataset with validation split from oldest games"""
    print("\n" + "="*80)
    print("TRAINING XGBOOST MODEL")
    print("="*80)

    model = OUModel()

    print(f"Training XGBoost with optimized parameters...")
    print(f"  Dataset: {len(features_df)} games")
    print(f"  Validation split: Oldest 5% of games (2021 season)")
    print(f"  Training split: Newest 95% of games (2022-2025)")
    print(f"Parameters: {OPTIMIZED_XGB_PARAMS}")

    # Prepare features with chronological order
    if 'date' in features_df.columns:
        features_df = features_df.sort('date')

    X, feature_cols, game_info = model.prepare_features(features_df)
    y = np.array(game_info["actual_total"], dtype=float)

    # Drop rows with NaN targets
    valid_idx = ~np.isnan(y)
    X = X[valid_idx]
    y = y[valid_idx]

    # Validation split from OLDEST games, training from NEWEST games
    n_samples = len(X)
    n_val = int(n_samples * 0.05)
    n_train = max(1, n_samples - n_val)

    # Split: oldest 5% as validation, newest 95% as training
    X_val, X_train = X[:n_val], X[n_val:]
    y_val, y_train = y[:n_val], y[n_val:]

    # Train the model with validation set
    model.model = xgb.XGBRegressor(**{
        "objective": "reg:squarederror",
        "eval_metric": "mae",
        "learning_rate": OPTIMIZED_XGB_PARAMS['learning_rate'],
        "max_depth": OPTIMIZED_XGB_PARAMS['max_depth'],
        "min_child_weight": OPTIMIZED_XGB_PARAMS['min_child_weight'],
        "subsample": OPTIMIZED_XGB_PARAMS['subsample'],
        "colsample_bytree": OPTIMIZED_XGB_PARAMS['colsample_bytree'],
        "reg_alpha": OPTIMIZED_XGB_PARAMS['reg_alpha'],
        "reg_lambda": OPTIMIZED_XGB_PARAMS['reg_lambda'],
        "n_estimators": OPTIMIZED_XGB_PARAMS['n_estimators'],
        "tree_method": "hist",
        "n_jobs": -1,
        "random_state": 42,
        "early_stopping_rounds": 50,
    })

    model.model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    model.feature_names = feature_cols
    model.is_trained = True

    # Calculate metrics
    train_pred = model.model.predict(X_train)
    val_pred = model.model.predict(X_val)

    train_mae = float(np.mean(np.abs(y_train - train_pred)))
    val_mae = float(np.mean(np.abs(y_val - val_pred)))
    train_rmse = float(np.sqrt(np.mean((y_train - train_pred) ** 2)))
    val_rmse = float(np.sqrt(np.mean((y_val - val_pred) ** 2)))

    print(f"\nTraining Results:")
    print(f"  Training MAE: {train_mae:.4f}")
    print(f"  Validation MAE: {val_mae:.4f}")
    print(f"  Training RMSE: {train_rmse:.4f}")
    print(f"  Validation RMSE: {val_rmse:.4f}")

    # Save model
    model.save_model(save_path)
    print(f"\nModel saved to: {save_path}")


def train_lightgbm(features_df: pl.DataFrame, save_path: str) -> None:
    """Train LightGBM model on full dataset with validation split from oldest games"""
    print("\n" + "="*80)
    print("TRAINING LIGHTGBM MODEL")
    print("="*80)

    model = LGBModel()

    print(f"Training LightGBM with optimized parameters...")
    print(f"  Dataset: {len(features_df)} games")
    print(f"  Validation split: Oldest 5% of games (2021 season)")
    print(f"  Training split: Newest 95% of games (2022-2025)")
    print(f"Parameters: {OPTIMIZED_LGB_PARAMS}")

    # Prepare features with chronological order
    if 'date' in features_df.columns:
        features_df = features_df.sort('date')

    X, feature_cols, game_info = model.prepare_features(features_df)
    y = np.array(game_info["actual_total"], dtype=float)

    # Drop rows with NaN targets
    valid_idx = ~np.isnan(y)
    X = X[valid_idx]
    y = y[valid_idx]

    # Validation split from OLDEST games, training from NEWEST games
    n_samples = len(X)
    n_val = int(n_samples * 0.05)
    n_train = max(1, n_samples - n_val)

    # Split: oldest 5% as validation, newest 95% as training
    X_val, X_train = X[:n_val], X[n_val:]
    y_val, y_train = y[:n_val], y[n_val:]

    # Train the model with validation set
    model.model = lgb.LGBMRegressor(**{
        "objective": "regression",
        "metric": "mae",
        "learning_rate": OPTIMIZED_LGB_PARAMS['learning_rate'],
        "max_depth": OPTIMIZED_LGB_PARAMS['max_depth'],
        "num_leaves": OPTIMIZED_LGB_PARAMS['num_leaves'],
        "min_child_samples": OPTIMIZED_LGB_PARAMS['min_child_samples'],
        "subsample": OPTIMIZED_LGB_PARAMS['subsample'],
        "colsample_bytree": OPTIMIZED_LGB_PARAMS['colsample_bytree'],
        "reg_alpha": OPTIMIZED_LGB_PARAMS['reg_alpha'],
        "reg_lambda": OPTIMIZED_LGB_PARAMS['reg_lambda'],
        "n_estimators": 400,
        "random_state": 42,
        "verbose": -1,
    })

    model.model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(50),
            lgb.log_evaluation(period=0)
        ]
    )

    model.feature_names = feature_cols
    model.is_trained = True

    # Calculate metrics
    train_pred = model.model.predict(X_train)
    val_pred = model.model.predict(X_val)

    train_mae = float(np.mean(np.abs(y_train - train_pred)))
    val_mae = float(np.mean(np.abs(y_val - val_pred)))
    train_rmse = float(np.sqrt(np.mean((y_train - train_pred) ** 2)))
    val_rmse = float(np.sqrt(np.mean((y_val - val_pred) ** 2)))

    print(f"\nTraining Results:")
    print(f"  Training MAE: {train_mae:.4f}")
    print(f"  Validation MAE: {val_mae:.4f}")
    print(f"  Training RMSE: {train_rmse:.4f}")
    print(f"  Validation RMSE: {val_rmse:.4f}")

    # Save model
    model.save_model(save_path)
    print(f"\nModel saved to: {save_path}")


def train_catboost(features_df: pl.DataFrame, save_path: str) -> None:
    """Train CatBoost model on full dataset with validation split from oldest games"""
    print("\n" + "="*80)
    print("TRAINING CATBOOST MODEL")
    print("="*80)

    model = CatModel()

    print(f"Training CatBoost with optimized parameters...")
    print(f"  Dataset: {len(features_df)} games")
    print(f"  Validation split: Oldest 5% of games (2021 season)")
    print(f"  Training split: Newest 95% of games (2022-2025)")
    print(f"Parameters: {OPTIMIZED_CAT_PARAMS}")

    # Prepare features with chronological order
    if 'date' in features_df.columns:
        features_df = features_df.sort('date')

    X, feature_cols, game_info = model.prepare_features(features_df)
    y = np.array(game_info["actual_total"], dtype=float)

    # Drop rows with NaN targets
    valid_idx = ~np.isnan(y)
    X = X[valid_idx]
    y = y[valid_idx]

    # Validation split from OLDEST games, training from NEWEST games
    n_samples = len(X)
    n_val = int(n_samples * 0.05)
    n_train = max(1, n_samples - n_val)

    # Split: oldest 5% as validation, newest 95% as training
    X_val, X_train = X[:n_val], X[n_val:]
    y_val, y_train = y[:n_val], y[n_val:]

    # Train the model with validation set
    model.model = cb.CatBoostRegressor(**{
        "loss_function": "MAE",
        "eval_metric": "MAE",
        "learning_rate": OPTIMIZED_CAT_PARAMS['learning_rate'],
        "depth": OPTIMIZED_CAT_PARAMS['depth'],
        "iterations": OPTIMIZED_CAT_PARAMS['iterations'],
        "l2_leaf_reg": OPTIMIZED_CAT_PARAMS['l2_leaf_reg'],
        "subsample": OPTIMIZED_CAT_PARAMS['subsample'],
        "colsample_bylevel": OPTIMIZED_CAT_PARAMS['colsample_bylevel'],
        "random_state": 42,
        "verbose": False,
    })

    model.model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        early_stopping_rounds=50,
        verbose=False
    )

    model.feature_names = feature_cols
    model.is_trained = True

    # Calculate metrics
    train_pred = model.model.predict(X_train)
    val_pred = model.model.predict(X_val)

    train_mae = float(np.mean(np.abs(y_train - train_pred)))
    val_mae = float(np.mean(np.abs(y_val - val_pred)))
    train_rmse = float(np.sqrt(np.mean((y_train - train_pred) ** 2)))
    val_rmse = float(np.sqrt(np.mean((y_val - val_pred) ** 2)))

    print(f"\nTraining Results:")
    print(f"  Training MAE: {train_mae:.4f}")
    print(f"  Validation MAE: {val_mae:.4f}")
    print(f"  Training RMSE: {train_rmse:.4f}")
    print(f"  Validation RMSE: {val_rmse:.4f}")

    # Save model
    model.save_model(save_path)
    print(f"\nModel saved to: {save_path}")


def main():
    print("="*80)
    print("OU MODEL TRAINING - FULL DATASET")
    print("="*80)

    # Load all features
    features_df = load_all_features()

    # Create save directory
    save_dir = os.path.join(models_dir, 'overunder', 'saved')
    os.makedirs(save_dir, exist_ok=True)

    # Define save paths
    xgb_path = os.path.join(save_dir, 'xgboost_model.pkl')
    lgb_path = os.path.join(save_dir, 'lightgbm_model.pkl')
    cat_path = os.path.join(save_dir, 'catboost_model.pkl')

    # Train all models
    train_xgboost(features_df, xgb_path)
    train_lightgbm(features_df, lgb_path)
    train_catboost(features_df, cat_path)

    print("\n" + "="*80)
    print("ALL MODELS TRAINED AND SAVED")
    print("="*80)
    print(f"\nModels saved in: {save_dir}")
    print(f"  - {os.path.basename(xgb_path)}")
    print(f"  - {os.path.basename(lgb_path)}")
    print(f"  - {os.path.basename(cat_path)}")


if __name__ == "__main__":
    main()
