#!/usr/bin/env python3
"""
Over/Under XGBoost Model - Bayesian Optimization and Training
Loads all featuresYYYY.csv files, optimizes hyperparameters, trains final model
Saves to models/overunder/saved/xgboost_model.pkl
"""

import os
import sys
from pathlib import Path
import polars as pl
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args

# Add parent directory to path
ncaamb_dir = Path(__file__).parent.parent
sys.path.insert(0, str(ncaamb_dir))

print("="*80)
print("OVER/UNDER XGBOOST - BAYESIAN OPTIMIZATION (250 ITERATIONS)")
print("="*80 + "\n")


def load_all_features():
    """Load all featuresYYYY.csv files"""
    features_dir = ncaamb_dir
    all_features = []

    print("Loading all features files...")
    for features_file in sorted(features_dir.glob("features*.csv")):
        year = features_file.stem.replace("features", "")
        if year.isdigit():
            print(f"  Loading {features_file.name}...")
            try:
                df = pl.read_csv(features_file)
                print(f"    ✓ Loaded {len(df)} games")
                all_features.append(df)
            except Exception as e:
                print(f"    ✗ Error: {e}")

    if not all_features:
        print("✗ No features files found!")
        return None

    combined_df = pl.concat(all_features)
    print(f"✓ Combined: {len(combined_df)} total games\n")
    return combined_df


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


def prepare_train_test_split(df: pl.DataFrame, feature_cols: list):
    """Split data for optimization: train on oldest to 2nd-most-recent, test on most recent"""
    # Filter out games without actual_total
    df = df.filter(pl.col('actual_total').is_not_null())

    # Dynamically find years
    df = df.with_columns(pl.col('date').str.slice(0, 4).alias('year'))
    years = sorted(df.select(pl.col('year')).unique().to_series().to_list())
    most_recent_year = years[-1]
    second_most_recent = years[-2] if len(years) >= 2 else None
    third_most_recent = years[-3] if len(years) >= 3 else None

    # For optimization: test on MOST RECENT, train on all others
    train_df = df.filter(pl.col('year') != most_recent_year)
    test_df = df.filter(pl.col('year') == most_recent_year)

    X_train = train_df.select(feature_cols).fill_null(0).to_numpy()
    y_train = train_df.select('actual_total').to_numpy().ravel()

    X_test = test_df.select(feature_cols).fill_null(0).to_numpy()
    y_test = test_df.select('actual_total').to_numpy().ravel()

    # Weight training data: (most_recent - 1) = 4x, (most_recent - 2) = 2x, others = 1x
    train_years = train_df.select('year').to_numpy().ravel()
    sample_weights = np.ones(len(train_years))
    if second_most_recent:
        sample_weights[train_years == second_most_recent] = 4.0
    if third_most_recent:
        sample_weights[train_years == third_most_recent] = 2.0

    print(f"Optimization Train/Test Split:")
    print(f"  Years available: {years[0]}-{years[-1]}")
    print(f"  Test year (most recent): {most_recent_year}")
    print(f"  Train years: {sorted([y for y in years if y != most_recent_year])}")
    print(f"  Train: {len(X_train)} games")
    print(f"  Test:  {len(X_test)} games")
    if second_most_recent and third_most_recent:
        print(f"  Sample weights: {second_most_recent}=4x, {third_most_recent}=2x, others=1x")
    elif second_most_recent:
        print(f"  Sample weights: {second_most_recent}=4x, others=1x")
    print(f"  Target range: [{y_train.min():.1f}, {y_train.max():.1f}]\n")

    return X_train, y_train, X_test, y_test, sample_weights, years


# Define hyperparameter search space
space = [
    Integer(3, 10, name='max_depth'),
    Real(0.01, 0.3, prior='log-uniform', name='learning_rate'),
    Integer(100, 1000, name='n_estimators'),
    Real(0.5, 1.0, name='subsample'),
    Real(0.5, 1.0, name='colsample_bytree'),
    Real(0.0, 10.0, name='reg_alpha'),
    Real(0.0, 10.0, name='reg_lambda'),
    Real(1.0, 10.0, name='gamma'),
    Integer(1, 10, name='min_child_weight')
]


@use_named_args(space)
def objective(**params):
    """Objective function for Bayesian optimization"""
    global X_train, y_train, X_test, y_test, sample_weights, iteration_count

    iteration_count += 1
    print(f"\n[Iteration {iteration_count}/250]")
    print(f"  Params: {params}")

    # Train model with sample weights
    model = XGBRegressor(
        objective='reg:squarederror',
        eval_metric='mae',
        random_state=42,
        n_jobs=-1,
        **params
    )

    model.fit(X_train, y_train, sample_weight=sample_weights, verbose=False)

    # Evaluate
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)

    # Minimize: test MAE + overfitting penalty
    overfit_penalty = max(0, test_mae - train_mae - 1.0) * 0.5
    score = test_mae + overfit_penalty

    print(f"  Train MAE: {train_mae:.2f}")
    print(f"  Test MAE:  {test_mae:.2f}")
    print(f"  Overfit:   {test_mae - train_mae:.2f}")
    print(f"  Score:     {score:.2f}")

    return score


if __name__ == "__main__":
    # Load data
    print("STEP 1: Loading Data")
    print("-"*80)
    df = load_all_features()
    if df is None:
        sys.exit(1)

    # Filter for O/U data
    print("STEP 2: Filtering for O/U Data")
    print("-"*80)
    before = len(df)
    df = df.filter(
        pl.col('actual_total').is_not_null() &
        pl.col('betonline_ou_line').is_not_null()
    )
    after = len(df)
    print(f"Removed {before - after} games without O/U data")
    print(f"Kept {after} games\n")

    # Identify features
    print("STEP 3: Identifying Features")
    print("-"*80)
    feature_cols = identify_feature_columns(df)
    print(f"Found {len(feature_cols)} feature columns\n")

    # Split data
    print("STEP 4: Train/Test Split")
    print("-"*80)
    X_train, y_train, X_test, y_test, sample_weights, years = prepare_train_test_split(df, feature_cols)

    # Run Bayesian optimization
    print("STEP 5: Bayesian Optimization (250 iterations)")
    print("-"*80)
    iteration_count = 0

    result = gp_minimize(
        objective,
        space,
        n_calls=250,
        random_state=42,
        verbose=False,
        n_jobs=1
    )

    print("\n" + "="*80)
    print("OPTIMIZATION COMPLETE")
    print("="*80)
    print(f"\nBest MAE: {result.fun:.2f}")
    print(f"Best params:")
    for param_name, param_value in zip([s.name for s in space], result.x):
        print(f"  {param_name}: {param_value}")

    # Train final model with best params
    print("\n" + "="*80)
    print("STEP 6: Training Final Model")
    print("="*80)

    best_params = {s.name: val for s, val in zip(space, result.x)}
    final_model = XGBRegressor(
        objective='reg:squarederror',
        eval_metric='mae',
        random_state=42,
        n_jobs=-1,
        **best_params
    )

    # Train on ALL data with sample weights prioritizing most recent
    # Recreate full dataset with year column
    df_with_year = df.with_columns(pl.col('date').str.slice(0, 4).alias('year'))
    df_with_year = df_with_year.filter(pl.col('actual_total').is_not_null())

    X_all = df_with_year.select(feature_cols).fill_null(0).to_numpy()
    y_all = df_with_year.select('actual_total').to_numpy().ravel()

    # Weight ALL data: most_recent=4x, second_most_recent=2x, others=1x
    all_years = df_with_year.select('year').to_numpy().ravel()
    weights_all = np.ones(len(all_years))
    most_recent = years[-1]
    second_most_recent = years[-2] if len(years) >= 2 else None
    weights_all[all_years == most_recent] = 4.0
    if second_most_recent:
        weights_all[all_years == second_most_recent] = 2.0

    print(f"Training final model on features{years[0]}-{years[-1]} ({len(X_all)} total games)...")
    print(f"  Weights: {most_recent}=4x, {second_most_recent if second_most_recent else 'N/A'}=2x, others=1x")
    final_model.fit(X_all, y_all, sample_weight=weights_all, verbose=False)

    final_mae = mean_absolute_error(y_all, final_model.predict(X_all))
    final_r2 = r2_score(y_all, final_model.predict(X_all))
    print(f"Final model MAE: {final_mae:.2f}")
    print(f"Final model R²:  {final_r2:.4f}")

    # Save model
    print("\n" + "="*80)
    print("STEP 7: Saving Model")
    print("="*80)

    save_dir = ncaamb_dir / "models" / "overunder" / "saved"
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / "xgboost_model.pkl"

    final_model.save_model(str(save_path))
    print(f"✓ Saved XGBoost model trained on features{years[0]}-{years[-1]}")
    print(f"  File: {save_path}")

    # Save hyperparameters
    params_path = save_dir / "xgboost_hyperparameters.txt"
    with open(params_path, 'w') as f:
        f.write("XGBoost O/U Model Hyperparameters\n")
        f.write("="*50 + "\n\n")
        for param_name, param_value in best_params.items():
            f.write(f"{param_name}: {param_value}\n")
        f.write(f"\nFinal MAE: {final_mae:.2f}\n")
        f.write(f"Final R²: {final_r2:.4f}\n")
        f.write(f"Training Games: {len(X_all)}\n")

    print(f"✓ Saved hyperparameters to: {params_path}")
    print("\n" + "="*80)
    print("DONE!")
    print("="*80)
