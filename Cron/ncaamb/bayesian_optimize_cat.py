#!/usr/bin/env python3
"""
Bayesian Optimization for CatBoost Over/Under model using Optuna

This script:
1. Loads features from multiple years
2. Uses Bayesian optimization (Optuna) to find best CatBoost parameters
3. Intelligently samples parameter space based on past results
4. Saves best parameters and trial history
"""
import polars as pl
import os
import sys
import optuna
from optuna.samplers import TPESampler
import pandas as pd

# Add models directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'models'))

from models.cat_model import CatModel


def apply_data_quality_filters(df, min_bookmakers=2):
    """Apply data quality filters to remove low-quality rows"""
    initial_count = len(df)
    stats = {'initial': initial_count, 'min_bookmakers': min_bookmakers}

    # Filter 1: Remove rows where either team is missing leaderboard data
    df = df.filter(
        pl.col('team_1_adjoe').is_not_null() &
        pl.col('team_1_adjde').is_not_null() &
        pl.col('team_2_adjoe').is_not_null() &
        pl.col('team_2_adjde').is_not_null()
    )
    stats['after_leaderboard_check'] = len(df)

    # Filter 2: Remove null avg_ou_line (no odds data)
    df = df.filter(pl.col('avg_ou_line').is_not_null())
    stats['after_avg_ou_line'] = len(df)

    # Filter 3: Remove rows with insufficient bookmakers
    df = df.filter(pl.col('num_books_with_ou') >= min_bookmakers)
    stats['after_num_books'] = len(df)

    stats['filtered_out'] = initial_count - len(df)
    stats['percent_retained'] = (len(df) / initial_count * 100) if initial_count > 0 else 0

    return df, stats


def remove_rows_with_too_many_nulls(df, null_threshold=0.2):
    """Remove rows where null percentage exceeds threshold"""
    initial_count = len(df)

    # Metadata columns to exclude from null check
    metadata_cols = {'game_id', 'date', 'team_1', 'team_2'}
    feature_cols = [c for c in df.columns if c not in metadata_cols]

    # Calculate null count per row for feature columns only
    null_counts = df.select(feature_cols).select([
        pl.sum_horizontal(pl.all().is_null()).alias('null_count')
    ])

    total_feature_cols = len(feature_cols)
    null_pct = null_counts['null_count'] / total_feature_cols if total_feature_cols > 0 else 0

    # Filter out rows exceeding threshold
    valid_rows = null_pct <= null_threshold
    df = df.filter(valid_rows)

    removed_count = initial_count - len(df)
    return df, {'initial': initial_count, 'removed': removed_count, 'retained': len(df)}


def load_features():
    """Load and concatenate features from multiple years"""
    print("\n1. Loading features from multiple years...")
    dfs = []

    for year in [2021, 2022, 2023, 2024, 2025]:
        filename = f"features{year}.csv"
        try:
            df = pl.read_csv(filename)
            dfs.append(df)
            print(f"   Loaded {filename}: {len(df)} games")
        except FileNotFoundError:
            print(f"   ⚠️  {filename} not found, skipping...")

    # Also try to load features.csv if it exists
    try:
        df = pl.read_csv("features.csv")
        dfs.append(df)
        print(f"   Loaded features.csv: {len(df)} games")
    except FileNotFoundError:
        pass

    if not dfs:
        raise FileNotFoundError("No feature files found (features.csv, features2021.csv, features2022.csv, features2023.csv, features2024.csv)")

    # Concatenate all dataframes
    features_df = pl.concat(dfs)
    print(f"\n   Total combined: {len(features_df)} games with {len(features_df.columns)} features")

    return features_df


def objective(trial, features_df):
    """
    Objective function for Optuna optimization.
    Returns test MAE (lower is better).
    """
    try:
        # Define parameter search space for CatBoost
        learning_rate = trial.suggest_float('learning_rate', 0.01, 0.15, log=True)
        depth = trial.suggest_int('depth', 3, 10)
        iterations = trial.suggest_int('iterations', 50, 300)
        l2_leaf_reg = trial.suggest_float('l2_leaf_reg', 1.0, 10.0)
        subsample = trial.suggest_float('subsample', 0.5, 1.0)
        colsample_bylevel = trial.suggest_float('colsample_bylevel', 0.5, 1.0)

        # Train model
        model = CatModel()
        metrics = model.train(
            features_df,
            test_size=0.2,
            learning_rate=learning_rate,
            depth=depth,
            iterations=iterations,
            l2_leaf_reg=l2_leaf_reg,
            subsample=subsample,
            colsample_bylevel=colsample_bylevel,
        )

        # Return test MAE (Optuna minimizes the objective)
        test_mae = metrics['test_mae']

        # Report intermediate value for pruning (early stopping if not promising)
        trial.report(test_mae, step=0)

        return test_mae

    except Exception as e:
        print(f"   Error in trial: {str(e)}")
        return float('inf')


def main():
    print("="*80)
    print("BAYESIAN OPTIMIZATION - CATBOOST MODEL (Optuna)")
    print("="*80)

    # Load features
    features_df = load_features()

    # Apply data quality filters
    print("\n2. Applying data quality filters...")
    MIN_BOOKMAKERS = 2
    features_df, filter_stats = apply_data_quality_filters(features_df, min_bookmakers=MIN_BOOKMAKERS)
    print(f"   Games after leaderboard/odds/bookmaker filters: {len(features_df)}")

    # Remove rows with too many nulls
    features_df, null_stats = remove_rows_with_too_many_nulls(features_df, null_threshold=0.2)
    print(f"   Games after removing high-null rows: {len(features_df)}")

    # Ensure actual_total is not null
    features_df = features_df.filter(pl.col('actual_total').is_not_null())
    print(f"   Games with valid target (actual_total): {len(features_df)}")

    print(f"\n   Final dataset: {len(features_df)} games ready for optimization")

    # Create Optuna study
    print("\n3. Starting Bayesian Optimization (Optuna)...")
    print("   Using TPE sampler (Tree-structured Parzen Estimator)")
    print("   Testing 300 trials with CatBoost\n")

    # Create study with TPE sampler
    sampler = TPESampler(seed=42)
    study = optuna.create_study(
        sampler=sampler,
        direction='minimize',  # Minimize MAE
        study_name='catboost_optimization'
    )

    # Optimize
    n_trials = 300
    study.optimize(
        lambda trial: objective(trial, features_df),
        n_trials=n_trials,
        show_progress_bar=True,
        n_jobs=1  # Use 1 job for stability
    )

    # Get best trial
    best_trial = study.best_trial
    print("\n" + "="*80)
    print("OPTIMIZATION COMPLETE - CATBOOST")
    print("="*80)

    print(f"\nBest Trial: #{best_trial.number}")
    print(f"Best Test MAE: {best_trial.value:.4f} points")

    print(f"\nBest Hyperparameters:")
    for key, value in best_trial.params.items():
        if isinstance(value, float):
            print(f"   {key:<20s}: {value:.6f}")
        else:
            print(f"   {key:<20s}: {value}")

    # Get all trials as dataframe
    trials_df = study.trials_dataframe()

    # Save results
    print(f"\n4. Saving results...")
    trials_df.to_csv("bayesian_optimization_cat_results.csv", index=False)
    print(f"   Saved all {len(trials_df)} trials to bayesian_optimization_cat_results.csv")

    # Show top 10 trials
    print(f"\n" + "="*80)
    print(f"TOP 10 TRIALS - CATBOOST")
    print(f"="*80)
    top_10 = trials_df.nsmallest(10, 'value')[['number', 'value', 'params_learning_rate', 'params_depth',
                                                   'params_iterations', 'params_l2_leaf_reg',
                                                   'params_subsample', 'params_colsample_bylevel']]

    print("\n" + " "*5 + f"{'Trial':<7} {'Test MAE':<12} {'LR':<10} {'Depth':<7} {'Iter':<7} {'L2Reg':<7} {'SS':<7} {'CS':<7}")
    print("="*90)

    for idx, row in top_10.iterrows():
        print(f"   {int(row['number']):<7} {row['value']:<12.4f} {row['params_learning_rate']:<10.6f} {int(row['params_depth']):<7} {int(row['params_iterations']):<7} {row['params_l2_leaf_reg']:<7.2f} {row['params_subsample']:<7.3f} {row['params_colsample_bylevel']:<7.3f}")

    # Summary statistics
    print(f"\n" + "="*80)
    print(f"OPTIMIZATION STATISTICS - CATBOOST")
    print(f"="*80)
    print(f"Total trials: {len(trials_df)}")
    print(f"Best MAE: {trials_df['value'].min():.4f}")
    print(f"Mean MAE: {trials_df['value'].mean():.4f}")
    print(f"Median MAE: {trials_df['value'].median():.4f}")
    print(f"Worst MAE: {trials_df['value'].max():.4f}")
    print(f"Improvement: {trials_df['value'].max() - trials_df['value'].min():.4f} points")

    print(f"\nResults saved to: bayesian_optimization_cat_results.csv")
    print(f"Study name: catboost_optimization")


if __name__ == "__main__":
    main()
