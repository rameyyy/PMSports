#!/usr/bin/env python3
"""
XGBoost Moneyline Model
Train on 2021-2024 data, test on 2025 data
"""

import os
import sys
from pathlib import Path
import polars as pl
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Add parent directory to path for imports
ncaamb_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ncaamb_dir))


def load_features_by_year(years: list) -> pl.DataFrame:
    """
    Load feature files from specified years

    Args:
        years: List of years to load (e.g., ['2021', '2022', '2023', '2024'])

    Returns:
        Polars DataFrame with combined features
    """
    features_dir = ncaamb_dir
    all_features = []

    print(f"Loading features for years: {years}")
    for year in years:
        features_file = features_dir / f"features{year}.csv"

        if features_file.exists():
            print(f"  Loading features{year}.csv...")
            try:
                df = pl.read_csv(features_file)
                print(f"    ✓ Loaded {len(df)} games")
                all_features.append(df)
            except Exception as e:
                print(f"    ✗ Error loading {year}: {e}")
        else:
            print(f"    ✗ File not found: {features_file}")

    if not all_features:
        print("  ✗ No feature files loaded!")
        return None

    # Combine all years
    combined_df = pl.concat(all_features)
    print(f"✓ Combined: {len(combined_df)} total games\n")

    return combined_df


def filter_low_quality_games(df: pl.DataFrame, min_data_quality: float = 0.5) -> pl.DataFrame:
    """
    Filter out early season games with mostly null values (no game history)

    Args:
        df: Polars DataFrame with features
        min_data_quality: Minimum data quality score (0-1) to keep a game

    Returns:
        Filtered DataFrame
    """
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
    """
    Create binary target variable for moneyline (1 = team_1 wins, 0 = team_2 wins)
    Only for games with actual scores

    Args:
        df: Polars DataFrame with game results

    Returns:
        DataFrame with 'ml_target' column
    """
    # Filter to only games with scores (completed games)
    df_with_scores = df.filter(
        pl.col('team_1_score').is_not_null() &
        pl.col('team_2_score').is_not_null()
    )

    # Create target: 1 if team_1 wins, 0 if team_2 wins
    df_with_scores = df_with_scores.with_columns(
        pl.when(pl.col('team_1_score') > pl.col('team_2_score'))
            .then(1)
            .otherwise(0)
            .alias('ml_target')
    )

    print(f"Created target for {len(df_with_scores)} games with results")
    print(f"  Team 1 wins: {df_with_scores.filter(pl.col('ml_target') == 1).height}")
    print(f"  Team 2 wins: {df_with_scores.filter(pl.col('ml_target') == 0).height}\n")

    return df_with_scores


def identify_feature_columns(df: pl.DataFrame) -> list:
    """
    Identify numeric feature columns (exclude metadata and target)
    """
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

    print(f"Identified {len(feature_cols)} numeric feature columns\n")
    return feature_cols


def prepare_training_data(df: pl.DataFrame, feature_cols: list) -> tuple:
    """
    Prepare X (features) and y (target) for training
    Fill NaN values with 0
    """
    X = df.select(feature_cols).fill_null(0)
    y = df.select('ml_target')

    print(f"Training data shape: X={X.shape}, y={y.shape}\n")

    return X, y


def train_xgboost_model(X_train: pl.DataFrame, y_train: pl.DataFrame,
                       model_dir: Path = None) -> XGBClassifier:
    """
    Train XGBoost classifier for moneyline predictions
    """
    if model_dir is None:
        model_dir = Path(__file__).parent

    model_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("TRAINING XGBOOST MONEYLINE MODEL (2021-2024)")
    print("="*80 + "\n")

    # Convert to numpy
    X_np = X_train.to_numpy()
    y_np = y_train.to_numpy().ravel()

    print(f"Training on {len(X_np)} samples...")
    model = XGBClassifier(
        n_estimators=200,
        max_depth=7,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss',
        verbosity=0
    )

    model.fit(X_np, y_np)
    print(f"✓ Model training complete\n")

    # Show feature importance
    print("Feature importance (top 15):")
    feature_importance = list(zip(X_train.columns, model.feature_importances_))
    feature_importance.sort(key=lambda x: x[1], reverse=True)

    for i, (feat, importance) in enumerate(feature_importance[:15], 1):
        print(f"  {i:2}. {feat:40} {importance:.4f}")

    print()

    # Save model
    model_path = model_dir / "xgboost_model.pkl"
    model.save_model(str(model_path))
    print(f"✓ Model saved to {model_path}\n")

    return model


def evaluate_model(y_true: pl.DataFrame, y_pred: np.ndarray, y_pred_proba: np.ndarray,
                  dataset_name: str = ""):
    """
    Evaluate model performance
    """
    y_true_np = y_true.to_numpy().ravel()

    accuracy = accuracy_score(y_true_np, y_pred)
    precision = precision_score(y_true_np, y_pred, zero_division=0)
    recall = recall_score(y_true_np, y_pred, zero_division=0)
    f1 = f1_score(y_true_np, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true_np, y_pred).ravel()

    print(f"{'='*80}")
    print(f"EVALUATION - {dataset_name}")
    print(f"{'='*80}\n")

    print(f"Accuracy:  {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1 Score:  {f1:.3f}\n")

    print(f"Confusion Matrix:")
    print(f"  True Negatives:  {tn:4d} (Team 2 wins, predicted Team 2)")
    print(f"  False Positives: {fp:4d} (Team 2 wins, predicted Team 1)")
    print(f"  False Negatives: {fn:4d} (Team 1 wins, predicted Team 2)")
    print(f"  True Positives:  {tp:4d} (Team 1 wins, predicted Team 1)\n")

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': (tn, fp, fn, tp)
    }


def main():
    print("\n")
    print("="*80)
    print("MONEYLINE MODEL - XGBOOST TRAIN/TEST PIPELINE")
    print("="*80 + "\n")

    # Load training data (2021-2024)
    print("STEP 1: Loading Training Data (2021-2024)")
    print("-"*80 + "\n")
    train_df = load_features_by_year(['2021', '2022', '2023', '2024'])

    if train_df is None or len(train_df) == 0:
        print("Failed to load training features")
        return

    # Filter low quality games
    train_df = filter_low_quality_games(train_df, min_data_quality=0.5)

    # Create target variable
    train_df = create_target_variable(train_df)

    # Identify features
    feature_cols = identify_feature_columns(train_df)

    # Prepare training data
    X_train, y_train = prepare_training_data(train_df, feature_cols)

    # Train model
    print("\nSTEP 2: Training Model")
    print("-"*80 + "\n")
    model = train_xgboost_model(X_train, y_train)

    # Evaluate on training data
    print("STEP 3: Evaluating on Training Data")
    print("-"*80 + "\n")
    y_pred_train = model.predict(X_train.to_numpy())
    y_pred_proba_train = model.predict_proba(X_train.to_numpy())
    train_metrics = evaluate_model(y_train, y_pred_train, y_pred_proba_train,
                                    "Training Set (2021-2024)")

    # Load test data (2025)
    print("\nSTEP 4: Loading Test Data (2025)")
    print("-"*80 + "\n")
    test_df = load_features_by_year(['2025'])

    if test_df is None or len(test_df) == 0:
        print("No test data (2025) available - skipping test evaluation")
        print("\n" + "="*80)
        print("✅ Training complete!")
        print("="*80 + "\n")
        return

    # Filter low quality games in test set
    test_df = filter_low_quality_games(test_df, min_data_quality=0.5)

    # Create target variable for test set
    test_df = create_target_variable(test_df)

    if len(test_df) == 0:
        print("No test games with results available")
        print("\n" + "="*80)
        print("✅ Training complete!")
        print("="*80 + "\n")
        return

    # Prepare test data
    X_test, y_test = prepare_training_data(test_df, feature_cols)

    # Evaluate on test data
    print("\nSTEP 5: Evaluating on Test Data")
    print("-"*80 + "\n")
    y_pred_test = model.predict(X_test.to_numpy())
    y_pred_proba_test = model.predict_proba(X_test.to_numpy())
    test_metrics = evaluate_model(y_test, y_pred_test, y_pred_proba_test,
                                   "Test Set (2025)")

    # Summary comparison
    print("\n" + "="*80)
    print("SUMMARY - TRAIN vs TEST")
    print("="*80 + "\n")

    print(f"Training Accuracy:  {train_metrics['accuracy']:.3f}")
    print(f"Test Accuracy:      {test_metrics['accuracy']:.3f}")
    print(f"Difference:         {train_metrics['accuracy'] - test_metrics['accuracy']:.3f}")
    print()

    print("="*80)
    print("✅ Training and evaluation complete!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
