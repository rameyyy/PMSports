"""
Train XGBoost model on NCAA basketball game features
"""

import polars as pl
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import pickle
import os
from datetime import datetime


def prepare_features(df: pl.DataFrame) -> tuple:
    """
    Prepare features and target for XGBoost.

    Args:
        df: Polars DataFrame with features

    Returns:
        (X, y) - Feature matrix and target vector
    """
    # Select feature columns (exclude identifiers and target)
    feature_cols = [
        'team_1_rank', 'team_2_rank', 'rank_diff',
        'team_1_winpct', 'team_2_winpct', 'record_diff',
        'team_1_barthag', 'team_2_barthag', 'barthag_diff',
        'team_1_hist_count', 'team_2_hist_count',
        'consensus_ml_home', 'implied_away_prob',
        'avg_spread_home', 'spread_variance',
        'num_books',
    ]

    # Convert to numpy arrays
    X = df.select(feature_cols).to_numpy()
    y = df.select('team_1_win').to_numpy().ravel()

    # Handle any remaining NaNs
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    return X, y, feature_cols


def train_xgboost(df: pl.DataFrame, test_size: float = 0.33, random_state: int = 42) -> dict:
    """
    Train XGBoost model on features with 67% train / 33% test split.

    Args:
        df: Polars DataFrame with features
        test_size: Proportion of data for testing (default 0.33)
        random_state: Random seed for reproducibility

    Returns:
        Dictionary with model, metrics, feature names, and results
    """
    print("="*100)
    print("TRAINING XGBOOST MODEL")
    print("="*100)

    # Prepare data
    print("\nPreparing features...")
    X, y, feature_cols = prepare_features(df)
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target distribution: {np.bincount(y.astype(int))}")
    print(f"Class balance: {np.mean(y):.2%} Team 1 wins")

    # Train-test split
    print(f"\nSplitting data: {100*(1-test_size):.0f}% train, {100*test_size:.0f}% test")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"Train set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")

    # Train XGBoost model
    print("\nTraining XGBoost model...")
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        eval_metric='logloss',
        verbosity=0,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    # Predictions
    print("\nMaking predictions...")
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_train_pred_proba = model.predict_proba(X_train)[:, 1]
    y_test_pred_proba = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    print("\n" + "-"*100)
    print("TRAINING METRICS")
    print("-"*100)
    train_acc = accuracy_score(y_train, y_train_pred)
    train_prec = precision_score(y_train, y_train_pred)
    train_rec = recall_score(y_train, y_train_pred)
    train_f1 = f1_score(y_train, y_train_pred)
    train_auc = roc_auc_score(y_train, y_train_pred_proba)

    print(f"Accuracy:  {train_acc:.4f}")
    print(f"Precision: {train_prec:.4f}")
    print(f"Recall:    {train_rec:.4f}")
    print(f"F1 Score:  {train_f1:.4f}")
    print(f"ROC AUC:   {train_auc:.4f}")

    print("\n" + "-"*100)
    print("TEST METRICS")
    print("-"*100)
    test_acc = accuracy_score(y_test, y_test_pred)
    test_prec = precision_score(y_test, y_test_pred)
    test_rec = recall_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    test_auc = roc_auc_score(y_test, y_test_pred_proba)

    print(f"Accuracy:  {test_acc:.4f}")
    print(f"Precision: {test_prec:.4f}")
    print(f"Recall:    {test_rec:.4f}")
    print(f"F1 Score:  {test_f1:.4f}")
    print(f"ROC AUC:   {test_auc:.4f}")

    # Feature importance
    print("\n" + "-"*100)
    print("TOP 10 FEATURE IMPORTANCE")
    print("-"*100)
    importance = model.feature_importances_
    feature_importance = sorted(
        zip(feature_cols, importance),
        key=lambda x: x[1],
        reverse=True
    )
    for i, (feat, imp) in enumerate(feature_importance[:10], 1):
        print(f"{i:2d}. {feat:<25} {imp:.4f}")

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
    print("\n" + "-"*100)
    print("CONFUSION MATRIX (TEST SET)")
    print("-"*100)
    print(f"True Negatives:  {tn:4d}  (Team 1 losses correctly predicted)")
    print(f"False Positives: {fp:4d}  (Team 1 losses wrongly predicted as wins)")
    print(f"False Negatives: {fn:4d}  (Team 1 wins wrongly predicted as losses)")
    print(f"True Positives:  {tp:4d}  (Team 1 wins correctly predicted)")

    results = {
        'model': model,
        'feature_cols': feature_cols,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'y_train_pred': y_train_pred,
        'y_test_pred': y_test_pred,
        'y_train_pred_proba': y_train_pred_proba,
        'y_test_pred_proba': y_test_pred_proba,
        'metrics': {
            'train': {
                'accuracy': train_acc,
                'precision': train_prec,
                'recall': train_rec,
                'f1': train_f1,
                'auc': train_auc,
            },
            'test': {
                'accuracy': test_acc,
                'precision': test_prec,
                'recall': test_rec,
                'f1': test_f1,
                'auc': test_auc,
            }
        },
        'feature_importance': feature_importance,
        'confusion_matrix': {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp},
    }

    return results


def save_model(results: dict, output_dir: str = ".") -> str:
    """
    Save trained model and metadata.

    Args:
        results: Dictionary with model and results
        output_dir: Directory to save model files

    Returns:
        Path to saved model
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(output_dir, f"xgboost_model_{timestamp}.pkl")

    # Save just the model (not the full results dict)
    with open(model_path, 'wb') as f:
        pickle.dump(results['model'], f)

    print(f"\n✓ Model saved to {model_path}")
    return model_path


def load_model(model_path: str) -> xgb.XGBClassifier:
    """
    Load a trained XGBoost model.

    Args:
        model_path: Path to saved model file

    Returns:
        Loaded XGBoost model
    """
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model
