#!/usr/bin/env python3
"""
March Madness — Logistic Regression model
Predicts P(team_1 beats team_2).
C (inverse regularization) tuned via 5-fold LogisticRegressionCV.
Features are StandardScaler-normalized before training.
Saves model, scaler, and metadata to this directory.

Run from ncaamb/ directory:
    python marchmadness/models/ridge/train.py
"""

import sys
import pickle
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import log_loss, accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

here         = Path(__file__).parent
marchmadness = here.parent.parent
ncaamb_dir   = marchmadness.parent
sys.path.insert(0, str(ncaamb_dir))
sys.path.insert(0, str(marchmadness))

from data_utils import load_all_features, get_feature_columns, build_splits, build_full_dataset


def tune_and_train(X_all, y_all, weights_all, X_test, y_test):
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)
    X_test_s = scaler.transform(X_test)

    print("  Running LogisticRegressionCV (10 Cs, 5-fold CV)...")
    Cs = np.logspace(-3, 3, 10)
    lr_cv = LogisticRegressionCV(
        Cs=Cs, cv=5, scoring='neg_log_loss',
        max_iter=1000, random_state=42, n_jobs=-1,
    )
    lr_cv.fit(X_scaled, y_all, sample_weight=weights_all)
    best_C = lr_cv.C_[0]
    print(f"  Best C: {best_C:.4f}")

    lr = LogisticRegression(C=best_C, max_iter=1000, random_state=42)
    lr.fit(X_scaled, y_all, sample_weight=weights_all)

    train_proba = lr.predict_proba(X_scaled)[:, 1]
    test_proba  = lr.predict_proba(X_test_s)[:, 1]

    train_ll  = log_loss(y_all,  train_proba)
    test_ll   = log_loss(y_test, test_proba)
    train_acc = accuracy_score(y_all,  lr.predict(X_scaled))
    test_acc  = accuracy_score(y_test, lr.predict(X_test_s))
    auc       = roc_auc_score(y_all, train_proba)

    print(f"  Train  — LogLoss: {train_ll:.4f}  Acc: {train_acc:.4f}")
    print(f"  Test   — LogLoss: {test_ll:.4f}  Acc: {test_acc:.4f}  AUC: {auc:.4f}")
    return lr, scaler, best_C, train_ll, test_ll, train_acc, test_acc, auc


def main():
    print("\n" + "=" * 70)
    print("MARCH MADNESS — LOGISTIC REGRESSION  (P(team_1 wins))")
    print("=" * 70 + "\n")

    print("STEP 1: Loading data")
    print("-" * 70)
    df = load_all_features()

    print("STEP 2: Feature columns")
    print("-" * 70)
    feature_cols = get_feature_columns(df)
    print(f"  {len(feature_cols)} feature columns\n")

    print("STEP 3: Train/test split")
    print("-" * 70)
    X_train, y_train, X_test, y_test, weights, years = build_splits(df, feature_cols)
    X_all, y_all, weights_all = build_full_dataset(df, feature_cols, years)
    print()

    print("STEP 4: Tune C + train on all data")
    print("-" * 70)
    lr, scaler, best_C, train_ll, test_ll, train_acc, test_acc, auc = tune_and_train(
        X_all, y_all, weights_all, X_test, y_test
    )
    print()

    print("STEP 5: Saving")
    print("-" * 70)

    model_path = here / "logistic_model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump({'model': lr, 'scaler': scaler}, f)
    print(f"  ✅ Model    → {model_path}")

    feat_path = here / "feature_columns.txt"
    feat_path.write_text('\n'.join(feature_cols))
    print(f"  ✅ Features → {feat_path}")

    params_path = here / "hyperparameters.txt"
    with open(params_path, 'w') as f:
        f.write(f"March Madness Logistic Regression — features{years[0]}-{years[-1]}\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Target:         P(team_1 wins)\n")
        f.write(f"Training games: {len(X_all)}\n")
        f.write(f"Feature count:  {len(feature_cols)}\n\n")
        f.write(f"C: {best_C:.4f}\n\n")
        f.write(f"Train LogLoss:  {train_ll:.4f}\n")
        f.write(f"Test  LogLoss:  {test_ll:.4f}\n")
        f.write(f"Train Accuracy: {train_acc:.4f}\n")
        f.write(f"Test  Accuracy: {test_acc:.4f}\n")
        f.write(f"AUC:            {auc:.4f}\n")
    print(f"  ✅ Params   → {params_path}\n")

    print("=" * 70)
    print(f"DONE  TestLogLoss={test_ll:.4f}  TestAcc={test_acc:.4f}  AUC={auc:.4f}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
