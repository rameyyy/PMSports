"""Ensemble: XGBoost + LightGBM averaged probabilities.

Each model is independently tuned via Optuna with its own native
hyperparameter space. XGBoost params from best_params.json,
LightGBM params from best_params_lgb.json. Final prediction =
simple average of the two probability outputs.
"""
import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")

import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).parent))
from train import load, time_split, mirror_augment

OUT = Path(__file__).parent


def train_xgb(X_train, y_train, X_test, y_test, params: dict) -> np.ndarray:
    model = xgb.XGBClassifier(
        **params,
        n_estimators          = 2000,
        eval_metric           = "logloss",
        early_stopping_rounds = 30,
        random_state          = 42,
        n_jobs                = -1,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=100)
    print(f"  XGB best iteration: {model.best_iteration}")
    return model.predict_proba(X_test)[:, 1]


def train_lgb(X_train, y_train, X_test, y_test, params: dict) -> np.ndarray:
    final_params = dict(
        **params,
        objective      = "binary",
        metric         = "binary_logloss",
        subsample_freq = 1,
        n_estimators   = 2000,
        verbosity      = -1,
        random_state   = 42,
        n_jobs         = -1,
    )
    model = lgb.LGBMClassifier(**final_params)
    model.fit(
        X_train, y_train,
        eval_set  = [(X_test, y_test)],
        callbacks = [lgb.early_stopping(30, verbose=False), lgb.log_evaluation(100)],
    )
    print(f"  LGB best iteration: {model.best_iteration_}")
    return model.predict_proba(X_test)[:, 1]


def evaluate(name: str, proba: np.ndarray, y_test: np.ndarray):
    acc = accuracy_score(y_test, (proba >= 0.5).astype(int))
    auc = roc_auc_score(y_test, proba)
    ll  = log_loss(y_test, proba)
    print(f"  {name:<12} test={acc:.4f} ({acc*100:.1f}%)  AUC={auc:.4f}  logloss={ll:.4f}")
    return acc, auc, ll


def ensemble():
    for path, label in [(OUT / "best_params.json", "XGBoost"), (OUT / "best_params_lgb.json", "LightGBM")]:
        if not path.exists():
            print(f"{path.name} not found — run tune.py / tune_lgb.py first.")
            return

    with open(OUT / "best_params.json") as f:
        xgb_params = json.load(f)
    with open(OUT / "best_params_lgb.json") as f:
        lgb_params = json.load(f)

    print("XGBoost params :", {k: round(v, 4) if isinstance(v, float) else v for k, v in xgb_params.items()})
    print("LightGBM params:", {k: round(v, 4) if isinstance(v, float) else v for k, v in lgb_params.items()})
    print()

    X, y, feature_cols, dates = load()
    X_train, X_test, y_train, y_test = time_split(X, y, dates)
    X_train_aug, y_train_aug = mirror_augment(X_train, y_train, feature_cols)

    print(f"Train : {len(y_train_aug):,} rows (augmented)")
    print(f"Test  : {len(y_test):,} rows (held out)\n")

    print("Training XGBoost...")
    xgb_proba = train_xgb(X_train_aug, y_train_aug, X_test, y_test, xgb_params)

    print("\nTraining LightGBM...")
    lgb_proba = train_lgb(X_train_aug, y_train_aug, X_test, y_test, lgb_params)

    ensemble_proba = (xgb_proba + lgb_proba) / 2.0

    print(f"\n{'='*60}")
    print("Individual model results:")
    evaluate("XGBoost",  xgb_proba,      y_test)
    evaluate("LightGBM", lgb_proba,      y_test)
    print()
    print("Ensemble (equal average):")
    ens_acc, ens_auc, ens_ll = evaluate("Ensemble", ensemble_proba, y_test)
    print(f"\nFinal — Test accuracy: {ens_acc*100:.1f}%  AUC: {ens_auc:.4f}  log-loss: {ens_ll:.4f}")
    print(f"{'='*60}\n")

    results = {
        "test_accuracy": float(ens_acc),
        "test_auc":      float(ens_auc),
        "test_logloss":  float(ens_ll),
        "xgb_auc":       float(roc_auc_score(y_test, xgb_proba)),
        "lgb_auc":       float(roc_auc_score(y_test, lgb_proba)),
    }
    with open(OUT / "ensemble_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {OUT / 'ensemble_results.json'}")


if __name__ == "__main__":
    ensemble()
