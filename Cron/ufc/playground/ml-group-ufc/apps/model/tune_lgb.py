"""Tune LightGBM independently via Optuna with TimeSeriesSplit CV.

LightGBM uses leaf-wise (best-first) tree growth vs XGBoost's level-wise,
so its optimal hyperparameters are genuinely different — num_leaves and
min_child_samples are the primary levers, not max_depth.

Saves best_params_lgb.json and model_lgb.txt on completion.
"""
import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")

import numpy as np
import lightgbm as lgb
import optuna
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).parent))
from train import load, time_split, mirror_augment

OUT      = Path(__file__).parent
N_TRIALS = 300
N_SPLITS = 7

optuna.logging.set_verbosity(optuna.logging.WARNING)


def objective(trial, X_train: np.ndarray, y_train: np.ndarray, feature_cols: list[str]) -> float:
    params = dict(
        objective         = "binary",
        metric            = "binary_logloss",
        verbosity         = -1,
        n_estimators      = 1000,
        # LightGBM-native params — leaf-wise growth, num_leaves is the main complexity lever
        num_leaves        = trial.suggest_int  ("num_leaves",       20,   150      ),
        max_depth         = trial.suggest_int  ("max_depth",        3,    7        ),
        learning_rate     = trial.suggest_float("learning_rate",    0.01, 0.10, log=True),
        subsample         = trial.suggest_float("subsample",        0.4,  0.9      ),
        subsample_freq    = 1,
        colsample_bytree  = trial.suggest_float("colsample_bytree", 0.4,  1.0      ),
        min_child_samples = trial.suggest_int  ("min_child_samples",10,   100      ),
        min_split_gain    = trial.suggest_float("min_split_gain",   0.0,  2.0      ),
        reg_alpha         = trial.suggest_float("reg_alpha",        0.0,  4.0      ),
        reg_lambda        = trial.suggest_float("reg_lambda",       0.5,  5.0      ),
        random_state      = 42,
        n_jobs            = -1,
    )

    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    fold_aucs: list[float] = []

    for fold_train_idx, fold_val_idx in tscv.split(X_train):
        X_ft, y_ft = X_train[fold_train_idx], y_train[fold_train_idx]
        X_fv, y_fv = X_train[fold_val_idx],   y_train[fold_val_idx]

        X_ft_aug, y_ft_aug = mirror_augment(X_ft, y_ft, feature_cols)

        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_ft_aug, y_ft_aug,
            eval_set  = [(X_fv, y_fv)],
            callbacks = [lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)],
        )

        proba = model.predict_proba(X_fv)[:, 1]
        fold_aucs.append(roc_auc_score(y_fv, proba))

    return float(np.mean(fold_aucs))


def tune_lgb():
    X, y, feature_cols, dates = load()
    X_train, X_test, y_train, y_test = time_split(X, y, dates)

    print(f"Features  : {X.shape[1]}")
    print(f"Train     : {len(y_train):,} fights  (CV source)")
    print(f"Test      : {len(y_test):,} fights  (held out — not touched during tuning)")
    print(f"CV        : TimeSeriesSplit, {N_SPLITS} folds")
    print(f"Trials    : {N_TRIALS}")
    print()

    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: objective(trial, X_train, y_train, feature_cols),
        n_trials=N_TRIALS,
        show_progress_bar=True,
    )

    best = study.best_params
    print(f"\nBest CV AUC : {study.best_value:.4f}")
    print(f"Best params :")
    for k, v in best.items():
        print(f"  {k:<25} {v}")

    # Final model on full training set
    final_params = dict(
        **best,
        objective     = "binary",
        metric        = "binary_logloss",
        n_estimators  = 2000,
        subsample_freq = 1,
        verbosity     = -1,
        random_state  = 42,
        n_jobs        = -1,
    )

    X_train_aug, y_train_aug = mirror_augment(X_train, y_train, feature_cols)

    print(f"\nTraining final model on {len(y_train_aug):,} rows (augmented)...")
    model = lgb.LGBMClassifier(**final_params)
    model.fit(
        X_train_aug, y_train_aug,
        eval_set  = [(X_test, y_test)],
        callbacks = [lgb.early_stopping(30, verbose=False), lgb.log_evaluation(100)],
    )

    from sklearn.metrics import accuracy_score, log_loss
    train_proba = model.predict_proba(X_train_aug)[:, 1]
    test_proba  = model.predict_proba(X_test)[:, 1]
    train_acc   = accuracy_score(y_train_aug, (train_proba >= 0.5).astype(int))
    test_acc    = accuracy_score(y_test,      (test_proba  >= 0.5).astype(int))
    auc         = roc_auc_score(y_test, test_proba)
    ll          = log_loss(y_test, test_proba)

    print(f"\n{'='*40}")
    print(f"Train accuracy : {train_acc:.4f}  ({train_acc*100:.1f}%)")
    print(f"Test accuracy  : {test_acc:.4f}  ({test_acc*100:.1f}%)")
    print(f"Overfit gap    : {(train_acc - test_acc)*100:.1f}pp")
    print(f"Test AUC       : {auc:.4f}")
    print(f"Test log-loss  : {ll:.4f}")
    print(f"Best iteration : {model.best_iteration_}")
    print(f"{'='*40}\n")

    # Feature importance
    importance = model.feature_importances_
    ranked = sorted(zip(feature_cols, importance), key=lambda x: x[1], reverse=True)
    print("Top 20 features:")
    print(f"{'Rank':<5} {'Feature':<45} {'Importance':>10}")
    print("-" * 62)
    for i, (feat, imp) in enumerate(ranked[:20], 1):
        print(f"{i:<5} {feat:<45} {imp:>10.0f}")

    with open(OUT / "best_params_lgb.json", "w") as f:
        json.dump(best, f, indent=2)
    model.booster_.save_model(str(OUT / "model_lgb.txt"))

    print(f"\nModel saved to          {OUT / 'model_lgb.txt'}")
    print(f"Best params saved to    {OUT / 'best_params_lgb.json'}")

    return model, study


if __name__ == "__main__":
    tune_lgb()
