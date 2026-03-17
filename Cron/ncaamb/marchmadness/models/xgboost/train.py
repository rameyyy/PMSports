#!/usr/bin/env python3
"""
March Madness — XGBoost binary classifier
Predicts P(team_1 beats team_2).
Bayesian optimization (150 iterations) minimizing log loss on held-out year.
Saves model + metadata to this directory.

Run from ncaamb/ directory:
    python marchmadness/models/xgboost/train.py
"""

import sys
from pathlib import Path

import xgboost as xgb
from sklearn.metrics import log_loss, accuracy_score, roc_auc_score
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args

here         = Path(__file__).parent
marchmadness = here.parent.parent
ncaamb_dir   = marchmadness.parent
sys.path.insert(0, str(ncaamb_dir))
sys.path.insert(0, str(marchmadness))

from data_utils import load_all_features, get_feature_columns, build_splits, build_full_dataset

N_CALLS = 150

space = [
    Integer(3,   8,    name='max_depth'),
    Real(0.01,   0.3,  prior='log-uniform', name='learning_rate'),
    Integer(100, 800,  name='n_estimators'),
    Real(1.0,    10.0, name='min_child_weight'),
    Real(0.5,    1.0,  name='subsample'),
    Real(0.5,    1.0,  name='colsample_bytree'),
    Real(0.0,    10.0, name='reg_alpha'),
    Real(0.0,    10.0, name='reg_lambda'),
]


def run_optimization(X_train, y_train, X_test, y_test, sample_weights):
    iter_count = [0]

    @use_named_args(space)
    def objective(**params):
        iter_count[0] += 1
        print(f"  [{iter_count[0]}/{N_CALLS}] depth={params['max_depth']} "
              f"lr={params['learning_rate']:.4f} n={params['n_estimators']}", end="  ")

        model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            random_state=42,
            tree_method='hist',
            verbosity=0,
            use_label_encoder=False,
            **params,
        )
        model.fit(X_train, y_train, sample_weight=sample_weights,
                  eval_set=[(X_test, y_test)], verbose=False)

        train_ll = log_loss(y_train, model.predict_proba(X_train)[:, 1])
        test_ll  = log_loss(y_test,  model.predict_proba(X_test)[:, 1])
        test_acc = accuracy_score(y_test, model.predict(X_test))
        overfit  = max(0, test_ll - train_ll - 0.02) * 0.5
        score    = test_ll + overfit
        print(f"train_ll={train_ll:.4f}  test_ll={test_ll:.4f}  acc={test_acc:.3f}  score={score:.4f}")
        return score

    result = gp_minimize(objective, space, n_calls=N_CALLS, random_state=42, verbose=False)
    best = {s.name: v for s, v in zip(space, result.x)}
    print(f"\n  Best score: {result.fun:.4f}")
    print(f"  Best params: {best}\n")
    return best, result.fun


def train_final(X_all, y_all, weights_all, best_params):
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=42,
        tree_method='hist',
        verbosity=0,
        use_label_encoder=False,
        **best_params,
    )
    model.fit(X_all, y_all, sample_weight=weights_all)
    proba = model.predict_proba(X_all)[:, 1]
    ll  = log_loss(y_all, proba)
    acc = accuracy_score(y_all, model.predict(X_all))
    auc = roc_auc_score(y_all, proba)
    return model, ll, acc, auc


def main():
    print("\n" + "=" * 70)
    print("MARCH MADNESS — XGBOOST  (P(team_1 wins))")
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

    print(f"STEP 4: Bayesian optimization ({N_CALLS} iterations)")
    print("-" * 70)
    best_params, best_score = run_optimization(X_train, y_train, X_test, y_test, weights)

    print("STEP 5: Final training on all data")
    print("-" * 70)
    model, ll, acc, auc = train_final(X_all, y_all, weights_all, best_params)
    print(f"  Log Loss: {ll:.4f}  Accuracy: {acc:.4f}  AUC: {auc:.4f}\n")

    print("STEP 6: Saving")
    print("-" * 70)

    model_path = here / "xgboost_model.json"
    model.save_model(str(model_path))
    print(f"  ✅ Model    → {model_path}")

    feat_path = here / "feature_columns.txt"
    feat_path.write_text('\n'.join(feature_cols))
    print(f"  ✅ Features → {feat_path}")

    params_path = here / "hyperparameters.txt"
    with open(params_path, 'w') as f:
        f.write(f"March Madness XGBoost — features{years[0]}-{years[-1]}\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Target:         P(team_1 wins)\n")
        f.write(f"Training games: {len(X_all)}\n")
        f.write(f"Feature count:  {len(feature_cols)}\n")
        f.write(f"Best opt score: {best_score:.4f}\n\n")
        for k, v in best_params.items():
            f.write(f"{k}: {v}\n")
        f.write(f"\nFinal Log Loss: {ll:.4f}\n")
        f.write(f"Final Accuracy: {acc:.4f}\n")
        f.write(f"Final AUC:      {auc:.4f}\n")
    print(f"  ✅ Params   → {params_path}\n")

    print("=" * 70)
    print(f"DONE  LogLoss={ll:.4f}  Accuracy={acc:.4f}  AUC={auc:.4f}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
