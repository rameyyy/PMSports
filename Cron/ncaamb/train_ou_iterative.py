#!/usr/bin/env python3
"""
Train Over/Under model on feature set with parameter sweep

This script:
1) Loads ou_features.csv
2) Applies data-quality filters
3) Ensures chronological order by date (if 'date' column exists)
4) Runs a parameter sweep over sensible XGBoost settings (chronological split)
   - Prints each combo's Test MAE and the best-so-far MAE + best params
   - Saves full sweep to ou_param_sweep.csv
5) Retrains a final model using the best params
6) Prints feature importance
7) Saves trained model and predictions
"""

import os
import sys
import itertools
from typing import Dict, List

import polars as pl
import numpy as np

# Add models directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "models"))

from models.ou_model import OUModel  # noqa: E402


# ---------------------------
# Data-quality filters
# ---------------------------
def apply_data_quality_filters(df: pl.DataFrame) -> (pl.DataFrame, Dict):
    """
    Apply data quality filters to remove low-quality rows

    Filters:
    1) Remove rows where avg_ou_line is null (no odds data)
    2) Remove rows where num_books_with_ou < 2 (need at least 2 bookmakers)
    """
    initial_count = len(df)
    stats = {"initial": initial_count}

    # Filter 1: Remove null avg_ou_line (no odds data)
    df = df.filter(pl.col("avg_ou_line").is_not_null())
    stats["after_avg_ou_line"] = len(df)

    # Filter 2: Remove rows with < 2 bookmakers for O/U
    df = df.filter(pl.col("num_books_with_ou") >= 2)
    stats["after_num_books"] = len(df)

    # Calculate filtered out
    stats["filtered_out"] = initial_count - len(df)
    stats["percent_retained"] = (len(df) / initial_count * 100) if initial_count > 0 else 0.0
    return df, stats


# ---------------------------
# Parameter sweep
# ---------------------------
def param_sweep(features_df: pl.DataFrame, base_random_state: int = 42) -> List[Dict]:
    """
    Run a compact grid of XGBoost params using OUModel.train()
    Maintains the chronological split done inside train() via test_size.

    Returns:
        A list of dicts (one per run) sorted by test_mae ascending.
    """
    # IMPORTANT: Do NOT include 'random_state' in the grid; it's passed once in the call
    grid = {
        "learning_rate":    [0.03, 0.05, 0.08],
        "max_depth":        [4, 6, 8],
        "min_child_weight": [1, 3, 6],
        "subsample":        [0.6, 0.8],
        "colsample_bytree": [0.6, 0.8],
        "reg_alpha":        [0.0, 0.5, 1.0],
        "reg_lambda":       [0.5, 1.0, 2.0],
        "n_estimators":     [400, 800],   # upper bound; early stopping will cut it
        "tree_method":      ["hist"],
        # no random_state here
    }

    keys = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))
    total = len(combos)

    print(f"\nPARAM SWEEP: evaluating {total} combinations...\n")

    rows = []
    best_row = None
    best_test_mae = float("inf")

    for i, values in enumerate(combos, 1):
        params = dict(zip(keys, values))

        m = OUModel()
        metrics = m.train(
            features_df,
            test_size=0.2,
            random_state=base_random_state,  # pass it ONCE here
            **params
        )

        row = {
            "run": i,
            **params,
            "train_mae": float(metrics["train_mae"]),
            "test_mae":  float(metrics["test_mae"]),
            "train_rmse": float(metrics["train_rmse"]),
            "test_rmse":  float(metrics["test_rmse"]),
            "n_train": int(metrics["n_train"]),
            "n_test":  int(metrics["n_test"]),
        }
        rows.append(row)

        # Per-combo printout
        print(
            f"[{i:>4}/{total}] Test MAE={row['test_mae']:.3f} | "
            f"lr={params['learning_rate']} depth={params['max_depth']} mcw={params['min_child_weight']} "
            f"sub={params['subsample']} col={params['colsample_bytree']} a={params['reg_alpha']} "
            f"l={params['reg_lambda']} n_est={params['n_estimators']}"
        )

        # Track best so far
        if row["test_mae"] < best_test_mae:
            best_test_mae = row["test_mae"]
            best_row = row
            # Print best-so-far snapshot with full params
            print("    ➜ NEW BEST so far:")
            print(f"      test_mae={best_test_mae:.3f}")
            print("      best_params={")
            for k in keys:
                print(f"        {k}: {row[k]}")
            print("      }")

    # Rank by test MAE
    rows_sorted = sorted(rows, key=lambda r: r["test_mae"])

    # Save full sweep
    pl.DataFrame(rows_sorted).write_csv("ou_param_sweep.csv")
    print("\nSaved full sweep results -> ou_param_sweep.csv")

    # Final best recap
    if best_row is not None:
        print("\nBEST COMBINATION (by Test MAE):")
        print(f"  test_mae={best_row['test_mae']:.3f}, train_mae={best_row['train_mae']:.3f}")
        for k in keys:
            print(f"  {k}: {best_row[k]}")

    return rows_sorted


def print_top20_feature_importance(model: OUModel):
    try:
        importance = model.get_feature_importance(top_n=20)
    except Exception as e:
        print(f"   (Skipping feature importance; error: {e})")
        return

    if not hasattr(model, "feature_names") or model.feature_names is None:
        print("   (No feature_names found; cannot map indices to names.)")
        return

    feature_name_map = {f"f{i}": model.feature_names[i] for i in range(len(model.feature_names))}

    print("\n5. Top 20 Most Important Features:")
    i = 1
    for feat, score in importance.items():
        feat_name = feature_name_map.get(feat, str(feat))
        print(f"   {i:2d}. {feat_name:<50s} {score:>8.1f}")
        i += 1


# ---------------------------
# Main
# ---------------------------
def main():
    print("=" * 80)
    print("TRAINING OVER/UNDER MODEL (with parameter sweep)")
    print("=" * 80)

    # 1) Load features
    print("\n1. Loading features...")
    features_path = "ou_features.csv"
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"Could not find {features_path} in working directory.")
    features_df = pl.read_csv(features_path)
    print(f"   Loaded {len(features_df)} games with {len(features_df.columns)} columns")

    # 2) Apply data quality filters
    print("\n2. Applying data quality filters...")
    features_df, filter_stats = apply_data_quality_filters(features_df)
    print(f"   Initial games: {filter_stats['initial']}")
    print(f"   After removing null avg_ou_line: {filter_stats['after_avg_ou_line']}")
    print(f"   After requiring >=2 bookmakers: {filter_stats['after_num_books']}")
    print(f"   Final games: {len(features_df)}")
    print(f"   Filtered out: {filter_stats['filtered_out']} ({100 - filter_stats['percent_retained']:.1f}%)")
    print(f"   Retained: {filter_stats['percent_retained']:.1f}%")

    # 2b) Ensure chronological order by date if present
    if "date" in features_df.columns:
        try:
            features_df = features_df.with_columns(
                pl.col("date").str.strptime(pl.Date, strict=False)
            ).sort("date")
            print("   Sorted by date for chronological split.")
        except Exception:
            # If parsing fails, at least do a lexical sort to be consistent
            features_df = features_df.sort("date")
            print("   Sorted lexically by 'date' (parse failed).")

    # 3) Parameter sweep
    print("\n3. Running parameter sweep...")
    sweep_results = param_sweep(features_df, base_random_state=42)
    best = sweep_results[0]
    best_params = {k: best[k] for k in [
        "learning_rate", "max_depth", "min_child_weight",
        "subsample", "colsample_bytree", "reg_alpha", "reg_lambda",
        "n_estimators", "tree_method"
    ]}
    print("\nBest combo selected for final training:")
    for k, v in best_params.items():
        print(f"   {k}: {v}")

    # 4) Final train with best params
    print("\n4. Training final model with best params...")
    model = OUModel()
    metrics = model.train(
        features_df,
        test_size=0.2,
        random_state=42,  # single pass
        **best_params
    )

    print("\n   Final Training Results:")
    print(f"   - Training MAE: {metrics['train_mae']:.2f} points")
    print(f"   - Test MAE: {metrics['test_mae']:.2f} points")
    print(f"   - Training RMSE: {metrics['train_rmse']:.2f} points")
    print(f"   - Test RMSE: {metrics['test_rmse']:.2f} points")
    print(f"   - Train games: {metrics['n_train']}")
    print(f"   - Test games: {metrics['n_test']}")

    # 5) Feature importance
    print_top20_feature_importance(model)

    # 6) Save model
    print("\n6. Saving model...")
    model_path = "ou_model.json"
    try:
        model.save_model(model_path)
        print(f"   Saved model -> {model_path}")
    except Exception as e:
        print(f"   (Skipping save; error: {e})")

    # 7) Predictions on filtered games
    print("\n7. Making predictions on filtered games...")
    predictions = model.predict(features_df)

    print("\n   Sample Predictions (first 10 games):")
    print("   " + "-" * 100)
    print(f"   {'Game ID':<35} {'Team 1':<15} {'Team 2':<15} {'Actual':<8} {'Predicted':<10} {'Error':<8}")
    print("   " + "-" * 100)

    for i in range(min(10, len(predictions["game_id"]))):
        game_id = predictions["game_id"][i]
        t1 = predictions["team_1"][i][:12] if predictions["team_1"][i] else ""
        t2 = predictions["team_2"][i][:12] if predictions["team_2"][i] else ""
        actual = predictions["actual_total"][i]
        pred = predictions["predicted_total"][i]
        err = predictions["prediction_error"][i]

        if actual is not None and not (isinstance(actual, float) and np.isnan(actual)):
            print(f"   {game_id:<35} {t1:<15} {t2:<15} {actual:>7.1f} {pred:>9.1f} {err:>7.1f}")
        else:
            print(f"   {game_id:<35} {t1:<15} {t2:<15} {'N/A':>7} {pred:>9.1f} {'N/A':>7}")

    # 8) Save predictions
    print("\n8. Saving predictions to CSV...")
    try:
        pred_df = pl.DataFrame({
            "game_id": predictions["game_id"],
            "date": predictions["date"],
            "team_1": predictions["team_1"],
            "team_2": predictions["team_2"],
            "actual_total": predictions["actual_total"],
            "predicted_total": predictions["predicted_total"],
            "prediction_error": predictions["prediction_error"],
        })
        pred_df.write_csv("ou_predictions.csv")
        print("   Saved to ou_predictions.csv")
    except Exception as e:
        print(f"   (Skipping prediction CSV; error: {e})")

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"\nModel: {model_path}")
    print("Sweep: ou_param_sweep.csv")
    print("Predictions: ou_predictions.csv")


if __name__ == "__main__":
    main()
