# models/cat_model.py

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np
import polars as pl
import catboost as cb


class CatModel:
    def __init__(self) -> None:
        self.model: Optional[cb.CatBoostRegressor] = None
        self.feature_names: Optional[List[str]] = None
        self.is_trained: bool = False

    # -----------------------------
    # Feature preparation
    # -----------------------------
    def prepare_features(
        self, features_df: pl.DataFrame
    ) -> Tuple[np.ndarray, List[str], Dict[str, List]]:
        """
        Build X (numpy), feature_cols (list), and game_info (dict of lists).
        Assumes target column is 'actual_total' and identifier columns exist.
        Uses all numeric columns except the target as features.
        """
        if not isinstance(features_df, pl.DataFrame):
            raise TypeError("features_df must be a Polars DataFrame")

        target_col = "actual_total"
        df = features_df

        # Best-effort: cast string-ish numeric columns to floats
        for c in df.columns:
            if c == target_col:
                continue
            if df[c].dtype == pl.String:
                try:
                    df = df.with_columns(pl.col(c).cast(pl.Float64).fill_null(strategy="forward"))
                except Exception:
                    pass

        numeric_types = (pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int16, pl.Int8)
        numeric_cols = [c for c, dt in zip(df.columns, df.dtypes) if dt in numeric_types]

        # Exclude target and any score-related columns that would cause data leakage
        # Also exclude metadata columns (identifiers, not features)
        exclude_cols = {target_col, 'team_1_score', 'team_2_score', 'game_id', 'date', 'team_1', 'team_2'}
        feature_cols = [c for c in numeric_cols if c not in exclude_cols]

        X = df.select(feature_cols).to_numpy()
        game_info = {
            "game_id": df["game_id"].to_list() if "game_id" in df.columns else [None] * len(df),
            "date": df["date"].to_list() if "date" in df.columns else [None] * len(df),
            "team_1": df["team_1"].to_list() if "team_1" in df.columns else [None] * len(df),
            "team_2": df["team_2"].to_list() if "team_2" in df.columns else [None] * len(df),
            "actual_total": (
                df[target_col].to_list() if target_col in df.columns else [np.nan] * len(df)
            ),
        }
        return X, feature_cols, game_info

    # -----------------------------
    # Training
    # -----------------------------
    def train(
        self,
        features_df: pl.DataFrame,
        test_size: float = 0.2,
        **cat_params
    ) -> Dict:
        """
        Train CatBoost model on features.

        Args:
            features_df: Polars DataFrame with computed features
            test_size: Fraction of data to use for validation (chronological split on most recent)
            **cat_params: CatBoost parameters

        Returns:
            Dictionary with training metrics
        """
        # Sort by date (assuming 'date' column exists) to ensure chronological order
        if 'date' in features_df.columns:
            features_df = features_df.sort('date')

        # Prepare features/target
        X, feature_cols, game_info = self.prepare_features(features_df)
        y = np.array(game_info["actual_total"], dtype=float)

        # Drop rows with NaN targets
        valid_idx = ~np.isnan(y)
        X = X[valid_idx]
        y = y[valid_idx]

        # Chronological split: train on older games, test on most recent games
        n_samples = len(X)
        n_test = int(n_samples * test_size)
        n_train = max(1, n_samples - n_test)

        X_train, X_test = X[:n_train], X[n_train:]
        y_train, y_test = y[:n_train], y[n_train:]

        # Base params
        default_params = {
            "loss_function": "MAE",
            "eval_metric": "MAE",
            "learning_rate": 0.05,
            "depth": 6,
            "iterations": 400,
            "min_child_samples": 1,
            "subsample": 0.7,
            "colsample_bylevel": 0.7,
            "l2_leaf_reg": 3.0,
            "random_state": 42,
            "verbose": False,
        }

        # Merge overrides
        default_params.update(cat_params)

        # Build model
        self.model = cb.CatBoostRegressor(**default_params)

        # Fit
        self.model.fit(
            X_train,
            y_train,
            eval_set=(X_test, y_test),
            early_stopping_rounds=50,
            verbose=False
        )

        self.feature_names = feature_cols
        self.is_trained = True

        # Metrics
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)

        train_mae = float(np.mean(np.abs(y_train - train_pred))) if len(y_train) else float("nan")
        test_mae = float(np.mean(np.abs(y_test - test_pred))) if len(y_test) else float("nan")
        train_rmse = float(np.sqrt(np.mean((y_train - train_pred) ** 2))) if len(y_train) else float("nan")
        test_rmse = float(np.sqrt(np.mean((y_test - test_pred) ** 2))) if len(y_test) else float("nan")

        return {
            "train_mae": train_mae,
            "test_mae": test_mae,
            "train_rmse": train_rmse,
            "test_rmse": test_rmse,
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
        }

    # -----------------------------
    # Prediction
    # -----------------------------
    def predict(self, features_df: pl.DataFrame) -> Dict:
        """
        Predict totals on new data.
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained before making predictions")

        X, feature_cols, game_info = self.prepare_features(features_df)
        predictions = self.model.predict(X)

        actual = np.array(game_info["actual_total"], dtype=float)
        valid_actual = ~np.isnan(actual)

        return {
            "game_id": game_info["game_id"],
            "date": game_info["date"],
            "team_1": game_info["team_1"],
            "team_2": game_info["team_2"],
            "predicted_total": predictions.tolist(),
            "actual_total": game_info["actual_total"],
            "prediction_error": [
                float(actual[i] - predictions[i]) if valid_actual[i] else None
                for i in range(len(predictions))
            ],
        }

    # -----------------------------
    # Utilities
    # -----------------------------
    def get_feature_importance(self, top_n: int = 20) -> Dict[str, float]:
        """
        Return feature importance dict.
        """
        if self.model is None:
            raise ValueError("Model is not trained")

        if hasattr(self.model, "feature_importances_") and self.feature_names:
            arr = list(self.model.feature_importances_)
            pairs = list(zip(self.feature_names, arr))
            pairs.sort(key=lambda kv: kv[1], reverse=True)
            if top_n is not None:
                pairs = pairs[:top_n]
            return dict(pairs)

        return {}

    def save_model(self, path: str) -> None:
        """
        Save the CatBoost model to file.
        """
        if self.model is None:
            raise ValueError("No model to save")
        self.model.save_model(path)
