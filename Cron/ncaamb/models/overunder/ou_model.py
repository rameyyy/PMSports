# models/ou_model.py

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np
import polars as pl
import xgboost as xgb


class OUModel:
    def __init__(self) -> None:
        self.model: Optional[xgb.XGBRegressor] = None
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
        **xgb_params
    ) -> Dict:
        """
        Train XGBoost model on features.

        Args:
            features_df: Polars DataFrame with computed features
            test_size: Fraction of data to use for validation (chronological split on most recent)
            **xgb_params: XGBoost parameters (may include 'random_state')

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
            "objective": "reg:squarederror",
            "eval_metric": "mae",          # put eval_metric in params (not in fit)
            "learning_rate": 0.05,
            "max_depth": 6,
            "min_child_weight": 3,
            "subsample": 0.7,
            "colsample_bytree": 0.7,
            "reg_alpha": 1.0,
            "reg_lambda": 1.0,
            "n_estimators": 400,           # upper bound; early stopping will choose best
            "tree_method": "hist",
            "n_jobs": -1,
        }

        # Resolve random_state once (default 42 if not provided)
        rs = int(xgb_params.pop("random_state", 42))
        default_params["random_state"] = rs

        # Merge remaining overrides
        default_params.update(xgb_params)

        # Add early_stopping_rounds to params for XGBoost 3.x
        default_params["early_stopping_rounds"] = 50

        # Build model
        self.model = xgb.XGBRegressor(**default_params)

        # Fit
        self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
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
        Return feature importance dict (prefer booster 'gain', fallback to sklearn importances).
        """
        if self.model is None:
            raise ValueError("Model is not trained")

        try:
            booster = self.model.get_booster()
            score = booster.get_score(importance_type="gain")  # keys like 'f0','f1',...
            items = sorted(score.items(), key=lambda kv: kv[1], reverse=True)
            if top_n is not None:
                items = items[:top_n]
            return dict(items)
        except Exception:
            pass

        if hasattr(self.model, "feature_importances_") and self.feature_names:
            arr = list(self.model.feature_importances_)
            pairs = list(zip([f"f{i}" for i in range(len(arr))], arr))
            pairs.sort(key=lambda kv: kv[1], reverse=True)
            if top_n is not None:
                pairs = pairs[:top_n]
            return dict(pairs)

        return {}

    def save_model(self, path: str) -> None:
        """
        Save the XGBoost model to file (JSON/UBJ/TXT supported by xgboost).
        """
        if self.model is None:
            raise ValueError("No model to save")
        self.model.save_model(path)
