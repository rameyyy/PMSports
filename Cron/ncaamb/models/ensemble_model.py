# models/ensemble_model.py

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np
import polars as pl

from models.ou_model import OUModel
from models.lgb_model import LGBModel


class EnsembleModel:
    """
    Ensemble of XGBoost and LightGBM models.
    Combines predictions using weighted averaging (default: equal weights).
    """
    def __init__(self, xgb_weight: float = 0.5, lgb_weight: float = 0.5) -> None:
        self.xgb_model: Optional[OUModel] = None
        self.lgb_model: Optional[LGBModel] = None
        self.xgb_weight = xgb_weight
        self.lgb_weight = lgb_weight
        self.is_trained: bool = False

    # Normalize weights
        total_weight = xgb_weight + lgb_weight
        self.xgb_weight = xgb_weight / total_weight
        self.lgb_weight = lgb_weight / total_weight

    # -----------------------------------------
    # Training
    # -----------------------------------------
    def train(
        self,
        features_df: pl.DataFrame,
        test_size: float = 0.2,
        xgb_params: Optional[Dict] = None,
        lgb_params: Optional[Dict] = None,
    ) -> Dict:
        """
        Train both XGBoost and LightGBM models.

        Args:
            features_df: Polars DataFrame with computed features
            test_size: Fraction of data to use for validation
            xgb_params: XGBoost hyperparameters
            lgb_params: LightGBM hyperparameters

        Returns:
            Dictionary with training metrics for both models
        """
        if xgb_params is None:
            xgb_params = {}
        if lgb_params is None:
            lgb_params = {}

        # Train XGBoost
        print("   Training XGBoost model...")
        self.xgb_model = OUModel()
        xgb_metrics = self.xgb_model.train(features_df, test_size=test_size, **xgb_params)

        # Train LightGBM
        print("   Training LightGBM model...")
        self.lgb_model = LGBModel()
        lgb_metrics = self.lgb_model.train(features_df, test_size=test_size, **lgb_params)

        self.is_trained = True

        # Return ensemble metrics
        ensemble_test_mae = (
            self.xgb_weight * xgb_metrics["test_mae"] +
            self.lgb_weight * lgb_metrics["test_mae"]
        )

        return {
            "xgb_train_mae": xgb_metrics["train_mae"],
            "xgb_test_mae": xgb_metrics["test_mae"],
            "lgb_train_mae": lgb_metrics["train_mae"],
            "lgb_test_mae": lgb_metrics["test_mae"],
            "ensemble_test_mae": ensemble_test_mae,
            "n_train": xgb_metrics["n_train"],
            "n_test": xgb_metrics["n_test"],
            "xgb_weight": self.xgb_weight,
            "lgb_weight": self.lgb_weight,
        }

    # -----------------------------------------
    # Prediction
    # -----------------------------------------
    def predict(self, features_df: pl.DataFrame) -> Dict:
        """
        Make ensemble predictions by averaging XGBoost and LightGBM predictions.
        """
        if not self.is_trained or self.xgb_model is None or self.lgb_model is None:
            raise ValueError("Both models must be trained before making predictions")

        # Get predictions from both models
        xgb_pred = self.xgb_model.predict(features_df)
        lgb_pred = self.lgb_model.predict(features_df)

        # Ensemble predictions (weighted average)
        xgb_pred_array = np.array(xgb_pred["predicted_total"])
        lgb_pred_array = np.array(lgb_pred["predicted_total"])
        ensemble_pred = (self.xgb_weight * xgb_pred_array) + (self.lgb_weight * lgb_pred_array)

        # Ensemble error
        actual = np.array(xgb_pred["actual_total"], dtype=float)
        valid_actual = ~np.isnan(actual)

        return {
            "game_id": xgb_pred["game_id"],
            "date": xgb_pred["date"],
            "team_1": xgb_pred["team_1"],
            "team_2": xgb_pred["team_2"],
            "predicted_total": ensemble_pred.tolist(),
            "xgb_predicted_total": xgb_pred["predicted_total"],
            "lgb_predicted_total": lgb_pred["predicted_total"],
            "actual_total": xgb_pred["actual_total"],
            "prediction_error": [
                float(actual[i] - ensemble_pred[i]) if valid_actual[i] else None
                for i in range(len(ensemble_pred))
            ],
        }

    # -----------------------------------------
    # Utilities
    # -----------------------------------------
    def get_feature_importance_comparison(self, top_n: int = 20) -> Dict[str, Dict]:
        """
        Compare feature importance between both models.
        """
        if self.xgb_model is None or self.lgb_model is None:
            raise ValueError("Models must be trained first")

        xgb_importance = self.xgb_model.get_feature_importance(top_n=top_n)
        lgb_importance = self.lgb_model.get_feature_importance(top_n=top_n)

        return {
            "xgb": xgb_importance,
            "lgb": lgb_importance,
        }
