# combined_model.py

import polars as pl
import numpy as np
from typing import Dict
from .fight_history_model import FightHistoryPredictor
from .attributes_model import AttributePredictor

class CombinedPredictor:
    """
    Combines physical attributes and fight history predictions.
    Uses weighted ensemble based on model strengths.
    """
    
    def __init__(self, hist_weight: float = 0.54, attr_weight: float = 0.46):
        """
        Args:
            hist_weight: Weight for fight history model (0-1)
            attr_weight: Weight for attributes model (0-1)
        """
        self.hist_predictor = FightHistoryPredictor()
        self.attr_predictor = AttributePredictor()
        
        # Normalize weights
        total = hist_weight + attr_weight
        self.hist_weight = hist_weight / total
        self.attr_weight = attr_weight / total
        
    def predict_dataframe(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Predict on a dataframe with both attributes and fight history features.
        
        Expected columns:
        - Attributes: f1_reach_in, f2_reach_in, f1_height_in, f2_height_in, f1_age, f2_age
          OR: f1_dob, f2_dob, fight_date (will calculate ages)
        - History: diff_* features (whatever your fight history model expects)
        """
        # Calculate ages if not present
        if 'f1_age' not in df.columns:
            df = df.with_columns([
                ((pl.col("fight_date") - pl.col("f1_dob")).dt.total_days() / 365.25).alias("f1_age"),
                ((pl.col("fight_date") - pl.col("f2_dob")).dt.total_days() / 365.25).alias("f2_age")
            ])
        
        # Get predictions from both models
        df_with_attr = self.attr_predictor.predict_dataframe(df)
        df_with_both = self.hist_predictor.predict_dataframe(df_with_attr)
        
        # Combine predictions
        combined_probs = []
        
        for row in df_with_both.iter_rows(named=True):
            attr_prob = row['attr_f1_win_prob']
            hist_prob = row['hist_f1_win_prob']
            
            # Weighted average
            combined_prob = (
                self.attr_weight * attr_prob +
                self.hist_weight * hist_prob
            )
            combined_probs.append(combined_prob)
        
        # Add combined predictions
        result_df = df_with_both.with_columns([
            pl.Series('combined_f1_win_prob', combined_probs),
            pl.Series('combined_f2_win_prob', [1 - p for p in combined_probs]),
            pl.Series('combined_predicted_winner', [1 if p > 0.5 else 2 for p in combined_probs]),
            pl.Series('combined_confidence', [abs(p - 0.5) * 2 for p in combined_probs]),
        ])
        
        return result_df
    
    def evaluate_all_models(self, df: pl.DataFrame, actual_winner_col: str = 'winner_id') -> Dict:
        """
        Evaluate all three models (attributes, history, combined) on the same data.
        
        Returns:
            Dictionary with evaluation results for each model
        """
        # Make predictions with all models
        predictions_df = self.predict_dataframe(df)
        
        # Evaluate each model
        attr_results = self.attr_predictor.evaluate_predictions(predictions_df, actual_winner_col)
        hist_results = self.hist_predictor.evaluate_predictions(predictions_df, actual_winner_col)
        combined_results = self._evaluate_combined(predictions_df, actual_winner_col)
        
        return {
            'attributes': attr_results,
            'history': hist_results,
            'combined': combined_results,
            'predictions_df': predictions_df
        }
    
    def _evaluate_combined(self, df: pl.DataFrame, actual_winner_col: str = 'winner_id') -> Dict:
        """Evaluate combined model predictions."""
        
        # Convert winner_id to 1/2 format
        eval_df = df.with_columns([
            pl.when(pl.col(actual_winner_col) == pl.col('fighter1_id'))
              .then(pl.lit(1))
              .when(pl.col(actual_winner_col) == pl.col('fighter2_id'))
              .then(pl.lit(2))
              .otherwise(None)
              .alias('actual_winner')
        ]).filter(pl.col('actual_winner').is_not_null())
        
        # Overall accuracy
        correct = eval_df.filter(
            pl.col('combined_predicted_winner') == pl.col('actual_winner')
        ).height
        total = eval_df.height
        accuracy = correct / total if total > 0 else 0
        
        # Accuracy by confidence level
        high_conf = eval_df.filter(pl.col('combined_confidence') > 0.4)
        high_conf_accuracy = (
            high_conf.filter(pl.col('combined_predicted_winner') == pl.col('actual_winner')).height /
            high_conf.height if high_conf.height > 0 else 0
        )
        
        med_conf = eval_df.filter(
            (pl.col('combined_confidence') > 0.2) & (pl.col('combined_confidence') <= 0.4)
        )
        med_conf_accuracy = (
            med_conf.filter(pl.col('combined_predicted_winner') == pl.col('actual_winner')).height /
            med_conf.height if med_conf.height > 0 else 0
        )
        
        low_conf = eval_df.filter(pl.col('combined_confidence') <= 0.2)
        low_conf_accuracy = (
            low_conf.filter(pl.col('combined_predicted_winner') == pl.col('actual_winner')).height /
            low_conf.height if low_conf.height > 0 else 0
        )
        
        return {
            'overall_accuracy': accuracy,
            'total_predictions': total,
            'correct_predictions': correct,
            'high_confidence_accuracy': high_conf_accuracy,
            'high_confidence_count': high_conf.height,
            'medium_confidence_accuracy': med_conf_accuracy,
            'medium_confidence_count': med_conf.height,
            'low_confidence_accuracy': low_conf_accuracy,
            'low_confidence_count': low_conf.height,
        }


def analyze_calibration(predictions_df: pl.DataFrame) -> Dict:
    """
    Analyze how well calibrated the predictions are.
    For each predicted probability (50-100%), show actual accuracy.
    """
    # Convert winner to 1/2 format
    df = predictions_df.with_columns([
        pl.when(pl.col('winner_id') == pl.col('fighter1_id'))
          .then(pl.lit(1))
          .when(pl.col('winner_id') == pl.col('fighter2_id'))
          .then(pl.lit(2))
          .otherwise(None)
          .alias('actual_winner')
    ]).filter(pl.col('actual_winner').is_not_null())
    
    # Get the maximum probability (since we predict the more likely winner)
    df = df.with_columns([
        pl.max_horizontal(['combined_f1_win_prob', 'combined_f2_win_prob']).alias('max_prob'),
        (pl.col('combined_predicted_winner') == pl.col('actual_winner')).alias('correct')
    ])
    
    # Round probability to nearest 1% for grouping
    df = df.with_columns([
        (pl.col('max_prob') * 100).round(0).alias('prob_bucket')
    ])
    
    # Group by probability bucket and calculate accuracy
    calibration = (
        df.group_by('prob_bucket')
        .agg([
            pl.count().alias('count'),
            pl.col('correct').mean().alias('actual_accuracy'),
            pl.col('max_prob').mean().alias('avg_predicted_prob')
        ])
        .sort('prob_bucket')
    )
    
    return calibration


def print_calibration_report(calibration_df: pl.DataFrame):
    """Print a calibration report showing predicted vs actual accuracy."""
    
    print("\n" + "=" * 80)
    print("CALIBRATION ANALYSIS: Predicted Probability vs Actual Accuracy")
    print("=" * 80)
    print(f"\n{'Predicted %':<15} {'Count':<10} {'Actual Accuracy':<20} {'Difference':<15}")
    print("-" * 80)
    
    for row in calibration_df.iter_rows(named=True):
        prob_bucket = int(row['prob_bucket'])
        count = row['count']
        actual_acc = row['actual_accuracy'] * 100
        avg_pred = row['avg_predicted_prob'] * 100
        diff = actual_acc - avg_pred
        
        # Color code the difference
        if abs(diff) < 2:
            status = "✅"  # Well calibrated
        elif abs(diff) < 5:
            status = "⚠️"  # Slightly off
        else:
            status = "❌"  # Poorly calibrated
        
        print(f"{prob_bucket}%{'':<12} {count:<10} {actual_acc:>6.2f}%{'':<13} {diff:>+6.2f}% {status}")
    
    # Calculate overall calibration error
    calibration_error = calibration_df.with_columns([
        (pl.col('actual_accuracy') - pl.col('avg_predicted_prob')).abs().alias('error'),
        pl.col('count')
    ])
    
    weighted_error = (
        calibration_error.select([
            (pl.col('error') * pl.col('count')).sum(),
            pl.col('count').sum()
        ])
    )
    
    total_count = weighted_error.select(pl.col('count')).item()
    total_error = weighted_error.select(pl.col('error')).item()
    mean_calibration_error = total_error / total_count if total_count > 0 else 0
    
    print("\n" + "-" * 80)
    print(f"Mean Calibration Error: {mean_calibration_error * 100:.2f}%")
    print(f"(Lower is better - ideal is 0%)")
    print("=" * 80)


def run_combined_evaluation(df: pl.DataFrame, hist_weight: float = 0.48, attr_weight: float = 0.52):
    """
    Main function to evaluate the combined model.
    
    Args:
        df: DataFrame with all necessary columns (attributes + diff_* features + winner info)
        hist_weight: Weight for fight history model
        attr_weight: Weight for attributes model
    """
    
    # Initialize combined predictor
    predictor = CombinedPredictor(hist_weight=hist_weight, attr_weight=attr_weight)
    
    # Evaluate all models
    results = predictor.evaluate_all_models(df, actual_winner_col='winner_id')
    
    predictions_df = results['predictions_df']
    
    # Analyze calibration
    calibration_df = analyze_calibration(predictions_df)
    
    # Print calibration report
    print_calibration_report(calibration_df)
    
    # Save predictions
    predictions_df.write_csv('combined_model_predictions.csv')
    
    return results


def run(df):
    """Wrapper function for easy import."""
    results = run_combined_evaluation(df)
    return results
