# attributes_model.py

import polars as pl
import numpy as np
from typing import Dict, Tuple

class AttributePredictor:
    """
    Predicts fight outcomes based on physical attributes (reach, height, age).
    Uses empirical win rates from historical data.
    """
    
    def __init__(self):
        # These lookup tables are derived from your analysis
        self.reach_win_rates = {
            -7: 0.526, -6: 0.412, -5: 0.414, -4: 0.459, -3: 0.545,
            -2: 0.540, -1: 0.527, 0: 0.476, 1: 0.518, 2: 0.485,
            3: 0.538, 4: 0.575, 5: 0.527, 6: 0.611, 7: 0.464, 8: 0.850
        }
        
        self.height_win_rates = {
            -5: 0.429, -4: 0.500, -3: 0.485, -2: 0.523, -1: 0.485,
            0: 0.512, 1: 0.515, 2: 0.556, 3: 0.524, 4: 0.580,
            5: 0.595, 6: 0.800
        }
        
        self.age_diff_win_rates = {
            -14: 0.789, -13: 0.800, -12: 0.750, -11: 0.750, -10: 0.714,
            -9: 0.597, -8: 0.643, -7: 0.634, -6: 0.613, -5: 0.562,
            -4: 0.579, -3: 0.573, -2: 0.523, -1: 0.538, 0: 0.517,
            1: 0.462, 2: 0.435, 3: 0.452, 4: 0.484, 5: 0.355,
            6: 0.397, 7: 0.373, 8: 0.294, 9: 0.290, 10: 0.400, 11: 0.583
        }
        
        self.age_performance = {
            23: 0.524, 24: 0.632, 25: 0.635, 26: 0.563, 27: 0.571,
            28: 0.550, 29: 0.576, 30: 0.509, 31: 0.546, 32: 0.511,
            33: 0.514, 34: 0.449, 35: 0.488, 36: 0.426, 37: 0.475,
            38: 0.362, 39: 0.376, 40: 0.385, 41: 0.298, 42: 0.429,
            43: 0.235, 44: 0.125
        }
        
        # Weights for each attribute (based on correlation strength)
        # Age has ~3x more predictive power than reach/height
        self.weights = {
            'reach': 0.18,
            'height': 0.26,
            'age_diff': 0.04,
            'age_absolute': 0.52
        }
        
        # Baseline (50/50 when all attributes are equal)
        self.baseline = 0.50
    
    def _get_reach_probability(self, reach_diff: float) -> float:
        """Get win probability based on reach difference."""
        reach_diff_rounded = round(reach_diff)
        
        # Clamp to available data range
        if reach_diff_rounded < -7:
            reach_diff_rounded = -7
        elif reach_diff_rounded > 8:
            reach_diff_rounded = 8
        
        return self.reach_win_rates.get(reach_diff_rounded, self.baseline)
    
    def _get_height_probability(self, height_diff: float) -> float:
        """Get win probability based on height difference."""
        height_diff_rounded = round(height_diff)
        
        # Clamp to available data range
        if height_diff_rounded < -5:
            height_diff_rounded = -5
        elif height_diff_rounded > 6:
            height_diff_rounded = 6
        
        return self.height_win_rates.get(height_diff_rounded, self.baseline)
    
    def _get_age_diff_probability(self, age_diff: float) -> float:
        """Get win probability based on age difference."""
        age_diff_rounded = round(age_diff)
        
        # Clamp to available data range
        if age_diff_rounded < -14:
            age_diff_rounded = -14
        elif age_diff_rounded > 11:
            age_diff_rounded = 11
        
        return self.age_diff_win_rates.get(age_diff_rounded, self.baseline)
    
    def _get_age_performance_factor(self, f1_age: float, f2_age: float) -> float:
        """
        Get relative performance factor based on absolute ages.
        Returns probability that fighter 1 wins based on absolute age performance.
        """
        f1_age_rounded = round(f1_age)
        f2_age_rounded = round(f2_age)
        
        # Clamp ages
        f1_age_rounded = max(23, min(44, f1_age_rounded))
        f2_age_rounded = max(23, min(44, f2_age_rounded))
        
        f1_perf = self.age_performance.get(f1_age_rounded, 0.50)
        f2_perf = self.age_performance.get(f2_age_rounded, 0.50)
        
        # Convert to relative probability
        # If both have same performance, return 0.5
        # If F1 has better performance, > 0.5
        total_perf = f1_perf + f2_perf
        if total_perf == 0:
            return 0.50
        
        return f1_perf / total_perf
    
    def predict_fight(self, 
                     f1_reach: float, f2_reach: float,
                     f1_height: float, f2_height: float,
                     f1_age: float, f2_age: float,
                     method: str = 'weighted') -> Dict:
        """
        Predict fight outcome based on physical attributes.
        
        Args:
            f1_reach: Fighter 1 reach in inches
            f2_reach: Fighter 2 reach in inches
            f1_height: Fighter 1 height in inches
            f2_height: Fighter 2 height in inches
            f1_age: Fighter 1 age in years
            f2_age: Fighter 2 age in years
            method: 'weighted' or 'simple_average'
        
        Returns:
            Dictionary with prediction details
        """
        # Calculate differences
        reach_diff = f1_reach - f2_reach
        height_diff = f1_height - f2_height
        age_diff = f1_age - f2_age
        
        # Get individual probabilities
        reach_prob = self._get_reach_probability(reach_diff)
        height_prob = self._get_height_probability(height_diff)
        age_diff_prob = self._get_age_diff_probability(age_diff)
        age_abs_prob = self._get_age_performance_factor(f1_age, f2_age)
        
        # Calculate final probability
        if method == 'weighted':
            # Weighted average based on predictive power
            f1_win_prob = (
                self.weights['reach'] * reach_prob +
                self.weights['height'] * height_prob +
                self.weights['age_diff'] * age_diff_prob +
                self.weights['age_absolute'] * age_abs_prob
            )
        else:
            # Simple average
            f1_win_prob = np.mean([reach_prob, height_prob, age_diff_prob, age_abs_prob])
        
        # Determine prediction
        predicted_winner = 1 if f1_win_prob > 0.5 else 2
        confidence = abs(f1_win_prob - 0.5) * 2  # Scale to 0-1
        
        return {
            'f1_win_probability': f1_win_prob,
            'f2_win_probability': 1 - f1_win_prob,
            'predicted_winner': predicted_winner,
            'confidence': confidence,
            'breakdown': {
                'reach_advantage': reach_diff,
                'reach_prob': reach_prob,
                'height_advantage': height_diff,
                'height_prob': height_prob,
                'age_difference': age_diff,
                'age_diff_prob': age_diff_prob,
                'age_absolute_prob': age_abs_prob,
                'f1_age': f1_age,
                'f2_age': f2_age
            }
        }
    
    def predict_dataframe(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Add attribute-based predictions to a dataframe of fights.
        
        Expected columns: f1_reach_in, f2_reach_in, f1_height_in, f2_height_in, 
                         f1_age, f2_age
        """
        predictions = []
        
        for row in df.iter_rows(named=True):
            pred = self.predict_fight(
                f1_reach=row.get('f1_reach_in'),
                f2_reach=row.get('f2_reach_in'),
                f1_height=row.get('f1_height_in'),
                f2_height=row.get('f2_height_in'),
                f1_age=row.get('f1_age'),
                f2_age=row.get('f2_age')
            )
            predictions.append(pred)
        
        # Add predictions to dataframe
        result_df = df.with_columns([
            pl.Series('attr_f1_win_prob', [p['f1_win_probability'] for p in predictions]),
            pl.Series('attr_f2_win_prob', [p['f2_win_probability'] for p in predictions]),
            pl.Series('attr_predicted_winner', [p['predicted_winner'] for p in predictions]),
            pl.Series('attr_confidence', [p['confidence'] for p in predictions]),
            pl.Series('attr_reach_advantage', [p['breakdown']['reach_advantage'] for p in predictions]),
            pl.Series('attr_height_advantage', [p['breakdown']['height_advantage'] for p in predictions]),
            pl.Series('attr_age_difference', [p['breakdown']['age_difference'] for p in predictions]),
        ])
        
        return result_df
    
    def evaluate_predictions(self, df: pl.DataFrame, actual_winner_col: str = 'winner_id') -> Dict:
        """
        Evaluate prediction accuracy on historical data.
        
        Args:
            df: DataFrame with predictions and actual outcomes
            actual_winner_col: Column name containing actual winner (1 or 2)
        """
        # Convert winner_id to 1/2 format if needed
        eval_df = df.with_columns([
            pl.when(pl.col(actual_winner_col) == pl.col('fighter1_id'))
              .then(pl.lit(1))
              .when(pl.col(actual_winner_col) == pl.col('fighter2_id'))
              .then(pl.lit(2))
              .otherwise(None)
              .alias('actual_winner')
        ]).filter(pl.col('actual_winner').is_not_null())
        
        # Calculate accuracy
        correct = eval_df.filter(
            pl.col('attr_predicted_winner') == pl.col('actual_winner')
        ).height
        
        total = eval_df.height
        accuracy = correct / total if total > 0 else 0
        
        # Calculate by confidence level
        high_conf = eval_df.filter(pl.col('attr_confidence') > 0.6)
        high_conf_accuracy = (
            high_conf.filter(pl.col('attr_predicted_winner') == pl.col('actual_winner')).height /
            high_conf.height if high_conf.height > 0 else 0
        )
        
        med_conf = eval_df.filter(
            (pl.col('attr_confidence') > 0.3) & (pl.col('attr_confidence') <= 0.6)
        )
        med_conf_accuracy = (
            med_conf.filter(pl.col('attr_predicted_winner') == pl.col('actual_winner')).height /
            med_conf.height if med_conf.height > 0 else 0
        )
        
        low_conf = eval_df.filter(pl.col('attr_confidence') <= 0.3)
        low_conf_accuracy = (
            low_conf.filter(pl.col('attr_predicted_winner') == pl.col('actual_winner')).height /
            low_conf.height if low_conf.height > 0 else 0
        )
        
        # Average prediction probability
        avg_prob = eval_df.select(pl.col('attr_f1_win_prob').mean()).item()
        
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
            'avg_prediction_probability': avg_prob
        }


def analyze_calibration(predictions_df: pl.DataFrame) -> pl.DataFrame:
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
        pl.max_horizontal(['attr_f1_win_prob', 'attr_f2_win_prob']).alias('max_prob'),
        (pl.col('attr_predicted_winner') == pl.col('actual_winner')).alias('correct')
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


def print_calibration_report(calibration_df: pl.DataFrame, model_name: str = "ATTRIBUTES MODEL"):
    """Print a calibration report showing predicted vs actual accuracy."""
    
    print("\n" + "=" * 80)
    print(f"CALIBRATION ANALYSIS: {model_name}")
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


def print_prediction_report(prediction: Dict, f1_name: str = "Fighter 1", f2_name: str = "Fighter 2"):
    """Pretty print a single fight prediction."""
    print("=" * 80)
    print(f"FIGHT PREDICTION: {f1_name} vs {f2_name}")
    print("=" * 80)
    
    winner_name = f1_name if prediction['predicted_winner'] == 1 else f2_name
    print(f"\n🥊 PREDICTED WINNER: {winner_name}")
    print(f"   Confidence: {prediction['confidence']*100:.1f}%")
    print(f"\n📊 WIN PROBABILITIES:")
    print(f"   {f1_name}: {prediction['f1_win_probability']*100:.1f}%")
    print(f"   {f2_name}: {prediction['f2_win_probability']*100:.1f}%")
    
    bd = prediction['breakdown']
    print(f"\n📏 ATTRIBUTE BREAKDOWN:")
    print(f"   Reach:  {f1_name} has {bd['reach_advantage']:+.1f}\" advantage")
    print(f"           → Win probability from reach: {bd['reach_prob']*100:.1f}%")
    print(f"   Height: {f1_name} has {bd['height_advantage']:+.1f}\" advantage")
    print(f"           → Win probability from height: {bd['height_prob']*100:.1f}%")
    print(f"   Age:    {f1_name} is {bd['age_difference']:+.1f} years different")
    print(f"           → Win probability from age diff: {bd['age_diff_prob']*100:.1f}%")
    print(f"           → Absolute age performance: {bd['age_absolute_prob']*100:.1f}%")
    print(f"           ({f1_name}: {bd['f1_age']:.1f}yo, {f2_name}: {bd['f2_age']:.1f}yo)")
    print("=" * 80)


def print_evaluation_report(eval_results: Dict):
    """Pretty print evaluation results."""
    print("=" * 80)
    print("ATTRIBUTE MODEL EVALUATION")
    print("=" * 80)
    print(f"\n📊 OVERALL PERFORMANCE:")
    print(f"   Accuracy: {eval_results['overall_accuracy']*100:.2f}%")
    print(f"   Correct:  {eval_results['correct_predictions']} / {eval_results['total_predictions']}")
    
    print(f"\n🎯 PERFORMANCE BY CONFIDENCE LEVEL:")
    print(f"   High Confidence (>60%): {eval_results['high_confidence_accuracy']*100:.2f}% ({eval_results['high_confidence_count']} fights)")
    print(f"   Med Confidence (30-60%): {eval_results['medium_confidence_accuracy']*100:.2f}% ({eval_results['medium_confidence_count']} fights)")
    print(f"   Low Confidence (<30%):   {eval_results['low_confidence_accuracy']*100:.2f}% ({eval_results['low_confidence_count']} fights)")
    
    print("=" * 80)


def run_evaluation():
    """Main function to evaluate the attributes model."""
    
    print("Loading fight data...")
    df = pl.read_parquet('fight_features_extracted.parquet')
    print(f"Loaded {len(df)} fights\n")
    
    # Calculate ages if not present
    if 'f1_age' not in df.columns:
        print("Calculating ages from DOB...")
        df = df.with_columns([
            ((pl.col("fight_date") - pl.col("f1_dob")).dt.total_days() / 365.25).alias("f1_age"),
            ((pl.col("fight_date") - pl.col("f2_dob")).dt.total_days() / 365.25).alias("f2_age")
        ])
    
    # Initialize predictor
    predictor = AttributePredictor()
    
    # Make predictions
    print("Generating predictions...")
    predictions_df = predictor.predict_dataframe(df)
    
    # Evaluate
    print("\nEvaluating model...\n")
    eval_results = predictor.evaluate_predictions(predictions_df, actual_winner_col='winner_id')
    
    # Print report
    print_evaluation_report(eval_results)
    
    # Calibration analysis
    calibration_df = analyze_calibration(predictions_df)
    print_calibration_report(calibration_df, "ATTRIBUTES MODEL")
    
    # Save predictions
    predictions_df.write_parquet('attributes_predictions.parquet')
    print("\n💾 Predictions saved to: attributes_predictions.parquet")
    
    return predictions_df, eval_results