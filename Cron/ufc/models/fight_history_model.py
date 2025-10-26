# fight_history_model.py

import polars as pl
import numpy as np
from typing import Dict, Tuple

class FightHistoryPredictor:
    """
    Predicts fight outcomes based on fight history features.
    Uses empirical correlations and weighted scoring system.
    """
    
    def __init__(self):
        # Feature weights based on correlation strength
        # Top features from analysis
        self.feature_weights = {
            'diff_win_rate': 0.1,
            'diff_finish_quality_score': 0.1,
            'diff_current_streak': 0.05,
            'diff_recent_form_5': 0.05,
            'diff_avg_ctrl_time': 0.1,
            'diff_avg_td_landed': 0.05,
            'diff_dominant_decision_rate': 0.45,
            'diff_sig_str_accuracy': 0.05,
            'diff_avg_head_str_landed': 0.05,
        }
        
        # Normalize weights to sum to 1
        total_weight = sum(self.feature_weights.values())
        self.feature_weights = {k: v/total_weight for k, v in self.feature_weights.items()}
        
        self.baseline = 0.50
    
    def _calculate_feature_probability(self, feature_name: str, value: float) -> float:
        """
        Convert a feature differential value to a win probability.
        Based on quintile analysis from the data.
        """
        
        # Approximate mappings based on quintile analysis
        # These are rough estimates - you could make these more precise
        
        if feature_name == 'diff_win_rate':
            # Q1: -0.28 → 38%, Q5: +0.33 → 62%
            if value < -0.20:
                return 0.38
            elif value < -0.05:
                return 0.48
            elif value < 0.05:
                return 0.50
            elif value < 0.20:
                return 0.54
            else:
                return 0.62
        
        elif feature_name == 'diff_finish_quality_score':
            # Q1: -0.97 → 41%, Q5: +1.21 → 60%
            if value < -0.50:
                return 0.41
            elif value < -0.10:
                return 0.45
            elif value < 0.20:
                return 0.53
            elif value < 0.70:
                return 0.56
            else:
                return 0.60
        
        elif feature_name == 'diff_current_streak':
            # Q1: -7 → 40%, Q5: +8 → 57%
            if value < -3:
                return 0.40
            elif value < -1:
                return 0.48
            elif value < 2:
                return 0.52
            elif value < 5:
                return 0.54
            else:
                return 0.57
        
        elif feature_name == 'diff_recent_form_5':
            # Q1: -0.44 → 41%, Q5: +0.53 → 59%
            if value < -0.30:
                return 0.41
            elif value < -0.05:
                return 0.48
            elif value < 0.10:
                return 0.50
            elif value < 0.40:
                return 0.52
            else:
                return 0.59
        
        elif feature_name == 'diff_avg_ctrl_time':
            # Q1: -181 → 42%, Q5: +174 → 57%
            if value < -100:
                return 0.42
            elif value < -30:
                return 0.48
            elif value < 30:
                return 0.52
            elif value < 100:
                return 0.53
            else:
                return 0.57
        
        elif feature_name == 'diff_avg_td_landed':
            # Q1: -1.93 → 45%, Q5: +1.75 → 56%
            if value < -1.0:
                return 0.45
            elif value < -0.30:
                return 0.48
            elif value < 0.30:
                return 0.51
            elif value < 1.0:
                return 0.54
            else:
                return 0.56
        
        else:
            # For other features, use simple linear scaling
            # Assume correlation of ~0.10 and scale accordingly
            prob = self.baseline + (value * 0.05)  # Conservative scaling
            return np.clip(prob, 0.20, 0.80)
    
    def predict_fight(self, fight_features: Dict) -> Dict:
        """
        Predict fight outcome based on fight history features.
        
        Args:
            fight_features: Dictionary containing diff_* features
        
        Returns:
            Dictionary with prediction details
        """
        
        # Calculate weighted probability
        weighted_prob = 0.0
        feature_contributions = {}
        used_weight = 0.0
        
        for feat_name, weight in self.feature_weights.items():
            if feat_name in fight_features and fight_features[feat_name] is not None:
                value = fight_features[feat_name]
                
                # Skip NaN or infinite values
                if np.isnan(value) or np.isinf(value):
                    continue
                
                feat_prob = self._calculate_feature_probability(feat_name, value)
                weighted_prob += weight * feat_prob
                used_weight += weight
                
                feature_contributions[feat_name] = {
                    'value': value,
                    'probability': feat_prob,
                    'weight': weight
                }
        
        # Normalize by used weight
        if used_weight > 0:
            f1_win_prob = weighted_prob / used_weight
        else:
            f1_win_prob = self.baseline
        
        # Apply confidence boosters for strong signals
        confidence_boost = 0.0
        
        # Boost 1: Momentum alignment (streak + recent form aligned)
        if ('diff_current_streak' in fight_features and 
            'diff_recent_form_5' in fight_features):
            
            streak = fight_features.get('diff_current_streak', 0)
            form = fight_features.get('diff_recent_form_5', 0)
            
            if streak is not None and form is not None:
                # If both positive and strong, boost confidence
                if streak > 2 and form > 0.3:
                    confidence_boost += 0.03
                    f1_win_prob += 0.02
                # If both negative and strong, boost confidence (for f2)
                elif streak < -2 and form < -0.3:
                    confidence_boost += 0.03
                    f1_win_prob -= 0.02
        
        # Boost 2: Dominant fighter (high win rate + high finish quality)
        if ('diff_win_rate' in fight_features and 
            'diff_finish_quality_score' in fight_features):
            
            win_rate = fight_features.get('diff_win_rate', 0)
            finish_quality = fight_features.get('diff_finish_quality_score', 0)
            
            if win_rate is not None and finish_quality is not None:
                # If both strongly favor f1
                if win_rate > 0.2 and finish_quality > 0.5:
                    confidence_boost += 0.04
                    f1_win_prob += 0.03
                # If both strongly favor f2
                elif win_rate < -0.2 and finish_quality < -0.5:
                    confidence_boost += 0.04
                    f1_win_prob -= 0.03
        
        # Boost 3: Complete fighter (striking + grappling)
        if ('diff_sig_str_accuracy' in fight_features and 
            'diff_avg_td_landed' in fight_features):
            
            striking = fight_features.get('diff_sig_str_accuracy', 0)
            grappling = fight_features.get('diff_avg_td_landed', 0)
            
            if striking is not None and grappling is not None:
                # If both favor same fighter
                if striking > 0.05 and grappling > 0.5:
                    confidence_boost += 0.02
                    f1_win_prob += 0.01
                elif striking < -0.05 and grappling < -0.5:
                    confidence_boost += 0.02
                    f1_win_prob -= 0.01
        
        # Clip probability to reasonable range
        f1_win_prob = np.clip(f1_win_prob, 0.15, 0.85)
        
        # Calculate confidence (distance from 50/50)
        base_confidence = abs(f1_win_prob - 0.5) * 2
        confidence = min(base_confidence + confidence_boost, 1.0)
        
        # Determine prediction
        predicted_winner = 1 if f1_win_prob > 0.5 else 2
        
        return {
            'f1_win_probability': f1_win_prob,
            'f2_win_probability': 1 - f1_win_prob,
            'predicted_winner': predicted_winner,
            'confidence': confidence,
            'feature_contributions': feature_contributions,
            'confidence_boosts_applied': confidence_boost
        }
    
    def predict_dataframe(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Add fight history-based predictions to a dataframe.
        
        Expected columns: diff_* features from fight history analysis
        """
        predictions = []
        
        for row in df.iter_rows(named=True):
            # Extract all diff_ features
            fight_features = {k: v for k, v in row.items() if k.startswith('diff_')}
            
            pred = self.predict_fight(fight_features)
            predictions.append(pred)
        
        # Add predictions to dataframe
        result_df = df.with_columns([
            pl.Series('hist_f1_win_prob', [p['f1_win_probability'] for p in predictions]),
            pl.Series('hist_f2_win_prob', [p['f2_win_probability'] for p in predictions]),
            pl.Series('hist_predicted_winner', [p['predicted_winner'] for p in predictions]),
            pl.Series('hist_confidence', [p['confidence'] for p in predictions]),
        ])
        
        return result_df
    
    def evaluate_predictions(self, df: pl.DataFrame, actual_winner_col: str = 'winner_id') -> Dict:
        """
        Evaluate prediction accuracy on historical data.
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
            pl.col('hist_predicted_winner') == pl.col('actual_winner')
        ).height
        
        total = eval_df.height
        accuracy = correct / total if total > 0 else 0
        
        # Calculate by confidence level
        high_conf = eval_df.filter(pl.col('hist_confidence') > 0.3)
        high_conf_accuracy = (
            high_conf.filter(pl.col('hist_predicted_winner') == pl.col('actual_winner')).height /
            high_conf.height if high_conf.height > 0 else 0
        )
        
        med_conf = eval_df.filter(
            (pl.col('hist_confidence') > 0.15) & (pl.col('hist_confidence') <= 0.3)
        )
        med_conf_accuracy = (
            med_conf.filter(pl.col('hist_predicted_winner') == pl.col('actual_winner')).height /
            med_conf.height if med_conf.height > 0 else 0
        )
        
        low_conf = eval_df.filter(pl.col('hist_confidence') <= 0.15)
        low_conf_accuracy = (
            low_conf.filter(pl.col('hist_predicted_winner') == pl.col('actual_winner')).height /
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
        pl.max_horizontal(['hist_f1_win_prob', 'hist_f2_win_prob']).alias('max_prob'),
        (pl.col('hist_predicted_winner') == pl.col('actual_winner')).alias('correct')
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


def print_calibration_report(calibration_df: pl.DataFrame, model_name: str = "FIGHT HISTORY MODEL"):
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


def calculate_probabilistic_metrics(df: pl.DataFrame):
    """Calculate MSE, Brier Score, and other probabilistic metrics."""
    
    # Add actual outcome column
    df = df.with_columns([
        pl.when(pl.col('winner_id') == pl.col('fighter1_id'))
          .then(pl.lit(1.0))
          .otherwise(pl.lit(0.0))
          .alias('f1_actually_won')
    ])
    
    predictions = df.select("hist_f1_win_prob").to_numpy().flatten()
    actuals = df.select("f1_actually_won").to_numpy().flatten()
    
    n = len(predictions)
    
    # MSE / Brier Score
    mse = np.mean((predictions - actuals) ** 2)
    rmse = np.sqrt(mse)
    brier_score = mse
    
    # Log Loss
    predictions_clipped = np.clip(predictions, 1e-15, 1 - 1e-15)
    log_loss = -np.mean(
        actuals * np.log(predictions_clipped) + 
        (1 - actuals) * np.log(1 - predictions_clipped)
    )
    
    # Baseline comparison
    baseline_mse = np.mean((0.5 - actuals) ** 2)
    baseline_log_loss = -np.mean(actuals * np.log(0.5) + (1 - actuals) * np.log(0.5))
    
    return {
        'mse': mse,
        'rmse': rmse,
        'brier_score': brier_score,
        'log_loss': log_loss,
        'baseline_mse': baseline_mse,
        'baseline_log_loss': baseline_log_loss,
        'n': n
    }


def print_evaluation_report(eval_results: Dict, metrics: Dict = None):
    """Pretty print evaluation results."""
    
    print("=" * 80)
    print("FIGHT HISTORY MODEL EVALUATION")
    print("=" * 80)
    
    print(f"\n📊 ACCURACY METRICS:")
    print(f"   Overall Accuracy: {eval_results['overall_accuracy']*100:.2f}%")
    print(f"   Correct:  {eval_results['correct_predictions']} / {eval_results['total_predictions']}")
    
    print(f"\n🎯 PERFORMANCE BY CONFIDENCE LEVEL:")
    print(f"   High Confidence (>30%): {eval_results['high_confidence_accuracy']*100:.2f}% ({eval_results['high_confidence_count']} fights)")
    print(f"   Med Confidence (15-30%): {eval_results['medium_confidence_accuracy']*100:.2f}% ({eval_results['medium_confidence_count']} fights)")
    print(f"   Low Confidence (<15%):   {eval_results['low_confidence_accuracy']*100:.2f}% ({eval_results['low_confidence_count']} fights)")
    
    if metrics:
        print(f"\n📈 PROBABILISTIC METRICS:")
        print(f"   Mean Squared Error (MSE):    {metrics['mse']:.6f}")
        print(f"   Root Mean Squared Error:      {metrics['rmse']:.6f}")
        print(f"   Brier Score:                  {metrics['brier_score']:.6f}")
        print(f"   Log Loss:                     {metrics['log_loss']:.6f}")
        
        print(f"\n📊 COMPARISON TO BASELINE:")
        print(f"   Baseline MSE:        {metrics['baseline_mse']:.6f}")
        print(f"   Baseline Log Loss:   {metrics['baseline_log_loss']:.6f}")
        print(f"\n   Model vs Baseline:")
        print(f"   MSE Improvement:      {(1 - metrics['mse']/metrics['baseline_mse'])*100:+.2f}%")
        print(f"   Log Loss Improvement: {(1 - metrics['log_loss']/metrics['baseline_log_loss'])*100:+.2f}%")
    
    print("=" * 80)


def run_evaluation():
    """Main function to evaluate the fight history model."""
    
    print("Loading fight features...")
    features_df = pl.read_parquet('fight_features_extracted.parquet')
    print(f"Loaded {len(features_df)} fights\n")
    
    # Initialize predictor
    predictor = FightHistoryPredictor()
    
    # Make predictions
    print("Generating predictions...")
    predictions_df = predictor.predict_dataframe(features_df)
    
    # Evaluate
    print("\nEvaluating model...\n")
    eval_results = predictor.evaluate_predictions(predictions_df, actual_winner_col='winner_id')
    
    # Calculate probabilistic metrics
    metrics = calculate_probabilistic_metrics(predictions_df)
    
    # Print report
    print_evaluation_report(eval_results, metrics)
    
    # Calibration analysis
    calibration_df = analyze_calibration(predictions_df)
    print_calibration_report(calibration_df, "FIGHT HISTORY MODEL")
    
    # Save predictions
    predictions_df.write_parquet('fight_history_predictions.parquet')
    print("\n💾 Predictions saved to: fight_history_predictions.parquet")
    
    return predictions_df, eval_results, metrics