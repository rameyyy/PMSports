from models.ufc_mma.utils import create_connection
import polars as pl
import numpy as np

def get_october_18_predictions(connection):
    """
    Get all predictions for fights on October 18, 2025 with fighter names.
    Focuses on Weighted Avg Probability ensemble and logs to file.
    
    Parameters:
    -----------
    connection : mysql.connector.connection.MySQLConnection
        Active MySQL connection
    
    Returns:
    --------
    pl.DataFrame
        Predictions for Oct 18 with fighter names and all model predictions
    """
    
    query = """
    SELECT 
        p.fight_id,
        p.fight_date,
        f.fighter1_name,
        f.fighter2_name,
        p.actual_winner,
        p.predicted_winner,
        p.prediction_confidence,
        
        -- Individual models
        p.logistic_pred,
        p.logistic_f1_prob,
        p.xgboost_pred,
        p.xgboost_f1_prob,
        p.gradient_pred,
        p.gradient_f1_prob,
        p.homemade_pred,
        p.homemade_f1_prob,
        
        -- Ensemble models
        p.ensemble_majorityvote_pred,
        p.ensemble_majorityvote_f1_prob,
        p.ensemble_weightedvote_pred,
        p.ensemble_weightedvote_f1_prob,
        p.ensemble_avgprob_pred,
        p.ensemble_avgprob_f1_prob,
        p.ensemble_weightedavgprob_pred,
        p.ensemble_weightedavgprob_f1_prob
        
    FROM ufc.predictions p
    JOIN ufc.fights f ON p.fight_id = f.fight_id
    WHERE p.fight_date = '2025-10-18'
    ORDER BY p.fight_id;
    """
    
    cursor = connection.cursor()
    cursor.execute(query)
    
    columns = [desc[0] for desc in cursor.description]
    rows = cursor.fetchall()
    cursor.close()
    
    if not rows:
        print("\n⚠️  No predictions found for October 18, 2025")
        return None
    
    df = pl.DataFrame(rows, schema=columns, orient='row')
    
    # Get confidence calibration data for weighted avg prob
    calibration_data = get_model_calibration_data(connection, 'ensemble_weightedavgprob')
    
    print("\n" + "="*80)
    print("PREDICTIONS FOR OCTOBER 18, 2025")
    print("Using: Ensemble Weighted Avg Probability")
    print("="*80)
    print(f"\nTotal fights: {len(df)}")
    
    # Open log file
    with open('oct18_predictions.txt', 'w') as log_file:
        log_file.write("="*80 + "\n")
        log_file.write("PREDICTIONS FOR OCTOBER 18, 2025\n")
        log_file.write("Model: Ensemble Weighted Avg Probability\n")
        log_file.write("="*80 + "\n\n")
        
        # Display each fight with predictions
        for idx, row in enumerate(df.iter_rows(named=True), 1):
            fighter1 = row['fighter1_name']
            fighter2 = row['fighter2_name']
            
            # Weighted avg prob prediction
            pred = row['ensemble_weightedavgprob_pred']
            prob = row['ensemble_weightedavgprob_f1_prob']
            
            # Determine winner and confidence
            if pred == 1:
                predicted_fighter = fighter1
                confidence = prob
            else:
                predicted_fighter = fighter2
                confidence = 1 - prob
            
            confidence_pct = confidence * 100
            
            # Find matching calibration bracket
            bracket_info = find_calibration_bracket(prob, calibration_data)
            
            # Print to console
            print(f"\n{'='*80}")
            print(f"Fight #{idx}: {fighter1} vs {fighter2}")
            print(f"{'='*80}")
            print(f"⭐ Prediction: {predicted_fighter}")
            print(f"   Confidence: {confidence_pct:.1f}%")
            
            if bracket_info:
                print(f"   Historical Accuracy: {bracket_info['accuracy']:.1f}% (based on {bracket_info['count']} fights in {bracket_info['bracket']} range)")
            else:
                print(f"   Historical Accuracy: N/A (insufficient data)")
            
            if row['actual_winner'] is not None:
                actual = fighter1 if row['actual_winner'] == 1 else fighter2
                result = "✅ CORRECT" if pred == row['actual_winner'] else "❌ INCORRECT"
                print(f"   Actual Result: {actual} won - {result}")
            
            # Write to log file
            log_file.write(f"Fight #{idx}: {fighter1} vs {fighter2}\n")
            log_file.write(f"Ensemble Weighted Avg Prob Prediction: {predicted_fighter}\n")
            log_file.write(f"Confidence: {confidence_pct:.1f}%\n")
            
            if bracket_info:
                log_file.write(f"Accuracy: {bracket_info['accuracy']:.1f}%\n")
                log_file.write(f"Accuracy Sample: {bracket_info['count']} fights\n")
            else:
                log_file.write(f"Accuracy: N/A (insufficient data)\n")
                log_file.write(f"Accuracy Sample: 0 fights\n")
            
            if row['actual_winner'] is not None:
                actual = fighter1 if row['actual_winner'] == 1 else fighter2
                result = "CORRECT" if pred == row['actual_winner'] else "INCORRECT"
                log_file.write(f"Actual Result: {actual} won - {result}\n")
            
            log_file.write("\n" + "-"*80 + "\n\n")
    
    print(f"\n✅ Predictions logged to: oct18_predictions.txt")
    
    return df


def get_model_calibration_data(connection, model_key):
    """
    Get calibration data for a specific model.
    
    Parameters:
    -----------
    connection : mysql.connector.connection.MySQLConnection
        Active MySQL connection
    model_key : str
        Model identifier (e.g., 'ensemble_weightedavgprob')
    
    Returns:
    --------
    dict : Calibration data by bracket
    """
    
    query = f"""
    SELECT 
        {model_key}_f1_prob,
        {model_key}_correct
    FROM ufc.predictions
    WHERE actual_winner IS NOT NULL;
    """
    
    cursor = connection.cursor()
    cursor.execute(query)
    
    columns = [desc[0] for desc in cursor.description]
    rows = cursor.fetchall()
    cursor.close()
    
    if not rows:
        return {}
    
    df = pl.DataFrame(rows, schema=columns, orient='row')
    
    prob_col = f"{model_key}_f1_prob"
    correct_col = f"{model_key}_correct"
    
    # Define 2% brackets
    brackets = [
        (0.50, 0.52, "50-52%"),
        (0.52, 0.54, "52-54%"),
        (0.54, 0.56, "54-56%"),
        (0.56, 0.58, "56-58%"),
        (0.58, 0.60, "58-60%"),
        (0.60, 0.62, "60-62%"),
        (0.62, 0.64, "62-64%"),
        (0.64, 0.66, "64-66%"),
        (0.66, 0.68, "66-68%"),
        (0.68, 0.70, "68-70%"),
        (0.70, 0.72, "70-72%"),
        (0.72, 0.74, "72-74%"),
        (0.74, 0.76, "74-76%"),
        (0.76, 0.78, "76-78%"),
        (0.78, 0.80, "78-80%"),
        (0.80, 0.82, "80-82%"),
        (0.82, 0.84, "82-84%"),
        (0.84, 0.86, "84-86%"),
        (0.86, 0.88, "86-88%"),
        (0.88, 0.90, "88-90%"),
        (0.90, 0.92, "90-92%"),
        (0.92, 0.94, "92-94%"),
        (0.94, 0.96, "94-96%"),
        (0.96, 0.98, "96-98%"),
        (0.98, 1.01, "98-100%")
    ]
    
    calibration = {}
    
    for min_conf, max_conf, label in brackets:
        # Get predictions in this bracket (considering both F1 win and F2 win)
        bracket_df = df.filter(
            ((pl.col(prob_col) >= min_conf) & (pl.col(prob_col) < max_conf)) |
            ((pl.col(prob_col) <= (1 - min_conf)) & (pl.col(prob_col) > (1 - max_conf)))
        )
        
        count = len(bracket_df)
        
        if count > 0:
            correct = bracket_df[correct_col].sum()
            accuracy = (correct / count) * 100
            
            calibration[label] = {
                'count': count,
                'accuracy': accuracy,
                'bracket': label
            }
    
    return calibration


def find_calibration_bracket(prob, calibration_data):
    """
    Find the calibration bracket for a given probability.
    
    Parameters:
    -----------
    prob : float
        Probability value (0-1)
    calibration_data : dict
        Calibration data from get_model_calibration_data
    
    Returns:
    --------
    dict : Bracket info or None
    """
    
    # Convert prob to confidence (distance from 0.5)
    if prob >= 0.5:
        conf = prob
    else:
        conf = 1 - prob
    
    # Find matching bracket
    brackets = [
        (0.50, 0.52, "50-52%"),
        (0.52, 0.54, "52-54%"),
        (0.54, 0.56, "54-56%"),
        (0.56, 0.58, "56-58%"),
        (0.58, 0.60, "58-60%"),
        (0.60, 0.62, "60-62%"),
        (0.62, 0.64, "62-64%"),
        (0.64, 0.66, "64-66%"),
        (0.66, 0.68, "66-68%"),
        (0.68, 0.70, "68-70%"),
        (0.70, 0.72, "70-72%"),
        (0.72, 0.74, "72-74%"),
        (0.74, 0.76, "74-76%"),
        (0.76, 0.78, "76-78%"),
        (0.78, 0.80, "78-80%"),
        (0.80, 0.82, "80-82%"),
        (0.82, 0.84, "82-84%"),
        (0.84, 0.86, "84-86%"),
        (0.86, 0.88, "86-88%"),
        (0.88, 0.90, "88-90%"),
        (0.90, 0.92, "90-92%"),
        (0.92, 0.94, "92-94%"),
        (0.94, 0.96, "94-96%"),
        (0.96, 0.98, "96-98%"),
        (0.98, 1.01, "98-100%")
    ]
    
    for min_conf, max_conf, label in brackets:
        if min_conf <= conf < max_conf:
            if label in calibration_data:
                return calibration_data[label]
            return None
    
    return None


def analyze_confidence_accuracy_by_model(connection):
    """
    Analyze the accuracy of each model at different confidence levels.
    Shows how well each model's confidence correlates with actual accuracy.
    
    Parameters:
    -----------
    connection : mysql.connector.connection.MySQLConnection
        Active MySQL connection
    
    Returns:
    --------
    dict : Accuracy by confidence bracket for each model
    """
    
    # Get all non-legacy predictions with results
    query = """
    SELECT 
        logistic_f1_prob,
        logistic_correct,
        xgboost_f1_prob,
        xgboost_correct,
        gradient_f1_prob,
        gradient_correct,
        homemade_f1_prob,
        homemade_correct,
        ensemble_weightedvote_f1_prob,
        ensemble_weightedvote_correct,
        ensemble_avgprob_f1_prob,
        ensemble_avgprob_correct,
        ensemble_weightedavgprob_f1_prob,
        ensemble_weightedavgprob_correct
    FROM ufc.predictions
    WHERE actual_winner IS NOT NULL;
    """
    
    cursor = connection.cursor()
    cursor.execute(query)
    
    columns = [desc[0] for desc in cursor.description]
    rows = cursor.fetchall()
    cursor.close()
    
    if not rows:
        print("\n⚠️  No completed predictions to analyze")
        return None
    
    df = pl.DataFrame(rows, schema=columns, orient='row')
    
    print("\n" + "="*80)
    print("CONFIDENCE vs ACCURACY ANALYSIS")
    print("="*80)
    print(f"\nAnalyzing {len(df)} completed predictions")
    print("Shows how well each model's confidence predicts actual accuracy\n")
    
    # Define confidence brackets (2% intervals)
    brackets = [
        (0.50, 0.52, "50-52%"),
        (0.52, 0.54, "52-54%"),
        (0.54, 0.56, "54-56%"),
        (0.56, 0.58, "56-58%"),
        (0.58, 0.60, "58-60%"),
        (0.60, 0.62, "60-62%"),
        (0.62, 0.64, "62-64%"),
        (0.64, 0.66, "64-66%"),
        (0.66, 0.68, "66-68%"),
        (0.68, 0.70, "68-70%"),
        (0.70, 0.72, "70-72%"),
        (0.72, 0.74, "72-74%"),
        (0.74, 0.76, "74-76%"),
        (0.76, 0.78, "76-78%"),
        (0.78, 0.80, "78-80%"),
        (0.80, 0.82, "80-82%"),
        (0.82, 0.84, "82-84%"),
        (0.84, 0.86, "84-86%"),
        (0.86, 0.88, "86-88%"),
        (0.88, 0.90, "88-90%"),
        (0.90, 0.92, "90-92%"),
        (0.92, 0.94, "92-94%"),
        (0.94, 0.96, "94-96%"),
        (0.96, 0.98, "96-98%"),
        (0.98, 1.01, "98-100%")
    ]
    
    models = [
        ('logistic', 'Logistic Regression'),
        ('xgboost', 'XGBoost'),
        ('gradient', 'Gradient Boost'),
        ('homemade', 'Homemade'),
        ('ensemble_weightedvote', 'Weighted Vote'),
        ('ensemble_avgprob', 'Avg Probability'),
        ('ensemble_weightedavgprob', 'Weighted Avg Prob')
    ]
    
    results = {}
    
    for model_key, model_name in models:
        prob_col = f"{model_key}_f1_prob"
        correct_col = f"{model_key}_correct"
        
        print(f"\n{'='*80}")
        print(f"📊 {model_name.upper()}")
        print(f"{'='*80}")
        print(f"\n{'Confidence':<15} {'Count':<10} {'Correct':<10} {'Accuracy':<12} {'Calibration':<15}")
        print("-" * 80)
        
        model_results = []
        
        for min_conf, max_conf, label in brackets:
            # Get predictions in this confidence bracket
            # Confidence is distance from 0.5, so convert F1 prob to confidence
            bracket_df = df.filter(
                ((pl.col(prob_col) >= min_conf) & (pl.col(prob_col) < max_conf)) |
                ((pl.col(prob_col) <= (1 - min_conf)) & (pl.col(prob_col) > (1 - max_conf)))
            )
            
            count = len(bracket_df)
            
            if count > 0:
                correct = bracket_df[correct_col].sum()
                accuracy = (correct / count) * 100
                
                # Expected accuracy at midpoint of bracket
                expected = ((min_conf + max_conf) / 2) * 100
                calibration = accuracy - expected
                calibration_str = f"{calibration:+.1f}%"
                
                print(f"{label:<15} {count:<10} {correct:<10} {accuracy:>6.1f}%      {calibration_str:<15}")
                
                model_results.append({
                    'bracket': label,
                    'count': count,
                    'correct': correct,
                    'accuracy': accuracy,
                    'expected': expected,
                    'calibration': calibration
                })
            else:
                print(f"{label:<15} {count:<10} {'N/A':<10} {'N/A':<12} {'N/A':<15}")
        
        results[model_name] = model_results
        
        # Overall stats
        total_correct = df[correct_col].sum()
        total_count = len(df)
        overall_acc = (total_correct / total_count) * 100
        
        print("-" * 80)
        print(f"{'OVERALL':<15} {total_count:<10} {total_correct:<10} {overall_acc:>6.1f}%")
    
    print("\n" + "="*80)
    print("CALIBRATION GUIDE")
    print("="*80)
    print("Calibration = Actual Accuracy - Expected Accuracy")
    print("  Positive (+): Model is underconfident (better than it thinks)")
    print("  Negative (-): Model is overconfident (worse than it thinks)")
    print("  Near 0:       Model is well-calibrated")
    
    return results


def main(connection):
    """
    Main function to analyze October 18 predictions.
    
    Parameters:
    -----------
    connection : mysql.connector.connection.MySQLConnection
        Active MySQL connection
    """
    
    # Show October 18 predictions
    oct18_df = get_october_18_predictions(connection)
    
    # Analyze confidence vs accuracy across all historical predictions
    print("\n\n")
    confidence_analysis = analyze_confidence_accuracy_by_model(connection)
    
    return oct18_df, confidence_analysis


if __name__ == "__main__":
    conn = create_connection()
    oct18_df, confidence_analysis = main(conn)
