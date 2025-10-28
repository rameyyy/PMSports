from .utils import create_connection, fetch_query, run_query, get_model_accuracies_batched
import numpy as np

def calculate_model_accuracies():
    """
    Calculate model accuracies and average confidences, then update the model_accuracies table.
    This should be run as a cron job to keep stats updated.
    """
    conn = create_connection()
    
    try:
        # First, create the model_accuracies table if it doesn't exist
        create_table_query = """
        CREATE TABLE IF NOT EXISTS ufc.model_accuracies (
            model_name VARCHAR(100) NOT NULL PRIMARY KEY,
            total_predictions INT NOT NULL,
            correct_predictions INT NOT NULL,
            accuracy FLOAT NOT NULL,
            avg_confidence FLOAT NOT NULL,
            avg_sample_size FLOAT NOT NULL
        );
        """
        run_query(conn, create_table_query, None)
        print("Model accuracies table ready")
        
        # Define models (excluding AlgoPicks as it's handled separately)
        models_config = [
            ('XGBoost', 'xgboost'),
            ('Gradient', 'gradient'),
            ('Logistic', 'logistic'),
            ('Ensemble Avg Prob', 'ensemble_avgprob'),
            ('Ensemble Weight Avg Prob', 'ensemble_weightedavgprob')
        ]
        
        models_data = []
        
        # Process each model (except AlgoPicks)
        for display_name, model_name in models_config:
            # Get basic stats
            stats_query = f"""
            SELECT 
                '{display_name}' AS model_name,
                COUNT(*) AS total_predictions,
                SUM(CASE WHEN {model_name}_correct = 1 THEN 1 ELSE 0 END) AS correct_predictions,
                COUNT(CASE WHEN {model_name}_f1_prob IS NOT NULL THEN 1 END) AS sample_size
            FROM ufc.predictions
            WHERE legacy = 0 AND {model_name}_correct IS NOT NULL;
            """
            stats = fetch_query(conn, stats_query, None)[0]

            # Get all probabilities and convert to predicted winner's confidence (>0.5)
            probs_query = f"""
            SELECT {model_name}_f1_prob
            FROM ufc.predictions
            WHERE legacy = 0 AND {model_name}_correct IS NOT NULL AND {model_name}_f1_prob IS NOT NULL;
            """
            probs_result = fetch_query(conn, probs_query, None)

            # Convert all probs to be > 0.5 (predicted winner's confidence)
            probs = []
            for row in probs_result:
                prob = float(row[f'{model_name}_f1_prob'])  # Dictionary access with column name
                # If prob < 0.5, it's predicting fighter 2, so convert to fighter 2's confidence
                if prob < 0.5:
                    prob = 1.0 - prob
                probs.append(prob)
            
            # Get unique probability values for batched accuracy calculation
            # Round to 2 decimals and get unique values
            unique_probs = list(set([round(p, 2) for p in probs]))

            if unique_probs:
                # Use batched function to get accuracy at each confidence level
                accuracy_by_prob = get_model_accuracies_batched(model_name, unique_probs, window=0.01)

                # Calculate weighted average confidence
                # Weight by how many predictions fall in each band
                total_weighted_accuracy = 0
                total_weight = 0

                for prob, data in accuracy_by_prob.items():
                    correct = data['correct']
                    prob_range = data['prob_range']

                    if prob_range > 0:
                        # Accuracy at this confidence level
                        band_accuracy = (correct / prob_range) * 100
                        # Weight by number of predictions in this band
                        total_weighted_accuracy += band_accuracy * prob_range
                        total_weight += prob_range

                avg_confidence = total_weighted_accuracy / total_weight if total_weight > 0 else 0.0
            else:
                avg_confidence = 0.0

            models_data.append({
                **stats,
                'avg_confidence': avg_confidence
            })
        
        # Handle AlgoPicks separately (from prediction_simplified table)
        algopicks_query = """
        SELECT 
            'AlgoPicks' AS model_name,
            COUNT(*) AS total_predictions,
            SUM(CASE WHEN correct = 1 THEN 1 ELSE 0 END) AS correct_predictions,
            COUNT(CASE WHEN algopick_probability IS NOT NULL THEN 1 END) AS sample_size
        FROM ufc.prediction_simplified
        WHERE correct IS NOT NULL;
        """
        algopicks_stats = fetch_query(conn, algopicks_query, None)[0]

        # For AlgoPicks, average the algopick_probability where correct IS NOT NULL
        algopicks_conf_query = """
        SELECT AVG(algopick_probability) AS avg_conf
        FROM ufc.prediction_simplified
        WHERE correct IS NOT NULL AND algopick_probability IS NOT NULL;
        """
        algopicks_conf_result = fetch_query(conn, algopicks_conf_query, None)[0]
        algopicks_conf = float(algopicks_conf_result['avg_conf']) if algopicks_conf_result['avg_conf'] else 0.0

        models_data.append({**algopicks_stats, 'avg_confidence': algopicks_conf})
        
        # Now insert/update the model_accuracies table
        for model in models_data:
            # Convert Decimal to float
            total_predictions = float(model['total_predictions'])
            correct_predictions = float(model['correct_predictions'])
            avg_confidence = float(model['avg_confidence']) if model['avg_confidence'] else 0.0
            sample_size = float(model['sample_size'])
            
            accuracy = round((correct_predictions * 100.0 / total_predictions), 2)
            avg_confidence = round(avg_confidence, 2)
            
            upsert_query = """
            INSERT INTO ufc.model_accuracies 
                (model_name, total_predictions, correct_predictions, accuracy, avg_confidence, avg_sample_size)
            VALUES 
                (%s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                total_predictions = VALUES(total_predictions),
                correct_predictions = VALUES(correct_predictions),
                accuracy = VALUES(accuracy),
                avg_confidence = VALUES(avg_confidence),
                avg_sample_size = VALUES(avg_sample_size);
            """
            
            params = (
                model['model_name'],
                int(total_predictions),
                int(correct_predictions),
                accuracy,
                avg_confidence,
                sample_size
            )
            
            run_query(conn, upsert_query, params)
            print(f"Updated {model['model_name']}: {accuracy}% accuracy, {avg_confidence}% avg confidence")
        
        print("Model accuracies updated successfully!")
        
    except Exception as e:
        print(f"Error calculating model accuracies: {e}")
        raise
    finally:
        if conn:
            conn.close()