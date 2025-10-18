import polars as pl
import pickle
import numpy as np
from models.ufc_mma import attr_fighthist_combined_model, utils

def load_all_models():
    """Load all saved models from disk."""
    
    print("Loading models...")
    
    models = {}
    
    # Load scaler
    with open('models/ufc_mma/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    # Load feature columns
    with open('models/ufc_mma/feature_columns.pkl', 'rb') as f:
        feature_cols = pickle.load(f)
    
    # Load ML models
    for name in ['logistic', 'xgboost', 'gradient_boost']:
        with open(f'models/ufc_mma/{name}_model.pkl', 'rb') as f:
            models[name] = pickle.load(f)
        print(f"   ✅ Loaded {name}")
    
    # Load metadata
    with open('models/ufc_mma/model_metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    
    print(f"   ✅ All models loaded!")
    print(f"   Trained on {metadata['total_fights']} fights")
    
    return {
        'models': models,
        'scaler': scaler,
        'feature_cols': feature_cols,
        'metadata': metadata
    }


def generate_all_predictions(differential_df, combined_df, loaded_models):
    """
    Generate predictions from all models for entire dataset.
    
    Returns:
    --------
    pl.DataFrame with columns:
        - fight_id
        - fight_date
        - actual_winner (1 or 0)
        - logistic_f1_prob
        - xgboost_f1_prob
        - gradient_f1_prob
        - homemade_f1_prob
        - ensemble_majorityvote_f1_prob
        - ensemble_weightedvote_f1_prob
        - ensemble_avgprob_f1_prob
        - ensemble_weightedavgprob_f1_prob
        - predicted_winner (from best ensemble)
    """
    
    print("="*80)
    print("GENERATING PREDICTIONS FOR ALL FIGHTS")
    print("="*80)
    
    # Extract loaded components
    models = loaded_models['models']
    scaler = loaded_models['scaler']
    feature_cols = loaded_models['feature_cols']
    
    # ==================== PREPARE FEATURES ====================
    print("\n1. Preparing features...")
    
    df_sorted = differential_df.sort('fight_date')
    
    exclude_cols = ['fight_id', 'fight_date', 'target', 'weight_class', 'stance_matchup']
    df_encoded = df_sorted.to_dummies(columns=['weight_class', 'stance_matchup'])
    
    # Ensure all feature columns exist
    for col in feature_cols:
        if col not in df_encoded.columns:
            df_encoded = df_encoded.with_columns(pl.lit(0).alias(col))
    
    X = df_encoded.select(feature_cols).to_numpy()
    X_scaled = scaler.transform(X)
    
    print(f"   Processing {len(df_sorted)} fights")
    
    # ==================== ML MODEL PREDICTIONS ====================
    print("\n2. Getting ML model predictions...")
    
    ml_predictions = {}
    ml_probabilities = {}
    
    for name, model in models.items():
        print(f"   Predicting with {name}...")
        ml_predictions[name] = model.predict(X_scaled)
        ml_probabilities[name] = model.predict_proba(X_scaled)[:, 1]
    
    # ==================== HOMEMADE MODEL PREDICTIONS ====================
    print("\n3. Getting homemade model predictions...")
    
    # Run homemade model on combined_df
    # Need to ensure combined_df is sorted same way as differential_df
    combined_sorted = combined_df.join(
        df_sorted.select(['fight_id', 'fight_date']),
        on='fight_id',
        how='inner'
    ).sort('fight_date')
    
    homemade_results = attr_fighthist_combined_model.run(combined_sorted)
    homemade_predictions_df = homemade_results['predictions_df'].sort('fight_date')
    
    # Extract homemade predictions
    homemade_preds = (homemade_predictions_df['combined_predicted_winner'] == 1).to_numpy().astype(int)
    homemade_probs = homemade_predictions_df['combined_f1_win_prob'].to_numpy()
    
    print(f"   Homemade predictions: {len(homemade_preds)}")
    
    # ==================== CREATE ENSEMBLE PREDICTIONS ====================
    print("\n4. Creating ensemble predictions...")
    
    all_predictions = {
        'logistic': ml_predictions['logistic'],
        'xgboost': ml_predictions['xgboost'],
        'gradient_boost': ml_predictions['gradient_boost'],
        'homemade': homemade_preds
    }
    
    all_probabilities = {
        'logistic': ml_probabilities['logistic'],
        'xgboost': ml_probabilities['xgboost'],
        'gradient_boost': ml_probabilities['gradient_boost'],
        'homemade': homemade_probs
    }
    
    all_accuracies = {
        'logistic': 0.6351,
        'xgboost': 0.6517,
        'gradient_boost': 0.6611,
        'homemade': 0.6564
    }
    
    # Ensemble 1: Majority Voting (with probability tiebreaker)
    votes = np.array([all_predictions[name] for name in all_predictions.keys()])
    vote_counts = votes.sum(axis=0)
    avg_proba_tiebreaker = np.mean([all_probabilities[name] for name in all_probabilities.keys()], axis=0)
    
    ensemble_majorityvote_pred = np.where(
        vote_counts == 2,
        (avg_proba_tiebreaker > 0.5).astype(int),
        (vote_counts > 2).astype(int)
    )
    ensemble_majorityvote_prob = avg_proba_tiebreaker
    
    # Ensemble 2: Weighted Voting
    total_weight = sum(all_accuracies.values())
    weighted_votes = np.zeros(len(X))
    for name in all_predictions.keys():
        weighted_votes += all_predictions[name] * all_accuracies[name]
    ensemble_weightedvote_pred = (weighted_votes > (total_weight / 2)).astype(int)
    ensemble_weightedvote_prob = weighted_votes / total_weight
    
    # Ensemble 3: Probability Averaging
    ensemble_avgprob_prob = np.mean([all_probabilities[name] for name in all_probabilities.keys()], axis=0)
    ensemble_avgprob_pred = (ensemble_avgprob_prob > 0.5).astype(int)
    
    # Ensemble 4: Weighted Probability Averaging
    weights = {name: acc / total_weight for name, acc in all_accuracies.items()}
    ensemble_weightedavgprob_prob = sum([all_probabilities[name] * weights[name] for name in all_probabilities.keys()])
    ensemble_weightedavgprob_pred = (ensemble_weightedavgprob_prob > 0.5).astype(int)
    
    # ==================== CREATE RESULTS DATAFRAME ====================
    print("\n5. Building results dataframe...")
    
    results_df = df_sorted.select(['fight_id', 'fight_date', 'target']).with_columns([
        # Rename target to actual_winner for clarity
        pl.col('target').alias('actual_winner'),
        
        # Individual model probabilities (for Fighter 1)
        pl.Series('logistic_f1_prob', ml_probabilities['logistic']),
        pl.Series('xgboost_f1_prob', ml_probabilities['xgboost']),
        pl.Series('gradient_f1_prob', ml_probabilities['gradient_boost']),
        pl.Series('homemade_f1_prob', homemade_probs),
        
        # Ensemble probabilities
        pl.Series('ensemble_majorityvote_f1_prob', ensemble_majorityvote_prob),
        pl.Series('ensemble_weightedvote_f1_prob', ensemble_weightedvote_prob),
        pl.Series('ensemble_avgprob_f1_prob', ensemble_avgprob_prob),
        pl.Series('ensemble_weightedavgprob_f1_prob', ensemble_weightedavgprob_prob),
        
        # Individual model predictions
        pl.Series('logistic_pred', ml_predictions['logistic']),
        pl.Series('xgboost_pred', ml_predictions['xgboost']),
        pl.Series('gradient_pred', ml_predictions['gradient_boost']),
        pl.Series('homemade_pred', homemade_preds),
        
        # Ensemble predictions
        pl.Series('ensemble_majorityvote_pred', ensemble_majorityvote_pred),
        pl.Series('ensemble_weightedvote_pred', ensemble_weightedvote_pred),
        pl.Series('ensemble_avgprob_pred', ensemble_avgprob_pred),
        pl.Series('ensemble_weightedavgprob_pred', ensemble_weightedavgprob_pred),
        
        # Best ensemble as predicted_winner
        pl.Series('predicted_winner', ensemble_majorityvote_pred),  # Majority Vote was best
    ]).drop('target')  # Remove duplicate column
    
    # Add confidence scores (distance from 0.5)
    results_df = results_df.with_columns([
        (pl.col('ensemble_majorityvote_f1_prob') - 0.5).abs().mul(2).alias('prediction_confidence')
    ])
    
    # Add correctness flags for ALL models
    results_df = results_df.with_columns([
        # Individual models
        (pl.col('logistic_pred') == pl.col('actual_winner')).alias('logistic_correct'),
        (pl.col('xgboost_pred') == pl.col('actual_winner')).alias('xgboost_correct'),
        (pl.col('gradient_pred') == pl.col('actual_winner')).alias('gradient_correct'),
        (pl.col('homemade_pred') == pl.col('actual_winner')).alias('homemade_correct'),
        
        # Ensemble methods
        (pl.col('ensemble_majorityvote_pred') == pl.col('actual_winner')).alias('ensemble_majorityvote_correct'),
        (pl.col('ensemble_weightedvote_pred') == pl.col('actual_winner')).alias('ensemble_weightedvote_correct'),
        (pl.col('ensemble_avgprob_pred') == pl.col('actual_winner')).alias('ensemble_avgprob_correct'),
        (pl.col('ensemble_weightedavgprob_pred') == pl.col('actual_winner')).alias('ensemble_weightedavgprob_correct'),
        
        # Overall correct (using best ensemble)
        (pl.col('ensemble_majorityvote_pred') == pl.col('actual_winner')).alias('correct'),
    ])
    
    print("\n" + "="*80)
    print("PREDICTION SUMMARY")
    print("="*80)
    
    total = len(results_df)
    
    print(f"\nTotal fights: {total}")
    
    print(f"\nIndividual Model Accuracies:")
    print(f"  Logistic:       {results_df['logistic_correct'].sum():4d}/{total} ({results_df['logistic_correct'].mean()*100:.2f}%)")
    print(f"  XGBoost:        {results_df['xgboost_correct'].sum():4d}/{total} ({results_df['xgboost_correct'].mean()*100:.2f}%)")
    print(f"  Gradient Boost: {results_df['gradient_correct'].sum():4d}/{total} ({results_df['gradient_correct'].mean()*100:.2f}%)")
    print(f"  Homemade:       {results_df['homemade_correct'].sum():4d}/{total} ({results_df['homemade_correct'].mean()*100:.2f}%)")
    
    print(f"\nEnsemble Method Accuracies:")
    print(f"  Majority Vote:            {results_df['ensemble_majorityvote_correct'].sum():4d}/{total} ({results_df['ensemble_majorityvote_correct'].mean()*100:.2f}%) ⭐")
    print(f"  Weighted Vote:            {results_df['ensemble_weightedvote_correct'].sum():4d}/{total} ({results_df['ensemble_weightedvote_correct'].mean()*100:.2f}%)")
    print(f"  Average Probability:      {results_df['ensemble_avgprob_correct'].sum():4d}/{total} ({results_df['ensemble_avgprob_correct'].mean()*100:.2f}%)")
    print(f"  Weighted Avg Probability: {results_df['ensemble_weightedavgprob_correct'].sum():4d}/{total} ({results_df['ensemble_weightedavgprob_correct'].mean()*100:.2f}%)")
    
    print(f"\n🏆 Best Model: Majority Vote Ensemble")
    
    return results_df


def push_predictions_to_sql(predictions_df, connection):
    """
    Push predictions DataFrame to MySQL database.
    
    Parameters:
    -----------
    predictions_df : pl.DataFrame
        DataFrame with all predictions
    connection : mysql.connector.connection.MySQLConnection
        Active MySQL connection object
    
    Returns:
    --------
    int : Number of rows inserted
    """
    
    print("\n" + "="*80)
    print("PUSHING PREDICTIONS TO MySQL")
    print("="*80)
    
    # Add legacy and fight_data_coverage columns
    df_for_sql = predictions_df.with_columns([
        pl.lit(True).alias('legacy'),
        pl.lit(1.0).alias('fight_data_coverage')
    ])
    
    # Convert to pandas for easier MySQL insertion
    import pandas as pd
    df_pandas = df_for_sql.to_pandas()
    
    # Convert boolean columns to int (MySQL BOOLEAN is stored as TINYINT)
    bool_cols = [
        'logistic_correct', 'xgboost_correct', 'gradient_correct', 'homemade_correct',
        'ensemble_majorityvote_correct', 'ensemble_weightedvote_correct',
        'ensemble_avgprob_correct', 'ensemble_weightedavgprob_correct',
        'correct', 'legacy'
    ]
    
    for col in bool_cols:
        if col in df_pandas.columns:
            df_pandas[col] = df_pandas[col].astype(int)
    
    print(f"\nPreparing to insert {len(df_pandas)} rows...")
    
    # Create INSERT statement
    columns = df_pandas.columns.tolist()
    placeholders = ', '.join(['%s'] * len(columns))
    column_names = ', '.join(columns)
    
    insert_query = f"""
        INSERT INTO ufc.predictions ({column_names})
        VALUES ({placeholders})
        ON DUPLICATE KEY UPDATE
            fight_date = VALUES(fight_date),
            actual_winner = VALUES(actual_winner),
            logistic_f1_prob = VALUES(logistic_f1_prob),
            xgboost_f1_prob = VALUES(xgboost_f1_prob),
            gradient_f1_prob = VALUES(gradient_f1_prob),
            homemade_f1_prob = VALUES(homemade_f1_prob),
            ensemble_majorityvote_f1_prob = VALUES(ensemble_majorityvote_f1_prob),
            ensemble_weightedvote_f1_prob = VALUES(ensemble_weightedvote_f1_prob),
            ensemble_avgprob_f1_prob = VALUES(ensemble_avgprob_f1_prob),
            ensemble_weightedavgprob_f1_prob = VALUES(ensemble_weightedavgprob_f1_prob),
            logistic_pred = VALUES(logistic_pred),
            xgboost_pred = VALUES(xgboost_pred),
            gradient_pred = VALUES(gradient_pred),
            homemade_pred = VALUES(homemade_pred),
            ensemble_majorityvote_pred = VALUES(ensemble_majorityvote_pred),
            ensemble_weightedvote_pred = VALUES(ensemble_weightedvote_pred),
            ensemble_avgprob_pred = VALUES(ensemble_avgprob_pred),
            ensemble_weightedavgprob_pred = VALUES(ensemble_weightedavgprob_pred),
            predicted_winner = VALUES(predicted_winner),
            prediction_confidence = VALUES(prediction_confidence),
            logistic_correct = VALUES(logistic_correct),
            xgboost_correct = VALUES(xgboost_correct),
            gradient_correct = VALUES(gradient_correct),
            homemade_correct = VALUES(homemade_correct),
            ensemble_majorityvote_correct = VALUES(ensemble_majorityvote_correct),
            ensemble_weightedvote_correct = VALUES(ensemble_weightedvote_correct),
            ensemble_avgprob_correct = VALUES(ensemble_avgprob_correct),
            ensemble_weightedavgprob_correct = VALUES(ensemble_weightedavgprob_correct),
            correct = VALUES(correct),
            legacy = VALUES(legacy),
            fight_data_coverage = VALUES(fight_data_coverage)
    """
    
    cursor = connection.cursor()
    
    try:
        # Insert in batches of 1000 for efficiency
        batch_size = 1000
        total_rows = len(df_pandas)
        rows_inserted = 0
        
        for start_idx in range(0, total_rows, batch_size):
            end_idx = min(start_idx + batch_size, total_rows)
            batch = df_pandas.iloc[start_idx:end_idx]
            
            # Convert batch to list of tuples
            data = [tuple(row) for row in batch.values]
            
            # Execute batch insert
            cursor.executemany(insert_query, data)
            connection.commit()
            
            rows_inserted += len(batch)
            print(f"   Inserted {rows_inserted}/{total_rows} rows ({rows_inserted/total_rows*100:.1f}%)")
        
        print(f"\n✅ Successfully inserted {rows_inserted} predictions into MySQL!")
        
        # Verify insertion
        cursor.execute("SELECT COUNT(*) FROM ufc.predictions WHERE legacy = 1")
        count = cursor.fetchone()[0]
        print(f"   Total legacy predictions in database: {count}")
        
        return rows_inserted
        
    except Exception as e:
        connection.rollback()
        print(f"\n❌ Error inserting data: {e}")
        raise
        
    finally:
        cursor.close()
    
    print("\n✅ Done! Check 'all_model_predictions.csv' for full results")

# ==================== MAIN SCRIPT ====================
if __name__ == "__main__":
    
    print("Loading data...")
    differential_df = pl.read_csv('trainingset.csv')
    combined_df = pl.read_parquet('fight_features_extracted.parquet')
    
    print(f"Loaded {len(differential_df)} fights\n")
    
    # Load models
    loaded_models = load_all_models()
    
    # Generate all predictions
    predictions_df = generate_all_predictions(differential_df, combined_df, loaded_models)
    
    # Save to CSV
    output_file = 'all_model_predictions.csv'
    predictions_df.write_csv(output_file)
    print(f"\n✅ Predictions saved to: {output_file}")
    
    # Show sample
    print("\nSample predictions (first 10 rows):")
    print(predictions_df.select([
        'fight_id',
        'fight_date',
        'actual_winner',
        'predicted_winner',
        'logistic_f1_prob',
        'xgboost_f1_prob',
        'gradient_f1_prob',
        'homemade_f1_prob',
        'ensemble_majorityvote_f1_prob',
        'prediction_confidence',
        'correct'
    ]).head(10))
    
    print("\n✅ Done! Check 'all_model_predictions.csv' for full results")
    conn = utils.create_connection()
    push_predictions_to_sql(predictions_df, conn)