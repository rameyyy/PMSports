import polars as pl
import pickle
import numpy as np
from .utils import create_connection, fetch_query

def get_model_accuracies() -> dict:
    """
    Calculates accuracy for each model (logistic, gradient_boost, xgboost)
    as SUM(correct=1)/COUNT(non-null correct), ignoring NULLs.
    """

    query = """
    SELECT 
        CAST(SUM(CASE WHEN logistic_correct = 1 THEN 1 ELSE 0 END) AS DECIMAL(10,6)) 
            / NULLIF(COUNT(logistic_correct), 0) AS logistic_acc,

        CAST(SUM(CASE WHEN gradient_correct = 1 THEN 1 ELSE 0 END) AS DECIMAL(10,6)) 
            / NULLIF(COUNT(gradient_correct), 0) AS gradient_boost_acc,

        CAST(SUM(CASE WHEN xgboost_correct = 1 THEN 1 ELSE 0 END) AS DECIMAL(10,6)) 
            / NULLIF(COUNT(xgboost_correct), 0) AS xgboost_acc
    FROM ufc.predictions;
    """
    
    conn = create_connection()
    df = fetch_query(conn, query)

    return {
        "logistic": float(df[0, "logistic_acc"]),
        "xgboost": float(df[0, "xgboost_acc"]),
        "gradient_boost": float(df[0, "gradient_boost_acc"])
    }


def load_saved_models():
    """
    Load all saved models from disk.
    
    Returns:
    --------
    dict
        Dictionary containing:
        - 'models': dict of trained ML models
        - 'scaler': fitted StandardScaler
        - 'feature_cols': list of feature column names
        - 'metadata': training metadata
    """
    print("Loading saved models...")
    
    models = {}
    
    # Load scaler
    with open('models/ufc_mma/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    print("   ✅ Loaded scaler")
    
    # Load feature columns
    with open('models/ufc_mma/feature_columns.pkl', 'rb') as f:
        feature_cols = pickle.load(f)
    print("   ✅ Loaded feature columns")
    
    # Load ML models
    for name in ['logistic', 'xgboost', 'gradient_boost']:
        with open(f'models/ufc_mma/{name}_model.pkl', 'rb') as f:
            models[name] = pickle.load(f)
        print(f"   ✅ Loaded {name} model")
    
    # Load metadata
    with open('models/ufc_mma/model_metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    print("   ✅ Loaded metadata")
    
    print(f"\n✅ All models loaded successfully!")
    print(f"   Trained on: {metadata['training_date']}")
    print(f"   Total fights: {metadata['total_fights']}")
    print(f"   Date range: {metadata['date_range']}")
    
    return {
        'models': models,
        'scaler': scaler,
        'feature_cols': feature_cols,
        'metadata': metadata
    }


def predict_with_saved_models(flat_df, loaded_models=None):
    """
    Generate predictions using saved models on flat differential features.
    
    Parameters:
    -----------
    flat_df : pl.DataFrame
        DataFrame from process_snapshots_to_flat_features()
        Must have differential features and categorical columns
    loaded_models : dict, optional
        Pre-loaded models. If None, will load from disk.
    
    Returns:
    --------
    pl.DataFrame
        Original dataframe with added prediction columns:
        - logistic_pred (0 or 1)
        - logistic_prob (probability fighter 1 wins)
        - xgboost_pred
        - xgboost_prob
        - gradient_boost_pred
        - gradient_boost_prob
        - ensemble_majorityvote_pred
        - ensemble_majorityvote_prob
        - ensemble_weightedvote_pred
        - ensemble_weightedvote_prob
        - ensemble_avgprob_pred
        - ensemble_avgprob_prob
        - ensemble_weightedavgprob_pred
        - ensemble_weightedavgprob_prob
        - predicted_winner (best ensemble prediction)
        - prediction_confidence (0-1, how confident)
    
    Usage:
        predictions_df = predict_with_saved_models(flat_df)
    """
    
    print("\n" + "="*80)
    print("GENERATING PREDICTIONS WITH SAVED MODELS")
    print("="*80)
    
    # Load models if not provided
    if loaded_models is None:
        loaded_models = load_saved_models()
    
    models = loaded_models['models']
    scaler = loaded_models['scaler']
    feature_cols = loaded_models['feature_cols']
    
    print(f"\nProcessing {len(flat_df)} fights...")
    
    # ==================== PREPARE FEATURES ====================
    print("\n1. Preparing features...")
    
    # One-hot encode categorical columns
    exclude_cols = ['fight_id', 'fight_date', 'target', 'weight_class', 'stance_matchup']
    df_encoded = flat_df.to_dummies(columns=['weight_class', 'stance_matchup'])
    
    # Ensure all feature columns exist (add missing ones as 0)
    for col in feature_cols:
        if col not in df_encoded.columns:
            df_encoded = df_encoded.with_columns(pl.lit(0).alias(col))
    
    # Extract features in correct order
    X = df_encoded.select(feature_cols).to_numpy()
    
    # Scale features
    X_scaled = scaler.transform(X)
    
    print(f"   Features prepared: {X.shape}")
    
    # ==================== GENERATE PREDICTIONS ====================
    print("\n2. Generating predictions from ML models...")
    
    predictions = {}
    probabilities = {}
    
    for name, model in models.items():
        print(f"   Predicting with {name}...")
        predictions[name] = model.predict(X_scaled)
        probabilities[name] = model.predict_proba(X_scaled)[:, 1]
    
    # ==================== CREATE ENSEMBLE PREDICTIONS ====================
    print("\n3. Creating ensemble predictions...")
    
    # Model accuracies (from training/validation)
    accuracies = get_model_accuracies()
    
    # Ensemble 1: Majority Vote (with probability tiebreaker)
    votes = np.array([predictions[name] for name in predictions.keys()])
    vote_counts = votes.sum(axis=0)
    avg_prob_tiebreaker = np.mean([probabilities[name] for name in probabilities.keys()], axis=0)
    
    ensemble_majorityvote_pred = np.where(
        vote_counts == 1.5,  # Tie (3 models, so 1.5 is impossible but we check == 2 for real tie in 3-way)
        (avg_prob_tiebreaker > 0.5).astype(int),
        (vote_counts >= 2).astype(int)  # Majority (2 or more out of 3)
    )
    ensemble_majorityvote_prob = avg_prob_tiebreaker
    
    # Ensemble 2: Weighted Vote
    total_weight = sum(accuracies.values())
    weighted_votes = np.zeros(len(X))
    for name in predictions.keys():
        weighted_votes += predictions[name] * accuracies[name]
    ensemble_weightedvote_pred = (weighted_votes > (total_weight / 2)).astype(int)
    ensemble_weightedvote_prob = weighted_votes / total_weight
    
    # Ensemble 3: Average Probability
    ensemble_avgprob_prob = np.mean([probabilities[name] for name in probabilities.keys()], axis=0)
    ensemble_avgprob_pred = (ensemble_avgprob_prob > 0.5).astype(int)
    
    # Ensemble 4: Weighted Average Probability
    weights = {name: acc / total_weight for name, acc in accuracies.items()}
    ensemble_weightedavgprob_prob = sum([probabilities[name] * weights[name] for name in probabilities.keys()])
    ensemble_weightedavgprob_pred = (ensemble_weightedavgprob_prob > 0.5).astype(int)
    
    # ==================== ADD PREDICTIONS TO DATAFRAME ====================
    print("\n4. Adding predictions to dataframe...")
    
    result_df = flat_df.with_columns([
        # Individual model predictions
        pl.Series('logistic_pred', predictions['logistic']),
        pl.Series('logistic_prob', probabilities['logistic']),
        
        pl.Series('xgboost_pred', predictions['xgboost']),
        pl.Series('xgboost_prob', probabilities['xgboost']),
        
        pl.Series('gradient_boost_pred', predictions['gradient_boost']),
        pl.Series('gradient_boost_prob', probabilities['gradient_boost']),
        
        # Ensemble predictions
        pl.Series('ensemble_majorityvote_pred', ensemble_majorityvote_pred),
        pl.Series('ensemble_majorityvote_prob', ensemble_majorityvote_prob),
        
        pl.Series('ensemble_weightedvote_pred', ensemble_weightedvote_pred),
        pl.Series('ensemble_weightedvote_prob', ensemble_weightedvote_prob),
        
        pl.Series('ensemble_avgprob_pred', ensemble_avgprob_pred),
        pl.Series('ensemble_avgprob_prob', ensemble_avgprob_prob),
        
        pl.Series('ensemble_weightedavgprob_pred', ensemble_weightedavgprob_pred),
        pl.Series('ensemble_weightedavgprob_prob', ensemble_weightedavgprob_prob),
        
        # Best prediction (Majority Vote was best in testing)
        pl.Series('predicted_winner', ensemble_majorityvote_pred),
        
        # Confidence (distance from 0.5, scaled 0-1)
        pl.Series('prediction_confidence', np.abs(ensemble_majorityvote_prob - 0.5) * 2),
    ])
    
    # Add correctness if target exists
    # Add correctness if target exists AND is not null
    if 'target' in result_df.columns:
        result_df = result_df.with_columns([
            pl.when(pl.col('target').is_null())
            .then(None)
            .otherwise(pl.col('logistic_pred') == pl.col('target'))
            .alias('logistic_correct'),
            
            pl.when(pl.col('target').is_null())
            .then(None)
            .otherwise(pl.col('xgboost_pred') == pl.col('target'))
            .alias('xgboost_correct'),
            
            pl.when(pl.col('target').is_null())
            .then(None)
            .otherwise(pl.col('gradient_boost_pred') == pl.col('target'))
            .alias('gradient_boost_correct'),
            
            pl.when(pl.col('target').is_null())
            .then(None)
            .otherwise(pl.col('ensemble_majorityvote_pred') == pl.col('target'))
            .alias('ensemble_majorityvote_correct'),
            
            pl.when(pl.col('target').is_null())
            .then(None)
            .otherwise(pl.col('ensemble_weightedvote_pred') == pl.col('target'))
            .alias('ensemble_weightedvote_correct'),
            
            pl.when(pl.col('target').is_null())
            .then(None)
            .otherwise(pl.col('ensemble_avgprob_pred') == pl.col('target'))
            .alias('ensemble_avgprob_correct'),
            
            pl.when(pl.col('target').is_null())
            .then(None)
            .otherwise(pl.col('ensemble_weightedavgprob_pred') == pl.col('target'))
            .alias('ensemble_weightedavgprob_correct'),
            
            pl.when(pl.col('target').is_null())
            .then(None)
            .otherwise(pl.col('predicted_winner') == pl.col('target'))
            .alias('correct'),
        ])
        
        print("\n" + "="*80)
        print("PREDICTION SUMMARY")
        print("="*80)
        
        total = len(result_df)
        
        print(f"\nTotal fights: {total}")
        print(f"\nIndividual Model Accuracies:")
        print(f"  Logistic:       {result_df['logistic_correct'].sum():4d}/{total} ({result_df['logistic_correct'].mean()*100:.2f}%)")
        print(f"  XGBoost:        {result_df['xgboost_correct'].sum():4d}/{total} ({result_df['xgboost_correct'].mean()*100:.2f}%)")
        print(f"  Gradient Boost: {result_df['gradient_boost_correct'].sum():4d}/{total} ({result_df['gradient_boost_correct'].mean()*100:.2f}%)")
        
        print(f"\nEnsemble Method Accuracies:")
        print(f"  Majority Vote:            {result_df['ensemble_majorityvote_correct'].sum():4d}/{total} ({result_df['ensemble_majorityvote_correct'].mean()*100:.2f}%) ⭐")
        print(f"  Weighted Vote:            {result_df['ensemble_weightedvote_correct'].sum():4d}/{total} ({result_df['ensemble_weightedvote_correct'].mean()*100:.2f}%)")
        print(f"  Average Probability:      {result_df['ensemble_avgprob_correct'].sum():4d}/{total} ({result_df['ensemble_avgprob_correct'].mean()*100:.2f}%)")
        print(f"  Weighted Avg Probability: {result_df['ensemble_weightedavgprob_correct'].sum():4d}/{total} ({result_df['ensemble_weightedavgprob_correct'].mean()*100:.2f}%)")
    
    print("\n" + "="*80)
    print(f"✅ PREDICTIONS COMPLETE: {len(result_df)} fights")
    print("="*80)
    
    # Return only essential columns
    essential_cols = [
        'fight_id',
        'fight_date',
        'logistic_pred',
        'logistic_prob',
        'xgboost_pred',
        'xgboost_prob',
        'gradient_boost_pred',
        'gradient_boost_prob',
        'ensemble_majorityvote_pred',
        'ensemble_majorityvote_prob',
        'ensemble_weightedvote_pred',
        'ensemble_weightedvote_prob',
        'ensemble_avgprob_pred',
        'ensemble_avgprob_prob',
        'ensemble_weightedavgprob_pred',
        'ensemble_weightedavgprob_prob',
        'predicted_winner',
        'prediction_confidence'
    ]
    
    return result_df.select(essential_cols)


def quick_predict(flat_df):
    """
    Convenience function: Load models and predict in one call.
    
    Parameters:
    -----------
    flat_df : pl.DataFrame
        DataFrame from process_snapshots_to_flat_features()
    
    Returns:
    --------
    pl.DataFrame
        Dataframe with predictions added
    
    Usage:
        predictions = quick_predict(flat_df)
    """
    loaded_models = load_saved_models()
    return predict_with_saved_models(flat_df, loaded_models)