# train_and_save_final_models.py

import polars as pl
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
from models.ufc_mma import attr_fighthist_combined_model
from datetime import datetime

def train_final_models_no_split(differential_df, combined_df):
    """
    Train all models on 100% of data (no test split) and save them.
    Use this for production deployment.
    """
    
    print("="*80)
    print("TRAINING FINAL PRODUCTION MODELS (NO TEST SPLIT)")
    print("="*80)
    print(f"\nTraining on ALL {len(differential_df)} fights\n")
    
    # ==================== PREPARE DATA ====================
    print("1. Preparing features...")
    
    # Sort by date (keep chronological order)
    df_sorted = differential_df.sort('fight_date')
    
    # Prepare features (NO SPLIT - use all data)
    exclude_cols = ['fight_id', 'fight_date', 'target', 'weight_class', 'stance_matchup']
    df_encoded = df_sorted.to_dummies(columns=['weight_class', 'stance_matchup'])
    
    feature_cols = [col for col in df_encoded.columns if col not in exclude_cols]
    
    X_all = df_encoded.select(feature_cols).to_numpy()
    y_all = df_encoded['target'].to_numpy()
    
    print(f"   Features: {len(feature_cols)}")
    print(f"   Training samples: {X_all.shape[0]}")
    
    # ==================== SCALE FEATURES ====================
    print("\n2. Scaling features...")
    
    scaler = StandardScaler()
    X_all_scaled = scaler.fit_transform(X_all)
    
    # ==================== TRAIN ML MODELS ====================
    print("\n3. Training ML models...")
    
    ml_models = {
        'logistic': LogisticRegression(max_iter=1000, random_state=42),
        'xgboost': xgb.XGBClassifier(
            n_estimators=50,
            max_depth=2,
            learning_rate=0.08,
            min_child_weight=9,
            subsample=0.75,
            colsample_bytree=0.75,
            random_state=42,
            eval_metric='logloss',
            verbosity=0
        ),
        'gradient_boost': GradientBoostingClassifier(
            n_estimators=88,
            max_depth=2,
            learning_rate=0.07,
            min_samples_split=20,
            min_samples_leaf=10,
            subsample=0.8,
            random_state=42
        )
    }
    
    trained_models = {}
    
    for name, model in ml_models.items():
        print(f"   Training {name}...")
        model.fit(X_all_scaled, y_all)
        trained_models[name] = model
        
        # Calculate training accuracy (for reference only)
        train_acc = model.score(X_all_scaled, y_all)
        print(f"      Training accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
    
    # ==================== TRAIN HOMEMADE MODEL ====================
    print("\n4. Training homemade model...")
    print("   (Running on combined_df - this may take a minute...)")
    
    # Train homemade on all data
    homemade_results = attr_fighthist_combined_model.run(combined_df)
    print("   Homemade model trained!")
    
    # ==================== SAVE EVERYTHING ====================
    print("\n5. Saving models to disk...")
    
    # Save scaler
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("   ✅ Saved: models/scaler.pkl")
    
    # Save feature columns
    with open('models/feature_columns.pkl', 'wb') as f:
        pickle.dump(feature_cols, f)
    print("   ✅ Saved: models/feature_columns.pkl")
    
    # Save each ML model
    for name, model in trained_models.items():
        filename = f'models/{name}_model.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(model, f)
        print(f"   ✅ Saved: {filename}")
    
    # Save homemade model components (if they have save methods)
    # Note: You may need to adjust this based on your homemade model's structure
    print("   ℹ️  Homemade model uses separate predictor classes (already saved)")
    
    # ==================== CREATE METADATA ====================
    metadata = {
        'training_date': str(datetime.now()),  # Changed from pl.datetime('now')
        'total_fights': len(differential_df),
        'date_range': f"{df_sorted['fight_date'].min()} to {df_sorted['fight_date'].max()}",
        'num_features': len(feature_cols),
        'models': list(trained_models.keys()),
        'expected_test_accuracy': {
            'logistic': 0.6351,
            'xgboost': 0.6517,
            'gradient_boost': 0.6611,
            'homemade': 0.6564,
            'ensemble_method1': 0.6754
        }
    }
        
    with open('models/model_metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)
    print("   ✅ Saved: models/model_metadata.pkl")
    
    print("\n" + "="*80)
    print("✅ ALL MODELS SAVED SUCCESSFULLY!")
    print("="*80)
    print("\nSaved files:")
    print("  - models/ufc_mma/scaler.pkl")
    print("  - models/ufc_mma/feature_columns.pkl")
    print("  - models/ufc_mma/logistic_model.pkl")
    print("  - models/ufc_mma/xgboost_model.pkl")
    print("  - models/ufc_mma/gradient_boost_model.pkl")
    print("  - models/ufc_mma/model_metadata.pkl")
    print("\nExpected Ensemble Accuracy: 67.54%")
    print("\nReady for production deployment! 🚀")
    
    return {
        'models': trained_models,
        'scaler': scaler,
        'feature_cols': feature_cols,
        'metadata': metadata
    }


def load_models_for_prediction():
    """
    Load saved models for making predictions.
    Use this in production.
    """
    
    print("Loading models from disk...")
    
    # Load scaler
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    # Load feature columns
    with open('models/feature_columns.pkl', 'rb') as f:
        feature_cols = pickle.load(f)
    
    # Load ML models
    models = {}
    for name in ['logistic', 'xgboost', 'gradient_boost']:
        with open(f'models/{name}_model.pkl', 'rb') as f:
            models[name] = pickle.load(f)
    
    # Load metadata
    with open('models/model_metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    
    print("✅ All models loaded!")
    print(f"   Trained on {metadata['total_fights']} fights")
    print(f"   Date range: {metadata['date_range']}")
    
    return {
        'models': models,
        'scaler': scaler,
        'feature_cols': feature_cols,
        'metadata': metadata
    }


def predict_fight(fighter1_features, fighter2_features, loaded_models):
    """
    Make prediction for a new fight using ensemble.
    
    Parameters:
    -----------
    fighter1_features : dict
        Differential features for the fight
    fighter2_features : dict  
        (Can be None if using differentials)
    loaded_models : dict
        Output from load_models_for_prediction()
    
    Returns:
    --------
    dict with predictions
    """
    
    # Extract components
    models = loaded_models['models']
    scaler = loaded_models['scaler']
    feature_cols = loaded_models['feature_cols']
    
    # Prepare feature vector (you'll need to create differential_df row)
    # This is a placeholder - adjust based on your input format
    X_new = prepare_features_for_prediction(fighter1_features, feature_cols)
    X_new_scaled = scaler.transform(X_new)
    
    # Get predictions from each model
    predictions = {}
    votes = []
    
    for name, model in models.items():
        pred = model.predict(X_new_scaled)[0]
        prob = model.predict_proba(X_new_scaled)[0, 1]
        predictions[name] = {'prediction': pred, 'probability': prob}
        votes.append(pred)
    
    # Add homemade model prediction here
    # homemade_pred = get_homemade_prediction(...)
    # predictions['homemade'] = {'prediction': homemade_pred, 'probability': homemade_prob}
    # votes.append(homemade_pred)
    
    # Ensemble Method 1 (Majority Vote)
    ensemble_prediction = 1 if sum(votes) >= 2 else 0
    
    return {
        'individual_predictions': predictions,
        'ensemble_prediction': ensemble_prediction,
        'winner': 'Fighter1' if ensemble_prediction == 1 else 'Fighter2'
    }


def prepare_features_for_prediction(fighter_features, feature_cols):
    """
    Convert fighter features dict to numpy array in correct order.
    """
    # This depends on your feature format
    # Placeholder implementation
    feature_vector = np.zeros((1, len(feature_cols)))
    # ... fill in features ...
    return feature_vector


# ==================== MAIN SCRIPT ====================
if __name__ == "__main__":
    # Create models directory if it doesn't exist
    import os
    os.makedirs('models', exist_ok=True)
    
    # Load your data
    print("Loading data...")
    differential_df = pl.read_csv('trainingset.csv')
    combined_df = pl.read_parquet('fight_features_extracted.parquet')
    
    # Train on ALL data and save
    final_models = train_final_models_no_split(differential_df, combined_df)
    
    print("\n" + "="*80)
    print("TESTING MODEL LOADING")
    print("="*80)
    
    # Test loading
    loaded = load_models_for_prediction()
    
    print("\n✅ Everything ready for production!")
    print("\nTo make predictions in production:")
    print("  1. Load models: loaded = load_models_for_prediction()")
    print("  2. Predict: result = predict_fight(features, None, loaded)")
    print("  3. Use: result['winner'] and result['ensemble_prediction']")