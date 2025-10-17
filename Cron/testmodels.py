import polars as pl
import pickle
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def load_and_evaluate_models(data_path='trainset.csv'):
    """
    Load saved models and evaluate them on the full dataset.
    """
    
    print("="*80)
    print("LOADING SAVED MODELS AND EVALUATING ON FULL DATASET")
    print("="*80)
    
    # ==================== LOAD DATA ====================
    print("\nLoading dataset...")
    df = pl.read_csv(data_path)
    print(f"Loaded {len(df)} fights from {data_path}")
    
    # ==================== LOAD MODELS ====================
    print("\nLoading saved models...")
    
    models = {}
    model_files = {
        'logistic': 'models/ufc_mma/logistic_model.pkl',
        'xgboost': 'models/ufc_mma/xgboost_model.pkl',
        'gradient_boost': 'models/ufc_mma/gradient_boost_model.pkl',
        'scaler': 'models/ufc_mma/scaler.pkl',
        'feature_columns': 'models/ufc_mma/feature_columns.pkl',
        'model_metadata': 'models/ufc_mma/model_metadata.pkl'
    }
    
    for name, filepath in model_files.items():
        try:
            with open(filepath, 'rb') as f:
                models[name] = pickle.load(f)
            print(f"  ✓ Loaded {name}")
        except FileNotFoundError:
            print(f"  ✗ Could not find {filepath}")
            return
        except Exception as e:
            print(f"  ✗ Error loading {name}: {e}")
            return
    
    # ==================== PREPARE FEATURES ====================
    print("\nPreparing features...")
    
    # Load feature columns from separate file
    if 'feature_columns' in models:
        feature_cols = models['feature_columns']
        print(f"✓ Loaded {len(feature_cols)} features from feature_columns.pkl")
    else:
        print("⚠️  WARNING: feature_columns.pkl not found!")
        print("   Attempting to reconstruct (may cause incorrect predictions)")
        exclude_cols = ['fight_id', 'fight_date', 'target', 'weight_class', 'stance_matchup']
        df_encoded_temp = df.to_dummies(columns=['weight_class', 'stance_matchup'])
        feature_cols = [col for col in df_encoded_temp.columns if col not in exclude_cols]
    
    # Prepare data same way as training
    exclude_cols = ['fight_id', 'fight_date', 'target', 'weight_class', 'stance_matchup']
    df_encoded = df.to_dummies(columns=['weight_class', 'stance_matchup'])
    
    # Ensure all required feature columns exist
    for col in feature_cols:
        if col not in df_encoded.columns:
            df_encoded = df_encoded.with_columns(pl.lit(0).alias(col))
    
    # Extract features and target
    X = df_encoded.select(feature_cols).to_numpy()
    y = df_encoded['target'].to_numpy()
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Fighter 1 wins: {y.sum()} ({y.sum()/len(y)*100:.1f}%)")
    print(f"Fighter 2 wins: {(1-y).sum()} ({(1-y).sum()/len(y)*100:.1f}%)")
    
    # Scale features
    print("\nScaling features...")
    X_scaled = models['scaler'].transform(X)
    
    # ==================== EVALUATE EACH MODEL ====================
    print("\n" + "="*80)
    print("MODEL EVALUATION RESULTS")
    print("="*80)
    
    all_predictions = {}
    all_probabilities = {}
    
    model_list = ['logistic', 'xgboost', 'gradient_boost']
    
    for model_name in model_list:
        print(f"\n{model_name.upper().replace('_', ' ')}")
        print("-"*80)
        
        model = models[model_name]
        
        # Make predictions
        y_pred = model.predict(X_scaled)
        y_proba = model.predict_proba(X_scaled)[:, 1]
        
        all_predictions[model_name] = y_pred
        all_probabilities[model_name] = y_proba
        
        # Calculate accuracy
        accuracy = accuracy_score(y, y_pred)
        print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Confusion matrix
        cm = confusion_matrix(y, y_pred)
        print(f"\nConfusion Matrix:")
        print(f"              Predicted F2  Predicted F1")
        print(f"Actual F2:         {cm[0,0]:4d}          {cm[0,1]:4d}")
        print(f"Actual F1:         {cm[1,0]:4d}          {cm[1,1]:4d}")
        
        # Classification report
        print(f"\nClassification Report:")
        report = classification_report(y, y_pred, target_names=['Fighter 2', 'Fighter 1'])
        print(report)
        
        # Confidence analysis
        print(f"Confidence Distribution (2% bins):")
        bins = [(i/100, (i+2)/100) for i in range(0, 100, 2)]
        for low, high in bins:
            mask = (y_proba >= low) & (y_proba < high)
            count = mask.sum()
            if count > 0:
                bin_accuracy = (y[mask] == (y_proba[mask] > 0.5).astype(int)).sum() / count
                avg_conf = y_proba[mask].mean()
                winner = "F2" if low < 0.5 else "F1"
                print(f"  {low:.0%}-{high:.0%}: {count:3d} fights (avg {avg_conf:.1%}), accuracy {bin_accuracy:.1%} (predicted {winner})")
    
    # ==================== CREATE ENSEMBLE PREDICTIONS ====================
    print("\n" + "="*80)
    print("ENSEMBLE PREDICTIONS")
    print("="*80)
    
    # Simple average of probabilities
    avg_proba = np.mean([all_probabilities[name] for name in model_list], axis=0)
    ensemble_pred = (avg_proba > 0.5).astype(int)
    ensemble_accuracy = accuracy_score(y, ensemble_pred)
    
    print(f"\nEnsemble (Average Probabilities):")
    print(f"Accuracy: {ensemble_accuracy:.4f} ({ensemble_accuracy*100:.2f}%)")
    
    # Confusion matrix
    cm = confusion_matrix(y, ensemble_pred)
    print(f"\nConfusion Matrix:")
    print(f"              Predicted F2  Predicted F1")
    print(f"Actual F2:         {cm[0,0]:4d}          {cm[0,1]:4d}")
    print(f"Actual F1:         {cm[1,0]:4d}          {cm[1,1]:4d}")
    
    # ==================== SAVE RESULTS ====================
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    results_df = df.select(['fight_id', 'fight_date', 'target']).with_columns([
        pl.Series('logistic_pred', all_predictions['logistic']),
        pl.Series('xgboost_pred', all_predictions['xgboost']),
        pl.Series('gradient_pred', all_predictions['gradient_boost']),
        pl.Series('ensemble_pred', ensemble_pred),
        pl.Series('logistic_f1_prob', all_probabilities['logistic']),
        pl.Series('xgboost_f1_prob', all_probabilities['xgboost']),
        pl.Series('gradient_f1_prob', all_probabilities['gradient_boost']),
        pl.Series('ensemble_f1_prob', avg_proba),
    ])
    
    # Add correct/incorrect columns
    results_df = results_df.with_columns([
        (pl.col('logistic_pred') == pl.col('target')).alias('logistic_correct'),
        (pl.col('xgboost_pred') == pl.col('target')).alias('xgboost_correct'),
        (pl.col('gradient_pred') == pl.col('target')).alias('gradient_correct'),
        (pl.col('ensemble_pred') == pl.col('target')).alias('ensemble_correct'),
    ])
    
    output_file = 'full_dataset_predictions.csv'
    results_df.write_csv(output_file)
    print(f"\n✓ Results saved to '{output_file}'")
    
    # ==================== SUMMARY ====================
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    print(f"\nTotal Fights Evaluated: {len(df)}")
    print(f"\nModel Accuracies:")
    print(f"  Logistic Regression:   {accuracy_score(y, all_predictions['logistic'])*100:.2f}%")
    print(f"  XGBoost:              {accuracy_score(y, all_predictions['xgboost'])*100:.2f}%")
    print(f"  Gradient Boost:       {accuracy_score(y, all_predictions['gradient_boost'])*100:.2f}%")
    print(f"  Ensemble (Avg Prob):  {ensemble_accuracy*100:.2f}%")
    
    best_model = max([
        ('Logistic', accuracy_score(y, all_predictions['logistic'])),
        ('XGBoost', accuracy_score(y, all_predictions['xgboost'])),
        ('Gradient Boost', accuracy_score(y, all_predictions['gradient_boost'])),
        ('Ensemble', ensemble_accuracy)
    ], key=lambda x: x[1])
    
    print(f"\n🏆 Best Model: {best_model[0]} with {best_model[1]*100:.2f}% accuracy")
    
    return results_df


if __name__ == "__main__":
    results = load_and_evaluate_models('trainingset.csv')
    print("\n✅ Evaluation complete!")