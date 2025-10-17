import polars as pl
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb

def train_and_evaluate_model(differential_df, model_type='logistic', test_split=0.2):
    """
    Train and evaluate ML model on MMA fight data with time-based split.
    
    Parameters:
    -----------
    differential_df : pl.DataFrame
        DataFrame with differential features from create_differential_features()
    model_type : str
        'logistic', 'xgboost', or 'gradient_boost'
    test_split : float
        Proportion of most recent data to use for testing (default 0.2 = 20%)
    
    Returns:
    --------
    dict
        Dictionary containing model, accuracy, and detailed results
    """
    
    print("="*80)
    print("MMA FIGHT PREDICTION MODEL TRAINING")
    print("="*80)
    
    # ==================== PREPARE DATA ====================
    print("\n1. Preparing data...")
    
    # Sort by date (critical for time-series split!)
    df_sorted = differential_df.sort('fight_date')
    
    # Calculate split point
    n_total = len(df_sorted)
    n_train = int(n_total * (1 - test_split))
    
    # Ensure no date overlap - find the actual date boundary
    split_date = df_sorted[n_train]['fight_date']
    
    # Split with strict date separation
    train_df = df_sorted.filter(pl.col('fight_date') < split_date)
    test_df = df_sorted.filter(pl.col('fight_date') >= split_date)
    
    print(f"   Total fights: {n_total}")
    print(f"   Training fights: {len(train_df)} ({len(train_df)/n_total*100:.1f}%)")
    print(f"   Testing fights: {len(test_df)} ({len(test_df)/n_total*100:.1f}%)")
    
    print(f"   Training date range: {train_df['fight_date'].min()} to {train_df['fight_date'].max()}")
    print(f"   Testing date range: {test_df['fight_date'].min()} to {test_df['fight_date'].max()}")
    
    # ==================== PREPARE FEATURES ====================
    print("\n2. Preparing features...")
    
    # Columns to exclude from features
    exclude_cols = ['fight_id', 'fight_date', 'target', 'weight_class', 'stance_matchup']
    
    # One-hot encode categorical features
    train_encoded = train_df.to_dummies(columns=['weight_class', 'stance_matchup'])
    test_encoded = test_df.to_dummies(columns=['weight_class', 'stance_matchup'])
    
    # Get feature columns (everything except excluded columns)
    feature_cols = [col for col in train_encoded.columns if col not in exclude_cols]
    
    # Ensure test set has same columns as train (in case of missing categories)
    for col in feature_cols:
        if col not in test_encoded.columns:
            test_encoded = test_encoded.with_columns(pl.lit(0).alias(col))
    
    # Extract features and target
    X_train = train_encoded.select(feature_cols).to_numpy()
    y_train = train_encoded['target'].to_numpy()
    
    X_test = test_encoded.select(feature_cols).to_numpy()
    y_test = test_encoded['target'].to_numpy()
    
    print(f"   Number of features: {len(feature_cols)}")
    print(f"   Training set shape: {X_train.shape}")
    print(f"   Test set shape: {X_test.shape}")
    
    # ==================== SCALE FEATURES ====================
    print("\n3. Scaling features...")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # ==================== TRAIN MODEL ====================
    print(f"\n4. Training {model_type} model...")
    
    if model_type == 'logistic':
        model = LogisticRegression(max_iter=1000, random_state=42)
    elif model_type == 'xgboost':
        model = xgb.XGBClassifier(
            n_estimators=50,        # Reduced from 100 to prevent overfitting
            max_depth=2,            # Reduced from 6 to prevent overfitting
            learning_rate=0.08,
            min_child_weight=9,     # Prevent overfitting
            subsample=0.75,          # Use 80% of data per tree
            colsample_bytree=0.75,   # Use 80% of features per tree
            random_state=42,
            eval_metric='logloss'
        )
    elif model_type == 'gradient_boost':
        model = GradientBoostingClassifier(
            n_estimators=88,        # Reduced from 100
            max_depth=2,            # Reduced from 5
            learning_rate=0.07,
            min_samples_split=20,   # Prevent overfitting
            min_samples_leaf=10,    # Prevent overfitting
            subsample=0.8,          # Use 80% of data per tree
            random_state=42
        )
    else:
        raise ValueError("model_type must be 'logistic', 'xgboost', or 'gradient_boost'")
    
    model.fit(X_train_scaled, y_train)
    print("   Training complete!")
    
    # ==================== MAKE PREDICTIONS ====================
    print("\n5. Making predictions...")
    
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    # Get probability predictions if available
    if hasattr(model, 'predict_proba'):
        y_test_proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        y_test_proba = None
    
    # ==================== EVALUATE MODEL ====================
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    print(f"\nTRAINING ACCURACY: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
    print(f"TESTING ACCURACY:  {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    if train_accuracy - test_accuracy > 0.1:
        print("\n⚠️  Warning: Large gap between train and test accuracy suggests overfitting")
    
    # Detailed classification report
    print("\n" + "-"*80)
    print("DETAILED CLASSIFICATION REPORT (Test Set):")
    print("-"*80)
    print(classification_report(y_test, y_test_pred, 
                                target_names=['Fighter 2 Wins', 'Fighter 1 Wins'],
                                digits=4))
    
    # Confusion matrix
    print("\nCONFUSION MATRIX (Test Set):")
    print("-"*80)
    cm = confusion_matrix(y_test, y_test_pred)
    print(f"                 Predicted F2 Win  |  Predicted F1 Win")
    print(f"Actual F2 Win         {cm[0][0]:>6}       |       {cm[0][1]:>6}")
    print(f"Actual F1 Win         {cm[1][0]:>6}       |       {cm[1][1]:>6}")
    
    # ==================== FEATURE IMPORTANCE ====================
    if model_type in ['xgboost', 'gradient_boost']:
        print("\n" + "-"*80)
        print("TOP 20 MOST IMPORTANT FEATURES:")
        print("-"*80)
        
        feature_importance = model.feature_importances_
        importance_df = pl.DataFrame({
            'feature': feature_cols,
            'importance': feature_importance
        }).sort('importance', descending=True)
        
        for i, row in enumerate(importance_df.head(20).iter_rows(named=True)):
            print(f"{i+1:2d}. {row['feature']:50s} {row['importance']:.6f}")
    
    # ==================== PREPARE DETAILED RESULTS ====================
    # Create results dataframe with predictions
    test_results = test_df.select(['fight_id', 'fight_date']).with_columns([
        pl.Series('actual_winner', ['Fighter1' if y == 1 else 'Fighter2' for y in y_test]),
        pl.Series('predicted_winner', ['Fighter1' if y == 1 else 'Fighter2' for y in y_test_pred]),
        pl.Series('correct', y_test == y_test_pred)
    ])
    
    if y_test_proba is not None:
        test_results = test_results.with_columns([
            pl.Series('confidence', y_test_proba)
        ])
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Model Type: {model_type}")
    print(f"Training Accuracy: {train_accuracy*100:.2f}%")
    print(f"Testing Accuracy: {test_accuracy*100:.2f}%")
    print(f"Correct Predictions: {(y_test == y_test_pred).sum()} / {len(y_test)}")
    print(f"Wrong Predictions: {(y_test != y_test_pred).sum()} / {len(y_test)}")
    
    # Return results
    return {
        'model': model,
        'scaler': scaler,
        'feature_cols': feature_cols,
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'y_test': y_test,
        'y_test_pred': y_test_pred,
        'y_test_proba': y_test_proba,
        'test_results': test_results,
        'confusion_matrix': cm
    }