# test_all_4_models.py

import polars as pl
import pickle
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
from Cron.models.ufc_mma import flat_df_to_model_ready_training
from models.ufc_mma import attr_fighthist_combined_model

def ensemble_all_models(differential_df, combined_df, test_split=0.25):
    """
    Combine all models with 4 different ensemble methods:
    1. Simple Majority Voting
    2. Weighted Voting (by accuracy)
    3. Probability Averaging
    4. Weighted Probability Averaging
    
    Returns DataFrame with all predictions for each fight.
    """
    
    print("="*80)
    print("TRAINING ALL MODELS AND CREATING ENSEMBLE")
    print("="*80)
    
    # ==================== SPLIT DATA ====================
    df_sorted = differential_df.sort('fight_date')
    n_total = len(df_sorted)
    split_date = df_sorted[int(n_total * (1 - test_split))]['fight_date']
    
    train_df = df_sorted.filter(pl.col('fight_date') < split_date)
    test_df = df_sorted.filter(pl.col('fight_date') >= split_date)
    
    print(f"\nTraining on {len(train_df)} fights (before {split_date})")
    print(f"Testing on {len(test_df)} fights (from {split_date} onwards)")
    
    # ==================== TRAIN ML MODELS (1, 2, 3) ====================
    print("\n" + "-"*80)
    print("Training ML Models (Logistic, XGBoost, Gradient Boost)...")
    print("-"*80)
    
    # Prepare features
    exclude_cols = ['fight_id', 'fight_date', 'target', 'weight_class', 'stance_matchup']
    train_encoded = train_df.to_dummies(columns=['weight_class', 'stance_matchup'])
    test_encoded = test_df.to_dummies(columns=['weight_class', 'stance_matchup'])
    
    feature_cols = [col for col in train_encoded.columns if col not in exclude_cols]
    
    # Ensure test has same columns
    for col in feature_cols:
        if col not in test_encoded.columns:
            test_encoded = test_encoded.with_columns(pl.lit(0).alias(col))
    
    X_train = train_encoded.select(feature_cols).to_numpy()
    y_train = train_encoded['target'].to_numpy()
    X_test = test_encoded.select(feature_cols).to_numpy()
    y_test = test_encoded['target'].to_numpy()
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train models
    ml_models = {
        'logistic': LogisticRegression(max_iter=1000, random_state=42),
        'xgboost': xgb.XGBClassifier(
            n_estimators=50, max_depth=2, learning_rate=0.08,
            min_child_weight=9, subsample=0.75, colsample_bytree=0.75,
            random_state=42, eval_metric='logloss'
        ),
        'gradient_boost': GradientBoostingClassifier(
            n_estimators=88, max_depth=2, learning_rate=0.07,
            min_samples_split=20, min_samples_leaf=10, subsample=0.8,
            random_state=42
        )
    }
    
    ml_predictions = {}
    ml_probabilities = {}
    
    for name, model in ml_models.items():
        print(f"  Training {name}...")
        model.fit(X_train_scaled, y_train)
        ml_predictions[name] = model.predict(X_test_scaled)
        ml_probabilities[name] = model.predict_proba(X_test_scaled)[:, 1]
        accuracy = accuracy_score(y_test, ml_predictions[name])
        print(f"    Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # ==================== GET HOMEMADE MODEL PREDICTIONS (4) ====================
    print("\n" + "-"*80)
    print("Running Homemade Combined Model...")
    print("-"*80)
    
    # Filter combined_df to only include test_df fight_ids
    test_fight_ids = test_df['fight_id'].to_list()
    combined_test_df = combined_df.filter(pl.col('fight_id').is_in(test_fight_ids))
    
    print(f"  Filtered combined_df to {len(combined_test_df)} test fights")
    
    # Run your combined model on TEST data only
    homemade_results = attr_fighthist_combined_model.run(combined_test_df)
    homemade_predictions_df = homemade_results['predictions_df']
    
    # Sort to match test_df order (by fight_date then fight_id)
    homemade_predictions_df = homemade_predictions_df.sort(['fight_date', 'fight_id'])
    test_df_sorted = test_df.sort(['fight_date', 'fight_id'])
    
    # Extract predictions
    homemade_preds = (homemade_predictions_df['combined_predicted_winner'] == 1).to_numpy().astype(int)
    homemade_probs = homemade_predictions_df['combined_f1_win_prob'].to_numpy()
    
    # Verify lengths match
    print(f"  y_test length: {len(y_test)}, homemade_preds length: {len(homemade_preds)}")
    
    if len(homemade_preds) != len(y_test):
        raise ValueError(f"Length mismatch! y_test: {len(y_test)}, homemade_preds: {len(homemade_preds)}")
    
    homemade_accuracy = accuracy_score(y_test, homemade_preds)
    print(f"  Homemade Combined Model Accuracy: {homemade_accuracy:.4f} ({homemade_accuracy*100:.2f}%)")
    
    # ==================== CREATE ENSEMBLES ====================
    print("\n" + "="*80)
    print("CREATING ENSEMBLE PREDICTIONS")
    print("="*80)
    
    # All 4 base models
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
        'logistic': accuracy_score(y_test, all_predictions['logistic']),
        'xgboost': accuracy_score(y_test, all_predictions['xgboost']),
        'gradient_boost': accuracy_score(y_test, all_predictions['gradient_boost']),
        'homemade': homemade_accuracy
    }
    
    # ==================== METHOD 1: SIMPLE MAJORITY VOTING WITH TIEBREAKER ====================
    print("\nMethod 1: Simple Majority Voting (with probability tiebreaker)")
    print("-"*80)

    # Each model votes 0 or 1, take the majority
    votes = np.array([all_predictions[name] for name in all_predictions.keys()])
    vote_counts = votes.sum(axis=0)  # Count how many models voted 1

    # Calculate average probability for tiebreaker
    avg_proba_tiebreaker = np.mean([all_probabilities[name] for name in all_probabilities.keys()], axis=0)

    # Use probability as tiebreaker when vote_counts == 2
    ensemble_pred_method1 = np.where(
        vote_counts == 2,  # If tie (2-2)
        (avg_proba_tiebreaker > 0.5).astype(int),  # Use average probability
        (vote_counts > 2).astype(int)  # Otherwise use majority (>2 means fighter1)
    )

    method1_accuracy = accuracy_score(y_test, ensemble_pred_method1)

    # Count ties
    ties = np.sum(vote_counts == 2)
    print(f"  Ties: {ties} out of {len(y_test)} fights ({ties/len(y_test)*100:.1f}%)")
    print(f"Ensemble Accuracy: {method1_accuracy:.4f} ({method1_accuracy*100:.2f}%)")
    improvement1 = method1_accuracy - max(all_accuracies.values())
    print(f"Improvement over best single model: {improvement1:+.4f} ({improvement1*100:+.2f}%)")
    
    # ==================== METHOD 2: WEIGHTED VOTING ====================
    print("\nMethod 2: Weighted Voting (by accuracy)")
    print("-"*80)
    
    # Weight each vote by model's accuracy
    weighted_votes = np.zeros(len(y_test))
    for name in all_predictions.keys():
        weighted_votes += all_predictions[name] * all_accuracies[name]
    
    # If weighted votes > half of total weight, predict 1
    total_weight = sum(all_accuracies.values())
    ensemble_pred_method2 = (weighted_votes > (total_weight / 2)).astype(int)
    
    method2_accuracy = accuracy_score(y_test, ensemble_pred_method2)
    print(f"Ensemble Accuracy: {method2_accuracy:.4f} ({method2_accuracy*100:.2f}%)")
    improvement2 = method2_accuracy - max(all_accuracies.values())
    print(f"Improvement over best single model: {improvement2:+.4f} ({improvement2*100:+.2f}%)")
    
    # ==================== METHOD 3: PROBABILITY AVERAGING ====================
    print("\nMethod 3: Probability Averaging (All 4 Models)")
    print("-"*80)
    
    avg_proba = np.mean([all_probabilities[name] for name in all_probabilities.keys()], axis=0)
    ensemble_pred_method3 = (avg_proba > 0.5).astype(int)
    method3_accuracy = accuracy_score(y_test, ensemble_pred_method3)
    
    print(f"Ensemble Accuracy: {method3_accuracy:.4f} ({method3_accuracy*100:.2f}%)")
    improvement3 = method3_accuracy - max(all_accuracies.values())
    print(f"Improvement over best single model: {improvement3:+.4f} ({improvement3*100:+.2f}%)")
    
    # ==================== METHOD 4: WEIGHTED PROBABILITY AVERAGING ====================
    print("\nMethod 4: Weighted Probability Averaging (by accuracy)")
    print("-"*80)
    
    # Weight each model by its accuracy
    weights = {name: acc / total_weight for name, acc in all_accuracies.items()}
    
    print("Model weights:")
    for name, weight in weights.items():
        print(f"  {name}: {weight:.4f}")
    
    weighted_proba = sum([all_probabilities[name] * weights[name] for name in all_probabilities.keys()])
    ensemble_pred_method4 = (weighted_proba > 0.5).astype(int)
    method4_accuracy = accuracy_score(y_test, ensemble_pred_method4)
    
    print(f"\nEnsemble Accuracy: {method4_accuracy:.4f} ({method4_accuracy*100:.2f}%)")
    improvement4 = method4_accuracy - max(all_accuracies.values())
    print(f"Improvement over best single model: {improvement4:+.4f} ({improvement4*100:+.2f}%)")
    
    # ==================== CREATE RESULTS DATAFRAME ====================
    print("\n" + "="*80)
    print("CREATING RESULTS DATAFRAME")
    print("="*80)
    
    results_df = test_df_sorted.select(['fight_id', 'fight_date', 'target']).with_columns([
        # Individual model predictions
        pl.Series('logistic_f1_prob', ml_probabilities['logistic']),
        pl.Series('xgboost_f1_prob', ml_probabilities['xgboost']),
        pl.Series('gradient_f1_prob', ml_probabilities['gradient_boost']),
        pl.Series('homemade_f1_prob', homemade_probs),
        
        # Ensemble predictions
        pl.Series('ensemble_method1_f1_prob', ensemble_pred_method1.astype(float)),  # Convert to prob for consistency
        pl.Series('ensemble_method2_f1_prob', (weighted_votes / total_weight)),
        pl.Series('ensemble_method3_f1_prob', avg_proba),
        pl.Series('ensemble_method4_f1_prob', weighted_proba),
        
        # Predicted winners
        pl.Series('logistic_pred', ml_predictions['logistic']),
        pl.Series('xgboost_pred', ml_predictions['xgboost']),
        pl.Series('gradient_pred', ml_predictions['gradient_boost']),
        pl.Series('homemade_pred', homemade_preds),
        pl.Series('ensemble_method1_pred', ensemble_pred_method1),
        pl.Series('ensemble_method2_pred', ensemble_pred_method2),
        pl.Series('ensemble_method3_pred', ensemble_pred_method3),
        pl.Series('ensemble_method4_pred', ensemble_pred_method4),
    ])
    
    # Add correct/incorrect columns
    results_df = results_df.with_columns([
        (pl.col('logistic_pred') == pl.col('target')).alias('logistic_correct'),
        (pl.col('xgboost_pred') == pl.col('target')).alias('xgboost_correct'),
        (pl.col('gradient_pred') == pl.col('target')).alias('gradient_correct'),
        (pl.col('homemade_pred') == pl.col('target')).alias('homemade_correct'),
        (pl.col('ensemble_method1_pred') == pl.col('target')).alias('ensemble_method1_correct'),
        (pl.col('ensemble_method2_pred') == pl.col('target')).alias('ensemble_method2_correct'),
        (pl.col('ensemble_method3_pred') == pl.col('target')).alias('ensemble_method3_correct'),
        (pl.col('ensemble_method4_pred') == pl.col('target')).alias('ensemble_method4_correct'),
    ])
    
    # ==================== FINAL SUMMARY ====================
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    
    print("\nIndividual Model Accuracies:")
    for name, acc in all_accuracies.items():
        print(f"  {name:20s}: {acc:.4f} ({acc*100:.2f}%)")
    
    print("\nEnsemble Accuracies:")
    print(f"  Method 1 (Majority Vote):       {method1_accuracy:.4f} ({method1_accuracy*100:.2f}%)")
    print(f"  Method 2 (Weighted Vote):       {method2_accuracy:.4f} ({method2_accuracy*100:.2f}%)")
    print(f"  Method 3 (Avg Prob):            {method3_accuracy:.4f} ({method3_accuracy*100:.2f}%)")
    print(f"  Method 4 (Weighted Avg Prob):   {method4_accuracy:.4f} ({method4_accuracy*100:.2f}%)")
    
    best_method = max([
        ('Method 1 (Majority Vote)', method1_accuracy),
        ('Method 2 (Weighted Vote)', method2_accuracy),
        ('Method 3 (Avg Prob)', method3_accuracy),
        ('Method 4 (Weighted Avg Prob)', method4_accuracy)
    ], key=lambda x: x[1])
    
    print(f"\n✅ Best Ensemble: {best_method[0]} with {best_method[1]*100:.2f}% accuracy")
    print(f"📈 Improvement: +{(best_method[1] - max(all_accuracies.values()))*100:.2f}%")
    
    # Overall best (including individual models)
    all_methods_accuracy = {
        **all_accuracies,
        'ensemble_method1': method1_accuracy,
        'ensemble_method2': method2_accuracy,
        'ensemble_method3': method3_accuracy,
        'ensemble_method4': method4_accuracy
    }
    
    overall_best = max(all_methods_accuracy.items(), key=lambda x: x[1])
    print(f"\n🏆 Overall Best Model: {overall_best[0]} with {overall_best[1]*100:.2f}% accuracy")
    
    # ==================== PREDICTION CONFIDENCE ANALYSIS ====================
    print("\n" + "="*80)
    print("PREDICTION CONFIDENCE ANALYSIS")
    print("="*80)
    
    # Analyze each model's prediction confidence vs actual accuracy
    models_to_analyze = {
        'Logistic': ml_probabilities['logistic'],
        'XGBoost': ml_probabilities['xgboost'],
        'Gradient Boost': ml_probabilities['gradient_boost'],
        'Homemade': homemade_probs,
        'Ensemble Method 3': avg_proba,
        'Ensemble Method 4': weighted_proba
    }
    
    for model_name, probs in models_to_analyze.items():
        print(f"\n{model_name}:")
        
        # Calculate average prediction confidence for fighter 1
        avg_prob_f1 = np.mean(probs)
        print(f"  Average F1 Win Probability: {avg_prob_f1:.1%} ({len(probs)} fights)")
        
        # Count how many predicted for F1 (prob > 0.5)
        f1_predictions = (probs > 0.5).sum()
        print(f"  Predicted F1 to win: {f1_predictions} fights ({f1_predictions/len(probs)*100:.1f}%)")
        
        # Actual F1 wins
        actual_f1_wins = y_test.sum()
        print(f"  Actual F1 wins: {actual_f1_wins} fights ({actual_f1_wins/len(y_test)*100:.1f}%)")
        
        # Calibration check: For fights where F1 was predicted to win, what % actually won?
        if f1_predictions > 0:
            f1_pred_mask = probs > 0.5
            f1_pred_accuracy = y_test[f1_pred_mask].sum() / f1_predictions
            print(f"  When predicted F1 wins: Actual accuracy = {f1_pred_accuracy:.1%}")
        
        # Confidence bins (2% increments)
        print(f"  Confidence distribution:")
        bins = [(i/100, (i+2)/100) for i in range(0, 100, 2)]
        for low, high in bins:
            mask = (probs >= low) & (probs < high)
            count = mask.sum()
            if count > 0:
                bin_accuracy = (y_test[mask] == (probs[mask] > 0.5).astype(int)).sum() / count
                avg_conf = probs[mask].mean()
                winner = "F2" if low < 0.5 else "F1"
                print(f"    {low:.0%}-{high:.0%}: {count:3d} fights (avg {avg_conf:.1%}), accuracy {bin_accuracy:.1%} (predicted {winner})")
    
    # ==================== PREDICTION CONFIDENCE ANALYSIS ====================
    print("\n" + "="*80)
    print("PREDICTION CONFIDENCE ANALYSIS")
    print("="*80)
    
    # Analyze each model's prediction confidence vs actual accuracy
    models_to_analyze = {
        'Logistic': ml_probabilities['logistic'],
        'XGBoost': ml_probabilities['xgboost'],
        'Gradient Boost': ml_probabilities['gradient_boost'],
        'Homemade': homemade_probs,
        'Ensemble Method 1': ensemble_pred_method1.astype(float),  # Binary, so just 0 or 1
        'Ensemble Method 2': (weighted_votes / total_weight),
        'Ensemble Method 3': avg_proba,
        'Ensemble Method 4': weighted_proba
    }
    
    for model_name, probs in models_to_analyze.items():
        print(f"\n{model_name}:")
        
        # Calculate average prediction confidence for fighter 1
        avg_prob_f1 = np.mean(probs)
        print(f"  Average F1 Win Probability: {avg_prob_f1:.1%} ({len(probs)} fights)")
        
        # Count how many predicted for F1 (prob > 0.5)
        f1_predictions = (probs > 0.5).sum()
        print(f"  Predicted F1 to win: {f1_predictions} fights ({f1_predictions/len(probs)*100:.1f}%)")
        
        # Actual F1 wins
        actual_f1_wins = y_test.sum()
        print(f"  Actual F1 wins: {actual_f1_wins} fights ({actual_f1_wins/len(y_test)*100:.1f}%)")
        
        # Calibration check: For fights where F1 was predicted to win, what % actually won?
        if f1_predictions > 0:
            f1_pred_mask = probs > 0.5
            f1_pred_accuracy = y_test[f1_pred_mask].sum() / f1_predictions
            print(f"  When predicted F1 wins: Actual accuracy = {f1_pred_accuracy:.1%}")
        
        # Confidence bins (2% increments)
        print(f"  Confidence distribution:")
        bins = [(i/100, (i+2)/100) for i in range(0, 100, 2)]
        for low, high in bins:
            mask = (probs >= low) & (probs < high)
            count = mask.sum()
            if count > 0:
                bin_accuracy = (y_test[mask] == (probs[mask] > 0.5).astype(int)).sum() / count
                avg_conf = probs[mask].mean()
                winner = "F2" if low < 0.5 else "F1"
                print(f"    {low:.0%}-{high:.0%}: {count:3d} fights (avg {avg_conf:.1%}), accuracy {bin_accuracy:.1%} (predicted {winner})")
    
    # ==================== FINAL SUMMARY ====================
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    
    print("\nIndividual Model Accuracies:")
    for name, acc in all_accuracies.items():
        print(f"  {name:20s}: {acc:.4f} ({acc*100:.2f}%)")
    return {
        'results_df': results_df,
        'models': ml_models,
        'scaler': scaler,
        'feature_cols': feature_cols,
        'accuracies': all_methods_accuracy
    }


# ==================== RUN IT ====================
if __name__ == "__main__":
    # Load your data
    print("Loading data...")
    # from models.ufc_mma import tuning_model
    # tuning_model.run()
    # exit()
    differential_df = pl.read_csv('trainingset.csv')
    combined_df = pl.read_parquet('fight_features_extracted.parquet')
    
    print(f"Loaded differential_df: {len(differential_df)} rows")
    print(f"Loaded combined_df: {len(combined_df)} rows")
    
    # Run ensemble
    ensemble_results = ensemble_all_models(differential_df, combined_df, test_split=0.2)
    
    # Save results
    ensemble_results['results_df'].write_csv('ensemble_predictions.csv')
    print("\n✅ Results saved to 'ensemble_predictions.csv'")
    
    # Show sample predictions
    print("\nSample predictions (first 10 fights):")
    print(ensemble_results['results_df'].head(10))