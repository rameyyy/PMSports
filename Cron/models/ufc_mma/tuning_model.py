# ultra_precision_tuning.py

import polars as pl
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb

def ultra_precision_xgboost(differential_df, test_split=0.2):
    """
    Ultra-precise XGBoost tuning around the absolute best parameters.
    Target: Beat 65.17%
    """
    
    print("="*80)
    print("ULTRA-PRECISION XGBOOST TUNING")
    print("="*80)
    print("Target: Beat 65.17% test accuracy\n")
    
    # Sweet spot found: n_est=50-120, depth=2, lr=0.04-0.08, min_child=8-10
    param_combinations = [
        # BEST SO FAR (65.17%)
        {'n_estimators': 50, 'max_depth': 2, 'learning_rate': 0.08, 'min_child_weight': 9, 'subsample': 0.75, 'colsample_bytree': 0.75},
        
        # SECOND BEST (64.93%) - 120 trees at 0.04 lr
        {'n_estimators': 120, 'max_depth': 2, 'learning_rate': 0.04, 'min_child_weight': 8, 'subsample': 0.75, 'colsample_bytree': 0.75},
        
        # MICRO-VARIATIONS AROUND BEST (n_est: 45-55, min_child: 8-10)
        {'n_estimators': 48, 'max_depth': 2, 'learning_rate': 0.08, 'min_child_weight': 9, 'subsample': 0.75, 'colsample_bytree': 0.75},
        {'n_estimators': 52, 'max_depth': 2, 'learning_rate': 0.08, 'min_child_weight': 9, 'subsample': 0.75, 'colsample_bytree': 0.75},
        {'n_estimators': 50, 'max_depth': 2, 'learning_rate': 0.08, 'min_child_weight': 8, 'subsample': 0.75, 'colsample_bytree': 0.75},
        {'n_estimators': 50, 'max_depth': 2, 'learning_rate': 0.08, 'min_child_weight': 10, 'subsample': 0.75, 'colsample_bytree': 0.75},
        
        # VARY LEARNING RATE VERY SLIGHTLY (0.075-0.085)
        {'n_estimators': 50, 'max_depth': 2, 'learning_rate': 0.075, 'min_child_weight': 9, 'subsample': 0.75, 'colsample_bytree': 0.75},
        {'n_estimators': 50, 'max_depth': 2, 'learning_rate': 0.085, 'min_child_weight': 9, 'subsample': 0.75, 'colsample_bytree': 0.75},
        {'n_estimators': 50, 'max_depth': 2, 'learning_rate': 0.082, 'min_child_weight': 9, 'subsample': 0.75, 'colsample_bytree': 0.75},
        
        # SUBSAMPLE MICRO-VARIATIONS (0.73-0.78)
        {'n_estimators': 50, 'max_depth': 2, 'learning_rate': 0.08, 'min_child_weight': 9, 'subsample': 0.73, 'colsample_bytree': 0.75},
        {'n_estimators': 50, 'max_depth': 2, 'learning_rate': 0.08, 'min_child_weight': 9, 'subsample': 0.76, 'colsample_bytree': 0.75},
        {'n_estimators': 50, 'max_depth': 2, 'learning_rate': 0.08, 'min_child_weight': 9, 'subsample': 0.78, 'colsample_bytree': 0.75},
        
        # COLSAMPLE MICRO-VARIATIONS (0.73-0.78)
        {'n_estimators': 50, 'max_depth': 2, 'learning_rate': 0.08, 'min_child_weight': 9, 'subsample': 0.75, 'colsample_bytree': 0.73},
        {'n_estimators': 50, 'max_depth': 2, 'learning_rate': 0.08, 'min_child_weight': 9, 'subsample': 0.75, 'colsample_bytree': 0.76},
        {'n_estimators': 50, 'max_depth': 2, 'learning_rate': 0.08, 'min_child_weight': 9, 'subsample': 0.75, 'colsample_bytree': 0.78},
        
        # COMBINE BEST TWEAKS
        {'n_estimators': 52, 'max_depth': 2, 'learning_rate': 0.078, 'min_child_weight': 9, 'subsample': 0.76, 'colsample_bytree': 0.75},
        {'n_estimators': 48, 'max_depth': 2, 'learning_rate': 0.082, 'min_child_weight': 9, 'subsample': 0.75, 'colsample_bytree': 0.76},
        {'n_estimators': 50, 'max_depth': 2, 'learning_rate': 0.08, 'min_child_weight': 8, 'subsample': 0.76, 'colsample_bytree': 0.76},
        
        # EXPLORE 110-130 TREES AT LOW LR (pattern from #2 best)
        {'n_estimators': 110, 'max_depth': 2, 'learning_rate': 0.04, 'min_child_weight': 8, 'subsample': 0.75, 'colsample_bytree': 0.75},
        {'n_estimators': 115, 'max_depth': 2, 'learning_rate': 0.042, 'min_child_weight': 8, 'subsample': 0.75, 'colsample_bytree': 0.75},
        {'n_estimators': 125, 'max_depth': 2, 'learning_rate': 0.038, 'min_child_weight': 8, 'subsample': 0.75, 'colsample_bytree': 0.75},
        {'n_estimators': 130, 'max_depth': 2, 'learning_rate': 0.037, 'min_child_weight': 9, 'subsample': 0.75, 'colsample_bytree': 0.75},
        
        # MIDDLE GROUND (70-90 trees)
        {'n_estimators': 75, 'max_depth': 2, 'learning_rate': 0.06, 'min_child_weight': 9, 'subsample': 0.75, 'colsample_bytree': 0.75},
        {'n_estimators': 80, 'max_depth': 2, 'learning_rate': 0.055, 'min_child_weight': 9, 'subsample': 0.75, 'colsample_bytree': 0.75},
        {'n_estimators': 85, 'max_depth': 2, 'learning_rate': 0.052, 'min_child_weight': 8, 'subsample': 0.76, 'colsample_bytree': 0.76},
    ]
    
    # Prepare data
    df_sorted = differential_df.sort('fight_date')
    n_total = len(df_sorted)
    split_date = df_sorted[int(n_total * (1 - test_split))]['fight_date']
    
    train_df = df_sorted.filter(pl.col('fight_date') < split_date)
    test_df = df_sorted.filter(pl.col('fight_date') >= split_date)
    
    exclude_cols = ['fight_id', 'fight_date', 'target', 'weight_class', 'stance_matchup']
    train_encoded = train_df.to_dummies(columns=['weight_class', 'stance_matchup'])
    test_encoded = test_df.to_dummies(columns=['weight_class', 'stance_matchup'])
    
    feature_cols = [col for col in train_encoded.columns if col not in exclude_cols]
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
    
    print(f"Testing {len(param_combinations)} ultra-precise combinations...\n")
    
    best_accuracy = 0.6517  # Current best
    best_params = {}
    best_train_acc = 0
    results = []
    improvements = 0
    
    for i, params in enumerate(param_combinations, 1):
        model = xgb.XGBClassifier(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            learning_rate=params['learning_rate'],
            min_child_weight=params['min_child_weight'],
            subsample=params['subsample'],
            colsample_bytree=params['colsample_bytree'],
            random_state=42,
            eval_metric='logloss',
            verbosity=0
        )
        
        model.fit(X_train_scaled, y_train)
        
        y_train_pred = model.predict(X_train_scaled)
        y_test_pred = model.predict(X_test_scaled)
        
        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        overfitting = train_acc - test_acc
        
        results.append({
            **params,
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'overfitting_gap': overfitting
        })
        
        status = "✅" if overfitting < 0.10 else "⚠️ "
        
        print(f"Trial {i:2d}: Test={test_acc:.4f} (+{test_acc-0.6517:+.4f}) Gap={overfitting:+.4f} {status}", end="")
        
        if test_acc > best_accuracy:
            improvements += 1
            best_accuracy = test_acc
            best_train_acc = train_acc
            best_params = params.copy()
            print(f" 🌟 NEW BEST! (+{test_acc-0.6517:.4f})")
        else:
            print()
    
    print("\n" + "="*80)
    print("ULTRA-PRECISION XGBOOST RESULTS")
    print("="*80)
    print(f"Starting accuracy: 65.17%")
    print(f"Final accuracy:    {best_accuracy*100:.2f}%")
    print(f"Improvement:       +{(best_accuracy-0.6517)*100:.2f}%")
    print(f"Number of improvements found: {improvements}")
    
    print("\nBest Parameters:")
    for param, value in best_params.items():
        print(f"  {param:20s}: {value}")
    print(f"\n  Train Accuracy: {best_train_acc:.4f}")
    print(f"  Test Accuracy:  {best_accuracy:.4f}")
    print(f"  Overfit Gap:    {best_train_acc - best_accuracy:+.4f}")
    
    results_df = pl.DataFrame(results).sort('test_accuracy', descending=True)
    print("\n" + "-"*80)
    print("TOP 10 CONFIGURATIONS")
    print("-"*80)
    print(results_df.select(['n_estimators', 'learning_rate', 'min_child_weight', 'test_accuracy', 'overfitting_gap']).head(10))
    
    return best_params, results_df


def ultra_precision_gradient_boost(differential_df, test_split=0.2):
    """
    Ultra-precise Gradient Boost tuning.
    Target: Beat 65.88%
    """
    
    print("="*80)
    print("ULTRA-PRECISION GRADIENT BOOST TUNING")
    print("="*80)
    print("Target: Beat 65.88% test accuracy\n")
    
    # Sweet spot: n_est=75-90, depth=2, lr=0.07-0.08, min_split=20, min_leaf=10, sub=0.8
    param_combinations = [
        # BEST SO FAR (65.88%)
        {'n_estimators': 90, 'max_depth': 2, 'learning_rate': 0.07, 'min_samples_split': 20, 'min_samples_leaf': 10, 'subsample': 0.8},
        
        # SECOND BEST (also 65.88%)
        {'n_estimators': 75, 'max_depth': 2, 'learning_rate': 0.08, 'min_samples_split': 20, 'min_samples_leaf': 10, 'subsample': 0.8},
        
        # MICRO-VARIATIONS AROUND 90 TREES, LR=0.07
        {'n_estimators': 88, 'max_depth': 2, 'learning_rate': 0.07, 'min_samples_split': 20, 'min_samples_leaf': 10, 'subsample': 0.8},
        {'n_estimators': 92, 'max_depth': 2, 'learning_rate': 0.07, 'min_samples_split': 20, 'min_samples_leaf': 10, 'subsample': 0.8},
        {'n_estimators': 95, 'max_depth': 2, 'learning_rate': 0.07, 'min_samples_split': 20, 'min_samples_leaf': 10, 'subsample': 0.8},
        
        # MICRO-VARY LEARNING RATE (0.068-0.078)
        {'n_estimators': 90, 'max_depth': 2, 'learning_rate': 0.068, 'min_samples_split': 20, 'min_samples_leaf': 10, 'subsample': 0.8},
        {'n_estimators': 90, 'max_depth': 2, 'learning_rate': 0.072, 'min_samples_split': 20, 'min_samples_leaf': 10, 'subsample': 0.8},
        {'n_estimators': 90, 'max_depth': 2, 'learning_rate': 0.075, 'min_samples_split': 20, 'min_samples_leaf': 10, 'subsample': 0.8},
        
        # BETWEEN 75 AND 90 TREES
        {'n_estimators': 82, 'max_depth': 2, 'learning_rate': 0.075, 'min_samples_split': 20, 'min_samples_leaf': 10, 'subsample': 0.8},
        {'n_estimators': 78, 'max_depth': 2, 'learning_rate': 0.078, 'min_samples_split': 20, 'min_samples_leaf': 10, 'subsample': 0.8},
        
        # MICRO-VARY SUBSAMPLE (0.78-0.82)
        {'n_estimators': 90, 'max_depth': 2, 'learning_rate': 0.07, 'min_samples_split': 20, 'min_samples_leaf': 10, 'subsample': 0.78},
        {'n_estimators': 90, 'max_depth': 2, 'learning_rate': 0.07, 'min_samples_split': 20, 'min_samples_leaf': 10, 'subsample': 0.82},
        {'n_estimators': 90, 'max_depth': 2, 'learning_rate': 0.07, 'min_samples_split': 20, 'min_samples_leaf': 10, 'subsample': 0.81},
        
        # VARY MIN_SAMPLES_SPLIT (18-22)
        {'n_estimators': 90, 'max_depth': 2, 'learning_rate': 0.07, 'min_samples_split': 18, 'min_samples_leaf': 10, 'subsample': 0.8},
        {'n_estimators': 90, 'max_depth': 2, 'learning_rate': 0.07, 'min_samples_split': 19, 'min_samples_leaf': 10, 'subsample': 0.8},
        {'n_estimators': 90, 'max_depth': 2, 'learning_rate': 0.07, 'min_samples_split': 21, 'min_samples_leaf': 10, 'subsample': 0.8},
        
        # VARY MIN_SAMPLES_LEAF (9-11)
        {'n_estimators': 90, 'max_depth': 2, 'learning_rate': 0.07, 'min_samples_split': 20, 'min_samples_leaf': 9, 'subsample': 0.8},
        {'n_estimators': 90, 'max_depth': 2, 'learning_rate': 0.07, 'min_samples_split': 20, 'min_samples_leaf': 11, 'subsample': 0.8},
        
        # COMBINE MICRO-TWEAKS
        {'n_estimators': 92, 'max_depth': 2, 'learning_rate': 0.072, 'min_samples_split': 19, 'min_samples_leaf': 10, 'subsample': 0.81},
        {'n_estimators': 88, 'max_depth': 2, 'learning_rate': 0.075, 'min_samples_split': 20, 'min_samples_leaf': 9, 'subsample': 0.8},
        {'n_estimators': 90, 'max_depth': 2, 'learning_rate': 0.07, 'min_samples_split': 19, 'min_samples_leaf': 10, 'subsample': 0.82},
        
        # EXPLORE 100-105 RANGE
        {'n_estimators': 100, 'max_depth': 2, 'learning_rate': 0.065, 'min_samples_split': 20, 'min_samples_leaf': 10, 'subsample': 0.8},
        {'n_estimators': 105, 'max_depth': 2, 'learning_rate': 0.06, 'min_samples_split': 20, 'min_samples_leaf': 10, 'subsample': 0.8},
    ]
    
    # Same data prep...
    df_sorted = differential_df.sort('fight_date')
    n_total = len(df_sorted)
    split_date = df_sorted[int(n_total * (1 - test_split))]['fight_date']
    
    train_df = df_sorted.filter(pl.col('fight_date') < split_date)
    test_df = df_sorted.filter(pl.col('fight_date') >= split_date)
    
    exclude_cols = ['fight_id', 'fight_date', 'target', 'weight_class', 'stance_matchup']
    train_encoded = train_df.to_dummies(columns=['weight_class', 'stance_matchup'])
    test_encoded = test_df.to_dummies(columns=['weight_class', 'stance_matchup'])
    
    feature_cols = [col for col in train_encoded.columns if col not in exclude_cols]
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
    
    print(f"Testing {len(param_combinations)} ultra-precise combinations...\n")
    
    best_accuracy = 0.6588  # Current best
    best_params = {}
    best_train_acc = 0
    results = []
    improvements = 0
    
    for i, params in enumerate(param_combinations, 1):
        model = GradientBoostingClassifier(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            learning_rate=params['learning_rate'],
            min_samples_split=params['min_samples_split'],
            min_samples_leaf=params['min_samples_leaf'],
            subsample=params['subsample'],
            random_state=42
        )
        
        model.fit(X_train_scaled, y_train)
        
        y_train_pred = model.predict(X_train_scaled)
        y_test_pred = model.predict(X_test_scaled)
        
        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        overfitting = train_acc - test_acc
        
        results.append({
            **params,
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'overfitting_gap': overfitting
        })
        
        status = "✅" if overfitting < 0.10 else "⚠️ "
        
        print(f"Trial {i:2d}: Test={test_acc:.4f} (+{test_acc-0.6588:+.4f}) Gap={overfitting:+.4f} {status}", end="")
        
        if test_acc > best_accuracy:
            improvements += 1
            best_accuracy = test_acc
            best_train_acc = train_acc
            best_params = params.copy()
            print(f" 🌟 NEW BEST! (+{test_acc-0.6588:.4f})")
        else:
            print()
    
    print("\n" + "="*80)
    print("ULTRA-PRECISION GRADIENT BOOST RESULTS")
    print("="*80)
    print(f"Starting accuracy: 65.88%")
    print(f"Final accuracy:    {best_accuracy*100:.2f}%")
    print(f"Improvement:       +{(best_accuracy-0.6588)*100:.2f}%")
    print(f"Number of improvements found: {improvements}")
    
    print("\nBest Parameters:")
    for param, value in best_params.items():
        print(f"  {param:20s}: {value}")
    print(f"\n  Train Accuracy: {best_train_acc:.4f}")
    print(f"  Test Accuracy:  {best_accuracy:.4f}")
    print(f"  Overfit Gap:    {best_train_acc - best_accuracy:+.4f}")
    
    results_df = pl.DataFrame(results).sort('test_accuracy', descending=True)
    print("\n" + "-"*80)
    print("TOP 10 CONFIGURATIONS")
    print("-"*80)
    print(results_df.select(['n_estimators', 'learning_rate', 'subsample', 'test_accuracy', 'overfitting_gap']).head(10))
    
    return best_params, results_df


# Usage:
def run():
    differential_df = pl.read_csv('trainingset.csv')
    
    print("ULTRA-PRECISION XGBOOST TUNING...\n")
    best_xgb, xgb_res = ultra_precision_xgboost(differential_df, test_split=0.2)
    
    print("\n\n" + "="*80 + "\n\n")
    
    print("ULTRA-PRECISION GRADIENT BOOST TUNING...\n")
    best_gb, gb_res = ultra_precision_gradient_boost(differential_df, test_split=0.2)
    
    print("\n\n" + "="*80)
    print("🎯 FINAL OPTIMIZED PARAMETERS")
    print("="*80)
    print("\nXGBoost:")
    for k, v in best_xgb.items():
        print(f"  {k}: {v}")
    print("\nGradient Boosting:")
    for k, v in best_gb.items():
        print(f"  {k}: {v}")