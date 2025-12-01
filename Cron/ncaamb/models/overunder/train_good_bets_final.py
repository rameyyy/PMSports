#!/usr/bin/env python3
"""
Train and save OU Good Bets Random Forest models for multiple sportsbooks
Prioritizes 2025 data in training
Train on 2021-2025, with 2025 data weighted heavily
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pickle
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
ncaamb_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ncaamb_dir))


def load_features_by_year(years: list):
    """Load feature files for specified years"""
    all_features = []
    year_labels = []

    print(f"Loading features for years: {years}")

    for year in years:
        features_file = ncaamb_dir / f"features{year}.csv"
        if features_file.exists():
            print(f"  Loading features{year}.csv...")
            try:
                df = pd.read_csv(features_file)
                print(f"    [OK] Loaded {len(df)} games")
                all_features.append(df)
                year_labels.extend([year] * len(df))
            except Exception as e:
                print(f"    [ERR] Error loading {year}: {e}")

    if not all_features:
        return None, None

    combined_df = pd.concat(all_features, ignore_index=True)
    combined_df['_year'] = year_labels

    # Convert numeric columns
    for col in combined_df.columns:
        if any(x in col.lower() for x in ['odds', 'line', 'spread', 'total', 'score', 'pace', 'efg', 'adj', 'avg']):
            combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')

    print(f"[OK] Combined: {len(combined_df)} total games\n")
    return combined_df, year_labels


def create_ou_target_variable(df, ou_line_col):
    """Create binary target variable for over/under betting"""
    df_filtered = df.dropna(subset=['actual_total', ou_line_col]).copy()

    before = len(df)
    after = len(df_filtered)
    print(f"Filtered for {ou_line_col}: removed {before - after}, kept {after}")

    df_filtered['ou_target'] = (df_filtered['actual_total'] > df_filtered[ou_line_col]).astype(int)

    over_count = (df_filtered['ou_target'] == 1).sum()
    under_count = (df_filtered['ou_target'] == 0).sum()

    print(f"Target distribution:")
    print(f"  Over hits (1):  {over_count}")
    print(f"  Under hits (0): {under_count}\n")

    return df_filtered


def load_trained_models():
    """Load pre-trained XGB, LGB, CatBoost models"""
    saved_dir = Path(__file__).parent / "saved"

    try:
        xgb_model = xgb.XGBRegressor()
        xgb_model.load_model(str(saved_dir / "xgboost_model.pkl"))
        lgb_model = lgb.Booster(model_file=str(saved_dir / "lightgbm_model.pkl"))
        cat_model = CatBoostRegressor()
        cat_model.load_model(str(saved_dir / "catboost_model.pkl"), format='cbm')
        print("[OK] Loaded pre-trained ensemble models\n")
        return xgb_model, lgb_model, cat_model
    except Exception as e:
        print(f"[ERR] Could not load pre-trained models: {e}\n")
        return None, None, None


def get_predictions_all_models(df, xgb_model, lgb_model, cat_model):
    """Get predictions from all 3 models"""
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    exclude = {'actual_total', 'team_1_score', 'team_2_score', 'game_id', 'date', 'team_1', 'team_2', 'ou_target', '_year'}
    feature_cols = [c for c in numeric_cols if c not in exclude]

    X = df[feature_cols].fillna(0)

    xgb_preds = xgb_model.predict(X)
    lgb_preds = lgb_model.predict(X, num_iteration=lgb_model.best_iteration)
    cat_preds = cat_model.predict(X)

    return xgb_preds, lgb_preds, cat_preds


def create_ou_good_bets_data(df, xgb_preds, lgb_preds, cat_preds, ou_line_col):
    """Create good bets features"""
    df_filtered = df.dropna(subset=['actual_total', ou_line_col]).copy()
    df_filtered['xgb_point_pred'] = xgb_preds[:len(df_filtered)]
    df_filtered['lgb_point_pred'] = lgb_preds[:len(df_filtered)]
    df_filtered['cat_point_pred'] = cat_preds[:len(df_filtered)]

    # Create confidence scores
    df_filtered['xgb_confidence_over'] = 1.0 / (1.0 + np.exp(-(df_filtered['xgb_point_pred'] - df_filtered[ou_line_col]) / 3.0))
    df_filtered['lgb_confidence_over'] = 1.0 / (1.0 + np.exp(-(df_filtered['lgb_point_pred'] - df_filtered[ou_line_col]) / 3.0))
    df_filtered['cat_confidence_over'] = 1.0 / (1.0 + np.exp(-(df_filtered['cat_point_pred'] - df_filtered[ou_line_col]) / 3.0))

    for col in ['xgb_confidence_over', 'lgb_confidence_over', 'cat_confidence_over']:
        df_filtered[col] = df_filtered[col].clip(0.01, 0.99)

    # Ensemble confidence
    df_filtered['ensemble_confidence_over'] = (
        0.441 * df_filtered['xgb_confidence_over'] +
        0.466 * df_filtered['lgb_confidence_over'] +
        0.093 * df_filtered['cat_confidence_over']
    )

    # Model std dev
    df_filtered['model_std_dev'] = np.std([
        df_filtered['xgb_confidence_over'].values,
        df_filtered['lgb_confidence_over'].values,
        df_filtered['cat_confidence_over'].values
    ], axis=0)

    # Build features
    feature_cols = [
        'xgb_confidence_over', 'lgb_confidence_over', 'cat_confidence_over',
        'ensemble_confidence_over', 'model_std_dev',
        ou_line_col, 'avg_ou_line', 'ou_line_variance'
    ]

    feature_cols = [c for c in feature_cols if c in df_filtered.columns]
    X = df_filtered[feature_cols].fillna(0)
    y = df_filtered['ou_target'].values

    return X.values, y, df_filtered, feature_cols


def main():
    print("="*100)
    print("TRAINING OU GOOD BETS MODELS - PRIORITIZING 2025 DATA")
    print("="*100 + "\n")

    # Load ensemble models
    print("[STEP 0] Loading pre-trained ensemble models")
    xgb_model, lgb_model, cat_model = load_trained_models()
    if not all([xgb_model, lgb_model, cat_model]):
        print("[-] Could not load models, exiting")
        return

    # Load all training data (2021-2025)
    print("[STEP 1] Loading all training data (2021-2025)")
    train_df_all, year_labels_all = load_features_by_year(['2021', '2022', '2023', '2024', '2025'])

    # Create sample weights that heavily weight 2025
    # 2025 gets 5x weight, others get 1x weight
    sample_weights_all = np.array([5.0 if year == '2025' else 1.0 for year in year_labels_all])

    print("[STEP 2] Getting ensemble predictions for all data")
    xgb_preds_all, lgb_preds_all, cat_preds_all = get_predictions_all_models(train_df_all, xgb_model, lgb_model, cat_model)

    # Train models for each sportsbook
    sportsbooks = [
        ('BetOnline', 'betonline_ou_line'),
        ('MyBookie', 'mybookie_ou_line'),
        ('Bovada', 'bovada_ou_line'),
    ]

    saved_dir = Path(__file__).parent / "saved"
    os.makedirs(saved_dir, exist_ok=True)

    results = {}

    for sportsbook_name, ou_line_col in sportsbooks:
        print(f"\n{'='*100}")
        print(f"TRAINING {sportsbook_name.upper()}")
        print(f"{'='*100}\n")

        # Create target for this sportsbook
        print(f"[STEP 1] Creating target for {sportsbook_name}")
        train_df_sb = create_ou_target_variable(train_df_all, ou_line_col)

        # Get year labels for filtered data (after dropna)
        filtered_indices = train_df_sb.index
        sample_weights_sb = sample_weights_all[filtered_indices]

        print(f"[STEP 2] Creating good bets features for {sportsbook_name}")
        X_train, y_train, _, _ = create_ou_good_bets_data(
            train_df_sb, xgb_preds_all[:len(train_df_all)], lgb_preds_all[:len(train_df_all)],
            cat_preds_all[:len(train_df_all)], ou_line_col
        )

        print(f"[STEP 3] Training Random Forest with sample weights (2025 prioritized)")
        rf_model = RandomForestClassifier(
            n_estimators=178,
            max_depth=5,
            min_samples_split=48,
            min_samples_leaf=44,
            random_state=42,
            n_jobs=-1
        )

        # Train with sample weights to prioritize 2025
        rf_model.fit(X_train, y_train, sample_weight=sample_weights_sb)

        # Evaluate on training data
        y_pred_train = rf_model.predict(X_train)
        train_acc = accuracy_score(y_train, y_pred_train)
        train_prec = precision_score(y_train, y_pred_train)
        train_recall = recall_score(y_train, y_pred_train)
        train_f1 = f1_score(y_train, y_pred_train)

        print(f"[OK] Training complete for {sportsbook_name}")
        print(f"    Train Accuracy: {train_acc:.4f}")
        print(f"    Train Precision: {train_prec:.4f}")
        print(f"    Train Recall: {train_recall:.4f}")
        print(f"    Train F1 Score: {train_f1:.4f}\n")

        # Save model
        print(f"[STEP 4] Saving {sportsbook_name} model")
        model_path = saved_dir / f"ou_good_bets_final_{sportsbook_name.lower()}.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(rf_model, f)
        print(f"[OK] Saved to {model_path}\n")

        results[sportsbook_name] = {
            'accuracy': train_acc,
            'precision': train_prec,
            'recall': train_recall,
            'f1': train_f1,
            'path': model_path
        }

    # Summary
    print("="*100)
    print("TRAINING COMPLETE - MODELS SAVED")
    print("="*100)
    print(f"\nSaved directory: {saved_dir}")
    print(f"\nModels trained (2021-2025 data, 2025 weighted 5x):\n")

    for sportsbook_name, metrics in results.items():
        print(f"{sportsbook_name}:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1 Score:  {metrics['f1']:.4f}")
        print(f"  Saved at:  {metrics['path']}\n")


if __name__ == "__main__":
    main()
