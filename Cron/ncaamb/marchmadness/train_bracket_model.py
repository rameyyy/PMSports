#!/usr/bin/env python3
"""
March Madness Bracket Model — No Odds Features
===============================================
Trains three models on features2021-2026.csv with all sportsbook/odds columns
removed so predictions can be made weeks before games (when odds don't exist).

Models:
  1. LightGBM        — Bayesian optimization via skopt
  2. XGBoost         — Bayesian optimization via skopt
  3. Ridge Regression — alpha tuned via cross-validation

Saved to:  marchmadness/saved/
           lightgbm_model.txt
           xgboost_model.json
           ridge_model.pkl
           feature_columns.txt
           hyperparameters.txt

Run from ncaamb/ directory:
    python marchmadness/train_bracket_model.py
"""

import sys
import pickle
from pathlib import Path

import numpy as np
import polars as pl
import lightgbm as lgb
import xgboost as xgb
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args

# ── Paths ─────────────────────────────────────────────────────────────────────
ncaamb_dir = Path(__file__).parent.parent
save_dir   = Path(__file__).parent / "saved"
save_dir.mkdir(parents=True, exist_ok=True)

# ── Odds columns to drop ──────────────────────────────────────────────────────
ODDS_COLS = [
    # Per-book: BetMGM
    'betmgm_ou_line', 'betmgm_over_odds', 'betmgm_under_odds',
    'betmgm_ml_team_1', 'betmgm_ml_team_2',
    'betmgm_spread_pts_team_1', 'betmgm_spread_odds_team_1',
    'betmgm_spread_pts_team_2', 'betmgm_spread_odds_team_2',
    # Per-book: BetOnline
    'betonline_ou_line', 'betonline_over_odds', 'betonline_under_odds',
    'betonline_ml_team_1', 'betonline_ml_team_2',
    'betonline_spread_pts_team_1', 'betonline_spread_odds_team_1',
    'betonline_spread_pts_team_2', 'betonline_spread_odds_team_2',
    # Per-book: Bovada
    'bovada_ou_line', 'bovada_over_odds', 'bovada_under_odds',
    'bovada_ml_team_1', 'bovada_ml_team_2',
    'bovada_spread_pts_team_1', 'bovada_spread_odds_team_1',
    'bovada_spread_pts_team_2', 'bovada_spread_odds_team_2',
    # Per-book: DraftKings
    'draftkings_ou_line', 'draftkings_over_odds', 'draftkings_under_odds',
    'draftkings_ml_team_1', 'draftkings_ml_team_2',
    'draftkings_spread_pts_team_1', 'draftkings_spread_odds_team_1',
    'draftkings_spread_pts_team_2', 'draftkings_spread_odds_team_2',
    # Per-book: FanDuel
    'fanduel_ou_line', 'fanduel_over_odds', 'fanduel_under_odds',
    'fanduel_ml_team_1', 'fanduel_ml_team_2',
    'fanduel_spread_pts_team_1', 'fanduel_spread_odds_team_1',
    'fanduel_spread_pts_team_2', 'fanduel_spread_odds_team_2',
    # Per-book: LowVig
    'lowvig_ou_line', 'lowvig_over_odds', 'lowvig_under_odds',
    'lowvig_ml_team_1', 'lowvig_ml_team_2',
    'lowvig_spread_pts_team_1', 'lowvig_spread_odds_team_1',
    'lowvig_spread_pts_team_2', 'lowvig_spread_odds_team_2',
    # Per-book: MyBookie
    'mybookie_ou_line', 'mybookie_over_odds', 'mybookie_under_odds',
    'mybookie_ml_team_1', 'mybookie_ml_team_2',
    'mybookie_spread_pts_team_1', 'mybookie_spread_odds_team_1',
    'mybookie_spread_pts_team_2', 'mybookie_spread_odds_team_2',
    # Aggregate odds
    'avg_ou_line', 'ou_line_variance', 'avg_over_odds', 'avg_under_odds',
    'num_books_with_ou',
    'avg_spread_pts_team_1', 'avg_spread_pts_team_2', 'spread_variance',
    'avg_spread_odds_team_1', 'avg_spread_odds_team_2',
    'avg_ml_team_1', 'avg_ml_team_2',
    'num_books_with_spread', 'num_books_with_ml',
    # Derived odds features
    'implied_team_1_score', 'implied_team_2_score', 'spread_ou_agreement',
    'hours_until_game_from_odds',
    'combined_expected_total_closest3rank',
    'combined_expected_total_closest5rank',
    'combined_expected_total_closest7rank',
]

# Columns that are metadata / target — never used as features
METADATA_COLS = {
    'game_id', 'date', 'season', 'team_1', 'team_2',
    'team_1_score', 'team_2_score', 'actual_total',
    'team_1_conference', 'team_2_conference',
    'team_1_is_home', 'team_2_is_home', 'location',
    'start_time', 'game_odds', 'ou_target',
}


# ─────────────────────────────────────────────────────────────────────────────
# Data loading & prep
# ─────────────────────────────────────────────────────────────────────────────

def load_all_features() -> pl.DataFrame:
    """Load features2021-2026.csv from ncaamb/, drop odds columns, filter completed games."""
    print("Loading features files...")
    dfs = []

    for year in range(2021, 2027):
        path = ncaamb_dir / f"features{year}.csv"
        if not path.exists():
            print(f"  ⚠ {path.name} not found — skipping")
            continue
        df = pl.read_csv(str(path))
        # Normalize schema: odds cols may be string in some years
        for col in df.columns:
            if any(x in col for x in ['_ml_team_', '_spread_pts_', '_spread_odds_', '_ou_']):
                try:
                    df = df.with_columns(pl.col(col).cast(pl.Float64, strict=False))
                except Exception:
                    pass
        dfs.append(df)
        print(f"  Loaded {path.name}: {len(df)} games")

    if not dfs:
        raise FileNotFoundError("No features files found in " + str(ncaamb_dir))

    combined = pl.concat(dfs, how="diagonal")
    print(f"  Combined: {len(combined)} total games\n")

    # Drop odds columns (only drop what actually exists)
    cols_to_drop = [c for c in ODDS_COLS if c in combined.columns]
    combined = combined.drop(cols_to_drop)
    print(f"  Dropped {len(cols_to_drop)} odds columns")

    # Filter to completed games
    combined = combined.filter(pl.col('actual_total').is_not_null())
    print(f"  Completed games (actual_total not null): {len(combined)}\n")

    return combined


def get_feature_columns(df: pl.DataFrame) -> list:
    """Return numeric columns that are not metadata/target."""
    feature_cols = []
    for col in df.columns:
        if col in METADATA_COLS:
            continue
        if col in ODDS_COLS:
            continue
        if df[col].dtype in (pl.Float32, pl.Float64, pl.Int32, pl.Int64):
            feature_cols.append(col)
    return feature_cols


def build_splits(df: pl.DataFrame, feature_cols: list):
    """
    Split by year: test = most recent year, train = everything else.
    Apply sample weights: most_recent-1 = 4x, most_recent-2 = 2x, others = 1x.
    """
    df = df.with_columns(pl.col('date').str.slice(0, 4).alias('_year'))
    years = sorted(df['_year'].unique().to_list())
    most_recent       = years[-1]
    second_most       = years[-2] if len(years) >= 2 else None
    third_most        = years[-3] if len(years) >= 3 else None

    train_df = df.filter(pl.col('_year') != most_recent)
    test_df  = df.filter(pl.col('_year') == most_recent)

    X_train = train_df.select(feature_cols).fill_null(0).to_numpy()
    y_train = train_df['actual_total'].to_numpy().astype(float)
    X_test  = test_df.select(feature_cols).fill_null(0).to_numpy()
    y_test  = test_df['actual_total'].to_numpy().astype(float)

    train_years = train_df['_year'].to_numpy()
    weights = np.ones(len(train_years))
    if second_most:
        weights[train_years == second_most] = 4.0
    if third_most:
        weights[train_years == third_most] = 2.0

    print(f"  Years available: {years[0]} – {years[-1]}")
    print(f"  Test  year : {most_recent}  ({len(X_test)} games)")
    print(f"  Train years: {sorted(y for y in years if y != most_recent)}  ({len(X_train)} games)")
    if second_most:
        print(f"  Sample weights: {second_most}=4x", end="")
        if third_most:
            print(f", {third_most}=2x", end="")
        print(", others=1x")

    return X_train, y_train, X_test, y_test, weights, years


def build_full_dataset(df: pl.DataFrame, feature_cols: list, years: list):
    """Build full dataset (all years) with sample weights for final model training."""
    df = df.with_columns(pl.col('date').str.slice(0, 4).alias('_year'))
    most_recent = years[-1]
    second_most = years[-2] if len(years) >= 2 else None

    X_all = df.select(feature_cols).fill_null(0).to_numpy()
    y_all = df['actual_total'].to_numpy().astype(float)

    all_years = df['_year'].to_numpy()
    weights_all = np.ones(len(all_years))
    weights_all[all_years == most_recent] = 4.0
    if second_most:
        weights_all[all_years == second_most] = 2.0

    return X_all, y_all, weights_all


# ─────────────────────────────────────────────────────────────────────────────
# LightGBM — Bayesian optimization
# ─────────────────────────────────────────────────────────────────────────────

lgb_space = [
    Integer(3,  10,   name='max_depth'),
    Real(0.01,  0.3,  prior='log-uniform', name='learning_rate'),
    Integer(100, 1000, name='n_estimators'),
    Integer(10,  200,  name='num_leaves'),
    Real(0.5,   1.0,  name='subsample'),
    Real(0.5,   1.0,  name='colsample_bytree'),
    Real(0.0,   10.0, name='reg_alpha'),
    Real(0.0,   10.0, name='reg_lambda'),
    Integer(5,  100,  name='min_child_samples'),
]


def optimize_lgb(X_train, y_train, X_test, y_test, sample_weights, n_calls=150):
    """Run Bayesian optimization for LightGBM."""
    iter_count = [0]

    @use_named_args(lgb_space)
    def objective(**params):
        iter_count[0] += 1
        print(f"  [LGB {iter_count[0]}/{n_calls}] depth={params['max_depth']} "
              f"lr={params['learning_rate']:.4f} n={params['n_estimators']}", end="  ")

        train_data = lgb.Dataset(X_train, label=y_train, weight=sample_weights)
        lgb_params = {
            'objective': 'regression', 'metric': 'mae',
            'verbose': -1, 'force_row_wise': True, 'random_state': 42,
            **params,
        }
        model = lgb.train(lgb_params, train_data, num_boost_round=params['n_estimators'])

        train_mae = mean_absolute_error(y_train, model.predict(X_train))
        test_mae  = mean_absolute_error(y_test,  model.predict(X_test))
        overfit   = max(0, test_mae - train_mae - 1.0) * 0.5
        score     = test_mae + overfit

        print(f"train={train_mae:.2f}  test={test_mae:.2f}  score={score:.2f}")
        return score

    result = gp_minimize(objective, lgb_space, n_calls=n_calls, random_state=42, verbose=False)
    best = {s.name: v for s, v in zip(lgb_space, result.x)}
    print(f"\n  Best LGB score: {result.fun:.4f}")
    print(f"  Best LGB params: {best}\n")
    return best


def train_lgb_final(X_all, y_all, weights_all, best_params):
    """Train final LightGBM on all data with best params."""
    train_data = lgb.Dataset(X_all, label=y_all, weight=weights_all)
    lgb_params = {
        'objective': 'regression', 'metric': 'mae',
        'verbose': -1, 'force_row_wise': True, 'random_state': 42,
        **best_params,
    }
    model = lgb.train(lgb_params, train_data, num_boost_round=best_params['n_estimators'])
    mae = mean_absolute_error(y_all, model.predict(X_all))
    r2  = r2_score(y_all, model.predict(X_all))
    print(f"  LGB final MAE: {mae:.4f}  R²: {r2:.4f}")
    return model, mae, r2


# ─────────────────────────────────────────────────────────────────────────────
# XGBoost — Bayesian optimization
# ─────────────────────────────────────────────────────────────────────────────

xgb_space = [
    Integer(3,  8,    name='max_depth'),
    Real(0.01,  0.3,  prior='log-uniform', name='learning_rate'),
    Integer(100, 800,  name='n_estimators'),
    Real(1,    10,    name='min_child_weight'),
    Real(0.5,   1.0,  name='subsample'),
    Real(0.5,   1.0,  name='colsample_bytree'),
    Real(0.0,  10.0,  name='reg_alpha'),
    Real(0.0,  10.0,  name='reg_lambda'),
]


def optimize_xgb(X_train, y_train, X_test, y_test, sample_weights, n_calls=150):
    """Run Bayesian optimization for XGBoost."""
    iter_count = [0]

    @use_named_args(xgb_space)
    def objective(**params):
        iter_count[0] += 1
        print(f"  [XGB {iter_count[0]}/{n_calls}] depth={params['max_depth']} "
              f"lr={params['learning_rate']:.4f} n={params['n_estimators']}", end="  ")

        model = xgb.XGBRegressor(
            objective='reg:absoluteerror',
            random_state=42,
            tree_method='hist',
            verbosity=0,
            **params,
        )
        model.fit(X_train, y_train, sample_weight=sample_weights,
                  eval_set=[(X_test, y_test)], verbose=False)

        train_mae = mean_absolute_error(y_train, model.predict(X_train))
        test_mae  = mean_absolute_error(y_test,  model.predict(X_test))
        overfit   = max(0, test_mae - train_mae - 1.0) * 0.5
        score     = test_mae + overfit

        print(f"train={train_mae:.2f}  test={test_mae:.2f}  score={score:.2f}")
        return score

    result = gp_minimize(objective, xgb_space, n_calls=n_calls, random_state=42, verbose=False)
    best = {s.name: v for s, v in zip(xgb_space, result.x)}
    print(f"\n  Best XGB score: {result.fun:.4f}")
    print(f"  Best XGB params: {best}\n")
    return best


def train_xgb_final(X_all, y_all, weights_all, best_params):
    """Train final XGBoost on all data with best params."""
    model = xgb.XGBRegressor(
        objective='reg:absoluteerror',
        random_state=42,
        tree_method='hist',
        verbosity=0,
        **best_params,
    )
    model.fit(X_all, y_all, sample_weight=weights_all)
    mae = mean_absolute_error(y_all, model.predict(X_all))
    r2  = r2_score(y_all, model.predict(X_all))
    print(f"  XGB final MAE: {mae:.4f}  R²: {r2:.4f}")
    return model, mae, r2


# ─────────────────────────────────────────────────────────────────────────────
# Ridge Regression — alpha tuned via RidgeCV
# ─────────────────────────────────────────────────────────────────────────────

def train_ridge(X_all, y_all, weights_all):
    """Tune and train Ridge regression. Scales features internally."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)

    alphas = np.logspace(-3, 5, 50)
    ridge_cv = RidgeCV(alphas=alphas, cv=5, scoring='neg_mean_absolute_error')
    ridge_cv.fit(X_scaled, y_all, sample_weight=weights_all)

    best_alpha = ridge_cv.alpha_
    print(f"  Ridge best alpha: {best_alpha:.4f}")

    # Train final Ridge with best alpha
    ridge = Ridge(alpha=best_alpha)
    ridge.fit(X_scaled, y_all, sample_weight=weights_all)

    mae = mean_absolute_error(y_all, ridge.predict(X_scaled))
    r2  = r2_score(y_all, ridge.predict(X_scaled))
    print(f"  Ridge final MAE: {mae:.4f}  R²: {r2:.4f}")
    return ridge, scaler, best_alpha, mae, r2


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 80)
    print("MARCH MADNESS BRACKET MODEL — NO ODDS FEATURES")
    print("=" * 80 + "\n")

    # ── 1. Load data ──────────────────────────────────────────────────────────
    print("STEP 1: Loading Data")
    print("-" * 80)
    df = load_all_features()

    # ── 2. Identify features ──────────────────────────────────────────────────
    print("STEP 2: Identifying Feature Columns")
    print("-" * 80)
    feature_cols = get_feature_columns(df)
    print(f"  Feature columns: {len(feature_cols)}\n")

    # ── 3. Train/test split ───────────────────────────────────────────────────
    print("STEP 3: Train / Test Split")
    print("-" * 80)
    X_train, y_train, X_test, y_test, sample_weights, years = build_splits(df, feature_cols)

    X_all, y_all, weights_all = build_full_dataset(df, feature_cols, years)
    print()

    # ── 4. Bayesian opt — LightGBM ────────────────────────────────────────────
    print("STEP 4: Bayesian Optimization — LightGBM (150 iterations)")
    print("-" * 80)
    lgb_best = optimize_lgb(X_train, y_train, X_test, y_test, sample_weights, n_calls=150)

    # ── 5. Bayesian opt — XGBoost ─────────────────────────────────────────────
    print("STEP 5: Bayesian Optimization — XGBoost (150 iterations)")
    print("-" * 80)
    xgb_best = optimize_xgb(X_train, y_train, X_test, y_test, sample_weights, n_calls=150)

    # ── 6. Train final models on all data ─────────────────────────────────────
    print("STEP 6: Training Final Models on All Data")
    print("-" * 80)
    print("  LightGBM...")
    lgb_model, lgb_mae, lgb_r2 = train_lgb_final(X_all, y_all, weights_all, lgb_best)

    print("  XGBoost...")
    xgb_model, xgb_mae, xgb_r2 = train_xgb_final(X_all, y_all, weights_all, xgb_best)

    print("  Ridge Regression...")
    ridge_model, ridge_scaler, ridge_alpha, ridge_mae, ridge_r2 = train_ridge(X_all, y_all, weights_all)
    print()

    # ── 7. Save models ────────────────────────────────────────────────────────
    print("STEP 7: Saving Models")
    print("-" * 80)

    lgb_path = save_dir / "lightgbm_model.txt"
    lgb_model.save_model(str(lgb_path))
    print(f"  ✅ LightGBM  → {lgb_path}")

    xgb_path = save_dir / "xgboost_model.json"
    xgb_model.save_model(str(xgb_path))
    print(f"  ✅ XGBoost   → {xgb_path}")

    ridge_path = save_dir / "ridge_model.pkl"
    with open(ridge_path, 'wb') as f:
        pickle.dump({'model': ridge_model, 'scaler': ridge_scaler}, f)
    print(f"  ✅ Ridge     → {ridge_path}")

    # Save feature column list
    feat_path = save_dir / "feature_columns.txt"
    with open(feat_path, 'w') as f:
        f.write('\n'.join(feature_cols))
    print(f"  ✅ Features  → {feat_path} ({len(feature_cols)} columns)")

    # Save hyperparameters + metrics
    hyper_path = save_dir / "hyperparameters.txt"
    with open(hyper_path, 'w') as f:
        f.write("March Madness Bracket Model — Hyperparameters & Metrics\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Training data: features{years[0]}-{years[-1]}.csv\n")
        f.write(f"Total games:   {len(X_all)}\n")
        f.write(f"Feature count: {len(feature_cols)}\n\n")

        f.write("LightGBM\n" + "-" * 30 + "\n")
        for k, v in lgb_best.items():
            f.write(f"  {k}: {v}\n")
        f.write(f"  Final MAE: {lgb_mae:.4f}\n")
        f.write(f"  Final R²:  {lgb_r2:.4f}\n\n")

        f.write("XGBoost\n" + "-" * 30 + "\n")
        for k, v in xgb_best.items():
            f.write(f"  {k}: {v}\n")
        f.write(f"  Final MAE: {xgb_mae:.4f}\n")
        f.write(f"  Final R²:  {xgb_r2:.4f}\n\n")

        f.write("Ridge Regression\n" + "-" * 30 + "\n")
        f.write(f"  alpha: {ridge_alpha:.4f}\n")
        f.write(f"  Final MAE: {ridge_mae:.4f}\n")
        f.write(f"  Final R²:  {ridge_r2:.4f}\n")

    print(f"  ✅ Params    → {hyper_path}\n")

    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"  LightGBM  MAE: {lgb_mae:.4f}  R²: {lgb_r2:.4f}")
    print(f"  XGBoost   MAE: {xgb_mae:.4f}  R²: {xgb_r2:.4f}")
    print(f"  Ridge     MAE: {ridge_mae:.4f}  R²: {ridge_r2:.4f}")
    print(f"\n  All models saved to: {save_dir}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
