"""Generate features from ufc327_snapshots.parquet and run the XGB+LGB ensemble.

Usage:
    cd apps/model && python predict_ufc327.py
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")
import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")

from pathlib import Path

import numpy as np
import polars as pl
import xgboost as xgb
import lightgbm as lgb

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
REPO  = Path(__file__).parent.parent.parent
DATA  = REPO / "data"
MODEL = Path(__file__).parent

sys.path.insert(0, str(REPO / "apps" / "features"))

# ------------------------------------------------------------------
# 1. Build features from the UFC 327 parquet
# ------------------------------------------------------------------
from extract import unnest_raw_df
from build_features import FightFeatures

raw_df = pl.read_parquet(DATA / "ufc327_snapshots.parquet")
fights_df, prior_fights_df, prior_rounds_df = unnest_raw_df(raw_df)

features = FightFeatures(fights_df, prior_fights_df, prior_rounds_df)
features.extract_fights_features()
features.extract_prior_fights_features()

ufc327_df = features.final_df

# ------------------------------------------------------------------
# 2. Keep only the upcoming fights (fight_id ends with _2026-04-11)
# ------------------------------------------------------------------
upcoming = ufc327_df.filter(
    pl.col("meta_root_fight_id").str.ends_with("_2026-04-11")
)

META_COLS = [
    "meta_root_fight_id", "meta_f1_id", "meta_f2_id",
    "meta_winner_id", "meta_loser_id",
    "meta_fight_date", "meta_end_time", "meta_method", "meta_fight_type",
]

# ------------------------------------------------------------------
# 3. Align feature columns to training set
# ------------------------------------------------------------------
train_df     = pl.read_csv(DATA / "features.csv")
feature_cols = [c for c in train_df.columns if c not in META_COLS]

# Add any missing columns as 0 (shouldn't normally happen)
for col in feature_cols:
    if col not in upcoming.columns:
        upcoming = upcoming.with_columns(pl.lit(None).cast(pl.Float32).alias(col))

X_pred = upcoming.select(feature_cols).to_numpy().astype(np.float32)

# ------------------------------------------------------------------
# 4. Load models and predict
# ------------------------------------------------------------------
xgb_model = xgb.XGBClassifier()
xgb_model.load_model(str(MODEL / "model.ubj"))

lgb_model = lgb.Booster(model_file=str(MODEL / "model_lgb.txt"))

xgb_proba = xgb_model.predict_proba(X_pred)[:, 1]
lgb_proba = lgb_model.predict(X_pred)
ensemble_proba = (xgb_proba + lgb_proba) / 2.0

# ------------------------------------------------------------------
# 5. Print results
# ------------------------------------------------------------------
# Grab fighter names from original parquet (keyed by fight_id)
name_map = {
    row["fight_id"]: (row["fighter1_name"], row["fighter2_name"])
    for row in raw_df.select(["fight_id", "fighter1_name", "fighter2_name"]).iter_rows(named=True)
}

print()
print("=" * 72)
print(f"{'UFC 327 — Model Predictions':^72}")
print("=" * 72)
print(f"{'Fighter 1 (f1)':<28}  {'Fighter 2 (f2)':<28}  {'f1 Win%':>7}  {'Pick'}")
print("-" * 72)

for i, row in enumerate(upcoming.iter_rows(named=True)):
    fid   = row["meta_root_fight_id"]
    f1n, f2n = name_map.get(fid, (row["meta_f1_id"], row["meta_f2_id"]))
    p     = ensemble_proba[i]
    pick  = f1n if p >= 0.5 else f2n
    print(f"{f1n:<28}  {f2n:<28}  {p*100:>6.1f}%  {pick}")

print("=" * 72)
print()
print("Individual model breakdown:")
print(f"{'Fighter 1':<28}  {'XGB f1%':>8}  {'LGB f1%':>8}  {'Ens f1%':>8}")
print("-" * 58)
for i, row in enumerate(upcoming.iter_rows(named=True)):
    fid  = row["meta_root_fight_id"]
    f1n, _ = name_map.get(fid, (row["meta_f1_id"], row["meta_f2_id"]))
    print(f"{f1n:<28}  {xgb_proba[i]*100:>7.1f}%  {lgb_proba[i]*100:>7.1f}%  {ensemble_proba[i]*100:>7.1f}%")
print("-" * 58)
