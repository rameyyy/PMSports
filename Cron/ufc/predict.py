"""
Generate AlgoPicks predictions for all upcoming UFC fights.

Flow:
  1. Query DB for upcoming events that have at least one fight assigned
  2. Build raw df with rounds (in-memory, no parquet)
  3. Run feature pipeline
  4. Load XGB + LGB ensemble, predict
  5. Print results + upsert to prediction_simplified

Usage:
    cd Cron/ufc && python predict.py
"""
import sys
import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")

from pathlib import Path
import numpy as np
import polars as pl
import xgboost as xgb
import lightgbm as lgb

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CRON_UFC  = Path(__file__).parent
ML_GROUP  = CRON_UFC / "playground" / "ml-group-ufc"
MODEL_DIR = ML_GROUP / "apps" / "model"
FEAT_DIR  = ML_GROUP / "apps" / "features"
DATA_DIR  = ML_GROUP / "data"

sys.path.insert(0, str(FEAT_DIR))
sys.path.insert(0, str(MODEL_DIR))

from extract import unnest_raw_df
from build_features import FightFeatures
from train import load  # only used for feature_cols reference

# ---------------------------------------------------------------------------
# DB helpers (reuse existing utils)
# ---------------------------------------------------------------------------
sys.path.insert(0, str(CRON_UFC / "models"))
from utils import create_connection, fetch_query

# Also need the raw-df builder
sys.path.insert(0, str(CRON_UFC))
from models.build_raw_df_all_fights_with_rounds import (
    get_all_fights,
    get_all_related_fights,
    build_pre_fight_snapshots,
    enrich_with_fighter_stats,
    enrich_with_fight_totals,
    enrich_with_round_data,
)


# ---------------------------------------------------------------------------
# Step 1: get upcoming fight IDs from DB
# ---------------------------------------------------------------------------
def get_upcoming_fight_ids(conn) -> set:
    """
    Return fight_ids for upcoming fights: fight_date >= today, no winner yet,
    and the event has at least one fight with both fighters decided (fighter ids
    present). This excludes events that are too far out with no fights assigned.
    """
    rows = fetch_query(conn, """
        SELECT f.fight_id
        FROM fights f
        JOIN events e ON f.event_id = e.event_id
        WHERE f.fight_date >= CURDATE()
          AND f.winner_id IS NULL
          AND f.fighter1_id IS NOT NULL
          AND f.fighter2_id IS NOT NULL
    """)
    return {r["fight_id"] for r in rows}


# ---------------------------------------------------------------------------
# Step 2: build raw df in-memory (same as build_raw_df_all_fights_with_rounds.run
#         but returns df instead of saving to parquet)
# ---------------------------------------------------------------------------
def build_raw_df(conn) -> pl.DataFrame:
    print("Loading all fights from DB...")
    all_fights = get_all_fights(conn)
    print(f"  {len(all_fights)} total fights")

    all_related = get_all_related_fights(conn, all_fights)
    print(f"  {len(all_related)} fights for fighter history")

    snapshots = build_pre_fight_snapshots(all_fights, all_related, min_prior_fights=1)
    print(f"  {len(snapshots)} snapshots (min 1 prior fight each)")

    enriched = enrich_with_fighter_stats(snapshots, conn)
    enriched = enrich_with_fight_totals(enriched, conn)
    enriched = enrich_with_round_data(enriched, conn)
    return enriched


# ---------------------------------------------------------------------------
# Step 3: feature columns from training CSV
# ---------------------------------------------------------------------------
META_COLS = [
    "meta_root_fight_id", "meta_f1_id", "meta_f2_id",
    "meta_winner_id", "meta_loser_id",
    "meta_fight_date", "meta_end_time", "meta_method", "meta_fight_type",
]

def get_feature_cols() -> list[str]:
    train_df = pl.read_csv(DATA_DIR / "features.csv")
    return [c for c in train_df.columns if c not in META_COLS]


# ---------------------------------------------------------------------------
# Push predictions to DB
# ---------------------------------------------------------------------------
def push_predictions(conn, feat_df, upcoming_raw, xgb_proba, lgb_proba, ens_proba):
    # Pull fighter metadata (nickname + img_link) keyed by fighter_id
    fighter_rows = fetch_query(conn, "SELECT fighter_id, nickname, img_link FROM fighters")
    fighter_meta = {r["fighter_id"]: r for r in fighter_rows}

    # Pull fight context (event_id, weight_class, fight_type, date) keyed by fight_id
    fight_ctx = {
        row["fight_id"]: row
        for row in upcoming_raw.select([
            "fight_id", "event_id", "fighter1_id", "fighter2_id",
            "fighter1_name", "fighter2_name", "fight_date", "weight_class", "fight_type",
        ]).iter_rows(named=True)
    }

    upsert_sql = """
        INSERT INTO prediction_simplified (
            fight_id, event_id,
            fighter1_id, fighter2_id,
            fighter1_name, fighter2_name,
            fighter1_nickname, fighter2_nickname,
            fighter1_img_link, fighter2_img_link,
            predicted_winner_id,
            f1_probability, xgb_f1_probability, lgb_f1_probability,
            correct, date, weight_class, fight_type
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
            NULL, %s, %s, %s
        )
        ON DUPLICATE KEY UPDATE
            event_id              = VALUES(event_id),
            fighter1_id           = VALUES(fighter1_id),
            fighter2_id           = VALUES(fighter2_id),
            fighter1_name         = VALUES(fighter1_name),
            fighter2_name         = VALUES(fighter2_name),
            fighter1_nickname     = VALUES(fighter1_nickname),
            fighter2_nickname     = VALUES(fighter2_nickname),
            fighter1_img_link     = VALUES(fighter1_img_link),
            fighter2_img_link     = VALUES(fighter2_img_link),
            predicted_winner_id   = VALUES(predicted_winner_id),
            f1_probability        = VALUES(f1_probability),
            xgb_f1_probability    = VALUES(xgb_f1_probability),
            lgb_f1_probability    = VALUES(lgb_f1_probability),
            date                  = VALUES(date),
            weight_class          = VALUES(weight_class),
            fight_type            = VALUES(fight_type)
    """

    cursor = conn.cursor()
    pushed = 0
    for i, row in enumerate(feat_df.iter_rows(named=True)):
        fid  = row["meta_root_fight_id"]
        ctx  = fight_ctx.get(fid)
        if not ctx:
            continue

        f1_id = ctx["fighter1_id"]
        f2_id = ctx["fighter2_id"]
        f1m   = fighter_meta.get(f1_id, {})
        f2m   = fighter_meta.get(f2_id, {})

        p         = float(ens_proba[i])
        winner_id = f1_id if p >= 0.5 else f2_id

        cursor.execute(upsert_sql, (
            fid, ctx["event_id"],
            f1_id, f2_id,
            ctx["fighter1_name"], ctx["fighter2_name"],
            f1m.get("nickname"), f2m.get("nickname"),
            f1m.get("img_link"), f2m.get("img_link"),
            winner_id,
            round(p, 4), round(float(xgb_proba[i]), 4), round(float(lgb_proba[i]), 4),
            ctx["fight_date"], ctx["weight_class"], ctx["fight_type"],
        ))
        pushed += 1

    conn.commit()
    cursor.close()
    print(f"Pushed {pushed} predictions to prediction_simplified")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    conn = create_connection()
    if conn is None:
        print("Could not connect to DB")
        sys.exit(1)

    # Step 1: which fights are upcoming and have both fighters set
    print("\nFetching upcoming fight IDs...")
    upcoming_ids = get_upcoming_fight_ids(conn)
    if not upcoming_ids:
        print("No upcoming fights found — nothing to predict.")
        conn.close()
        return
    print(f"  {len(upcoming_ids)} upcoming fights to predict")

    # Step 2: build full raw df
    print("\nBuilding raw df with rounds...")
    raw_df = build_raw_df(conn)
    # keep conn open for push step below

    # Step 3: filter to upcoming fights only
    upcoming_raw = raw_df.filter(pl.col("fight_id").is_in(upcoming_ids))
    if upcoming_raw.is_empty():
        print("❌ Upcoming fight IDs not found in raw df (fighters may lack prior fight history).")
        return
    print(f"  {len(upcoming_raw)} upcoming fights with sufficient fighter history")

    # Step 4: build features
    print("\nRunning feature pipeline...")
    fights_df, prior_fights_df, prior_rounds_df = unnest_raw_df(upcoming_raw)
    features = FightFeatures(fights_df, prior_fights_df, prior_rounds_df)
    features.extract_fights_features()
    features.extract_prior_fights_features()
    feat_df = features.final_df

    # Step 5: align columns to training feature set
    feature_cols = get_feature_cols()
    for col in feature_cols:
        if col not in feat_df.columns:
            feat_df = feat_df.with_columns(pl.lit(None).cast(pl.Float32).alias(col))

    X_pred = feat_df.select(feature_cols).to_numpy().astype(np.float32)

    # Step 6: load models and predict
    print("\nLoading models...")
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(str(MODEL_DIR / "model.ubj"))

    lgb_model = lgb.Booster(model_file=str(MODEL_DIR / "model_lgb.txt"))

    xgb_proba    = xgb_model.predict_proba(X_pred)[:, 1]
    lgb_proba    = lgb_model.predict(X_pred)
    ens_proba    = (xgb_proba + lgb_proba) / 2.0

    # Step 7: print results
    # Pull fighter names from the raw df
    name_map = {
        row["fight_id"]: (row["fighter1_name"], row["fighter2_name"])
        for row in upcoming_raw.select(["fight_id", "fighter1_name", "fighter2_name"]).iter_rows(named=True)
    }

    print()
    print("=" * 72)
    print(f"{'UFC AlgoPicks — Upcoming Predictions':^72}")
    print("=" * 72)
    print(f"{'Fighter 1 (f1)':<28}  {'Fighter 2 (f2)':<28}  {'f1 Win%':>7}  {'Pick'}")
    print("-" * 72)

    for i, row in enumerate(feat_df.iter_rows(named=True)):
        fid      = row["meta_root_fight_id"]
        f1n, f2n = name_map.get(fid, (row["meta_f1_id"], row["meta_f2_id"]))
        p        = ens_proba[i]
        pick     = f1n if p >= 0.5 else f2n
        print(f"{f1n:<28}  {f2n:<28}  {p*100:>6.1f}%  {pick}")

    print("=" * 72)
    print()
    print("Individual model breakdown:")
    print(f"{'Fighter 1':<28}  {'XGB f1%':>8}  {'LGB f1%':>8}  {'Ens f1%':>8}")
    print("-" * 58)
    for i, row in enumerate(feat_df.iter_rows(named=True)):
        fid  = row["meta_root_fight_id"]
        f1n, _ = name_map.get(fid, (row["meta_f1_id"], row["meta_f2_id"]))
        print(f"{f1n:<28}  {xgb_proba[i]*100:>7.1f}%  {lgb_proba[i]*100:>7.1f}%  {ens_proba[i]*100:>7.1f}%")
    print("-" * 58)

    # Step 8: push to DB
    print("\nPushing predictions to DB...")
    push_predictions(conn, feat_df, upcoming_raw, xgb_proba, lgb_proba, ens_proba)
    conn.close()


if __name__ == "__main__":
    main()
