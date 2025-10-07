import polars as pl
from .utils import create_connection, fetch_query


def get_valid_fighters(conn):
    query = """
    WITH all_fights AS (
        SELECT fighter1_id AS fighter_id FROM fights
        UNION ALL
        SELECT fighter2_id AS fighter_id FROM fights
    )
    SELECT fighter_id
    FROM all_fights
    GROUP BY fighter_id
    HAVING COUNT(*) >= 9;
    """
    df = pl.DataFrame(fetch_query(conn, query, params=None))
    df = df.rename({"fighter_id": "fighter_id"})
    return df


def load_fights(conn) -> pl.DataFrame:
    sql = """
        SELECT
            fight_id,
            fight_date,
            weight_class,
            fight_format,
            end_time,
            method,
            fighter1_id,
            fighter1_name,
            fighter2_id,
            fighter2_name,
            winner_id,
            loser_id
        FROM fights
        WHERE fight_date IS NOT NULL
        ORDER BY fight_date, fight_id;
    """
    return pl.DataFrame(fetch_query(conn, sql, params=None)).with_columns(
        pl.col("fight_date").cast(pl.Date)
    )


def build_base_two_rows(fights: pl.DataFrame) -> pl.DataFrame:
    f1 = fights.select(
        pl.col("fight_id"),
        pl.col("fight_date"),
        pl.col("weight_class"),
        pl.col("fight_format"),
        pl.col("end_time"),
        pl.col("method"),
        pl.col("fighter1_id").alias("fighter_id"),
        pl.col("fighter1_name").alias("fighter_name"),
        pl.col("fighter2_id").alias("opponent_id"),
        pl.col("fighter2_name").alias("opponent_name"),
        (pl.col("winner_id") == pl.col("fighter1_id")).cast(pl.Int8).alias("is_winner"),
    ).with_columns(pl.lit("f1").alias("fighter_side"))

    f2 = fights.select(
        pl.col("fight_id"),
        pl.col("fight_date"),
        pl.col("weight_class"),
        pl.col("fight_format"),
        pl.col("end_time"),
        pl.col("method"),
        pl.col("fighter2_id").alias("fighter_id"),
        pl.col("fighter2_name").alias("fighter_name"),
        pl.col("fighter1_id").alias("opponent_id"),
        pl.col("fighter1_name").alias("opponent_name"),
        (pl.col("winner_id") == pl.col("fighter2_id")).cast(pl.Int8).alias("is_winner"),
    ).with_columns(pl.lit("f2").alias("fighter_side"))

    base = pl.concat([f1, f2]).with_columns(
        pl.col("fight_date").cast(pl.Date)
    ).sort(["fight_date", "fight_id", "fighter_side"])
    return base


def filter_to_valid_matchups(base: pl.DataFrame, valid_fighters: pl.DataFrame) -> pl.DataFrame:
    valid = valid_fighters.select(pl.col("fighter_id")).unique()
    base = (
        base.join(valid.rename({"fighter_id": "fighter_id"}), on="fighter_id", how="inner")
        .join(valid.rename({"fighter_id": "opponent_id"}), on="opponent_id", how="inner")
    )
    return base


import polars as pl

def attach_prior_history(base: pl.DataFrame) -> pl.DataFrame:
    # 1) Sort chronologically within each fighter
    df = base.sort(["fighter_id", "fight_date", "fight_id", "fighter_side"])

    # 2) 0-based scalar index per fighter row (how many prior fights exist before this row)
    df = df.with_columns(
        (pl.col("fight_id").cum_count().over("fighter_id") - 1)
        .cast(pl.Int64)
        .alias("_idx")
    )

    # 3) Build per-fighter FULL lists via groupby->agg (returns List dtype columns)
    agg = (
        df.group_by("fighter_id", maintain_order=True)
          .agg([
              pl.col("fight_id").alias("_all_ids"),
              pl.col("fight_date").alias("_all_dates"),
              pl.col("opponent_id").alias("_all_opps"),
              pl.col("method").alias("_all_methods"),
              pl.col("is_winner").alias("_all_results"),
              pl.col("weight_class").alias("_all_wcs"),
              pl.col("fight_format").alias("_all_formats"),
              pl.col("end_time").alias("_all_endtimes"),
          ])
    )

    # 4) Join the lists back to every fighter row
    df = df.join(agg, on="fighter_id", how="left")

    # 5) Slice each List to include ONLY prior entries for this row
    # NOTE: use .list.slice(...), NOT .arr.slice(...)
    df = df.with_columns([
        pl.col("_all_ids").list.slice(0, pl.col("_idx")).alias("hist_fight_ids"),
        pl.col("_all_dates").list.slice(0, pl.col("_idx")).alias("hist_dates"),
        pl.col("_all_opps").list.slice(0, pl.col("_idx")).alias("hist_opponent_ids"),
        pl.col("_all_methods").list.slice(0, pl.col("_idx")).alias("hist_methods"),
        pl.col("_all_results").list.slice(0, pl.col("_idx")).alias("hist_is_winner"),
        pl.col("_all_wcs").list.slice(0, pl.col("_idx")).alias("hist_weight_classes"),
        pl.col("_all_formats").list.slice(0, pl.col("_idx")).alias("hist_fight_formats"),
        pl.col("_all_endtimes").list.slice(0, pl.col("_idx")).alias("hist_end_times"),
    ])

    # 6) Clean helpers
    df = df.drop([
        "_idx",
        "_all_ids","_all_dates","_all_opps","_all_methods",
        "_all_results","_all_wcs","_all_formats","_all_endtimes",
    ])

    return df




def build_fight_history_df(conn, valid_fighters: pl.DataFrame) -> pl.DataFrame:
    fights = load_fights(conn)
    base = build_base_two_rows(fights)
    base = filter_to_valid_matchups(base, valid_fighters)
    out = attach_prior_history(base)
    out = out.sort(["fight_date", "fight_id", "fighter_side"])
    return out


def flatten_nested(df: pl.DataFrame) -> pl.DataFrame:
    """Recursively flatten nested (struct or list) columns in a Polars DataFrame."""
    while True:
        # Find struct columns
        struct_cols = [c for c, dt in df.schema.items() if isinstance(dt, pl.datatypes.Struct)]
        if struct_cols:
            for c in struct_cols:
                df = df.unnest(c)
            continue

        # Find list columns
        list_cols = [c for c, dt in df.schema.items() if isinstance(dt, pl.datatypes.List)]
        if list_cols:
            for c in list_cols:
                # Convert lists to JSON strings instead of exploding to preserve row alignment
                df = df.with_columns(pl.col(c).cast(pl.Utf8))
            continue

        break
    return df


def run():
    conn = create_connection()
    validFightersDf = get_valid_fighters(conn)
    print(f"Valid fighters: {len(validFightersDf)}")

    fightHistoryDf = build_fight_history_df(conn, validFightersDf)
    print(fightHistoryDf.head(1).to_dicts())
    print(f"Total fight snapshots: {len(fightHistoryDf)}")
