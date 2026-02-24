"""
Build raw dataframe with ALL fights in database (minimum 1 prior fight per fighter).
Includes round-by-round data for richer feature extraction.

Usage:
    python -m Cron.ufc.models.build_raw_df_all_fights_with_rounds

Output:
    fight_snapshots_all_with_rounds.parquet - Complete fight history snapshots with round data
"""

import polars as pl
from .utils import create_connection, fetch_query


def get_all_fights(conn) -> pl.DataFrame:
    """Get ALL fights from the database."""
    query = """
    SELECT *
    FROM fights
    ORDER BY fight_date;
    """
    rows = fetch_query(conn, query, params=None)
    # Use infer_schema_length=None to scan all rows for schema inference
    return pl.DataFrame(rows, infer_schema_length=None)


def get_all_related_fights(conn, all_fights_df: pl.DataFrame) -> pl.DataFrame:
    """
    From all fights, return ALL fights that involve ANY fighter who appeared in the dataset.
    This ensures we have complete fight history for every fighter.
    """
    if all_fights_df.is_empty():
        return pl.DataFrame()

    # Collect all unique fighter IDs
    fighters_df = pl.concat([
        all_fights_df.select(pl.col("fighter1_id").alias("fighter_id")),
        all_fights_df.select(pl.col("fighter2_id").alias("fighter_id")),
    ], rechunk=True).unique(subset=["fighter_id"])

    fighters_df = fighters_df.with_columns(pl.col("fighter_id").cast(pl.Utf8))
    fighter_ids = fighters_df.get_column("fighter_id").to_list()

    if not fighter_ids:
        return pl.DataFrame()

    def _esc(v: str) -> str:
        return str(v).replace("'", "''")

    select_union = " UNION ALL ".join(
        [f"SELECT '{_esc(fid)}' AS fighter_id" for fid in fighter_ids]
    )

    query = f"""
    SELECT f.*
    FROM fights f
    JOIN ({select_union}) vf ON f.fighter1_id = vf.fighter_id OR f.fighter2_id = vf.fighter_id;
    """

    rows = fetch_query(conn, query, params=None)
    return pl.DataFrame(rows, infer_schema_length=None)


def build_pre_fight_snapshots(fightHistoryDf: pl.DataFrame, allRelatedFights: pl.DataFrame, min_prior_fights: int = 1) -> pl.DataFrame:
    """
    For each fight in fightHistoryDf, attach:
      - prior_f1: list[struct] of fighter1's fights strictly before this fight's date
      - prior_f2: list[struct] of fighter2's fights strictly before this fight's date
      - prior_cnt_f1 / prior_cnt_f2: counts of those lists

    Filters out fights where either fighter has fewer than min_prior_fights.
    """
    if fightHistoryDf.is_empty():
        return fightHistoryDf

    # Ensure proper date type
    allRelatedFights = allRelatedFights.with_columns(pl.col("fight_date").cast(pl.Date))
    fightHistoryDf = fightHistoryDf.with_columns(pl.col("fight_date").cast(pl.Date))

    # Build long-form fighter history
    optional_cols = [c for c in ("method", "weight_class", "end_time", "fight_format") if c in allRelatedFights.columns]
    base_cols = ["fight_id", "fight_date", "winner_id"] + optional_cols

    # Create fighter-level view
    ff1 = allRelatedFights.select(
        pl.col("fighter1_id").alias("fighter_id"),
        pl.col("fighter2_id").alias("opponent_id"),
        *[pl.col(c) for c in base_cols],
    )
    ff2 = allRelatedFights.select(
        pl.col("fighter2_id").alias("fighter_id"),
        pl.col("fighter1_id").alias("opponent_id"),
        *[pl.col(c) for c in base_cols],
    )
    fighter_fights = pl.concat([ff1, ff2], rechunk=True)

    # Add result column
    fighter_fights = (
        fighter_fights
        .with_columns(
            pl.when(pl.col("winner_id") == pl.col("fighter_id"))
              .then(pl.lit("win"))
              .when(pl.col("winner_id").is_null())
              .then(pl.lit("nc"))
              .otherwise(pl.lit("loss"))
              .alias("result")
        )
        .drop("winner_id")
        .sort(by=["fighter_id", "fight_date", "fight_id"])
    )

    # Create history struct
    history_struct_fields = [
        pl.col("fight_id"),
        pl.col("fight_date"),
        pl.col("opponent_id"),
        pl.col("result"),
    ] + ([pl.col("method")] if "method" in fighter_fights.columns else []) \
      + ([pl.col("weight_class")] if "weight_class" in fighter_fights.columns else []) \
      + ([pl.col("end_time")] if "end_time" in fighter_fights.columns else []) \
      + ([pl.col("fight_format")] if "fight_format" in fighter_fights.columns else [])

    fighter_fights = fighter_fights.with_columns(
        pl.struct(history_struct_fields).alias("fight_entry")
    )

    # Process fighter1
    f1_history = (
        fightHistoryDf
        .select("fight_id", "fighter1_id", "fight_date")
        .join(
            fighter_fights.select("fighter_id", "fight_date", "fight_entry"),
            left_on="fighter1_id",
            right_on="fighter_id",
            how="left"
        )
        .filter(pl.col("fight_date_right") < pl.col("fight_date"))
        .group_by("fight_id")
        .agg(pl.col("fight_entry").alias("prior_f1"))
    )

    # Process fighter2
    f2_history = (
        fightHistoryDf
        .select("fight_id", "fighter2_id", "fight_date")
        .join(
            fighter_fights.select("fighter_id", "fight_date", "fight_entry"),
            left_on="fighter2_id",
            right_on="fighter_id",
            how="left"
        )
        .filter(pl.col("fight_date_right") < pl.col("fight_date"))
        .group_by("fight_id")
        .agg(pl.col("fight_entry").alias("prior_f2"))
    )

    # Join back to original dataframe
    enriched = (
        fightHistoryDf
        .join(f1_history, on="fight_id", how="left")
        .join(f2_history, on="fight_id", how="left")
        .with_columns([
            pl.when(pl.col("prior_f1").is_null())
              .then(pl.lit([]))
              .otherwise(pl.col("prior_f1"))
              .alias("prior_f1"),
            pl.when(pl.col("prior_f2").is_null())
              .then(pl.lit([]))
              .otherwise(pl.col("prior_f2"))
              .alias("prior_f2"),
        ])
        .with_columns([
            pl.col("prior_f1").list.len().alias("prior_cnt_f1"),
            pl.col("prior_f2").list.len().alias("prior_cnt_f2"),
        ])
        .filter(
            (pl.col("prior_cnt_f1") >= min_prior_fights) & (pl.col("prior_cnt_f2") >= min_prior_fights)
        )
    )
    return enriched


def enrich_with_fighter_stats(enriched_df: pl.DataFrame, conn) -> pl.DataFrame:
    """
    Enriches the fight dataframe with fighter statistics (height, weight, reach, stance, dob)
    for both fighter1_id and fighter2_id in the main fight row, and for all fighters in the
    prior_f1 and prior_f2 history lists.
    """
    query = """
    SELECT
        fighter_id,
        height_in,
        weight_lbs,
        reach_in,
        stance,
        dob,
        win,
        loss,
        slpm,
        str_acc,
        sapm,
        str_def,
        td_avg,
        td_acc,
        td_def,
        sub_avg
    FROM fighters;
    """
    fighters_df = pl.DataFrame(fetch_query(conn, query, params=None), infer_schema_length=None)

    # Ensure proper types
    if "dob" in fighters_df.columns:
        fighters_df = fighters_df.with_columns(pl.col("dob").cast(pl.Date))

    # Convert decimal columns to float
    decimal_cols = ["slpm", "str_acc", "sapm", "str_def", "td_avg", "td_acc", "td_def", "sub_avg"]
    for col in decimal_cols:
        if col in fighters_df.columns:
            fighters_df = fighters_df.with_columns(pl.col(col).cast(pl.Float64))

    # Enrich main fight row with fighter1 stats
    enriched_df = enriched_df.join(
        fighters_df.rename({
            "fighter_id": "fighter1_id",
            "height_in": "f1_height_in",
            "weight_lbs": "f1_weight_lbs",
            "reach_in": "f1_reach_in",
            "stance": "f1_stance",
            "dob": "f1_dob",
            "win": "f1_win",
            "loss": "f1_loss",
            "slpm": "f1_slpm",
            "str_acc": "f1_str_acc",
            "sapm": "f1_sapm",
            "str_def": "f1_str_def",
            "td_avg": "f1_td_avg",
            "td_acc": "f1_td_acc",
            "td_def": "f1_td_def",
            "sub_avg": "f1_sub_avg"
        }),
        on="fighter1_id",
        how="left"
    )

    # Enrich main fight row with fighter2 stats
    enriched_df = enriched_df.join(
        fighters_df.rename({
            "fighter_id": "fighter2_id",
            "height_in": "f2_height_in",
            "weight_lbs": "f2_weight_lbs",
            "reach_in": "f2_reach_in",
            "stance": "f2_stance",
            "dob": "f2_dob",
            "win": "f2_win",
            "loss": "f2_loss",
            "slpm": "f2_slpm",
            "str_acc": "f2_str_acc",
            "sapm": "f2_sapm",
            "str_def": "f2_str_def",
            "td_avg": "f2_td_avg",
            "td_acc": "f2_td_acc",
            "td_def": "f2_td_def",
            "sub_avg": "f2_sub_avg"
        }),
        on="fighter2_id",
        how="left"
    )

    # Check what fields exist in the prior_f1 struct
    struct_fields = enriched_df.select(pl.col("prior_f1")).schema["prior_f1"].inner.fields
    field_names = [field.name for field in struct_fields]
    has_method = "method" in field_names
    has_weight_class = "weight_class" in field_names
    has_end_time = "end_time" in field_names
    has_fight_format = "fight_format" in field_names

    # Add row index to track original rows
    enriched_df = enriched_df.with_row_index("_row_idx")

    # Explode prior_f1, join with fighter stats, re-aggregate
    prior_f1_base = (
        enriched_df
        .select("_row_idx", "prior_f1")
        .explode("prior_f1")
        .with_columns(
            pl.col("prior_f1").struct.field("opponent_id").alias("opponent_id")
        )
        .join(
            fighters_df.rename({
                "fighter_id": "opponent_id",
                "height_in": "opp_height_in",
                "weight_lbs": "opp_weight_lbs",
                "reach_in": "opp_reach_in",
                "stance": "opp_stance",
                "dob": "opp_dob"
            }),
            on="opponent_id",
            how="left"
        )
    )

    # Build struct fields list dynamically
    struct_fields_list = [
        pl.col("prior_f1").struct.field("fight_id"),
        pl.col("prior_f1").struct.field("fight_date"),
        pl.col("opponent_id"),
        pl.col("prior_f1").struct.field("result"),
    ]

    if has_method:
        struct_fields_list.append(pl.col("prior_f1").struct.field("method"))
    if has_weight_class:
        struct_fields_list.append(pl.col("prior_f1").struct.field("weight_class"))
    if has_end_time:
        struct_fields_list.append(pl.col("prior_f1").struct.field("end_time"))
    if has_fight_format:
        struct_fields_list.append(pl.col("prior_f1").struct.field("fight_format"))

    struct_fields_list.extend([
        pl.col("opp_height_in"),
        pl.col("opp_weight_lbs"),
        pl.col("opp_reach_in"),
        pl.col("opp_stance"),
        pl.col("opp_dob")
    ])

    prior_f1_enriched = (
        prior_f1_base
        .with_columns(
            pl.struct(struct_fields_list).alias("prior_f1_enriched")
        )
        .group_by("_row_idx")
        .agg(pl.col("prior_f1_enriched").alias("prior_f1"))
    )

    # Same for prior_f2
    prior_f2_base = (
        enriched_df
        .select("_row_idx", "prior_f2")
        .explode("prior_f2")
        .with_columns(
            pl.col("prior_f2").struct.field("opponent_id").alias("opponent_id")
        )
        .join(
            fighters_df.rename({
                "fighter_id": "opponent_id",
                "height_in": "opp_height_in",
                "weight_lbs": "opp_weight_lbs",
                "reach_in": "opp_reach_in",
                "stance": "opp_stance",
                "dob": "opp_dob"
            }),
            on="opponent_id",
            how="left"
        )
    )

    struct_fields_list_f2 = [
        pl.col("prior_f2").struct.field("fight_id"),
        pl.col("prior_f2").struct.field("fight_date"),
        pl.col("opponent_id"),
        pl.col("prior_f2").struct.field("result"),
    ]

    if has_method:
        struct_fields_list_f2.append(pl.col("prior_f2").struct.field("method"))
    if has_weight_class:
        struct_fields_list_f2.append(pl.col("prior_f2").struct.field("weight_class"))
    if has_end_time:
        struct_fields_list_f2.append(pl.col("prior_f2").struct.field("end_time"))
    if has_fight_format:
        struct_fields_list_f2.append(pl.col("prior_f2").struct.field("fight_format"))

    struct_fields_list_f2.extend([
        pl.col("opp_height_in"),
        pl.col("opp_weight_lbs"),
        pl.col("opp_reach_in"),
        pl.col("opp_stance"),
        pl.col("opp_dob")
    ])

    prior_f2_enriched = (
        prior_f2_base
        .with_columns(
            pl.struct(struct_fields_list_f2).alias("prior_f2_enriched")
        )
        .group_by("_row_idx")
        .agg(pl.col("prior_f2_enriched").alias("prior_f2"))
    )

    # Join enriched histories back
    enriched_df = (
        enriched_df
        .drop(["prior_f1", "prior_f2"])
        .join(prior_f1_enriched, on="_row_idx", how="left")
        .join(prior_f2_enriched, on="_row_idx", how="left")
        .drop("_row_idx")
    )

    return enriched_df


def enrich_with_fight_totals(enriched_df: pl.DataFrame, conn) -> pl.DataFrame:
    """
    Enriches the prior_f1 and prior_f2 history lists with fight totals data.
    For each fight in the history, adds the fighter's stats and their opponent's stats.
    """
    query = """
    SELECT
        fight_id,
        fighter_id,
        body_attempts,
        body_landed,
        clinch_attempts,
        clinch_landed,
        ctrl_time_s,
        distance_attempts,
        distance_landed,
        ground_attempts,
        ground_landed,
        head_attempts,
        head_landed,
        kd,
        leg_attempts,
        leg_landed,
        rev,
        sig_str_attempts,
        sig_str_landed,
        sub_att,
        td_attempts,
        td_landed,
        total_str_attempts,
        total_str_landed
    FROM fight_totals;
    """
    fight_totals_df = pl.DataFrame(fetch_query(conn, query, params=None), infer_schema_length=None)

    # Check what fields exist in the prior_f1 struct
    struct_fields = enriched_df.select(pl.col("prior_f1")).schema["prior_f1"].inner.fields
    field_names = [field.name for field in struct_fields]
    has_method = "method" in field_names
    has_weight_class = "weight_class" in field_names
    has_end_time = "end_time" in field_names

    # Add row index to track original rows
    enriched_df = enriched_df.with_row_index("_row_idx")

    # Build prior_f1 with totals
    prior_f1_base = (
        enriched_df
        .select("_row_idx", "fighter1_id", "prior_f1")
        .explode("prior_f1")
        .with_columns([
            pl.col("prior_f1").struct.field("fight_id").alias("fight_id"),
            pl.col("prior_f1").struct.field("opponent_id").alias("opponent_id")
        ])
    )

    # Get fighter's stats and opponent's stats for each prior fight
    # Join with fighter's stats (where fighter_id in totals = fighter1_id from main fight)
    prior_f1_with_fighter_stats = prior_f1_base.join(
        fight_totals_df.rename({"fighter_id": "fighter1_id"}),
        on=["fight_id", "fighter1_id"],
        how="left"
    )

    # Join with opponent's stats
    prior_f1_with_stats = (
        prior_f1_with_fighter_stats
        .join(fight_totals_df.rename({
            "fighter_id": "opponent_id",
            "body_attempts": "opp_body_attempts", "body_landed": "opp_body_landed",
            "clinch_attempts": "opp_clinch_attempts", "clinch_landed": "opp_clinch_landed",
            "ctrl_time_s": "opp_ctrl_time_s", "distance_attempts": "opp_distance_attempts",
            "distance_landed": "opp_distance_landed", "ground_attempts": "opp_ground_attempts",
            "ground_landed": "opp_ground_landed", "head_attempts": "opp_head_attempts",
            "head_landed": "opp_head_landed", "kd": "opp_kd", "leg_attempts": "opp_leg_attempts",
            "leg_landed": "opp_leg_landed", "rev": "opp_rev", "sig_str_attempts": "opp_sig_str_attempts",
            "sig_str_landed": "opp_sig_str_landed", "sub_att": "opp_sub_att", "td_attempts": "opp_td_attempts",
            "td_landed": "opp_td_landed", "total_str_attempts": "opp_total_str_attempts",
            "total_str_landed": "opp_total_str_landed"
        }), on=["fight_id", "opponent_id"], how="left")
        .unique(subset=["_row_idx", "fight_id"])  # Deduplicate by row and fight
    )

    struct_list = [
        pl.col("prior_f1").struct.field("fight_id"), pl.col("prior_f1").struct.field("fight_date"),
        pl.col("opponent_id"), pl.col("prior_f1").struct.field("result"),
    ]
    if has_method: struct_list.append(pl.col("prior_f1").struct.field("method"))
    if has_weight_class: struct_list.append(pl.col("prior_f1").struct.field("weight_class"))
    if has_end_time: struct_list.append(pl.col("prior_f1").struct.field("end_time"))
    struct_list.append(pl.col("prior_f1").struct.field("fight_format"))

    if "opp_height_in" in prior_f1_with_stats.columns:
        struct_list.extend([
            pl.col("prior_f1").struct.field("opp_height_in"), pl.col("prior_f1").struct.field("opp_weight_lbs"),
            pl.col("prior_f1").struct.field("opp_reach_in"), pl.col("prior_f1").struct.field("opp_stance"),
            pl.col("prior_f1").struct.field("opp_dob")
        ])

    struct_list.extend([
        pl.col("body_attempts"), pl.col("body_landed"), pl.col("clinch_attempts"), pl.col("clinch_landed"),
        pl.col("ctrl_time_s"), pl.col("distance_attempts"), pl.col("distance_landed"), pl.col("ground_attempts"),
        pl.col("ground_landed"), pl.col("head_attempts"), pl.col("head_landed"), pl.col("kd"),
        pl.col("leg_attempts"), pl.col("leg_landed"), pl.col("rev"), pl.col("sig_str_attempts"),
        pl.col("sig_str_landed"), pl.col("sub_att"), pl.col("td_attempts"), pl.col("td_landed"),
        pl.col("total_str_attempts"), pl.col("total_str_landed"),
        pl.col("opp_body_attempts"), pl.col("opp_body_landed"), pl.col("opp_clinch_attempts"),
        pl.col("opp_clinch_landed"), pl.col("opp_ctrl_time_s"), pl.col("opp_distance_attempts"),
        pl.col("opp_distance_landed"), pl.col("opp_ground_attempts"), pl.col("opp_ground_landed"),
        pl.col("opp_head_attempts"), pl.col("opp_head_landed"), pl.col("opp_kd"),
        pl.col("opp_leg_attempts"), pl.col("opp_leg_landed"), pl.col("opp_rev"),
        pl.col("opp_sig_str_attempts"), pl.col("opp_sig_str_landed"), pl.col("opp_sub_att"),
        pl.col("opp_td_attempts"), pl.col("opp_td_landed"), pl.col("opp_total_str_attempts"),
        pl.col("opp_total_str_landed")
    ])

    prior_f1_enriched = (
        prior_f1_with_stats.with_columns(pl.struct(struct_list).alias("prior_f1_enriched"))
        .group_by("_row_idx").agg(pl.col("prior_f1_enriched").alias("prior_f1"))
    )

    # Same for prior_f2
    prior_f2_base = (
        enriched_df
        .select("_row_idx", "fighter2_id", "prior_f2")
        .explode("prior_f2")
        .with_columns([
            pl.col("prior_f2").struct.field("fight_id").alias("fight_id"),
            pl.col("prior_f2").struct.field("opponent_id").alias("opponent_id")
        ])
    )

    # Get fighter's stats and opponent's stats for each prior fight
    # Join with fighter's stats (where fighter_id in totals = fighter2_id from main fight)
    prior_f2_with_fighter_stats = prior_f2_base.join(
        fight_totals_df.rename({"fighter_id": "fighter2_id"}),
        on=["fight_id", "fighter2_id"],
        how="left"
    )

    # Join with opponent's stats
    prior_f2_with_stats = (
        prior_f2_with_fighter_stats
        .join(fight_totals_df.rename({
            "fighter_id": "opponent_id",
            "body_attempts": "opp_body_attempts", "body_landed": "opp_body_landed",
            "clinch_attempts": "opp_clinch_attempts", "clinch_landed": "opp_clinch_landed",
            "ctrl_time_s": "opp_ctrl_time_s", "distance_attempts": "opp_distance_attempts",
            "distance_landed": "opp_distance_landed", "ground_attempts": "opp_ground_attempts",
            "ground_landed": "opp_ground_landed", "head_attempts": "opp_head_attempts",
            "head_landed": "opp_head_landed", "kd": "opp_kd", "leg_attempts": "opp_leg_attempts",
            "leg_landed": "opp_leg_landed", "rev": "opp_rev", "sig_str_attempts": "opp_sig_str_attempts",
            "sig_str_landed": "opp_sig_str_landed", "sub_att": "opp_sub_att", "td_attempts": "opp_td_attempts",
            "td_landed": "opp_td_landed", "total_str_attempts": "opp_total_str_attempts",
            "total_str_landed": "opp_total_str_landed"
        }), on=["fight_id", "opponent_id"], how="left")
        .unique(subset=["_row_idx", "fight_id"])  # Deduplicate by row and fight
    )

    struct_list_f2 = [
        pl.col("prior_f2").struct.field("fight_id"), pl.col("prior_f2").struct.field("fight_date"),
        pl.col("opponent_id"), pl.col("prior_f2").struct.field("result"),
    ]
    if has_method: struct_list_f2.append(pl.col("prior_f2").struct.field("method"))
    if has_weight_class: struct_list_f2.append(pl.col("prior_f2").struct.field("weight_class"))
    if has_end_time: struct_list_f2.append(pl.col("prior_f2").struct.field("end_time"))
    struct_list_f2.append(pl.col("prior_f2").struct.field("fight_format"))

    if "opp_height_in" in prior_f2_with_stats.columns:
        struct_list_f2.extend([
            pl.col("prior_f2").struct.field("opp_height_in"), pl.col("prior_f2").struct.field("opp_weight_lbs"),
            pl.col("prior_f2").struct.field("opp_reach_in"), pl.col("prior_f2").struct.field("opp_stance"),
            pl.col("prior_f2").struct.field("opp_dob")
        ])

    struct_list_f2.extend([
        pl.col("body_attempts"), pl.col("body_landed"), pl.col("clinch_attempts"), pl.col("clinch_landed"),
        pl.col("ctrl_time_s"), pl.col("distance_attempts"), pl.col("distance_landed"), pl.col("ground_attempts"),
        pl.col("ground_landed"), pl.col("head_attempts"), pl.col("head_landed"), pl.col("kd"),
        pl.col("leg_attempts"), pl.col("leg_landed"), pl.col("rev"), pl.col("sig_str_attempts"),
        pl.col("sig_str_landed"), pl.col("sub_att"), pl.col("td_attempts"), pl.col("td_landed"),
        pl.col("total_str_attempts"), pl.col("total_str_landed"),
        pl.col("opp_body_attempts"), pl.col("opp_body_landed"), pl.col("opp_clinch_attempts"),
        pl.col("opp_clinch_landed"), pl.col("opp_ctrl_time_s"), pl.col("opp_distance_attempts"),
        pl.col("opp_distance_landed"), pl.col("opp_ground_attempts"), pl.col("opp_ground_landed"),
        pl.col("opp_head_attempts"), pl.col("opp_head_landed"), pl.col("opp_kd"),
        pl.col("opp_leg_attempts"), pl.col("opp_leg_landed"), pl.col("opp_rev"),
        pl.col("opp_sig_str_attempts"), pl.col("opp_sig_str_landed"), pl.col("opp_sub_att"),
        pl.col("opp_td_attempts"), pl.col("opp_td_landed"), pl.col("opp_total_str_attempts"),
        pl.col("opp_total_str_landed")
    ])

    prior_f2_enriched = (
        prior_f2_with_stats.with_columns(pl.struct(struct_list_f2).alias("prior_f2_enriched"))
        .group_by("_row_idx").agg(pl.col("prior_f2_enriched").alias("prior_f2"))
    )

    enriched_df = (
        enriched_df
        .drop(["prior_f1", "prior_f2"])
        .join(prior_f1_enriched, on="_row_idx", how="left")
        .join(prior_f2_enriched, on="_row_idx", how="left")
        .drop("_row_idx")
    )
    return enriched_df


def enrich_with_round_data(enriched_df: pl.DataFrame, conn) -> pl.DataFrame:
    """
    **NEW FEATURE**: Enriches the prior_f1 and prior_f2 history lists with round-by-round data.
    For each fight in the history, adds a list of rounds with detailed round statistics.
    """
    print("\n" + "="*80)
    print("ENRICHING WITH ROUND-BY-ROUND DATA")
    print("="*80)

    query = """
    SELECT
        fight_id,
        round_number,
        fighter_id,
        body_attempts,
        body_landed,
        clinch_attempts,
        clinch_landed,
        ctrl_time_s,
        distance_attempts,
        distance_landed,
        ground_attempts,
        ground_landed,
        head_attempts,
        head_landed,
        kd,
        leg_attempts,
        leg_landed,
        rev,
        sig_str_attempts,
        sig_str_landed,
        sub_att,
        td_attempts,
        td_landed,
        total_str_attempts,
        total_str_landed
    FROM fight_rounds
    ORDER BY fight_id, fighter_id, round_number;
    """
    fight_rounds_df = pl.DataFrame(fetch_query(conn, query, params=None), infer_schema_length=None)

    print(f"Loaded {len(fight_rounds_df)} round records from database")

    # Check what fields exist in the prior_f1 struct
    struct_fields = enriched_df.select(pl.col("prior_f1")).schema["prior_f1"].inner.fields
    field_names = [field.name for field in struct_fields]

    # Add row index to track original rows
    enriched_df = enriched_df.with_row_index("_row_idx")

    # ==================== Process prior_f1 with rounds ====================
    print("\nProcessing prior_f1 rounds...")

    prior_f1_base = (
        enriched_df
        .select("_row_idx", "fighter1_id", "prior_f1")
        .explode("prior_f1")
        .with_columns([
            pl.col("prior_f1").struct.field("fight_id").alias("fight_id"),
        ])
    )

    # Join fighter's round data - get ALL rounds for the fight (both fighters)
    prior_f1_with_rounds = (
        prior_f1_base
        .join(
            fight_rounds_df,  # Don't rename - keep fighter_id to distinguish fighters
            on="fight_id",    # Only join on fight_id to get both fighters' rounds
            how="left"
        )
    )

    # Create round struct - include fighter_id so we know whose stats these are
    round_struct_fields = [
        pl.col("fighter_id"),  # Add fighter_id to identify whose stats these are
        pl.col("round_number"),
        pl.col("kd"),
        pl.col("sig_str_landed"), pl.col("sig_str_attempts"),
        pl.col("total_str_landed"), pl.col("total_str_attempts"),
        pl.col("td_landed"), pl.col("td_attempts"),
        pl.col("sub_att"), pl.col("rev"), pl.col("ctrl_time_s"),
        pl.col("head_landed"), pl.col("head_attempts"),
        pl.col("body_landed"), pl.col("body_attempts"),
        pl.col("leg_landed"), pl.col("leg_attempts"),
        pl.col("distance_landed"), pl.col("distance_attempts"),
        pl.col("clinch_landed"), pl.col("clinch_attempts"),
        pl.col("ground_landed"), pl.col("ground_attempts"),
    ]

    # Group rounds into a list per fight
    prior_f1_rounds_grouped = (
        prior_f1_with_rounds
        .with_columns(pl.struct(round_struct_fields).alias("round_data"))
        .group_by(["_row_idx", "fight_id"])
        .agg(pl.col("round_data").alias("rounds"))
    )

    # Now rebuild the prior_f1 struct with all existing fields PLUS the rounds field
    prior_f1_full = (
        prior_f1_base
        .join(prior_f1_rounds_grouped, on=["_row_idx", "fight_id"], how="left")
    )

    # Build the final struct with all fields
    final_struct_fields = []
    for field in field_names:
        final_struct_fields.append(pl.col("prior_f1").struct.field(field))
    final_struct_fields.append(pl.col("rounds"))

    prior_f1_enriched = (
        prior_f1_full
        .with_columns(pl.struct(final_struct_fields).alias("prior_f1_with_rounds"))
        .group_by("_row_idx")
        .agg(pl.col("prior_f1_with_rounds").alias("prior_f1"))
    )

    # ==================== Process prior_f2 with rounds ====================
    print("Processing prior_f2 rounds...")

    prior_f2_base = (
        enriched_df
        .select("_row_idx", "fighter2_id", "prior_f2")
        .explode("prior_f2")
        .with_columns([
            pl.col("prior_f2").struct.field("fight_id").alias("fight_id"),
        ])
    )

    prior_f2_with_rounds = (
        prior_f2_base
        .join(
            fight_rounds_df,  # Don't rename - keep fighter_id to distinguish fighters
            on="fight_id",    # Only join on fight_id to get both fighters' rounds
            how="left"
        )
    )

    prior_f2_rounds_grouped = (
        prior_f2_with_rounds
        .with_columns(pl.struct(round_struct_fields).alias("round_data"))
        .group_by(["_row_idx", "fight_id"])
        .agg(pl.col("round_data").alias("rounds"))
    )

    prior_f2_full = (
        prior_f2_base
        .join(prior_f2_rounds_grouped, on=["_row_idx", "fight_id"], how="left")
    )

    # Build final struct for prior_f2
    final_struct_fields_f2 = []
    for field in field_names:
        final_struct_fields_f2.append(pl.col("prior_f2").struct.field(field))
    final_struct_fields_f2.append(pl.col("rounds"))

    prior_f2_enriched = (
        prior_f2_full
        .with_columns(pl.struct(final_struct_fields_f2).alias("prior_f2_with_rounds"))
        .group_by("_row_idx")
        .agg(pl.col("prior_f2_with_rounds").alias("prior_f2"))
    )

    # Join enriched histories back
    enriched_df = (
        enriched_df
        .drop(["prior_f1", "prior_f2"])
        .join(prior_f1_enriched, on="_row_idx", how="left")
        .join(prior_f2_enriched, on="_row_idx", how="left")
        .drop("_row_idx")
    )

    print(f"✅ Round-by-round data added to {len(enriched_df)} fight snapshots")
    print("="*80 + "\n")

    return enriched_df


def run(min_prior_fights: int = 1):
    """
    Build complete raw dataframe with ALL fights in database.
    Includes round-by-round data for richer feature extraction.

    Parameters:
    -----------
    min_prior_fights : int
        Minimum prior fights required (default: 1)

    Returns:
    --------
    pl.DataFrame
        Enriched fight data with complete fight history including rounds
    """
    conn = create_connection()

    print("\n" + "="*80)
    print("BUILDING RAW DATAFRAME - ALL FIGHTS WITH ROUND DATA")
    print("="*80)
    print(f"\nMinimum prior fights required: {min_prior_fights}")
    print(f"Target: ALL fights in database\n")

    # Step 1: Get all fights
    print("Step 1: Loading all fights from database...")
    all_fights = get_all_fights(conn)
    print(f"   ✅ Loaded {len(all_fights)} total fights")

    if all_fights.is_empty():
        print("❌ No fights found in database")
        return pl.DataFrame()

    # Step 2: Get all related fights for complete history
    print("\nStep 2: Loading complete fight history...")
    allRelatedFights = get_all_related_fights(conn, all_fights)
    print(f"   ✅ Loaded {len(allRelatedFights)} fights for complete history")

    # Step 3: Build pre-fight snapshots
    print(f"\nStep 3: Building pre-fight snapshots (min {min_prior_fights} prior fights)...")
    snapshotsDf = build_pre_fight_snapshots(all_fights, allRelatedFights, min_prior_fights)
    print(f"   ✅ Created {len(snapshotsDf)} fight snapshots")

    if snapshotsDf.is_empty():
        print(f"❌ No fights remain after requiring {min_prior_fights}+ prior fights")
        return pl.DataFrame()

    # Step 4: Enrich with fighter stats
    print("\nStep 4: Enriching with fighter statistics...")
    enrichedDfFightersData = enrich_with_fighter_stats(snapshotsDf, conn)
    print(f"   ✅ Added fighter stats (height, reach, stance, career stats, etc.)")

    # Step 5: Enrich with fight totals
    print("\nStep 5: Enriching with fight totals (aggregated stats)...")
    enrichedDfFightTotals = enrich_with_fight_totals(enrichedDfFightersData, conn)
    print(f"   ✅ Added fight totals (strikes, TDs, KDs, etc.)")

    # Step 6: Enrich with round-by-round data (NEW!)
    print("\nStep 6: Enriching with round-by-round data...")
    enrichedDfWithRounds = enrich_with_round_data(enrichedDfFightTotals, conn)
    print(f"   ✅ Added round-by-round statistics")

    # Step 7: Save as parquet
    output_file = "fight_snapshots_all_with_rounds.parquet"
    print(f"\nStep 7: Saving to {output_file}...")
    enrichedDfWithRounds.write_parquet(output_file)
    print(f"   ✅ Saved {len(enrichedDfWithRounds)} fight snapshots")

    # Summary
    print("\n" + "="*80)
    print("✅ COMPLETE - RAW DATAFRAME BUILT SUCCESSFULLY")
    print("="*80)
    print(f"\n📊 Summary:")
    print(f"   - Total fights in database: {len(all_fights)}")
    print(f"   - Fights with {min_prior_fights}+ prior fights: {len(enrichedDfWithRounds)}")
    print(f"   - Output file: {output_file}")
    print(f"   - File size: {enrichedDfWithRounds.estimated_size() / 1024 / 1024:.2f} MB (estimated)")
    print(f"\n🎯 Features included:")
    print(f"   ✓ Fight metadata (date, fighters, outcome)")
    print(f"   ✓ Fighter attributes (height, reach, stance, DOB)")
    print(f"   ✓ Career statistics (W-L record, overall averages)")
    print(f"   ✓ Complete fight history (prior_f1, prior_f2)")
    print(f"   ✓ Fight totals for each historical fight")
    print(f"   ✓ Round-by-round data for each historical fight")
    print(f"\n🔥 Ready for advanced feature engineering!\n")

    conn.close()
    return enrichedDfWithRounds


if __name__ == "__main__":
    run(min_prior_fights=1)
