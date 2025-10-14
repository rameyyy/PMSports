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


def get_fights_with_valid_fighters(conn, valid_fighters_df: pl.DataFrame) -> pl.DataFrame:
    if valid_fighters_df.is_empty():
        return pl.DataFrame()

    # Dedup + keep as strings
    valid_ids = sorted(set(valid_fighters_df.get_column("fighter_id").to_list()))

    # Build a SELECT…UNION ALL block with PROPER QUOTING
    # e.g., SELECT 'abc' AS fighter_id UNION ALL SELECT 'def' ...
    def _esc(v: str) -> str:
        # minimal SQL literal escape for single quotes
        return str(v).replace("'", "''")

    select_union = " UNION ALL ".join(
        [f"SELECT '{_esc(fid)}' AS fighter_id" for fid in valid_ids]
    )

    # Use the same derived table twice (no parameters needed)
    query = f"""
    SELECT f.*
    FROM fights f
    JOIN (
        {select_union}
    ) vf1 ON f.fighter1_id = vf1.fighter_id
    JOIN (
        {select_union}
    ) vf2 ON f.fighter2_id = vf2.fighter_id
    ;
    """

    rows = fetch_query(conn, query, params=None)
    return pl.DataFrame(rows)


def get_all_related_fights(conn, valid_fights_df: pl.DataFrame) -> pl.DataFrame:
    """
    From the fights returned by get_fights_with_valid_fighters, return ALL fights
    that involve ANY fighter who appeared in that set (either as fighter1 or fighter2).
    """

    if valid_fights_df.is_empty():
        return pl.DataFrame()

    # Collect all fighter IDs from both columns under a single name
    fighters_df = pl.concat(
        [
            valid_fights_df.select(pl.col("fighter1_id").alias("fighter_id")),
            valid_fights_df.select(pl.col("fighter2_id").alias("fighter_id")),
        ],
        rechunk=True,
    ).unique(subset=["fighter_id"])

    # If your IDs are strings/hex, ensure Utf8 for safe SQL literal building
    fighters_df = fighters_df.with_columns(pl.col("fighter_id").cast(pl.Utf8))

    fighter_ids = fighters_df.get_column("fighter_id").to_list()
    if not fighter_ids:
        return pl.DataFrame()

    # Build a derived table of IDs with proper quoting (MySQL-safe, no params)
    def _esc(v: str) -> str:
        return str(v).replace("'", "''")

    select_union = " UNION ALL ".join(
        [f"SELECT '{_esc(fid)}' AS fighter_id" for fid in fighter_ids]
    )

    # Pull every fight where either side is in our fighter set
    query = f"""
    SELECT f.*
    FROM fights f
    JOIN (
        {select_union}
    ) vf ON f.fighter1_id = vf.fighter_id OR f.fighter2_id = vf.fighter_id;
    """

    rows = fetch_query(conn, query, params=None)
    return pl.DataFrame(rows)


def build_pre_fight_snapshots(fightHistoryDf: pl.DataFrame, allRelatedFights: pl.DataFrame) -> pl.DataFrame:
    """
    For each fight in fightHistoryDf, attach:
      - prior_f1: list[struct] of fighter1's fights strictly before this fight's date
      - prior_f2: list[struct] of fighter2's fights strictly before this fight's date
      - prior_cnt_f1 / prior_cnt_f2: counts of those lists
    
    Filters out fights where either fighter has fewer than 4 prior fights.
    """
    if fightHistoryDf.is_empty():
        return fightHistoryDf

    # --- Ensure proper date type ---
    allRelatedFights = allRelatedFights.with_columns(pl.col("fight_date").cast(pl.Date))
    fightHistoryDf   = fightHistoryDf.with_columns(pl.col("fight_date").cast(pl.Date))

    # --- Build long-form fighter history ---
    optional_cols = [c for c in ("method", "weight_class") if c in allRelatedFights.columns]
    base_cols = ["fight_id", "fight_date", "winner_id"] + optional_cols

    # Create fighter-level view: each row is one fighter's participation in a fight
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
      + ([pl.col("weight_class")] if "weight_class" in fighter_fights.columns else [])

    fighter_fights = fighter_fights.with_columns(
        pl.struct(history_struct_fields).alias("fight_entry")
    )

    # --- Process fighter1 ---
    # Join fightHistoryDf with fighter_fights where fighter matches fighter1_id
    # and the fight date is before the current fight
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
        .agg(
            pl.col("fight_entry").alias("prior_f1")
        )
    )

    # --- Process fighter2 ---
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
        .agg(
            pl.col("fight_entry").alias("prior_f2")
        )
    )

    # --- Join back to original dataframe ---
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
        # Filter out fights where either fighter has fewer than 4 prior fights
        .filter(
            (pl.col("prior_cnt_f1") >= 4) & (pl.col("prior_cnt_f2") >= 4)
        )
    )

    return enriched

def enrich_with_fighter_stats(enriched_df: pl.DataFrame, conn) -> pl.DataFrame:
    """
    Enriches the fight dataframe with fighter statistics (height, weight, reach, stance, dob)
    for both fighter1_id and fighter2_id in the main fight row, and for all fighters in the 
    prior_f1 and prior_f2 history lists.
    """
    # Query to get all fighter stats
    query = """
    SELECT 
        fighter_id,
        height_in,
        weight_lbs,
        reach_in,
        stance,
        dob
    FROM fighters;
    """
    fighters_df = pl.DataFrame(fetch_query(conn, query, params=None))
    
    # Ensure proper types
    if "dob" in fighters_df.columns:
        fighters_df = fighters_df.with_columns(pl.col("dob").cast(pl.Date))
    
    # --- Enrich main fight row with fighter1 stats ---
    enriched_df = enriched_df.join(
        fighters_df.rename({
            "fighter_id": "fighter1_id",
            "height_in": "f1_height_in",
            "weight_lbs": "f1_weight_lbs",
            "reach_in": "f1_reach_in",
            "stance": "f1_stance",
            "dob": "f1_dob"
        }),
        on="fighter1_id",
        how="left"
    )
    
    # --- Enrich main fight row with fighter2 stats ---
    enriched_df = enriched_df.join(
        fighters_df.rename({
            "fighter_id": "fighter2_id",
            "height_in": "f2_height_in",
            "weight_lbs": "f2_weight_lbs",
            "reach_in": "f2_reach_in",
            "stance": "f2_stance",
            "dob": "f2_dob"
        }),
        on="fighter2_id",
        how="left"
    )
    
    # --- Check what fields exist in the prior_f1 struct ---
    # Get the struct schema from the first non-null value
    struct_fields = enriched_df.select(pl.col("prior_f1")).schema["prior_f1"].inner.fields
    field_names = [field.name for field in struct_fields]
    has_method = "method" in field_names
    has_weight_class = "weight_class" in field_names
    
    # --- Enrich prior_f1 and prior_f2 history lists ---
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
    
    # Explode prior_f2, join with fighter stats, re-aggregate
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
    
    # Build struct fields list for prior_f2
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
    # Query to get all fight totals
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
    fight_totals_df = pl.DataFrame(fetch_query(conn, query, params=None))
    
    # --- Check what fields exist in the prior_f1 struct ---
    struct_fields = enriched_df.select(pl.col("prior_f1")).schema["prior_f1"].inner.fields
    field_names = [field.name for field in struct_fields]
    has_method = "method" in field_names
    has_weight_class = "weight_class" in field_names
    
    # --- Add row index to track original rows ---
    enriched_df = enriched_df.with_row_index("_row_idx")
    
    # --- Process prior_f1 ---
    # We need to get both the fighter's stats and opponent's stats for each fight
    prior_f1_base = (
        enriched_df
        .select("_row_idx", "fighter1_id", "prior_f1")
        .explode("prior_f1")
        .with_columns([
            pl.col("prior_f1").struct.field("fight_id").alias("fight_id"),
            pl.col("prior_f1").struct.field("opponent_id").alias("opponent_id")
        ])
    )
    
    # Join fighter's own stats (fighter1_id for the main fight is the fighter in this historical fight)
    prior_f1_with_stats = (
        prior_f1_base
        .join(
            fight_totals_df.rename({
                "fighter_id": "fighter1_id",
                "body_attempts": "body_attempts",
                "body_landed": "body_landed",
                "clinch_attempts": "clinch_attempts",
                "clinch_landed": "clinch_landed",
                "ctrl_time_s": "ctrl_time_s",
                "distance_attempts": "distance_attempts",
                "distance_landed": "distance_landed",
                "ground_attempts": "ground_attempts",
                "ground_landed": "ground_landed",
                "head_attempts": "head_attempts",
                "head_landed": "head_landed",
                "kd": "kd",
                "leg_attempts": "leg_attempts",
                "leg_landed": "leg_landed",
                "rev": "rev",
                "sig_str_attempts": "sig_str_attempts",
                "sig_str_landed": "sig_str_landed",
                "sub_att": "sub_att",
                "td_attempts": "td_attempts",
                "td_landed": "td_landed",
                "total_str_attempts": "total_str_attempts",
                "total_str_landed": "total_str_landed"
            }),
            on=["fight_id", "fighter1_id"],
            how="left"
        )
        # Join opponent's stats
        .join(
            fight_totals_df.rename({
                "fighter_id": "opponent_id",
                "body_attempts": "opp_body_attempts",
                "body_landed": "opp_body_landed",
                "clinch_attempts": "opp_clinch_attempts",
                "clinch_landed": "opp_clinch_landed",
                "ctrl_time_s": "opp_ctrl_time_s",
                "distance_attempts": "opp_distance_attempts",
                "distance_landed": "opp_distance_landed",
                "ground_attempts": "opp_ground_attempts",
                "ground_landed": "opp_ground_landed",
                "head_attempts": "opp_head_attempts",
                "head_landed": "opp_head_landed",
                "kd": "opp_kd",
                "leg_attempts": "opp_leg_attempts",
                "leg_landed": "opp_leg_landed",
                "rev": "opp_rev",
                "sig_str_attempts": "opp_sig_str_attempts",
                "sig_str_landed": "opp_sig_str_landed",
                "sub_att": "opp_sub_att",
                "td_attempts": "opp_td_attempts",
                "td_landed": "opp_td_landed",
                "total_str_attempts": "opp_total_str_attempts",
                "total_str_landed": "opp_total_str_landed"
            }),
            on=["fight_id", "opponent_id"],
            how="left"
        )
    )
    
    # Build struct with all fields
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
    
    # Add fighter stats fields that already exist (from previous enrichment)
    if "opp_height_in" in prior_f1_with_stats.columns:
        struct_fields_list.extend([
            pl.col("prior_f1").struct.field("opp_height_in"),
            pl.col("prior_f1").struct.field("opp_weight_lbs"),
            pl.col("prior_f1").struct.field("opp_reach_in"),
            pl.col("prior_f1").struct.field("opp_stance"),
            pl.col("prior_f1").struct.field("opp_dob")
        ])
    
    # Add fight totals
    struct_fields_list.extend([
        pl.col("body_attempts"),
        pl.col("body_landed"),
        pl.col("clinch_attempts"),
        pl.col("clinch_landed"),
        pl.col("ctrl_time_s"),
        pl.col("distance_attempts"),
        pl.col("distance_landed"),
        pl.col("ground_attempts"),
        pl.col("ground_landed"),
        pl.col("head_attempts"),
        pl.col("head_landed"),
        pl.col("kd"),
        pl.col("leg_attempts"),
        pl.col("leg_landed"),
        pl.col("rev"),
        pl.col("sig_str_attempts"),
        pl.col("sig_str_landed"),
        pl.col("sub_att"),
        pl.col("td_attempts"),
        pl.col("td_landed"),
        pl.col("total_str_attempts"),
        pl.col("total_str_landed"),
        pl.col("opp_body_attempts"),
        pl.col("opp_body_landed"),
        pl.col("opp_clinch_attempts"),
        pl.col("opp_clinch_landed"),
        pl.col("opp_ctrl_time_s"),
        pl.col("opp_distance_attempts"),
        pl.col("opp_distance_landed"),
        pl.col("opp_ground_attempts"),
        pl.col("opp_ground_landed"),
        pl.col("opp_head_attempts"),
        pl.col("opp_head_landed"),
        pl.col("opp_kd"),
        pl.col("opp_leg_attempts"),
        pl.col("opp_leg_landed"),
        pl.col("opp_rev"),
        pl.col("opp_sig_str_attempts"),
        pl.col("opp_sig_str_landed"),
        pl.col("opp_sub_att"),
        pl.col("opp_td_attempts"),
        pl.col("opp_td_landed"),
        pl.col("opp_total_str_attempts"),
        pl.col("opp_total_str_landed")
    ])
    
    prior_f1_enriched = (
        prior_f1_with_stats
        .with_columns(
            pl.struct(struct_fields_list).alias("prior_f1_enriched")
        )
        .group_by("_row_idx")
        .agg(pl.col("prior_f1_enriched").alias("prior_f1"))
    )
    
    # --- Process prior_f2 (same logic) ---
    prior_f2_base = (
        enriched_df
        .select("_row_idx", "fighter2_id", "prior_f2")
        .explode("prior_f2")
        .with_columns([
            pl.col("prior_f2").struct.field("fight_id").alias("fight_id"),
            pl.col("prior_f2").struct.field("opponent_id").alias("opponent_id")
        ])
    )
    
    prior_f2_with_stats = (
        prior_f2_base
        .join(
            fight_totals_df.rename({
                "fighter_id": "fighter2_id",
                "body_attempts": "body_attempts",
                "body_landed": "body_landed",
                "clinch_attempts": "clinch_attempts",
                "clinch_landed": "clinch_landed",
                "ctrl_time_s": "ctrl_time_s",
                "distance_attempts": "distance_attempts",
                "distance_landed": "distance_landed",
                "ground_attempts": "ground_attempts",
                "ground_landed": "ground_landed",
                "head_attempts": "head_attempts",
                "head_landed": "head_landed",
                "kd": "kd",
                "leg_attempts": "leg_attempts",
                "leg_landed": "leg_landed",
                "rev": "rev",
                "sig_str_attempts": "sig_str_attempts",
                "sig_str_landed": "sig_str_landed",
                "sub_att": "sub_att",
                "td_attempts": "td_attempts",
                "td_landed": "td_landed",
                "total_str_attempts": "total_str_attempts",
                "total_str_landed": "total_str_landed"
            }),
            on=["fight_id", "fighter2_id"],
            how="left"
        )
        .join(
            fight_totals_df.rename({
                "fighter_id": "opponent_id",
                "body_attempts": "opp_body_attempts",
                "body_landed": "opp_body_landed",
                "clinch_attempts": "opp_clinch_attempts",
                "clinch_landed": "opp_clinch_landed",
                "ctrl_time_s": "opp_ctrl_time_s",
                "distance_attempts": "opp_distance_attempts",
                "distance_landed": "opp_distance_landed",
                "ground_attempts": "opp_ground_attempts",
                "ground_landed": "opp_ground_landed",
                "head_attempts": "opp_head_attempts",
                "head_landed": "opp_head_landed",
                "kd": "opp_kd",
                "leg_attempts": "opp_leg_attempts",
                "leg_landed": "opp_leg_landed",
                "rev": "opp_rev",
                "sig_str_attempts": "opp_sig_str_attempts",
                "sig_str_landed": "opp_sig_str_landed",
                "sub_att": "opp_sub_att",
                "td_attempts": "opp_td_attempts",
                "td_landed": "opp_td_landed",
                "total_str_attempts": "opp_total_str_attempts",
                "total_str_landed": "opp_total_str_landed"
            }),
            on=["fight_id", "opponent_id"],
            how="left"
        )
    )
    
    # Build struct for prior_f2
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
    
    if "opp_height_in" in prior_f2_with_stats.columns:
        struct_fields_list_f2.extend([
            pl.col("prior_f2").struct.field("opp_height_in"),
            pl.col("prior_f2").struct.field("opp_weight_lbs"),
            pl.col("prior_f2").struct.field("opp_reach_in"),
            pl.col("prior_f2").struct.field("opp_stance"),
            pl.col("prior_f2").struct.field("opp_dob")
        ])
    
    struct_fields_list_f2.extend([
        pl.col("body_attempts"),
        pl.col("body_landed"),
        pl.col("clinch_attempts"),
        pl.col("clinch_landed"),
        pl.col("ctrl_time_s"),
        pl.col("distance_attempts"),
        pl.col("distance_landed"),
        pl.col("ground_attempts"),
        pl.col("ground_landed"),
        pl.col("head_attempts"),
        pl.col("head_landed"),
        pl.col("kd"),
        pl.col("leg_attempts"),
        pl.col("leg_landed"),
        pl.col("rev"),
        pl.col("sig_str_attempts"),
        pl.col("sig_str_landed"),
        pl.col("sub_att"),
        pl.col("td_attempts"),
        pl.col("td_landed"),
        pl.col("total_str_attempts"),
        pl.col("total_str_landed"),
        pl.col("opp_body_attempts"),
        pl.col("opp_body_landed"),
        pl.col("opp_clinch_attempts"),
        pl.col("opp_clinch_landed"),
        pl.col("opp_ctrl_time_s"),
        pl.col("opp_distance_attempts"),
        pl.col("opp_distance_landed"),
        pl.col("opp_ground_attempts"),
        pl.col("opp_ground_landed"),
        pl.col("opp_head_attempts"),
        pl.col("opp_head_landed"),
        pl.col("opp_kd"),
        pl.col("opp_leg_attempts"),
        pl.col("opp_leg_landed"),
        pl.col("opp_rev"),
        pl.col("opp_sig_str_attempts"),
        pl.col("opp_sig_str_landed"),
        pl.col("opp_sub_att"),
        pl.col("opp_td_attempts"),
        pl.col("opp_td_landed"),
        pl.col("opp_total_str_attempts"),
        pl.col("opp_total_str_landed")
    ])
    
    prior_f2_enriched = (
        prior_f2_with_stats
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

def run():
    conn = create_connection()

    validFightersDf = get_valid_fighters(conn)
    fightHistoryDf   = get_fights_with_valid_fighters(conn, validFightersDf)
    allRelatedFights = get_all_related_fights(conn, fightHistoryDf)

    snapshotsDf = build_pre_fight_snapshots(fightHistoryDf, allRelatedFights)
    enrichedDfFightersData = enrich_with_fighter_stats(snapshotsDf, conn)
    enrichedDfFightTotals = enrich_with_fight_totals(enrichedDfFightersData, conn)

    print(f"Rows: {len(enrichedDfFightTotals)}")
    print(snapshotsDf.select(["fight_id","prior_cnt_f1","prior_cnt_f2", "fight_date"]).head())
    enrichedDfFightTotals.write_parquet("models/fight_snapshots.parquet")  # recommended over CSV for nested lists
    print(enrichedDfFightTotals.head(1).to_dicts()[0])

