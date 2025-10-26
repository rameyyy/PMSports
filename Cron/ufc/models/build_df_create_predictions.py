import polars as pl
from .utils import create_connection, fetch_query
from .fight_features import process_snapshots_to_features
from .attr_fighthist_combined_model import CombinedPredictor
from .snapshots_to_model_ready_df import process_snapshots_to_flat_features
from .load_models_and_predict import quick_predict


def get_fights_for_event(conn, event_id: str) -> pl.DataFrame:
    """Get all fights for a specific event."""
    query = """
    SELECT *
    FROM fights
    WHERE event_id = %s;
    """
    rows = fetch_query(conn, query, params=(event_id,))
    return pl.DataFrame(rows)


def get_all_related_fights(conn, event_fights_df: pl.DataFrame) -> pl.DataFrame:
    """Get ALL fights involving any fighter from the event."""
    if event_fights_df.is_empty():
        return pl.DataFrame()

    fighters_df = pl.concat([
        event_fights_df.select(pl.col("fighter1_id").alias("fighter_id")),
        event_fights_df.select(pl.col("fighter2_id").alias("fighter_id")),
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
    return pl.DataFrame(rows)


def build_pre_fight_snapshots(event_fights_df: pl.DataFrame, allRelatedFights: pl.DataFrame, min_prior_fights: int = 3) -> pl.DataFrame:
    """Build snapshots with prior fight history. Filters out fights where either fighter has < min_prior_fights."""
    if event_fights_df.is_empty():
        return event_fights_df

    allRelatedFights = allRelatedFights.with_columns(pl.col("fight_date").cast(pl.Date))
    event_fights_df = event_fights_df.with_columns(pl.col("fight_date").cast(pl.Date))

    optional_cols = [c for c in ("method", "weight_class", "end_time", "fight_format") if c in allRelatedFights.columns]
    base_cols = ["fight_id", "fight_date", "winner_id"] + optional_cols

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

    f1_history = (
        event_fights_df
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

    f2_history = (
        event_fights_df
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

    enriched = (
        event_fights_df
        .join(f1_history, on="fight_id", how="left")
        .join(f2_history, on="fight_id", how="left")
        .with_columns([
            pl.when(pl.col("prior_f1").is_null()).then(pl.lit([])).otherwise(pl.col("prior_f1")).alias("prior_f1"),
            pl.when(pl.col("prior_f2").is_null()).then(pl.lit([])).otherwise(pl.col("prior_f2")).alias("prior_f2"),
        ])
        .with_columns([
            pl.col("prior_f1").list.len().alias("prior_cnt_f1"),
            pl.col("prior_f2").list.len().alias("prior_cnt_f2"),
        ])
        .filter((pl.col("prior_cnt_f1") >= min_prior_fights) & (pl.col("prior_cnt_f2") >= min_prior_fights))
    )
    return enriched


def enrich_with_fighter_stats(enriched_df: pl.DataFrame, conn) -> pl.DataFrame:
    """Enrich with fighter stats."""
    query = """
    SELECT fighter_id, height_in, weight_lbs, reach_in, stance, dob,
           win, loss, slpm, str_acc, sapm, str_def, td_avg, td_acc, td_def, sub_avg
    FROM fighters;
    """
    fighters_df = pl.DataFrame(fetch_query(conn, query, params=None))
    
    if "dob" in fighters_df.columns:
        fighters_df = fighters_df.with_columns(pl.col("dob").cast(pl.Date))
    
    # Convert decimal columns to float
    decimal_cols = ["slpm", "str_acc", "sapm", "str_def", "td_avg", "td_acc", "td_def", "sub_avg"]
    for col in decimal_cols:
        if col in fighters_df.columns:
            fighters_df = fighters_df.with_columns(pl.col(col).cast(pl.Float64))
    
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
        on="fighter1_id", how="left"
    )
    
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
        on="fighter2_id", how="left"
    )
    
    struct_fields = enriched_df.select(pl.col("prior_f1")).schema["prior_f1"].inner.fields
    field_names = [field.name for field in struct_fields]
    has_method = "method" in field_names
    has_weight_class = "weight_class" in field_names
    has_end_time = "end_time" in field_names
    has_fight_format = "fight_format" in field_names
    
    enriched_df = enriched_df.with_row_index("_row_idx")
    
    prior_f1_base = (
        enriched_df.select("_row_idx", "prior_f1").explode("prior_f1")
        .with_columns(pl.col("prior_f1").struct.field("opponent_id").alias("opponent_id"))
        .join(fighters_df.rename({"fighter_id": "opponent_id", "height_in": "opp_height_in", "weight_lbs": "opp_weight_lbs",
                                  "reach_in": "opp_reach_in", "stance": "opp_stance", "dob": "opp_dob"}),
              on="opponent_id", how="left")
    )
    
    struct_fields_list = [
        pl.col("prior_f1").struct.field("fight_id"), pl.col("prior_f1").struct.field("fight_date"),
        pl.col("opponent_id"), pl.col("prior_f1").struct.field("result"),
    ]
    if has_method: struct_fields_list.append(pl.col("prior_f1").struct.field("method"))
    if has_weight_class: struct_fields_list.append(pl.col("prior_f1").struct.field("weight_class"))
    if has_end_time: struct_fields_list.append(pl.col("prior_f1").struct.field("end_time"))
    if has_fight_format: struct_fields_list.append(pl.col("prior_f1").struct.field("fight_format"))
    struct_fields_list.extend([pl.col("opp_height_in"), pl.col("opp_weight_lbs"), pl.col("opp_reach_in"), 
                               pl.col("opp_stance"), pl.col("opp_dob")])
    
    prior_f1_enriched = (
        prior_f1_base.with_columns(pl.struct(struct_fields_list).alias("prior_f1_enriched"))
        .group_by("_row_idx").agg(pl.col("prior_f1_enriched").alias("prior_f1"))
    )
    
    prior_f2_base = (
        enriched_df.select("_row_idx", "prior_f2").explode("prior_f2")
        .with_columns(pl.col("prior_f2").struct.field("opponent_id").alias("opponent_id"))
        .join(fighters_df.rename({"fighter_id": "opponent_id", "height_in": "opp_height_in", "weight_lbs": "opp_weight_lbs",
                                  "reach_in": "opp_reach_in", "stance": "opp_stance", "dob": "opp_dob"}),
              on="opponent_id", how="left")
    )
    
    struct_fields_list_f2 = [
        pl.col("prior_f2").struct.field("fight_id"), pl.col("prior_f2").struct.field("fight_date"),
        pl.col("opponent_id"), pl.col("prior_f2").struct.field("result"),
    ]
    if has_method: struct_fields_list_f2.append(pl.col("prior_f2").struct.field("method"))
    if has_weight_class: struct_fields_list_f2.append(pl.col("prior_f2").struct.field("weight_class"))
    if has_end_time: struct_fields_list_f2.append(pl.col("prior_f2").struct.field("end_time"))
    if has_fight_format: struct_fields_list_f2.append(pl.col("prior_f2").struct.field("fight_format"))
    struct_fields_list_f2.extend([pl.col("opp_height_in"), pl.col("opp_weight_lbs"), pl.col("opp_reach_in"),
                                  pl.col("opp_stance"), pl.col("opp_dob")])
    
    prior_f2_enriched = (
        prior_f2_base.with_columns(pl.struct(struct_fields_list_f2).alias("prior_f2_enriched"))
        .group_by("_row_idx").agg(pl.col("prior_f2_enriched").alias("prior_f2"))
    )
    
    enriched_df = (
        enriched_df.drop(["prior_f1", "prior_f2"])
        .join(prior_f1_enriched, on="_row_idx", how="left")
        .join(prior_f2_enriched, on="_row_idx", how="left")
        .drop("_row_idx")
    )
    return enriched_df


def enrich_with_fight_totals(enriched_df: pl.DataFrame, conn) -> pl.DataFrame:
    """Enrich with fight totals."""
    query = """
    SELECT fight_id, fighter_id, body_attempts, body_landed, clinch_attempts, clinch_landed, ctrl_time_s,
           distance_attempts, distance_landed, ground_attempts, ground_landed, head_attempts, head_landed, kd,
           leg_attempts, leg_landed, rev, sig_str_attempts, sig_str_landed, sub_att, td_attempts, td_landed,
           total_str_attempts, total_str_landed
    FROM fight_totals;
    """
    fight_totals_df = pl.DataFrame(fetch_query(conn, query, params=None))
    
    struct_fields = enriched_df.select(pl.col("prior_f1")).schema["prior_f1"].inner.fields
    field_names = [field.name for field in struct_fields]
    has_method = "method" in field_names
    has_weight_class = "weight_class" in field_names
    has_end_time = "end_time" in field_names
    
    enriched_df = enriched_df.with_row_index("_row_idx")
    
    # Build prior_f1 with totals
    prior_f1_base = (
        enriched_df.select("_row_idx", "fighter1_id", "prior_f1").explode("prior_f1")
        .with_columns([
            pl.col("prior_f1").struct.field("fight_id").alias("fight_id"),
            pl.col("prior_f1").struct.field("opponent_id").alias("opponent_id")
        ])
    )
    
    prior_f1_with_stats = (
        prior_f1_base
        .join(fight_totals_df.rename({"fighter_id": "fighter1_id"}), on=["fight_id", "fighter1_id"], how="left")
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
        enriched_df.select("_row_idx", "fighter2_id", "prior_f2").explode("prior_f2")
        .with_columns([
            pl.col("prior_f2").struct.field("fight_id").alias("fight_id"),
            pl.col("prior_f2").struct.field("opponent_id").alias("opponent_id")
        ])
    )
    
    prior_f2_with_stats = (
        prior_f2_base
        .join(fight_totals_df.rename({"fighter_id": "fighter2_id"}), on=["fight_id", "fighter2_id"], how="left")
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
        enriched_df.drop(["prior_f1", "prior_f2"])
        .join(prior_f1_enriched, on="_row_idx", how="left")
        .join(prior_f2_enriched, on="_row_idx", how="left")
        .drop("_row_idx")
    )
    return enriched_df


def combine_predictions(ml_predictions_df, homemade_predictions_df):
    """
    Combine ML model predictions with homemade model predictions.
    
    Parameters:
    -----------
    ml_predictions_df : pl.DataFrame
        Output from quick_predict() - has 0/1 predictions
    homemade_predictions_df : pl.DataFrame
        Output from homemade model - has 1/2 predictions
    
    Returns:
    --------
    pl.DataFrame
        Combined predictions ready for MySQL predictions table
    """
    
    print("\n" + "="*80)
    print("COMBINING MODEL PREDICTIONS")
    print("="*80)
    
    # Convert homemade predictions from 1/2 to 0/1 format
    # combined_predicted_winner: 1 = Fighter 1 wins, 2 = Fighter 2 wins
    # We need: 1 = Fighter 1 wins, 0 = Fighter 2 wins
    homemade_df = homemade_predictions_df.with_columns([
        (pl.col('combined_predicted_winner') == 1).cast(pl.Int32).alias('homemade_pred'),
        pl.col('combined_f1_win_prob').alias('homemade_f1_prob')
    ]).select(['fight_id', 'homemade_pred', 'homemade_f1_prob'])
    
    # Join ML predictions with homemade predictions
    combined_df = ml_predictions_df.join(
        homemade_df,
        on='fight_id',
        how='left'
    )
    
    # Add required columns for predictions table
    combined_df = combined_df.with_columns([
        pl.lit(0).cast(pl.Int8).alias('legacy'),  # tinyint(1) - 0 for non-legacy
        pl.lit(1.0).cast(pl.Float32).alias('fight_data_coverage'),  # float
        pl.lit(None).cast(pl.Int8).alias('actual_winner'),  # tinyint - NULL for future fights
        pl.lit(None).cast(pl.Int8).alias('correct')  # tinyint(1) - NULL since we don't know outcome yet
    ])
    
    # Add correctness columns for each model (all NULL since we don't know outcome)
    combined_df = combined_df.with_columns([
        pl.lit(None).cast(pl.Int8).alias('logistic_correct'),
        pl.lit(None).cast(pl.Int8).alias('xgboost_correct'),
        pl.lit(None).cast(pl.Int8).alias('gradient_correct'),
        pl.lit(None).cast(pl.Int8).alias('homemade_correct'),
        pl.lit(None).cast(pl.Int8).alias('ensemble_majorityvote_correct'),
        pl.lit(None).cast(pl.Int8).alias('ensemble_weightedvote_correct'),
        pl.lit(None).cast(pl.Int8).alias('ensemble_avgprob_correct'),
        pl.lit(None).cast(pl.Int8).alias('ensemble_weightedavgprob_correct'),
    ])
    
    # Rename columns to match table schema EXACTLY
    combined_df = combined_df.rename({
        'logistic_prob': 'logistic_f1_prob',
        'xgboost_prob': 'xgboost_f1_prob',
        'gradient_boost_prob': 'gradient_f1_prob',
        'gradient_boost_pred': 'gradient_pred',  # Table has 'gradient_pred' not 'gradient_boost_pred'
        'ensemble_majorityvote_prob': 'ensemble_majorityvote_f1_prob',
        'ensemble_weightedvote_prob': 'ensemble_weightedvote_f1_prob',
        'ensemble_avgprob_prob': 'ensemble_avgprob_f1_prob',
        'ensemble_weightedavgprob_prob': 'ensemble_weightedavgprob_f1_prob'
    })
    
    # Select columns in exact order matching predictions table
    final_df = combined_df.select([
        'fight_id',
        'fight_date',
        'actual_winner',
        'logistic_f1_prob',
        'xgboost_f1_prob',
        'gradient_f1_prob',
        'homemade_f1_prob',
        'ensemble_majorityvote_f1_prob',
        'ensemble_weightedvote_f1_prob',
        'ensemble_avgprob_f1_prob',
        'ensemble_weightedavgprob_f1_prob',
        'logistic_pred',
        'xgboost_pred',
        'gradient_pred',  # Changed from gradient_boost_pred
        'homemade_pred',
        'ensemble_majorityvote_pred',
        'ensemble_weightedvote_pred',
        'ensemble_avgprob_pred',
        'ensemble_weightedavgprob_pred',
        'predicted_winner',
        'prediction_confidence',
        'logistic_correct',
        'xgboost_correct',
        'gradient_correct',
        'homemade_correct',
        'ensemble_majorityvote_correct',
        'ensemble_weightedvote_correct',
        'ensemble_avgprob_correct',
        'ensemble_weightedavgprob_correct',
        'correct',
        'legacy',
        'fight_data_coverage'
    ])
    
    print(f"✅ Combined {len(final_df)} predictions")
    print(f"   Columns: {len(final_df.columns)}")
    
    return final_df


def push_predictions_to_mysql(predictions_df, connection):
    """
    Push combined predictions to MySQL predictions table.
    
    Parameters:
    -----------
    predictions_df : pl.DataFrame
        Combined predictions from combine_predictions()
    connection : mysql.connector.connection.MySQLConnection
        Active MySQL connection
    
    Returns:
    --------
    int : Number of rows inserted
    """
    
    print("\n" + "="*80)
    print("PUSHING PREDICTIONS TO MySQL")
    print("="*80)
    
    import pandas as pd
    
    # Convert to pandas
    df_pandas = predictions_df.to_pandas()
    
    # Convert boolean to int for MySQL
    df_pandas['legacy'] = df_pandas['legacy'].astype(int)
    
    print(f"\nPreparing to insert {len(df_pandas)} rows...")
    
    # Create INSERT statement with all columns
    columns = df_pandas.columns.tolist()
    placeholders = ', '.join(['%s'] * len(columns))
    column_names = ', '.join(columns)
    
    # Build UPDATE clause for ON DUPLICATE KEY
    # Don't update actual_winner or correctness columns - those should only be set by update_predictions_with_results()
    exclude_from_update = [
        'fight_id', 
        # 'actual_winner', 
        'correct', 
        'logistic_correct', 
        'xgboost_correct', 
        'gradient_correct', 
        'homemade_correct',
        'ensemble_majorityvote_correct', 
        'ensemble_weightedvote_correct', 
        'ensemble_avgprob_correct', 
        'ensemble_weightedavgprob_correct'
    ]
    
    update_clause = ', '.join([f"{col} = VALUES({col})" for col in columns if col not in exclude_from_update])
    
    insert_query = f"""
        INSERT INTO ufc.predictions ({column_names})
        VALUES ({placeholders})
        ON DUPLICATE KEY UPDATE {update_clause}
    """
    
    cursor = connection.cursor()
    
    try:
        # Insert in batches
        batch_size = 1000
        total_rows = len(df_pandas)
        rows_inserted = 0
        
        for start_idx in range(0, total_rows, batch_size):
            end_idx = min(start_idx + batch_size, total_rows)
            batch = df_pandas.iloc[start_idx:end_idx]
            
            # Convert to list of tuples
            data = [tuple(row) for row in batch.values]
            
            # Execute batch
            cursor.executemany(insert_query, data)
            connection.commit()
            
            rows_inserted += len(batch)
            print(f"   Inserted {rows_inserted}/{total_rows} rows ({rows_inserted/total_rows*100:.1f}%)")
        
        print(f"\n✅ Successfully inserted {rows_inserted} predictions!")
        
        # Verify insertion
        cursor.execute("SELECT COUNT(*) FROM ufc.predictions WHERE legacy = 0")
        count = cursor.fetchone()[0]
        print(f"   Total non-legacy predictions in database: {count}")
        
        return rows_inserted
        
    except Exception as e:
        connection.rollback()
        print(f"\n❌ Error inserting data: {e}")
        raise
        
    finally:
        cursor.close()


def process_and_push_predictions(ml_predictions_df, homemade_predictions_df, connection):
    """
    Convenience function: Combine predictions and push to MySQL in one call.
    
    Parameters:
    -----------
    ml_predictions_df : pl.DataFrame
        Output from quick_predict()
    homemade_predictions_df : pl.DataFrame
        Output from homemade model
    connection : mysql.connector.connection.MySQLConnection
        Active MySQL connection
    
    Returns:
    --------
    pl.DataFrame : Combined predictions that were pushed
    
    Usage:
        predictions = process_and_push_predictions(ml_preds, homemade_preds, conn)
    """
    
    # Combine predictions
    combined_df = combine_predictions(ml_predictions_df, homemade_predictions_df)
    
    # Push to MySQL
    push_predictions_to_mysql(combined_df, connection)
    
    print("\n" + "="*80)
    print("✅ COMPLETE: Predictions combined and pushed to MySQL")
    print("="*80)
    
    return combined_df

def update_predictions_with_results(connection):
    """
    Update predictions table with actual results from fights table.
    
    For any prediction where actual_winner is NULL:
    - Check the fights table for that fight_id
    - If winner_id is NULL or 'drawornc', skip
    - Otherwise, determine if Fighter 1 or Fighter 2 won
    - Update actual_winner and all correctness flags
    
    Parameters:
    -----------
    connection : mysql.connector.connection.MySQLConnection
        Active MySQL connection
    
    Returns:
    --------
    int : Number of predictions updated
    """
    
    print("\n" + "="*80)
    print("UPDATING PREDICTIONS WITH FIGHT RESULTS")
    print("="*80)
    
    cursor = connection.cursor(dictionary=True)
    
    # Get all predictions where actual_winner is NULL
    query = """
    SELECT p.fight_id
    FROM ufc.predictions p
    WHERE p.actual_winner IS NULL;
    """
    
    cursor.execute(query)
    pending_predictions = cursor.fetchall()
    
    if not pending_predictions:
        print("\n✅ No predictions need updating - all have results!")
        cursor.close()
        return 0
    
    print(f"\nFound {len(pending_predictions)} predictions without results")
    print("Checking fights table for outcomes...\n")
    
    fight_ids = [pred['fight_id'] for pred in pending_predictions]
    
    # Build placeholders for the IN clause
    placeholders = ','.join(['%s'] * len(fight_ids))
    
    # Get fight results from fights table
    fights_query = f"""
    SELECT 
        fight_id,
        fighter1_id,
        fighter2_id,
        winner_id
    FROM ufc.fights
    WHERE fight_id IN ({placeholders});
    """
    
    cursor.execute(fights_query, fight_ids)
    fights = cursor.fetchall()
    
    # Process each fight
    updates_made = 0
    skipped = 0
    
    for fight in fights:
        fight_id = fight['fight_id']
        winner_id = fight['winner_id']
        fighter1_id = fight['fighter1_id']
        fighter2_id = fight['fighter2_id']
        
        # Skip if no winner or draw/nc
        if winner_id is None or winner_id == 'drawornc':
            skipped += 1
            continue
        
        # Determine actual winner (1 = Fighter 1 won, 0 = Fighter 2 won)
        if winner_id == fighter1_id:
            actual_winner = 1
        elif winner_id == fighter2_id:
            actual_winner = 0
        else:
            print(f"⚠️  Warning: winner_id '{winner_id}' doesn't match fighter1_id or fighter2_id for fight {fight_id}")
            skipped += 1
            continue
        
        # Update the prediction with actual_winner and calculate correctness
        update_query = """
        UPDATE ufc.predictions
        SET 
            actual_winner = %s,
            logistic_correct = (logistic_pred = %s),
            xgboost_correct = (xgboost_pred = %s),
            gradient_correct = (gradient_pred = %s),
            homemade_correct = (homemade_pred = %s),
            ensemble_majorityvote_correct = (ensemble_majorityvote_pred = %s),
            ensemble_weightedvote_correct = (ensemble_weightedvote_pred = %s),
            ensemble_avgprob_correct = (ensemble_avgprob_pred = %s),
            ensemble_weightedavgprob_correct = (ensemble_weightedavgprob_pred = %s),
            correct = (predicted_winner = %s)
        WHERE fight_id = %s;
        """
        
        cursor.execute(update_query, (
            actual_winner,
            actual_winner,
            actual_winner,
            actual_winner,
            actual_winner,
            actual_winner,
            actual_winner,
            actual_winner,
            actual_winner,
            actual_winner,
            fight_id
        ))
        
        updates_made += 1
        print(f"   Updated fight {fight_id}: Fighter {'1' if actual_winner == 1 else '2'} won")
    
    # Commit all updates
    connection.commit()
    cursor.close()
    
    print("\n" + "="*80)
    print("UPDATE SUMMARY")
    print("="*80)
    print(f"✅ Updated: {updates_made} predictions")
    print(f"⏭️  Skipped: {skipped} fights (no result or draw/nc)")
    print(f"📊 Total processed: {len(fights)} fights")
    print("="*80)
    
    return updates_made

import mysql.connector
import polars as pl

def get_non_legacy_accuracies(connection):
    """
    Calculate and display accuracies for all models on non-legacy predictions.
    
    Parameters:
    -----------
    connection : mysql.connector.connection.MySQLConnection
        Active MySQL connection
    
    Returns:
    --------
    pl.DataFrame
        DataFrame with accuracy metrics for each model
    """
    
    print("\n" + "="*80)
    print("NON-LEGACY MODEL ACCURACIES")
    print("="*80)
    
    cursor = connection.cursor()
    
    # Get all non-legacy predictions where we know the outcome
    query = """
    SELECT 
        COUNT(*) as total_fights,
        SUM(CASE WHEN actual_winner IS NOT NULL THEN 1 ELSE 0 END) as fights_with_results,
        
        -- Individual Models
        SUM(logistic_correct) as logistic_correct,
        SUM(xgboost_correct) as xgboost_correct,
        SUM(gradient_correct) as gradient_correct,
        SUM(homemade_correct) as homemade_correct,
        
        -- Ensemble Models
        SUM(ensemble_majorityvote_correct) as ensemble_majorityvote_correct,
        SUM(ensemble_weightedvote_correct) as ensemble_weightedvote_correct,
        SUM(ensemble_avgprob_correct) as ensemble_avgprob_correct,
        SUM(ensemble_weightedavgprob_correct) as ensemble_weightedavgprob_correct,
        
        -- Overall
        SUM(correct) as overall_correct
    FROM ufc.predictions
    WHERE legacy = 0 AND actual_winner IS NOT NULL;
    """
    
    cursor.execute(query)
    result = cursor.fetchone()
    cursor.close()
    
    if result is None or result[1] == 0:
        print("\n⚠️  No non-legacy predictions with results found!")
        print("   (All non-legacy predictions are for future fights)")
        return None
    
    total_fights = result[0]
    fights_with_results = result[1]
    
    print(f"\nTotal non-legacy predictions: {total_fights}")
    print(f"Predictions with known results: {fights_with_results}")
    
    if fights_with_results == 0:
        print("\n⚠️  No results available yet for non-legacy predictions")
        return None
    
    # Calculate accuracies
    accuracies = {
        'Model': [
            'Logistic Regression',
            'XGBoost',
            'Gradient Boost',
            'Homemade',
            'Ensemble: Majority Vote',
            'Ensemble: Weighted Vote',
            'Ensemble: Average Probability',
            'Ensemble: Weighted Avg Prob',
            'Overall (Best Ensemble)'
        ],
        'Correct': [
            result[2],  # logistic_correct
            result[3],  # xgboost_correct
            result[4],  # gradient_correct
            result[5],  # homemade_correct
            result[6],  # ensemble_majorityvote_correct
            result[7],  # ensemble_weightedvote_correct
            result[8],  # ensemble_avgprob_correct
            result[9],  # ensemble_weightedavgprob_correct
            result[10]  # overall_correct
        ],
        'Total': [fights_with_results] * 9,
    }
    
    # Calculate accuracy percentages
    accuracies['Accuracy'] = [
        f"{(correct/fights_with_results)*100:.2f}%" if correct is not None else "N/A"
        for correct in accuracies['Correct']
    ]
    
    df = pl.DataFrame(accuracies)
    
    print("\n" + "="*80)
    print("ACCURACY BREAKDOWN")
    print("="*80)
    print()
    print(df)
    
    # Find best model
    best_idx = max(range(len(accuracies['Correct'])), 
                   key=lambda i: accuracies['Correct'][i] if accuracies['Correct'][i] is not None else 0)
    
    print("\n" + "="*80)
    print(f"🏆 Best Model: {accuracies['Model'][best_idx]}")
    print(f"   Accuracy: {accuracies['Accuracy'][best_idx]}")
    print("="*80)
    
    return df

def get_non_legacy_accuracies(connection):
    """
    Calculate and display accuracies for all models on non-legacy predictions.
    
    Parameters:
    -----------
    connection : mysql.connector.connection.MySQLConnection
        Active MySQL connection
    
    Returns:
    --------
    pl.DataFrame
        DataFrame with accuracy metrics for each model
    """
    
    print("\n" + "="*80)
    print("NON-LEGACY MODEL ACCURACIES")
    print("="*80)
    
    cursor = connection.cursor()
    
    # Get all non-legacy predictions where we know the outcome
    query = """
    SELECT 
        COUNT(*) as total_fights,
        SUM(CASE WHEN actual_winner IS NOT NULL THEN 1 ELSE 0 END) as fights_with_results,
        
        -- Individual Models
        SUM(logistic_correct) as logistic_correct,
        SUM(xgboost_correct) as xgboost_correct,
        SUM(gradient_correct) as gradient_correct,
        SUM(homemade_correct) as homemade_correct,
        
        -- Ensemble Models
        SUM(ensemble_majorityvote_correct) as ensemble_majorityvote_correct,
        SUM(ensemble_weightedvote_correct) as ensemble_weightedvote_correct,
        SUM(ensemble_avgprob_correct) as ensemble_avgprob_correct,
        SUM(ensemble_weightedavgprob_correct) as ensemble_weightedavgprob_correct,
        
        -- Overall
        SUM(correct) as overall_correct
    FROM ufc.predictions
    WHERE legacy = 0 AND actual_winner IS NOT NULL;
    """
    
    cursor.execute(query)
    result = cursor.fetchone()
    cursor.close()
    
    if result is None or result[1] == 0:
        print("\n⚠️  No non-legacy predictions with results found!")
        print("   (All non-legacy predictions are for future fights)")
        return None
    
    total_fights = result[0]
    fights_with_results = result[1]
    
    print(f"\nTotal non-legacy predictions: {total_fights}")
    print(f"Predictions with known results: {fights_with_results}")
    
    if fights_with_results == 0:
        print("\n⚠️  No results available yet for non-legacy predictions")
        return None
    
    # Calculate accuracies
    accuracies = {
        'Model': [
            'Logistic Regression',
            'XGBoost',
            'Gradient Boost',
            'Homemade',
            'Ensemble: Majority Vote',
            'Ensemble: Weighted Vote',
            'Ensemble: Average Probability',
            'Ensemble: Weighted Avg Prob',
            'Overall (Best Ensemble)'
        ],
        'Correct': [
            result[2],  # logistic_correct
            result[3],  # xgboost_correct
            result[4],  # gradient_correct
            result[5],  # homemade_correct
            result[6],  # ensemble_majorityvote_correct
            result[7],  # ensemble_weightedvote_correct
            result[8],  # ensemble_avgprob_correct
            result[9],  # ensemble_weightedavgprob_correct
            result[10]  # overall_correct
        ],
        'Total': [fights_with_results] * 9,
    }
    
    # Calculate accuracy percentages
    accuracies['Accuracy'] = [
        f"{(correct/fights_with_results)*100:.2f}%" if correct is not None else "N/A"
        for correct in accuracies['Correct']
    ]
    
    df = pl.DataFrame(accuracies)
    
    print("\n" + "="*80)
    print("ACCURACY BREAKDOWN")
    print("="*80)
    print()
    print(df)
    
    # Find best model
    best_idx = max(range(len(accuracies['Correct'])), 
                   key=lambda i: accuracies['Correct'][i] if accuracies['Correct'][i] is not None else 0)
    
    print("\n" + "="*80)
    print(f"🏆 Best Model: {accuracies['Model'][best_idx]}")
    print(f"   Accuracy: {accuracies['Accuracy'][best_idx]}")
    print("="*80)
    
    return df

def run(event_id: str, min_prior_fights: int = 3):
    """
    Build fight snapshots for a specific event.
    
    Parameters:
    -----------
    event_id : str
        The event ID to process
    min_prior_fights : int
        Minimum prior fights required (default: 3)
    
    Returns:
    --------
    pl.DataFrame
        Enriched fight data with prior fight history
    """
    conn = create_connection()
    
    print(f"Processing event: {event_id}")
    print(f"Minimum prior fights required: {min_prior_fights}")
    
    event_fights_df = get_fights_for_event(conn, event_id)
    print(f"Found {len(event_fights_df)} fights in event")
    
    if event_fights_df.is_empty():
        print("No fights found for this event")
        return pl.DataFrame()
    
    allRelatedFights = get_all_related_fights(conn, event_fights_df)
    print(f"Found {len(allRelatedFights)} total fights involving these fighters")
    
    snapshotsDf = build_pre_fight_snapshots(event_fights_df, allRelatedFights, min_prior_fights)
    print(f"After filtering for {min_prior_fights}+ prior fights: {len(snapshotsDf)} fights remain")
    
    if snapshotsDf.is_empty():
        print("No fights remain after filtering")
        return pl.DataFrame()
    
    enrichedDfFightersData = enrich_with_fighter_stats(snapshotsDf, conn)
    print("Enriched with fighter stats")
    
    enrichedDfFightTotals = enrich_with_fight_totals(enrichedDfFightersData, conn)
    print("Enriched with fight totals")
    
    print(f"\nFinal rows: {len(enrichedDfFightTotals)}")

    fight_features_df = process_snapshots_to_features(enrichedDfFightTotals)
    modelDiffsDf_nan = process_snapshots_to_flat_features(enrichedDfFightTotals)
    modelDiffsDf = modelDiffsDf_nan.drop('target')
    
    homemadeModel = CombinedPredictor()
    
    homemadeModelPredictions = homemadeModel.predict_dataframe(fight_features_df)
    modelsPredictions = quick_predict(modelDiffsDf)
    
    process_and_push_predictions(modelsPredictions, homemadeModelPredictions, conn)
    update_predictions_with_results(conn)
    get_non_legacy_accuracies(conn)