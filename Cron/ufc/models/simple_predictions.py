from .utils import create_connection, fetch_query
from .get_accuracies import get_model_accuracies_batched, get_all_model_accuracies
import re, math
from typing import Optional, List, Dict, Tuple
import polars as pl

# ------------------------
# Model config / helpers
# ------------------------

MODELS = [
    "logistic",
    "xgboost",
    "gradient",
    "homemade",
    # "ensemble_majorityvote",  # explicitly excluded
    "ensemble_weightedvote",
    "ensemble_avgprob",
    "ensemble_weightedavgprob",
]

def _colnames_for(model: str) -> tuple[str, str]:
    """Return (pred_col, f1_prob_col) for a model key."""
    return f"{model}_pred", f"{model}_f1_prob"

def _choose_best_model(acc: dict) -> str:
    """Pick argmax accuracy, excluding ensemble_majorityvote if present."""
    filtered = {k: v for k, v in acc.items() if k != "ensemble_majorityvote" and v is not None}
    return max(filtered, key=filtered.get) if filtered else "logistic"

# ---- Calibration helpers ----

def _weighted_true_accuracy_with_n(model: str, p_f1: float, window: float = 0.01):
    # Call once for both p and (1-p)
    results = get_model_accuracies_batched(model, [p_f1, 1.0 - p_f1], window)
    
    a = results[p_f1]
    b = results[1.0 - p_f1]
    
    n1 = a["prob_range"]
    c1 = a["correct"]
    n2 = b["prob_range"]
    c2 = b["correct"]
    
    N = n1 + n2
    if N == 0:
        return (None, 0)
    p_hat = (c1 + c2) / N
    return (round(p_hat, 4), N)

# ------------------------
# Name normalization (Polars-friendly)
# ------------------------

def _cap_segment(seg: str) -> str:
    if not seg:
        return seg
    return seg[0].upper() + seg[1:].lower()

def _cap_apostrophes(token: str) -> str:
    parts = token.split("'")
    parts = [_cap_segment(p) for p in parts]
    return "'".join(parts)

def _cap_hyphens(token: str) -> str:
    parts = token.split("-")
    parts = [_cap_apostrophes(p) for p in parts]  # handles O'Neil style inside
    return "-".join(parts)

_INITIAL_RE = re.compile(r"^[A-Za-z]\.$")
def _cap_initials(token: str) -> str:
    # a. -> A. ; b. -> B.
    if _INITIAL_RE.fullmatch(token):
        return token[0].upper() + "."
    return token

def _smart_title_name_py(name: Optional[str]) -> Optional[str]:
    if name is None:
        return None
    s = re.sub(r"\s+", " ", name.strip())
    if not s:
        return s

    out_tokens = []
    for tok in s.split(" "):
        t = _cap_initials(tok)
        if t == tok:  # not an initial
            t = _cap_hyphens(t)
            if "-" not in t and "'" not in t:
                t = _cap_segment(t)
        out_tokens.append(t)
    return " ".join(out_tokens)

def _normalize_name_expr(col: str) -> pl.Expr:
    return (
        pl.col(col)
          .cast(pl.Utf8, strict=False)
          .str.replace_all(r"\s+", " ")
          .str.strip_chars()
          .map_elements(_smart_title_name_py, return_dtype=pl.Utf8)
    )

def _normalize_names(df: pl.DataFrame) -> pl.DataFrame:
    name_cols = [c for c in df.columns if c in (
        "fighter1_name", "fighter2_name",
        "fighter1_nickname", "fighter2_nickname"
    )]
    if not name_cols:
        return df
    return df.with_columns([_normalize_name_expr(c).alias(c) for c in name_cols])

# ------------------------
# Main builder (Polars)
# ------------------------

def push_algopicks_to_sql(df: pl.DataFrame):
    """
    Push algopicks DataFrame to the predictions_simplified table.
    
    Args:
        df: Polars DataFrame from build_algopicks_rows()
        truncate: If True, truncate table before inserting
    """
    from .utils import create_connection
    
    conn = create_connection()
    cursor = conn.cursor()
    
    try:
        
        # Prepare insert query
        insert_query = """
        INSERT INTO ufc.prediction_simplified (
            fight_id, event_id, fighter1_id, fighter2_id,
            fighter1_name, fighter2_name,
            fighter1_nickname, fighter2_nickname,
            fighter1_img_link, fighter2_img_link,
            algopick_model, algopick_prediction,
            algopick_probability, correct,
            date, end_time, weight_class, fight_type, win_method, window_sample
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
        )
        ON DUPLICATE KEY UPDATE
            event_id = VALUES(event_id),
            fighter1_id = VALUES(fighter1_id),
            fighter2_id = VALUES(fighter2_id),
            fighter1_name = VALUES(fighter1_name),
            fighter2_name = VALUES(fighter2_name),
            fighter1_nickname = VALUES(fighter1_nickname),
            fighter2_nickname = VALUES(fighter2_nickname),
            fighter1_img_link = VALUES(fighter1_img_link),
            fighter2_img_link = VALUES(fighter2_img_link),
            algopick_model = VALUES(algopick_model),
            algopick_prediction = VALUES(algopick_prediction),
            algopick_probability = VALUES(algopick_probability),
            correct = VALUES(correct),
            date = VALUES(date),
            end_time = VALUES(end_time),
            weight_class = VALUES(weight_class),
            fight_type = VALUES(fight_type),
            win_method = VALUES(win_method),
            window_sample = VALUES(window_sample);
        """
        
        # Convert DataFrame to list of tuples
        rows = df.to_dicts()
        
        insert_count = 0
        for row in rows:
            prob = row.get("final_calibrated_confidence")
            if prob is not None:
                prob = round(prob * 100, 2)
            
            values = (
                str(row.get("fight_id")),
                row.get("event_id"),
                row.get("fighter1_id"),
                row.get("fighter2_id"),
                row.get("fighter1_name"),
                row.get("fighter2_name"),
                row.get("fighter1_nickname"),
                row.get("fighter2_nickname"),
                row.get("fighter1_img_link"),
                row.get("fighter2_img_link"),
                row.get("algopick_model"),
                row.get("algopick_prediction"),
                prob,
                row.get("correct"),
                row.get("date"),
                row.get("end_time"),
                row.get("weight_class"),
                row.get("fight_type"),  # Added fight_type
                row.get("win_method"),
                row.get("algopick_calib_n"),
            )
            
            cursor.execute(insert_query, values)
            insert_count += 1
        
        conn.commit()
        print(f"Successfully inserted {insert_count} rows into predictions_simplified")
        
    except Exception as e:
        conn.rollback()
        print(f"Error inserting data: {e}")
        raise
    finally:
        cursor.close()
        conn.close()

# Convert end_time to round format (e.g., "15:00" -> "Round 3", "4:58" -> "Round 1 at 4:58")
def _format_end_time(time_str):
    if time_str is None:
        return None
    
    # Parse time string (format: "M:SS" or "MM:SS")
    parts = time_str.split(":")
    if len(parts) != 2:
        return time_str
    
    try:
        minutes = int(parts[0])
        seconds = int(parts[1])
    except ValueError:
        return time_str
    
    # UFC rounds are 5 minutes each
    # If time is exactly 5:00, 10:00, 15:00, etc., it's end of a round
    total_seconds = minutes * 60 + seconds
    
    if total_seconds % 300 == 0:  # Exactly at round end (5:00, 10:00, 15:00, etc.)
        round_num = total_seconds // 300
        return f"Round {round_num}"
    else:
        # Determine which round based on total time
        round_num = (total_seconds // 300) + 1
        time_in_round = total_seconds % 300
        mins_in_round = time_in_round // 60
        secs_in_round = time_in_round % 60
        return f"Round {round_num} at {mins_in_round}:{secs_in_round:02d}"

# Usage:
def build_algopicks_rows(
    event_id: str,
    include_legacy_for_model_choice: bool = False,
    prob_window: float = 0.0065,
) -> pl.DataFrame:
    """
    Returns Polars DataFrame for your algopicks table for a specific event.

    - Chooses best model by overall accuracy (excluding ensemble_majorityvote)
    - Pulls fights with specified event_id
    - Normalizes names (smart title case)
    - algopick_prediction: 0 if model predicts F1, 1 if model predicts F2 (NO FLIPPING)
    - confidence_ge: raw model confidence (distance from 50%)
    - final_calibrated_confidence: calibrated probability, floored at 50.5% if below 50%
    - algopick_calib_n: N used to compute calibrated probability
    """
    # 1) Pick best model from full table accuracies (toggle legacy)
    acc = get_all_model_accuracies(include_legacy=include_legacy_for_model_choice)
    best_model = _choose_best_model(acc)
    pred_col, prob_col = _colnames_for(best_model)

    # 2) Pull necessary rows/columns in one query
    conn = create_connection()
    query = f"""
    SELECT 
        p.fight_id,
        f.event_id,
        f.fighter1_id,
        f.fighter2_id,
        f.fighter1_name,
        f.fighter2_name,
        f.fight_date   AS date,
        f.end_time,
        f.weight_class,
        f.fight_type,
        f.method       AS win_method,
        p.actual_winner,
        p.{pred_col}   AS model_pred,        -- 1 = predict F1 wins, 0 = predict F2 wins
        p.{prob_col}   AS model_f1_prob,     -- probability F1 wins from chosen model

        f1.img_link    AS fighter1_img_link,
        f1.nickname    AS fighter1_nickname,
        f2.img_link    AS fighter2_img_link,
        f2.nickname    AS fighter2_nickname
    FROM ufc.predictions p
    JOIN ufc.fights f
      ON f.fight_id = p.fight_id
    LEFT JOIN ufc.fighters f1
      ON f1.fighter_id = f.fighter1_id
    LEFT JOIN ufc.fighters f2
      ON f2.fighter_id = f.fighter2_id
    WHERE f.event_id = %s;
    """
    rows = fetch_query(conn, query, [event_id])
    if not rows:
        return pl.DataFrame()

    df = pl.DataFrame(rows)

    # 3) Polars transformations
    df = _normalize_names(df)

    # chosen model name
    df = df.with_columns(pl.lit(best_model).alias("algopick_model"))

    # algopick_prediction: flip encoding (model_pred 1->0, 0->1) - NO FLIPPING BASED ON CALIBRATION
    df = df.with_columns(
        pl.when(pl.col("model_pred").is_null())
          .then(None)
          .otherwise(pl.when(pl.col("model_pred") == 1).then(0).otherwise(1))
          .cast(pl.Int8)
          .alias("algopick_prediction")
    )

    # confidence_ge: distance from 50% (raw model confidence)
    df = df.with_columns(
        pl.when(pl.col("model_f1_prob").is_null())
          .then(None)
          .otherwise(
              (pl.col("model_f1_prob") - 0.5).abs()
          )
          .cast(pl.Float64)
          .alias("confidence_ge")
    )

    # ---- Calibration: algopick_probability & algopick_calib_n ----
    def _prob_and_n(p):
        if p is None:
            return {"prob": None, "n": 0}
        prob, n = _weighted_true_accuracy_with_n(best_model, float(p), window=prob_window)
        return {"prob": prob, "n": n}

    df = df.with_columns(
        pl.col("model_f1_prob")
        .map_elements(_prob_and_n,
                        return_dtype=pl.Struct({"prob": pl.Float64, "n": pl.Int64}))
        .alias("calib")
    ).with_columns(
        pl.col("calib").struct.field("prob").alias("algopick_probability"),
        pl.col("calib").struct.field("n").alias("algopick_calib_n")
    ).drop("calib")

    # final_calibrated_confidence: floor at 50.5% if calibrated probability < 50%
    df = df.with_columns(
        pl.when(pl.col("algopick_probability").is_null())
          .then(None)
          .when(pl.col("algopick_probability") < 0.5)
          .then(0.505)
          .otherwise(pl.col("algopick_probability"))
          .cast(pl.Float64)
          .alias("final_calibrated_confidence")
    )

    # correctness vs prediction (no flipping):
    # if actual_winner==1 => should_be=0 ; if ==0 => should_be=1
    df = df.with_columns(
        pl.when(pl.col("actual_winner").is_null())
          .then(None)
          .otherwise(
              (pl.col("algopick_prediction") ==
               pl.when(pl.col("actual_winner") == 1).then(0).otherwise(1))
              .cast(pl.Int8)
          )
          .alias("correct")
    )

    # Normalize win_method values
    df = df.with_columns(
        pl.when(pl.col("win_method") == "d_unan")
          .then(pl.lit("Unanimous Decision"))
          .when(pl.col("win_method") == "sub")
          .then(pl.lit("Submission"))
          .when(pl.col("win_method") == "unknown")
          .then(pl.lit("Unknown"))
          .when(pl.col("win_method") == "kotko")
          .then(pl.lit("KO/TKO"))
          .when(pl.col("win_method") == "d_split")
          .then(pl.lit("Split Decision"))
          .when(pl.col("win_method") == "d_maj")
          .then(pl.lit("Majority Decision"))
          .otherwise(pl.col("win_method"))
          .alias("win_method")
    )
    
    df = df.with_columns(
        pl.col("end_time")
          .map_elements(_format_end_time, return_dtype=pl.Utf8)
          .alias("end_time")
    )

    # Final selection
    out_cols = [
        "fight_id", "event_id",
        "fighter1_id", "fighter2_id",
        "fighter1_name", "fighter2_name",
        "fighter1_img_link", "fighter2_img_link",
        "fighter1_nickname", "fighter2_nickname",
        "algopick_model",
        "algopick_prediction",
        "confidence_ge",
        "final_calibrated_confidence",
        "algopick_probability",
        "algopick_calib_n",
        "correct",
        "date", "end_time", "weight_class", "fight_type", "win_method",
    ]
    out_cols = [c for c in out_cols if c in df.columns]

    finaldf = df.select(out_cols)
    push_algopicks_to_sql(finaldf)
    
def update_prediction_simplified_with_results(connection, event_id: str):
    """
    Update prediction_simplified table with results for all fights in a given event.
    
    Updates:
    - correct: if algopick_prediction==0, check if ps.fighter1_id==winner_id
               if algopick_prediction==1, check if ps.fighter2_id==winner_id
               if winner_id is NULL or 'drawornc', set correct=NULL
    - win_method: normalized from fights table, or "No Contest" if draw/nc
    - end_time: formatted as "Round X" or "Round X at M:SS"
    
    Parameters:
    -----------
    connection : mysql.connector.connection.MySQLConnection
        Active MySQL connection
    event_id : str
        The event_id to update predictions for
    
    Returns:
    --------
    int : Number of predictions updated
    """
    
    print("\n" + "="*80)
    print(f"UPDATING PREDICTION_SIMPLIFIED FOR EVENT: {event_id}")
    print("="*80)
    
    cursor = connection.cursor(dictionary=True)
    
    # Get all fights from this event with their results
    query = """
    SELECT 
        f.fight_id,
        f.winner_id,
        f.method,
        f.end_time,
        ps.fighter1_id,
        ps.fighter2_id,
        ps.algopick_prediction
    FROM ufc.fights f
    JOIN ufc.prediction_simplified ps ON ps.fight_id = f.fight_id
    WHERE f.event_id = %s;
    """
    
    cursor.execute(query, (event_id,))
    fights = cursor.fetchall()
    
    if not fights:
        print(f"\n⚠️  No fights found for event_id: {event_id}")
        cursor.close()
        return 0
    
    print(f"\nFound {len(fights)} fights to update\n")
    
    updates_made = 0
    no_contests = 0
    
    for fight in fights:
        fight_id = fight['fight_id']
        winner_id = fight['winner_id']
        fighter1_id = fight['fighter1_id']  # From prediction_simplified
        fighter2_id = fight['fighter2_id']  # From prediction_simplified
        method = fight['method']
        end_time = fight['end_time']
        algopick_prediction = fight['algopick_prediction']
        
        # Handle draw/no contest - explicitly set correct to NULL
        if winner_id is None or winner_id == 'drawornc':
            formatted_end_time = _format_end_time(end_time)
            
            update_query = """
            UPDATE ufc.prediction_simplified
            SET 
                correct = NULL,
                win_method = 'No Contest',
                end_time = %s
            WHERE fight_id = %s;
            """
            
            cursor.execute(update_query, (formatted_end_time, fight_id))
            no_contests += 1
            print(f"   ⚪ Fight {fight_id}: No Contest/Draw (correct set to NULL)")
            continue
        
        # Calculate correctness using winner_id and fighter IDs from prediction_simplified
        # algopick_prediction: 0 = predict Fighter 1, 1 = predict Fighter 2
        if algopick_prediction is not None:
            if algopick_prediction == 0:
                # Predicted Fighter 1 to win - compare ps.fighter1_id to winner_id
                correct = 1 if winner_id == fighter1_id else 0
            else:  # algopick_prediction == 1
                # Predicted Fighter 2 to win - compare ps.fighter2_id to winner_id
                correct = 1 if winner_id == fighter2_id else 0
        else:
            correct = None
        
        # Normalize win_method
        method_map = {
            'd_unan': 'Unanimous Decision',
            'sub': 'Submission',
            'unknown': 'Unknown',
            'kotko': 'KO/TKO',
            'd_split': 'Split Decision',
            'd_maj': 'Majority Decision'
        }
        normalized_method = method_map.get(method, method)
        
        # Format end_time
        formatted_end_time = _format_end_time(end_time)
        
        # Update the prediction_simplified row
        update_query = """
        UPDATE ufc.prediction_simplified
        SET 
            correct = %s,
            win_method = %s,
            end_time = %s
        WHERE fight_id = %s;
        """
        
        cursor.execute(update_query, (
            correct,
            normalized_method,
            formatted_end_time,
            fight_id
        ))
        
        updates_made += 1
        winner_name = "Fighter 1" if winner_id == fighter1_id else "Fighter 2"
        print(f"   ✅ Updated fight {fight_id}: {winner_name} won, Prediction {'✓ correct' if correct else '✗ incorrect'}")
    
    # Commit all updates
    connection.commit()
    cursor.close()
    
    print("\n" + "="*80)
    print("UPDATE SUMMARY")
    print("="*80)
    print(f"✅ Updated: {updates_made} predictions")
    print(f"⚪ No Contests: {no_contests} fights")
    print(f"📊 Total processed: {len(fights)} fights")
    print("="*80)
    
    return updates_made