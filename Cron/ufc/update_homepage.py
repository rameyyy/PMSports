"""
Update UFC homepage stats table (ufc_homepage).
Runs weekly on Sundays after scrape.py, predict.py, and oddsapi.py.

Steps:
  1. Update prediction_simplified.correct for completed fights
  2. Update ufc_homepage.pow_correct for settled POW picks
  3. Compute model accuracy + vegas accuracy from prediction_simplified
  4. Find next event + compute EV for every fighter → select POW
  5. Compute pow_avg_odds from all historical POW rows
  6. Compute cumulative pow_win / pow_loss
  7. Insert new row into ufc_homepage

EV formula (per $100 stake):
  EV = (P_win * profit) - (P_lose * 100)
  profit: positive odds → odds value, negative odds → 10000 / abs(odds)
  Valid POW range: 2 <= EV <= 30. Anything > 30 = model likely off, skip.

Vegas accuracy:
  Vegas pick = f1 if f1_odds < f2_odds, f2 if f2_odds < f1_odds.
  Equal odds = auto incorrect (counted in total, not in correct).
"""
import sys
from pathlib import Path
from datetime import date

sys.path.insert(0, str(Path(__file__).parent / "models"))
from utils import create_connection, run_query, fetch_query


# ---------------------------------------------------------------------------
# Odds helpers
# ---------------------------------------------------------------------------
def american_profit(odds: int) -> float:
    """Profit per $100 stake for american odds."""
    return float(odds) if odds > 0 else 10000.0 / abs(odds)


def american_to_decimal(odds: int) -> float:
    return (odds / 100.0) + 1 if odds > 0 else (100.0 / abs(odds)) + 1


def decimal_to_american(dec: float) -> int:
    if dec >= 2.0:
        return int(round((dec - 1) * 100))
    return int(round(-100.0 / (dec - 1)))


def calc_ev(prob: float, odds: int) -> float:
    """EV per $100 stake."""
    return (prob * american_profit(odds)) - ((1 - prob) * 100)


# ---------------------------------------------------------------------------
# Step 1: update prediction_simplified.correct
# ---------------------------------------------------------------------------
def update_prediction_correct(conn):
    run_query(conn, """
        UPDATE prediction_simplified ps
        JOIN fights f ON ps.fight_id = f.fight_id
        SET ps.correct = CASE
            WHEN f.winner_id = ps.predicted_winner_id THEN 1
            ELSE 0
        END,
        ps.win_method = f.method,
        ps.end_time   = f.end_time
        WHERE ps.correct IS NULL
          AND f.winner_id IS NOT NULL
          AND f.winner_id NOT IN ('drawornc', '', 'draw')
    """)
    print("Updated prediction_simplified.correct")


# ---------------------------------------------------------------------------
# Step 2: update ufc_homepage.pow_correct for settled POW picks
# ---------------------------------------------------------------------------
def update_pow_correct(conn):
    run_query(conn, """
        UPDATE ufc_homepage uh
        JOIN prediction_simplified ps ON uh.pow_fight_id = ps.fight_id
        SET uh.pow_correct = ps.correct
        WHERE uh.pow_correct IS NULL
          AND uh.pow_no_pick = 0
          AND ps.correct IS NOT NULL
    """)
    print("Updated ufc_homepage.pow_correct")


# ---------------------------------------------------------------------------
# Step 3: model accuracy
# ---------------------------------------------------------------------------
def get_model_accuracy(conn) -> dict:
    rows = fetch_query(conn, """
        SELECT
            SUM(correct = 1) AS correct_count,
            SUM(correct = 0) AS wrong_count
        FROM prediction_simplified
        WHERE correct IS NOT NULL
    """)
    r = rows[0] if rows else {}
    correct = int(r.get("correct_count") or 0)
    wrong   = int(r.get("wrong_count")   or 0)
    return {"correct": correct, "total": correct + wrong}


# ---------------------------------------------------------------------------
# Step 4: vegas accuracy (using f1_odds/f2_odds in prediction_simplified)
# ---------------------------------------------------------------------------
def get_vegas_accuracy(conn) -> dict:
    rows = fetch_query(conn, """
        SELECT
            SUM(CASE
                WHEN ps.f1_odds = ps.f2_odds THEN 0
                WHEN ps.f1_odds < ps.f2_odds AND f.winner_id = ps.fighter1_id THEN 1
                WHEN ps.f2_odds < ps.f1_odds AND f.winner_id = ps.fighter2_id THEN 1
                ELSE 0
            END) AS vegas_correct,
            COUNT(*) AS vegas_total
        FROM prediction_simplified ps
        JOIN fights f ON ps.fight_id = f.fight_id
        WHERE ps.correct IS NOT NULL
          AND ps.f1_odds IS NOT NULL
          AND ps.f2_odds IS NOT NULL
    """)
    r = rows[0] if rows else {}
    return {
        "correct": int(r.get("vegas_correct") or 0),
        "total":   int(r.get("vegas_total")   or 0),
    }


# ---------------------------------------------------------------------------
# Step 5: select Pick of the Week via EV
# ---------------------------------------------------------------------------
def select_pow(conn) -> dict | None:
    """
    For every fight in the next upcoming event (prediction_simplified, correct IS NULL),
    compute EV for both fighters. Pick the fighter with EV in [2, 30].
    If multiple qualify, pick highest EV. If none, return None.
    """
    fights = fetch_query(conn, """
        SELECT ps.fight_id, ps.fighter1_id, ps.fighter2_id,
               ps.fighter1_name, ps.fighter2_name,
               ps.f1_probability, ps.f1_odds, ps.f2_odds
        FROM prediction_simplified ps
        WHERE ps.correct IS NULL
          AND ps.f1_odds IS NOT NULL
          AND ps.f2_odds IS NOT NULL
          AND ps.date >= CURDATE()
          AND ps.date = (
              SELECT MIN(date) FROM prediction_simplified
              WHERE correct IS NULL AND date >= CURDATE()
          )
    """)

    if not fights:
        return None

    best_ev   = None
    best_row  = None
    best_is_f1 = True

    for row in fights:
        f1_prob = float(row["f1_probability"])
        f2_prob = 1.0 - f1_prob
        f1_odds = row["f1_odds"]
        f2_odds = row["f2_odds"]

        if f1_odds is None or f2_odds is None:
            continue

        ev1 = calc_ev(f1_prob, int(f1_odds))
        ev2 = calc_ev(f2_prob, int(f2_odds))

        for ev, is_f1 in [(ev1, True), (ev2, False)]:
            if 2 <= ev <= 30:
                if best_ev is None or ev > best_ev:
                    best_ev   = ev
                    best_row  = row
                    best_is_f1 = is_f1

    if best_row is None:
        return None

    return {
        "fight_id":   best_row["fight_id"],
        "f1_name":    best_row["fighter1_name"],
        "f2_name":    best_row["fighter2_name"],
        "pick_name":  best_row["fighter1_name"] if best_is_f1 else best_row["fighter2_name"],
        "pick_id":    best_row["fighter1_id"]   if best_is_f1 else best_row["fighter2_id"],
        "pick_odds":  int(best_row["f1_odds"])  if best_is_f1 else int(best_row["f2_odds"]),
        "ev":         round(best_ev, 2),
    }


# ---------------------------------------------------------------------------
# Step 6: pow_avg_odds from all historical POW rows (excluding no-pick rows)
# ---------------------------------------------------------------------------
def get_pow_avg_odds(conn) -> int | None:
    rows = fetch_query(conn, """
        SELECT pow_odds FROM ufc_homepage
        WHERE pow_no_pick = 0 AND pow_odds IS NOT NULL
    """)
    if not rows:
        return None
    decimals = [american_to_decimal(int(r["pow_odds"])) for r in rows]
    avg_dec  = sum(decimals) / len(decimals)
    return decimal_to_american(avg_dec)


# ---------------------------------------------------------------------------
# Step 7: cumulative pow_win / pow_loss + units
# ---------------------------------------------------------------------------
def get_pow_record(conn) -> dict:
    rows = fetch_query(conn, """
        SELECT pow_correct, pow_odds
        FROM ufc_homepage
        WHERE pow_no_pick = 0 AND pow_correct IS NOT NULL
    """)
    wins = losses = 0
    units = 0.0
    for r in rows:
        if int(r["pow_correct"]) == 1:
            wins += 1
            odds = int(r["pow_odds"])
            units += odds / 100.0 if odds > 0 else 100.0 / abs(odds)
        else:
            losses += 1
            units -= 1.0
    return {
        "win":   wins,
        "loss":  losses,
        "units": round(units, 2),
    }


# ---------------------------------------------------------------------------
# Step 8: next event info
# ---------------------------------------------------------------------------
def get_next_event(conn) -> dict | None:
    rows = fetch_query(conn, """
        SELECT e.title, e.date
        FROM events e
        JOIN fights f ON e.event_id = f.event_id
        WHERE e.date >= CURDATE()
          AND f.fighter1_id IS NOT NULL
          AND f.fighter2_id IS NOT NULL
        GROUP BY e.event_id
        ORDER BY e.date ASC
        LIMIT 1
    """)
    if not rows:
        return None
    return {"name": rows[0]["title"], "date": rows[0]["date"]}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    conn = create_connection()
    if conn is None:
        print("Could not connect to DB")
        sys.exit(1)

    print("\n--- Step 1: update prediction_simplified.correct ---")
    update_prediction_correct(conn)

    print("--- Step 2: update ufc_homepage.pow_correct ---")
    update_pow_correct(conn)

    print("--- Step 3: model accuracy ---")
    model = get_model_accuracy(conn)
    print(f"  {model['correct']}-{model['total'] - model['correct']} ({model['total']} total)")

    print("--- Step 4: vegas accuracy ---")
    vegas = get_vegas_accuracy(conn)
    print(f"  {vegas['correct']}-{vegas['total'] - vegas['correct']} ({vegas['total']} total)")

    print("--- Step 5: pick of the week ---")
    pow_pick = select_pow(conn)
    if pow_pick:
        print(f"  POW: {pow_pick['pick_name']} ({pow_pick['f1_name']} vs {pow_pick['f2_name']}) EV={pow_pick['ev']} odds={pow_pick['pick_odds']}")
    else:
        print("  No pick of the week (no fight with EV in 2-30 range)")

    print("--- Step 6: pow avg odds ---")
    avg_odds = get_pow_avg_odds(conn)
    print(f"  Avg odds: {avg_odds}")

    print("--- Step 7: pow record ---")
    record = get_pow_record(conn)
    print(f"  {record['win']}-{record['loss']} ({record['units']:+.2f}u)")

    print("--- Step 8: next event ---")
    next_event = get_next_event(conn)
    print(f"  {next_event}")

    print("--- Inserting homepage row ---")
    no_pick = 1 if pow_pick is None else 0
    run_query(conn, """
        INSERT INTO ufc_homepage (
            date_inserted,
            next_event_name, next_event_date,
            model_correct, model_total,
            vegas_correct, vegas_total,
            pow_fight_id, pow_f1_name, pow_f2_name,
            pow_pick, pow_pick_id, pow_odds, pow_avg_odds,
            pow_win, pow_loss, pow_units, pow_no_pick, pow_correct
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s, %s, %s,
            %s, %s, %s, %s, NULL
        )
    """, (
        date.today(),
        next_event["name"] if next_event else None,
        next_event["date"] if next_event else None,
        model["correct"], model["total"],
        vegas["correct"], vegas["total"],
        pow_pick["fight_id"]  if pow_pick else None,
        pow_pick["f1_name"]   if pow_pick else None,
        pow_pick["f2_name"]   if pow_pick else None,
        pow_pick["pick_name"] if pow_pick else None,
        pow_pick["pick_id"]   if pow_pick else None,
        pow_pick["pick_odds"] if pow_pick else None,
        avg_odds,
        record["win"], record["loss"], record["units"],
        no_pick,
    ))
    print("Done.")
    conn.close()


if __name__ == "__main__":
    main()
