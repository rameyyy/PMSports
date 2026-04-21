from .utils import create_connection, fetch_query, run_query


def _implied(american: int) -> float:
    if american > 0:
        return 100 / (american + 100)
    return abs(american) / (abs(american) + 100)


def _find_fight(conn, name1: str, name2: str, event_date: str) -> dict | None:
    """Find upcoming fight in fights table by fuzzy name + date match."""
    query = """
        SELECT fight_id, fighter1_id, fighter2_id, fighter1_name, fighter2_name
        FROM fights
        WHERE fight_date >= DATE_SUB(%s, INTERVAL 1 DAY)
          AND fight_date <= DATE_ADD(%s, INTERVAL 1 DAY)
          AND (
            (fighter1_name LIKE %s AND fighter2_name LIKE %s)
            OR (fighter1_name LIKE %s AND fighter2_name LIKE %s)
          )
        LIMIT 1
    """
    n1, n2 = f"%{name1.split()[-1]}%", f"%{name2.split()[-1]}%"
    rows = fetch_query(conn, query, (event_date, event_date, n1, n2, n2, n1))
    return rows[0] if rows else None


def _push_moneyline(conn, fight_id: str, bookmaker: str,
                    f1_id: str, f2_id: str,
                    f1_odds: int, f2_odds: int) -> None:
    run_query(conn, """
        INSERT INTO bookmaker_moneyline
            (fight_id, bookmaker, fighter1_id, fighter2_id,
             fighter1_odds, fighter2_odds, fighter1_implied, fighter2_implied)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            fighter1_id      = VALUES(fighter1_id),
            fighter2_id      = VALUES(fighter2_id),
            fighter1_odds    = VALUES(fighter1_odds),
            fighter2_odds    = VALUES(fighter2_odds),
            fighter1_implied = VALUES(fighter1_implied),
            fighter2_implied = VALUES(fighter2_implied)
    """, (fight_id, bookmaker, f1_id, f2_id,
          f1_odds, f2_odds,
          round(_implied(f1_odds) * 100, 2),
          round(_implied(f2_odds) * 100, 2)))


def _push_rounds(conn, fight_id: str, bookmaker: str,
                 line: float, over_price: int, under_price: int) -> None:
    run_query(conn, """
        INSERT INTO bookmaker_rounds
            (fight_id, bookmaker, line, over_price, under_price,
             over_implied, under_implied)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            line          = VALUES(line),
            over_price    = VALUES(over_price),
            under_price   = VALUES(under_price),
            over_implied  = VALUES(over_implied),
            under_implied = VALUES(under_implied)
    """, (fight_id, bookmaker, line, over_price, under_price,
          round(_implied(over_price) * 100, 2),
          round(_implied(under_price) * 100, 2)))


PREFERRED_BOOKS = ["draftkings", "fanduel", "betmgm", "bovada", "williamhill_us"]


def _update_prediction_odds(conn, fight_id: str, f1_odds: int, f2_odds: int) -> None:
    """Set f1_odds/f2_odds on prediction_simplified only if currently NULL — never overwrites."""
    run_query(conn, """
        UPDATE prediction_simplified
        SET f1_odds = %s, f2_odds = %s
        WHERE fight_id = %s
          AND f1_odds IS NULL
          AND f2_odds IS NULL
    """, (f1_odds, f2_odds, fight_id))


def process_odds(conn, data: list) -> None:
    processed = skipped = errors = 0

    for event in data:
        home = event.get("home_team")
        away = event.get("away_team")
        event_date = event.get("commence_time", "")[:10]

        fight = _find_fight(conn, home, away, event_date)
        if not fight:
            print(f"  [skip] No DB match: {home} vs {away}")
            skipped += 1
            continue

        fight_id = fight["fight_id"]
        f1_id    = fight["fighter1_id"]
        f2_id    = fight["fighter2_id"]
        f1_name  = (fight["fighter1_name"] or "").lower()

        # Track best h2h odds for prediction_simplified (preferred book first)
        pred_f1_odds = pred_f2_odds = None
        bookmakers_by_key = {bm["key"]: bm for bm in event.get("bookmakers", [])}
        book_order = PREFERRED_BOOKS + [k for k in bookmakers_by_key if k not in PREFERRED_BOOKS]

        for bm_key in book_order:
            bm = bookmakers_by_key.get(bm_key)
            if not bm:
                continue
            for market in bm.get("markets", []):
                outcomes = {o["name"]: o for o in market.get("outcomes", [])}

                if market["key"] == "h2h" and len(outcomes) == 2:
                    names = list(outcomes.keys())
                    if names[0].split()[-1].lower() in f1_name or f1_name in names[0].lower():
                        f1_odds = outcomes[names[0]]["price"]
                        f2_odds = outcomes[names[1]]["price"]
                    else:
                        f1_odds = outcomes[names[1]]["price"]
                        f2_odds = outcomes[names[0]]["price"]
                    _push_moneyline(conn, fight_id, bm_key, f1_id, f2_id, f1_odds, f2_odds)
                    # capture first (preferred) book for prediction_simplified
                    if pred_f1_odds is None:
                        pred_f1_odds, pred_f2_odds = f1_odds, f2_odds

                elif market["key"] == "totals" and "Over" in outcomes and "Under" in outcomes:
                    line        = outcomes["Over"]["point"]
                    over_price  = outcomes["Over"]["price"]
                    under_price = outcomes["Under"]["price"]
                    _push_rounds(conn, fight_id, bm_key, line, over_price, under_price)

        if pred_f1_odds is not None:
            _update_prediction_odds(conn, fight_id, pred_f1_odds, pred_f2_odds)

        processed += 1

    print(f"Odds: {processed} matched, {skipped} skipped, {errors} errors")


def run(json_data: list) -> None:
    conn = create_connection()
    try:
        process_odds(conn, json_data)
    finally:
        conn.close()
