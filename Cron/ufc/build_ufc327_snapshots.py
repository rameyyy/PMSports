"""
build_ufc327_snapshots.py

Builds ufc327_snapshots.parquet for tonight's UFC 327 card.
Pushes the fight rows from the Tapology event page, then builds
pre-fight snapshots using existing DB fight history — no full
fighter re-scrape needed.

Run from Cron/ufc/:
    python build_ufc327_snapshots.py
"""

import polars as pl
import sys, os
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(__file__))

from scrapes import create_connection, get_event_data, push_events
from scrapes.sqlpush import run_query, fetch_query
from scrapes.utils import parse_event_date
from models.build_raw_df_all_fights_with_rounds import (
    get_all_related_fights,
    build_pre_fight_snapshots,
    enrich_with_fighter_stats,
    enrich_with_fight_totals,
    enrich_with_round_data,
)

UFC327_URL = "https://www.tapology.com/fightcenter/events/137847-ufc-327"
EVENT_ID   = "137847-ufc-327"
OUTPUT_FILE = "ufc327_snapshots.parquet"


def push_ufc327_fights(event_data, conn):
    """
    Insert UFC 327 fights into the DB using only what Tapology gives us.
    Looks up fighter_ids by name from the fighters table.
    """
    fight_date, _ = parse_event_date(event_data["date"])
    inserted = 0

    for fight in event_data["fights"]:
        f1 = fight["fighter1"]
        f2 = fight["fighter2"]
        f1_name = f1["fighter_name"]
        f2_name = f2["fighter_name"]

        # Look up IDs by name (fuzzy not needed — exact from DB)
        def get_id(name):
            rows = fetch_query(conn,
                "SELECT fighter_id FROM ufc.fighters WHERE name = %s LIMIT 1",
                (name,))
            return rows[0]["fighter_id"] if rows else None

        f1_id = get_id(f1_name)
        f2_id = get_id(f2_name)

        if not f1_id or not f2_id:
            print(f"  Skipping (no DB id): {f1_name} vs {f2_name}  (f1={f1_id}, f2={f2_id})")
            continue

        overview = fight.get("fight_overview", {})
        fight_type_raw = (overview.get("fight_card_type") or "").lower()
        if "main event" in fight_type_raw:
            fight_type = "main"
            fight_format = 5
        elif "title" in fight_type_raw:
            fight_type = "title"
            fight_format = 5
        else:
            fight_type = None
            fight_format = 3

        fight_id = f"{f1_id}_{f2_id}_{fight_date}"

        cmd = """
            INSERT INTO fights (
                fight_id, event_id, fighter1_id, fighter2_id,
                fighter1_name, fighter2_name, fight_date,
                fight_format, fight_type, weight_class
            ) VALUES (
                %(fight_id)s, %(event_id)s, %(fighter1_id)s, %(fighter2_id)s,
                %(fighter1_name)s, %(fighter2_name)s, %(fight_date)s,
                %(fight_format)s, %(fight_type)s, %(weight_class)s
            )
            ON DUPLICATE KEY UPDATE
                event_id      = IF(VALUES(event_id) IS NULL, event_id, VALUES(event_id)),
                fighter1_name = IF(VALUES(fighter1_name) IS NULL, fighter1_name, VALUES(fighter1_name)),
                fighter2_name = IF(VALUES(fighter2_name) IS NULL, fighter2_name, VALUES(fighter2_name)),
                fight_format  = IF(VALUES(fight_format) IS NULL, fight_format, VALUES(fight_format)),
                fight_type    = IF(VALUES(fight_type) IS NULL, fight_type, VALUES(fight_type)),
                weight_class  = IF(VALUES(weight_class) IS NULL, weight_class, VALUES(weight_class))
        """
        params = {
            "fight_id": fight_id,
            "event_id": EVENT_ID,
            "fighter1_id": f1_id,
            "fighter2_id": f2_id,
            "fighter1_name": f1_name,
            "fighter2_name": f2_name,
            "fight_date": fight_date,
            "fight_format": fight_format,
            "fight_type": fight_type,
            "weight_class": overview.get("weight_class"),
        }
        run_query(conn, cmd, params)
        label = f"  + {f1_name} vs {f2_name}".encode('ascii', errors='replace').decode('ascii')
        print(label)
        inserted += 1

    return inserted


def main():
    conn = create_connection()
    print(f"UFC 327 snapshot builder\n{'='*50}")

    # 1. Fetch event page from Tapology
    print("\nStep 1: Fetching UFC 327 from Tapology...")
    data = get_event_data(UFC327_URL, getting_old_data=False)
    if data is None:
        data = get_event_data(UFC327_URL, getting_old_data=True)
    if data is None:
        print("Could not fetch event data.")
        return
    title = data['title'].encode('ascii', errors='replace').decode('ascii')
    print(f"  Got: {title} - {len(data['fights'])} fights")

    # 2. Push event + fight rows (no fighter scraping)
    print("\nStep 2: Pushing event and fight rows...")
    push_events(data, conn)
    n = push_ufc327_fights(data, conn)
    print(f"  {n} fights inserted/updated")

    # 3. Load those fights back from DB
    print("\nStep 3: Loading UFC 327 fights from DB...")
    rows = fetch_query(conn, "SELECT * FROM ufc.fights WHERE event_id = %s", (EVENT_ID,))
    if not rows:
        print("  Still no fights in DB — fighter IDs may not match.")
        return
    event_fights = pl.DataFrame(rows)
    print(f"  {len(event_fights)} fights loaded")

    # 4. Get full prior fight history for these fighters
    print("\nStep 4: Loading prior fight history...")
    all_related = get_all_related_fights(conn, event_fights)
    print(f"  {len(all_related)} historical fights loaded")

    # 5-7. Build snapshots + enrich
    print("\nStep 5: Building pre-fight snapshots...")
    snapshots = build_pre_fight_snapshots(event_fights, all_related, min_prior_fights=0)
    print(f"  {len(snapshots)} snapshots")

    print("\nStep 6: Enriching with fighter stats...")
    snapshots = enrich_with_fighter_stats(snapshots, conn)

    print("\nStep 7: Enriching with fight totals...")
    snapshots = enrich_with_fight_totals(snapshots, conn)

    print("\nStep 8: Enriching with round data...")
    snapshots = enrich_with_round_data(snapshots, conn)

    # 8. Save
    print(f"\nSaving to {OUTPUT_FILE}...")
    snapshots.write_parquet(OUTPUT_FILE)
    print(f"Done — {len(snapshots)} rows, {len(snapshots.columns)} columns")


if __name__ == "__main__":
    main()
