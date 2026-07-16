"""
backfill_last_2_months.py

Scrapes and pushes:
  1. Past UFC events within the cutoff window.
  2. All upcoming UFC events + their fighter/fight rows.

Run from Cron/ufc/:
    python backfill_last_2_months.py
"""

import time
from datetime import date, timedelta
from update_ufc_db import event_update_loop, get_new_upcoming_events, update_scrapes_for_upcoming_events
from scrapes import (
    create_connection,
    get_all_events,
    get_event_data,
    push_events,
)
from scrapes.utils import parse_event_date

CUTOFF_DATE = date.today() - timedelta(days=365)   # ~12 months back
MAX_PAGES = 10
SLEEP_BETWEEN_FETCHES = 2   # between individual event page fetches during collection
SLEEP_BETWEEN_EVENTS = 10   # between full event processing runs


def collect_past_events_auto() -> list[tuple]:
    """
    Page through Tapology results and return (url, data) pairs for events
    within the cutoff window. Caches data here so process step doesn't re-fetch.
    """
    collected = []
    for page in range(1, MAX_PAGES + 1):
        print(f"  Fetching past events page {page}...")
        urls = get_all_events(group="ufc", past=True, page=page)
        if not urls:
            print(f"  Page {page} returned no URLs, stopping.")
            break

        page_has_recent = False
        for url in urls:
            if any(u == url for u, _ in collected):
                continue
            time.sleep(SLEEP_BETWEEN_FETCHES)
            data = get_event_data(url, getting_old_data=True)
            if data is None:
                continue
            event_date_str = data.get("date")
            if not event_date_str:
                continue
            sql_date_str, _ = parse_event_date(event_date_str)
            event_date_obj = date.fromisoformat(sql_date_str)

            if event_date_obj >= CUTOFF_DATE:
                page_has_recent = True
                collected.append((url, data))
                print(f"    + {data['title']}  ({sql_date_str})")
            else:
                print(f"    - {data.get('title', url)}  ({sql_date_str})  [older than cutoff]")

        if not page_has_recent:
            print("  All remaining events are older than cutoff, stopping.")
            break

        time.sleep(SLEEP_BETWEEN_FETCHES)

    return collected


def process_events(event_pairs: list[tuple], conn) -> None:
    """
    Push event rows and run the full fighter/fight scrape loop.
    Passes cached event data so event pages are never re-fetched.
    """
    for i, (url, data) in enumerate(event_pairs, 1):
        print(f"\n[{i}/{len(event_pairs)}] {data['title']}  ({data.get('date', '?')})")
        push_events(data, conn)
        try:
            event_update_loop([url], conn, old_data=True, prefetched={url: data})
        except Exception as e:
            print(f"  ERROR processing event, skipping: {e.__class__.__name__}: {e}")
        time.sleep(SLEEP_BETWEEN_EVENTS)


def main():
    start = time.time()
    conn = create_connection()
    if not conn:
        print("Could not connect to database. Exiting.")
        return

    print(f"=== Backfill (cutoff: {CUTOFF_DATE}) ===\n")

    # ── 1. Collect past events (fetches event pages with rate-limit-friendly sleep)
    print("Step 1: Collecting past event URLs...")
    event_pairs = collect_past_events_auto()

    if event_pairs:
        print(f"\nProcessing {len(event_pairs)} past events...")
        process_events(event_pairs, conn)
    else:
        print("No past events found within cutoff.")

    # ── 2. Upcoming events
    print("\nStep 2: Upcoming events...")
    get_new_upcoming_events(conn=conn)
    update_scrapes_for_upcoming_events(conn=conn)

    elapsed = time.time() - start
    print(f"\nDone. Total time: {int(elapsed // 60)}m {elapsed % 60:.1f}s")


if __name__ == "__main__":
    main()
