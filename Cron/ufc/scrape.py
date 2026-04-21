from update_ufc_db import get_new_upcoming_events, update_scrapes_for_upcoming_events, update_last2_events_outcomes
from update_or_predict import update_bookmakers
from scrapes import create_connection
import time

if __name__ == "__main__":
    start_time = time.time()
    conn = create_connection()
    print("Beginning UFC scrape...")

    # Tapology / UFCStats scrapes
    get_new_upcoming_events(conn=conn)
    update_scrapes_for_upcoming_events(conn=conn)
    update_last2_events_outcomes(conn=conn)

    # Odds API — pull moneylines for all upcoming fights
    print("Fetching bookmaker odds...")
    update_bookmakers()

    elapsed = time.time() - start_time
    print(f"Scrape done in {int(elapsed // 60)}m {elapsed % 60:.2f}s")
