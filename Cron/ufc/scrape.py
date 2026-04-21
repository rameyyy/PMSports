from update_ufc_db import get_new_upcoming_events, update_scrapes_for_upcoming_events, update_last2_events_outcomes
from scrapes import create_connection
import time

if __name__ == "__main__":
    start_time = time.time()
    conn = create_connection()
    print("Beginning UFC scrape...")

    # Discover new upcoming events
    get_new_upcoming_events(conn=conn)
    # Refresh fighter/fight data for upcoming cards
    update_scrapes_for_upcoming_events(conn=conn)
    # Fill in outcomes for the last 2 completed events
    update_last2_events_outcomes(conn=conn)

    conn.close()
    elapsed = time.time() - start_time
    print(f"Scrape done in {int(elapsed // 60)}m {elapsed % 60:.2f}s")
