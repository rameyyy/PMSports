from bookmakers.sportsbook_api import get_mma_odds
from bookmakers.bookmaker_push import run
import time

if __name__ == "__main__":
    start = time.time()
    print("Fetching UFC odds...")
    data = get_mma_odds()
    print(f"Got {len(data)} fights from Odds API")
    run(data)
    print(f"Odds done in {time.time() - start:.1f}s")
