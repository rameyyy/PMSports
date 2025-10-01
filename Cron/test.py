API_KEY = 'ce0e0d3f9c33b0fb986a502cbb1dbd39'
import os
import requests
BASE = "https://api.the-odds-api.com/v4"

def get_mma_odds(
    regions=("us",),
    markets=("h2h",),
    odds_format="american"
):
    params = {
        "regions": ",".join(regions),
        "markets": ",".join(markets),
        "oddsFormat": odds_format,
        "apiKey": API_KEY,
    }
    url = f"{BASE}/sports/mma_mixed_martial_arts/odds"
    response = requests.get(url, params=params, timeout=20)
    response.raise_for_status()
    return response.json()

if __name__ == "__main__":
    data = get_mma_odds()
    # Print out the first event for preview
    import json

    # Suppose `rows` is the list of dicts you got from flattening the odds API response
    with open("ufc_odds.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    if data:
        from pprint import pprint
        pprint(data[0])
    else:
        print("No events returned.")
