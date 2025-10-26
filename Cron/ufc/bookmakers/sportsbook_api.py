import os
import requests
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv('ODDS_API_KEY')

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