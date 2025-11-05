"""Debug script to see raw API response"""
import os
from dotenv import load_dotenv
import requests
import json

# Load environment variables
ncaamb_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(ncaamb_dir, '.env')
load_dotenv(env_path)

ODDS_API_KEY = os.getenv("ODDS_API")
BASE_URL = "https://api.the-odds-api.com/v4"

url = f"{BASE_URL}/historical/sports/basketball_ncaab/odds"
params = {
    "apiKey": ODDS_API_KEY,
    "date": "2020-11-16T21:00:00Z",
    "regions": "us",
    "markets": "spreads,h2h,totals",
    "oddsFormat": "american"
}

response = requests.get(url, params=params, timeout=30)
data = response.json()

print("API Response structure:")
print(json.dumps(data, indent=2)[:2000])

if 'data' in data and len(data['data']) > 0:
    print("\n\nFirst event structure:")
    print(json.dumps(data['data'][0], indent=2))
