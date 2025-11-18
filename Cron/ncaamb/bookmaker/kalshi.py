import requests
from datetime import datetime

BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"

def get_upcoming_ncaa_games():
    """Get upcoming NCAA basketball games with odds"""
    
    # Get all open markets
    response = requests.get(
        f"{BASE_URL}/markets",
        params={
            "event_ticker": "KXNCAAB",  # NCAA Basketball
            "status": "open",
            "limit": 100
        }
    )
    
    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        return []
    
    markets = response.json().get("markets", [])
    
    games = []
    for market in markets:
        game = {
            "title": market.get("title"),
            "subtitle": market.get("subtitle"),
            "yes_price": market.get("yes_bid"),  # Current yes price (cents)
            "no_price": market.get("no_bid"),
            "yes_ask": market.get("yes_ask"),
            "no_ask": market.get("no_ask"),
            "close_time": market.get("close_time"),
            "ticker": market.get("ticker"),
            "volume": market.get("volume")
        }
        games.append(game)
    
    return games

def format_odds(games):
    """Format odds for display"""
    upcoming = []
    
    for game in games:
        # Parse close time
        close_time = datetime.fromisoformat(game["close_time"].replace("Z", "+00:00"))
        
        # Only include future games
        if close_time > datetime.now(close_time.tzinfo):
            yes_prob = game["yes_price"] / 100 if game["yes_price"] else 0
            american_odds = kalshi_to_american(yes_prob)
            
            upcoming.append({
                "matchup": game["title"],
                "details": game["subtitle"],
                "probability": f"{yes_prob:.1%}",
                "american_odds": american_odds,
                "game_time": close_time,
                "ticker": game["ticker"]
            })
    
    # Sort by game time
    upcoming.sort(key=lambda x: x["game_time"])
    return upcoming

def kalshi_to_american(prob):
    """Convert probability to American odds"""
    if prob == 0 or prob == 1:
        return "N/A"
    
    if prob >= 0.5:
        odds = int(-prob / (1 - prob) * 100)
    else:
        odds = int((1 - prob) / prob * 100)
    
    return f"{odds:+d}"

# Usage
if __name__ == "__main__":
    games = get_upcoming_ncaa_games()
    upcoming = format_odds(games)
    
    print(f"Upcoming NCAA Basketball Games ({len(upcoming)}):\n")
    for game in upcoming:
        print(f"{game['matchup']}")
        print(f"  {game['details']}")
        print(f"  Probability: {game['probability']} ({game['american_odds']})")
        print(f"  Game Time: {game['game_time'].strftime('%m/%d %I:%M %p')}")
        print()