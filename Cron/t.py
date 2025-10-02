import json
# from scrapes.ufc_mma 
# read JSON file
with open("ufc_odds.json", "r") as f:
    data = json.load(f)

# loop through and print teams
for event in data:
    home = event.get("home_team")
    away = event.get("away_team")
    print(f"Home: {home}, Away: {away}")