import json
from scrapes.ufc_mma.sqlpush import fetch_query, create_connection
# from scrapes.ufc_mma 
# read JSON file
with open("ufc_odds.json", "r") as f:
    data = json.load(f)

# loop through and print teams
conn = create_connection()
query = "SELECT name FROM ufc.fighters WHERE name LIKE %s"
for event in data:
    home = event.get("home_team")
    away = event.get("away_team")
    time = event.get('commence_time')
    bm = event.get('bookmakers')
    if not bm:
        continue
    if time[0:10] != '2025-10-18':
        continue
    
    homeMatch = fetch_query(conn, query, (home,))
    awayMatch = fetch_query(conn, query, (away,))
    print(f"Home: {home}, Away: {away}")
    if homeMatch:
        print(f'\tHome Matched: {home} = {homeMatch}')
    if awayMatch:
        print(f'\tAway Matched: {away} = {awayMatch}')