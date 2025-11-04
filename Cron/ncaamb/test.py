import mysql.connector
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv('.env')

try:
    conn = mysql.connector.connect(
        host=os.getenv("DB_HOST"),
        port=int(os.getenv("DB_PORT")),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("NCAAMB_DB"),
    )

    cursor = conn.cursor(dictionary=True)

    # Query for the Austin Peay vs Memphis game
    game_id = "20231230_Austin Peay_Memphis"
    cursor.execute("SELECT * FROM games WHERE game_id = %s", (game_id,))

    result = cursor.fetchone()

    if result:
        print(f"Game Found: {game_id}")
        print("-" * 100)
        for key, value in result.items():
            print(f"{key:<30} {value}")
    else:
        print(f"No game found with ID: {game_id}")
        print("\nSearching for games with 'Austin Peay' and 'Memphis'...")
        cursor.execute("SELECT game_id, date, team_1, team_2 FROM games WHERE (team_1 = 'Austin Peay' OR team_2 = 'Austin Peay') AND (team_1 = 'Memphis' OR team_2 = 'Memphis')")
        results = cursor.fetchall()
        if results:
            print(f"Found {len(results)} matching games:")
            for row in results:
                print(row)
        else:
            print("No matching games found")

    cursor.close()
    conn.close()

except Exception as e:
    print(f"Error: {e}")
