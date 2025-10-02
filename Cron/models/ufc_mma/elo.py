from .utils import create_connection, fetch_query
query = """
SELECT 
    fight_id, 
    fighter1_id, 
    fighter2_id, 
    winner_id, 
    fight_date, 
    method, 
    fight_format, 
    end_time
FROM ufc.fights
WHERE winner_id IS NOT NULL
  AND (fighter1_id = %s OR fighter2_id = %s)
"""
# fighter with id in query2 will be good to test elo system on
query2 = """
SELECT * FROM ufc.fighters
WHERE fighter_id = '08af939f41b5a57b ' 
"""
def raw_elo():
    conn = create_connection()
    fighter = fetch_query(conn, query2, None)
    fighter_id = fighter.get('fighter_id')
    fighters_fights = fetch_query(conn, query, (fighter_id, fighter_id))

