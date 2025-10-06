from .utils import create_connection, fetch_query, run_query
import time
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
WHERE (fighter1_id = %s OR fighter2_id = %s)
"""
#WHERE winner_id IS NOT NULL
#AND (fighter1_id = %s OR fighter2_id = %s)

# fighter with id in query2 will be good to test elo system on
query2 = """
SELECT * FROM ufc.fighters;
"""
ankalev = 'd802174b0c0c1f4e'
perera = 'e5549c82bfb5582d'

SQL = """
INSERT INTO ufc.advanced_fighter_stats (fighter_id, weighted_elo)
VALUES (%s, %s)
ON DUPLICATE KEY UPDATE
  weighted_elo = VALUES(weighted_elo);
"""
# params: (fighter_id, weighted_elo)

query3 = """
SELECT raw_elo FROM ufc.advanced_fighter_stats
WHERE fighter_id = %s
"""
def raw_elo():
    from . import calc_elo
    conn = create_connection()
    start = time.perf_counter()
    fighter = fetch_query(conn, query2, None)
    # fighters_id = fighter.get('fighter_id')
    # fighter_name = fighter.get('name')
    # fighters_fights = fetch_query(conn, query, (fighters_id, fighters_id))
    # elo = calc_raw_elo.get_weighted_elo(fighters_id, fighters_fights, conn)
    # print(fighter_name, elo)
    lenf = len(fighter)
    count = 1
    for fighter_info in fighter:
      fighter_id = fighter_info.get('fighter_id')
      fighters_fights = fetch_query(conn, query, (fighter_id, fighter_id))
      fighters_elo = calc_elo.get_weighted_elo(fighter_id, fighters_fights, conn)
      run_query(conn, SQL, (fighter_id, fighters_elo))
      print(f'{count}/{lenf}')
      count+=1

    end = time.perf_counter()
    print(f"Took {((end - start)/60):.3f} m")


