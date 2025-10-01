from scrapes.ufc_mma import get_all_events, get_event_data, get_fighter_data, push_events, EventNameIndex, push_fighter, create_connection, push_fights_upcoming
from scrapes.ufc_mma import push_fights, push_totals, push_rounds, fetch_query
import time
PUSHVAR = True
def main():
    conn = create_connection()
    total_fights = 0
    total_events = 0
    events_arr = []
    for page_num in range(1, 2):
        events = get_all_events(past=False, page=page_num)
        events_arr.extend(events)
    events_arr = set(events_arr)
    print(f"Found {len(events_arr)} events")
    for event_url in events_arr:
        data = get_event_data(event_url, getting_old_data=False)
        event_id = data.get('event_id')
        query = "SELECT * FROM ufc.events WHERE event_id = %s"
        params = (event_id,)
        rows = fetch_query(conn, query, params)
        if rows: # event data already in
            print(f'event: {event_id} already in SQL')
            # continue
        else:
            print(f'event {event_id} not in SQL yet, continuing..')
        idx = EventNameIndex(data)
        if PUSHVAR:
            push_events(data, conn)
        if data is not None:
            print(f"--- {event_url} ---")
            print(f"Title: {data['title']}")
            for i in range(0, len(data['fights'])):
                fighter1s_url = data['fights'][i]['fighter1']['link']
                fighter2s_url = data['fights'][i]['fighter2']['link']
                if not fighter1s_url or not fighter2s_url: # this could be there first fight so they wont have data
                    continue
                fighter1, fighter2 = data['fights'][i]['fighter1']['fighter_name'], data['fights'][i]['fighter2']['fighter_name']
                print(f'Fight {i+1}: {fighter1} vs {fighter2}')
                fighter1_career_stats, fights_arr1, upcoming1, resp_code1 = get_fighter_data(fighter1s_url, fighter1)
                fighter2_career_stats, fights_arr2, upcoming2, resp_code2 = get_fighter_data(fighter2s_url, fighter2)
                if resp_code1 == 200 and resp_code2 == 200:
                    print(f'http request response was valid')
                else:
                    print(f"over requesting server, responses stopped. resp codes: {resp_code1}, {resp_code2}\nstopping program on event_id = '{event_id}'")
                    exit()
                if not fighter1_career_stats or not fighter2_career_stats or not fights_arr1 or not fights_arr2: # fighter has not faught for ufc yet (debut)
                    continue
                if PUSHVAR:
                    push_fighter(idx, fighter1_career_stats, conn)
                    push_fighter(idx, fighter2_career_stats, conn)
                if fights_arr1:
                    for i in fights_arr1:
                        random_career_stats = i.get('ops_careerstats')
                        if random_career_stats:
                            if PUSHVAR:
                                push_fighter(idx, random_career_stats, conn)
                        if not i['winner_loser']['winner'] or not i['winner_loser']['loser']:
                            i['winner_loser']['winner'] = 'draw'
                            i['winner_loser']['loser'] = 'draw'
                            continue
                        if PUSHVAR:
                            push_fights(idx, i, fighter1_career_stats, conn)
                            push_totals(i, fighter1_career_stats, conn)
                            push_rounds(i, fighter1_career_stats, conn)
                if fights_arr2:
                    for i in fights_arr2:
                        random_career_stats = i.get('ops_careerstats')
                        if random_career_stats:
                            if PUSHVAR:
                                push_fighter(idx, random_career_stats, conn)
                        if not i['winner_loser']['winner'] or not i['winner_loser']['loser']:
                            print(i)
                            continue
                        if PUSHVAR:
                            push_fights(idx, i, fighter2_career_stats, conn)
                            push_totals(i, fighter2_career_stats, conn)
                            push_rounds(i, fighter2_career_stats, conn)
                if upcoming1:
                    for dict_data in upcoming1:
                        if PUSHVAR:
                            push_fighter(idx, dict_data['fighter2_careerstats'], conn)
                            push_fights_upcoming(idx, dict_data, conn)
                if upcoming2:
                    for dict_data in upcoming1:
                        if PUSHVAR:
                            push_fighter(idx, dict_data['fighter2_careerstats'], conn)
                            push_fights_upcoming(idx, dict_data, conn)
                
                total_fights += (len(fights_arr1) + len(fights_arr2))
            print()
            total_events += 1
            if total_events == 4:
                break

    print(f"\nTotal fights scraped: {total_fights}")
    print(f"Total events scraped: {total_events}")

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    runtime = end_time - start_time
    print(f"Runtime: {runtime:.2f} seconds")