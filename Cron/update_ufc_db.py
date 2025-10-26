from ufc.scrapes import *
import time

def get_new_upcoming_events(conn):
    upcoming_events_arr = get_all_events(past=False)
    events_not_in_db_yet = filter_new_events(upcoming_events_arr)
    if not events_not_in_db_yet:
        print('No new upcoming events found...')
        return
    print(f'Found {len(events_not_in_db_yet)} events not in db yet, pushing...')
    for event_url in events_not_in_db_yet:
        data = get_event_data(event_url, getting_old_data=False)
        if data is not None:
            push_events(data, conn)

def event_update_loop(event_urls, conn, old_data):
    for event_url in event_urls:
        data = get_event_data(event_url, getting_old_data=old_data)
        if data is None:
            print(f'No data found for event with URL: {event_url}')
            continue
        idx = EventNameIndex(data)
        for i in range(0, len(data['fights'])):
            fighter1s_url = data['fights'][i]['fighter1']['link']
            fighter2s_url = data['fights'][i]['fighter2']['link']
            if not fighter1s_url or not fighter2s_url: # this could be there first fight so they wont have data
                continue
            fighter1, fighter2 = data['fights'][i]['fighter1']['fighter_name'], data['fights'][i]['fighter2']['fighter_name']
            fighter1_career_stats, fights_arr1, upcoming1, resp_code1 = get_fighter_data(fighter1s_url, fighter1)
            fighter2_career_stats, fights_arr2, upcoming2, resp_code2 = get_fighter_data(fighter2s_url, fighter2)
            if upcoming1:
                for dict_data in upcoming1:
                    push_fighter(idx, dict_data['fighter2_careerstats'], conn)
                    push_fights_upcoming(idx, dict_data, conn)
            if upcoming2:
                for dict_data in upcoming2:
                    push_fighter(idx, dict_data['fighter2_careerstats'], conn)
                    push_fights_upcoming(idx, dict_data, conn)
            if resp_code1 == 200 and resp_code2 == 200:
                pass
            else:
                print(f"over requesting server, responses stopped. resp codes: {resp_code1}, {resp_code2}\nstopping program on event_url = '{event_url}'")
                time.sleep(20)
            if not fighter1_career_stats or not fighter2_career_stats or not fights_arr1 or not fights_arr2: # fighter has not faught for ufc yet (debut)
                continue
            push_fighter(idx, fighter1_career_stats, conn)
            push_fighter(idx, fighter2_career_stats, conn)
            if fights_arr1:
                for i in fights_arr1:
                    random_career_stats = i.get('ops_careerstats')
                    if random_career_stats:
                        push_fighter(idx, random_career_stats, conn)
                    if not i['winner_loser']['winner'] or not i['winner_loser']['loser']:
                        i['winner_loser']['winner'] = 'draw'
                        i['winner_loser']['loser'] = 'draw'
                    push_fights(idx, i, fighter1_career_stats, conn)
                    push_totals(i, fighter1_career_stats, conn)
                    push_rounds(i, fighter1_career_stats, conn)
            if fights_arr2:
                for i in fights_arr2:
                    random_career_stats = i.get('ops_careerstats')
                    if random_career_stats:
                        push_fighter(idx, random_career_stats, conn)
                    if not i['winner_loser']['winner'] or not i['winner_loser']['loser']:
                        i['winner_loser']['winner'] = 'draw'
                        i['winner_loser']['loser'] = 'draw'
                    push_fights(idx, i, fighter2_career_stats, conn)
                    push_totals(i, fighter2_career_stats, conn)
                    push_rounds(i, fighter2_career_stats, conn)

def update_scrapes_for_upcoming_events(conn):
    upcoming_event_urls = get_future_event_urls(conn)
    if not upcoming_event_urls:
        print('No upcoming events found in db...')
        return
    print(f'Updating {len(upcoming_event_urls)} events in DB\nscraping...')
    event_update_loop(upcoming_event_urls, conn, old_data=False)

def update_last2_events_outcomes(conn):
    last2_events = get_last_two_past_events(conn)
    if not last2_events:
        print('No previous events found in db...')
        return
    print(f'Updating last 2 event outcomes.\nscraping...')
    event_update_loop(last2_events, conn, old_data=True)