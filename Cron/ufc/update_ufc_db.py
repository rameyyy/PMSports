from scrapes import *
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

def event_update_loop(event_urls, conn, old_data, prefetched=None):
    """
    prefetched: optional dict mapping event_url -> already-fetched event data,
    so Tapology doesn't get re-hit for the event page.
    """
    for event_url in event_urls:
        if prefetched and event_url in prefetched:
            data = prefetched[event_url]
        else:
            data = get_event_data(event_url, getting_old_data=old_data)
        if data is None:
            print(f'No data found for event with URL: {event_url}')
            continue

        idx = EventNameIndex(data)

        # --- Phase 1: fetch all fighter data, collect UFCStats names ---
        fetched_bouts = []
        all_ufc_names = []
        for bout in data['fights']:
            f1_url = bout['fighter1']['link']
            f2_url = bout['fighter2']['link']
            if not f1_url or not f2_url:
                fetched_bouts.append(None)
                continue
            f1_name = bout['fighter1']['fighter_name']
            f2_name = bout['fighter2']['fighter_name']
            cs1, arr1, up1, rc1 = get_fighter_data(f1_url, f1_name)
            cs2, arr2, up2, rc2 = get_fighter_data(f2_url, f2_name)
            fetched_bouts.append((cs1, arr1, up1, rc1, cs2, arr2, up2, rc2))
            # collect every UFCStats name we'll need to match
            for cs in (cs1, cs2):
                if cs and cs.get('fighter_name'):
                    all_ufc_names.append(cs['fighter_name'])
            for arr in (arr1 or [], arr2 or []):
                for fight in arr:
                    ops = fight.get('ops_careerstats') or {}
                    if ops.get('fighter_name'):
                        all_ufc_names.append(ops['fighter_name'])
            if rc1 != 200 or rc2 != 200:
                print(f"over requesting server, resp codes: {rc1}, {rc2}\nstopping on '{event_url}'")
                time.sleep(20)

        # --- One Claude batch call for entire event ---
        idx.preload(all_ufc_names)

        # --- Phase 2: push using cached matches ---
        for bout_data, fetched in zip(data['fights'], fetched_bouts):
            if fetched is None:
                continue
            cs1, arr1, up1, rc1, cs2, arr2, up2, rc2 = fetched
            if up1:
                for dict_data in up1:
                    push_fighter(idx, dict_data['fighter2_careerstats'], conn)
                    push_fights_upcoming(idx, dict_data, conn)
            if up2:
                for dict_data in up2:
                    push_fighter(idx, dict_data['fighter2_careerstats'], conn)
                    push_fights_upcoming(idx, dict_data, conn)
            if not cs1 or not cs2 or not arr1 or not arr2:
                continue
            push_fighter(idx, cs1, conn)
            push_fighter(idx, cs2, conn)
            for fight in arr1:
                ops = fight.get('ops_careerstats')
                if ops:
                    push_fighter(idx, ops, conn)
                if not fight['winner_loser']['winner'] or not fight['winner_loser']['loser']:
                    fight['winner_loser']['winner'] = 'draw'
                    fight['winner_loser']['loser'] = 'draw'
                push_fights(idx, fight, cs1, conn)
                push_totals(fight, cs1, conn)
                push_rounds(fight, cs1, conn)
            for fight in arr2:
                ops = fight.get('ops_careerstats')
                if ops:
                    push_fighter(idx, ops, conn)
                if not fight['winner_loser']['winner'] or not fight['winner_loser']['loser']:
                    fight['winner_loser']['winner'] = 'draw'
                    fight['winner_loser']['loser'] = 'draw'
                push_fights(idx, fight, cs2, conn)
                push_totals(fight, cs2, conn)
                push_rounds(fight, cs2, conn)

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