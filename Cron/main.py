from scrapes.ufc_mma import get_all_events, get_event_data, get_fighter_data, push_events, EventNameIndex, push_fighter
import time

def main():
    total_fights = 0
    total_events = 0
    events_arr = []
    for page_num in range(1, 2):
        events = get_all_events(past=False, page=page_num)
        events_arr.extend(events)
    #print(f'Dupes: {len(events_arr)-len(set(events_arr))}')
    events_arr = set(events_arr)
    print(f"Found {len(events_arr)} events")
    # exit()
    for event_url in events_arr:
        data = get_event_data(event_url, getting_old_data=False)
        idx = EventNameIndex(data)
        # with open("event.json", "w") as f:
        #     json.dump(data, f, indent=4)  # indent=4 makes it pretty
        
        check = push_events(data)
        if check:
            print('worked')
        else:
            print('did not work')
        if data is not None:
            print(f"--- {event_url} ---\n")
            print(f"Title: {data['title']}")
            for i in range(0, len(data['fights'])):
                fighter1s_url = data['fights'][i]['fighter1']['link']
                fighter2s_url = data['fights'][i]['fighter2']['link']
                if not fighter1s_url or not fighter2s_url: # this could be there first fight so they wont have data
                    continue
                fighter1, fighter2 = data['fights'][i]['fighter1']['fighter_name'], data['fights'][i]['fighter2']['fighter_name']
                print(f'Fight {i+1}: {fighter1} vs {fighter2}')
                fighter1_career_stats, fights_arr1, upcoming1 = get_fighter_data(fighter1s_url, fighter1)
                fighter2_career_stats, fights_arr2, upcoming2 = get_fighter_data(fighter2s_url, fighter2)
                if fighter1_career_stats == None: # fighter has not faught for ufc yet (debut)
                    continue
                push_fighter(idx, fighter1_career_stats)
                push_fighter(idx, fighter2_career_stats)
                for i in fights_arr1:
                    random_career_stats = i.get('ops_careerstats')
                    if random_career_stats: push_fighter(idx, random_career_stats)
                for i in fights_arr2:
                    random_career_stats = i.get('ops_careerstats')
                    if random_career_stats: push_fighter(idx, random_career_stats)
                
                total_fights += (len(fights_arr1) + len(fights_arr2))
            total_events += 1

    print(f"\nTotal fights scraped: {total_fights}")
    print(f"Total events scraped: {total_events}")

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    runtime = end_time - start_time
    print(f"Runtime: {runtime:.2f} seconds")