from scrapes.ufc_mma import get_all_events, get_event_data, get_fighter_data
import time
import pprint

def main():
    total_fights = 0
    total_events = 0
    events_arr = []
    for page_num in range(1, 2):
        events = get_all_events(past=True, page=page_num)
        events_arr.extend(events)
    print(f'Dupes: {len(events_arr)-len(set(events_arr))}')
    events_arr = set(events_arr)
    print(f"Found {len(events_arr)} events")
    # exit()

    for event_url in events_arr:
        data = get_event_data(event_url, getting_old_data=True)
        if data is not None:
            print(f"\n--- {event_url} ---")
            print(f"Title: {data['title']}")
            print(f"Date: {data['date']}")
            print(f"Location: {data['location']}")
            print(f"Fights scraped: {len(data['fights'])}")
            pprint.pprint(data)
            break
            for i in range(0, len(data['fights'])):
                fighters_url = data['fights'][i]['fighter1']['link']
                fname = data['fights'][i]['fighter1']['fighter_name']
                fageatfight = data['fights'][i]['fighter1']['age_at_fight']
                print(f'{fname}: {fageatfight}')
                soup = get_fighter_data(fighters_url)
                print('\n')
            print('Sleeping for 5s...')
            time.sleep(5)
            total_fights += len(data['fights'])
            total_events += 1

    print(f"\nTotal fights scraped: {total_fights}")
    print(f"Total events scraped: {total_events}")

if __name__ == "__main__":
    main()
