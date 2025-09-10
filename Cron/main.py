from scrapes.ufc_mma import get_all_events, get_event_data
import pprint

def main():
    total_fights = 0
    total_events = 0
    events_arr = []
    for page_num in range(1,4):
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
            # pprint.pprint(data)
            total_fights += len(data['fights'])
            total_events += 1

    print(f"\nTotal fights scraped: {total_fights}")
    print(f"Total events scraped: {total_events}")

if __name__ == "__main__":
    main()
