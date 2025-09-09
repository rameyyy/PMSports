from scrapes.ufc_mma import get_all_events, get_event_data

def main():
    total_fights = 0
    events = get_all_events(past=True)
    print(f"Found {len(events)} events")

    for event_url in events:
        print(f"\n--- {event_url} ---")
        data = get_event_data(event_url)

        print(f"Title: {data['title']}")
        print(f"Date: {data['date']}")
        print(f"Location: {data['location']}")
        print(f"Fights scraped: {len(data['fights'])}")
        total_fights += len(data['fights'])

    print(f"\nTotal fights scraped: {total_fights}")

if __name__ == "__main__":
    main()
