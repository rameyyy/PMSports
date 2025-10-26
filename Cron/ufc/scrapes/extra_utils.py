from .sqlpush import create_connection, fetch_query

def filter_new_events(event_links: list[str]) -> list[str]:
    """
    Filter out event links that already exist in the ufc.events table.
    """
    if not event_links:
        return []
    
    placeholders = ', '.join(['%s'] * len(event_links))
    
    # Query to check which URLs already exist
    query = f"""
        SELECT event_url 
        FROM ufc.events 
        WHERE event_url IN ({placeholders})
    """
    
    conn = create_connection()
    existing_events = fetch_query(conn, query, event_links)
    
    # Create a set of existing URLs for O(1) lookup
    existing_urls = {row['event_url'] for row in existing_events}
    
    # Return only the URLs that are NOT in the database
    return [url for url in event_links if url not in existing_urls]

def get_future_event_urls(conn) -> list[str]:
    """
    Get all event URLs for events with dates after today.
    """
    query = """
        SELECT event_url 
        FROM ufc.events 
        WHERE date > CURDATE()
    """
    
    results = fetch_query(conn, query, [])
    
    # Extract just the URLs as a list of strings
    return [row['event_url'] for row in results]

def get_last_two_past_events(conn) -> list[str]:
    """
    Get the event URLs for the last 2 events that have already occurred.
    """
    query = """
        SELECT event_url 
        FROM ufc.events 
        WHERE date < CURDATE()
        ORDER BY date DESC
        LIMIT 4
    """
    
    results = fetch_query(conn, query, [])
    
    # Extract just the URLs as a list of strings
    return [row['event_url'] for row in results]