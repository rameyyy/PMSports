import mysql.connector
from mysql.connector import Error
from .utils import parse_event_date, _normalize_dob, tapology_fighter_profile_link
from dotenv import load_dotenv
import os

# load environment variables from .env
load_dotenv()

def create_connection():
    try:
        conn = mysql.connector.connect(
            host=os.getenv("DB_HOST"),
            port=int(os.getenv("DB_PORT", 3306)),  # default 3306 if not set
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            database=os.getenv("UFC_DB"),
        )
        if conn.is_connected():
            print("✅ Connected to database")
            return conn
    except mysql.connector.Error as e:
        print(f"❌ Error: {e}")
        return None


def run_query(connection, query, params=None):
    """Execute a query with optional parameters"""
    cursor = connection.cursor()
    try:
        cursor.execute(query, params or ())
        connection.commit()
        print("✅ Query executed successfully")
    except Error as e:
        if e.errno == 1062: # duplicate entry, updates keys if needed
            return True
        print(f"❌ Error running query: {e}")
        return False
    finally:
        cursor.close()


def fetch_query(connection, query, params=None):
    """Fetch results from a SELECT query"""
    cursor = connection.cursor(dictionary=True)  # dict rows
    try:
        cursor.execute(query, params or ())
        return cursor.fetchall()
    except Error as e:
        print(f"❌ Error fetching query: {e}")
        return []
    finally:
        cursor.close()


def push_events(dataset):
    conn = create_connection()

    if not conn:
        return False
    
    url = dataset['url']
    event_id = dataset['event_id']
    event_date_str = dataset['date']
    event_date, upcoming = parse_event_date(event_date_str)
    location = dataset['location']
    title = dataset['title']

    cmd = """
        INSERT INTO events (event_id, event_url, title, event_datestr, location, date)
        VALUES (%s, %s, %s, %s, %s, %s)
        """
    params = (event_id, url, title, event_date_str, location, event_date)
    check = run_query(conn, cmd, params)
    if check == False:
        return False
    return True


def push_fighter(idx, careerstats):
    """
    Push one fighter's stats into the fighters table.
    
    Args:
        idx (EventNameIndex): index built from the event JSON
        careerstats (dict): scraped fighter stats
    """
    conn = create_connection()

    if not conn:
        return False
    
    # --- Match fighter name ---
    fname = careerstats.get("fighter_name")
    match, score, fighter_eventset = idx.find(fname, threshold=0.82)
    if match:
        final_name = match
        nickname = fighter_eventset.get("nickname")
        img_link = fighter_eventset.get("img_link")
    else:
        final_name = fname
        print(f"match not found, best score: {score}")
        nickname = None
        img_link = None
    
    # --- Prepare insert payload ---
    fighter_data = {
        "fighter_id": careerstats.get("fighter_id"),
        "name": final_name,
        "nickname": nickname,
        "img_link": img_link,
        "height_in": careerstats.get("height"),
        "weight_lbs": careerstats.get("weight"),
        "reach_in": careerstats.get("reach"),
        "stance": careerstats.get("stance"),
        "dob": _normalize_dob(careerstats.get("dob")),
        "slpm": careerstats.get("slpm"),
        "str_acc": careerstats.get("str_acc"),
        "sapm": careerstats.get("sapm"),
        "str_def": careerstats.get("str_def"),
        "td_avg": careerstats.get("td_avg"),
        "td_acc": careerstats.get("td_acc"),
        "td_def": careerstats.get("td_def"),
        "sub_avg": careerstats.get("sub_avg"),
        "win": careerstats.get("win"),
        "loss": careerstats.get("loss"),
        "draw": careerstats.get("draw"),
    }

    # --- SQL upsert ---
    cmd = """
    INSERT INTO fighters (
        fighter_id, name, nickname, img_link,
        height_in, weight_lbs, reach_in, stance, dob,
        slpm, str_acc, sapm, str_def, td_avg, td_acc, td_def, sub_avg,
        win, loss, draw
    ) VALUES (
        %(fighter_id)s, %(name)s, %(nickname)s, %(img_link)s,
        %(height_in)s, %(weight_lbs)s, %(reach_in)s, %(stance)s, %(dob)s,
        %(slpm)s, %(str_acc)s, %(sapm)s, %(str_def)s, %(td_avg)s, %(td_acc)s, %(td_def)s, %(sub_avg)s,
        %(win)s, %(loss)s, %(draw)s
    )
    ON DUPLICATE KEY UPDATE
        name       = IF(VALUES(name) IS NULL, name, VALUES(name)),
        nickname   = IF(VALUES(nickname) IS NULL, nickname, VALUES(nickname)),
        img_link   = IF(VALUES(img_link) IS NULL, img_link, VALUES(img_link)),
        height_in  = VALUES(height_in),
        weight_lbs = VALUES(weight_lbs),
        reach_in   = VALUES(reach_in),
        stance     = VALUES(stance),
        dob        = VALUES(dob),
        slpm       = VALUES(slpm),
        str_acc    = VALUES(str_acc),
        sapm       = VALUES(sapm),
        str_def    = VALUES(str_def),
        td_avg     = VALUES(td_avg),
        td_acc     = VALUES(td_acc),
        td_def     = VALUES(td_def),
        sub_avg    = VALUES(sub_avg),
        win        = VALUES(win),
        loss       = VALUES(loss),
        draw       = VALUES(draw)
    """
    return run_query(conn, cmd, fighter_data)