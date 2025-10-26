import mysql.connector
from mysql.connector import Error
from .utils import parse_event_date, _normalize_dob, _mmss_to_seconds, _to_sql_date, normalize_name, tapology_fighter_profile_link, ufcstats_fight_details_link
from dotenv import load_dotenv
from .namematch import compare_names
import os

# load environment variables from .env
load_dotenv()

def create_connection():
    try:
        conn = mysql.connector.connect(
            host=os.getenv("DB_HOST"),
            port=int(os.getenv("DB_PORT")),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            database=os.getenv("UFC_DB"),
        )
        if conn.is_connected():
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


def push_events(dataset, conn):
    """
    Insert or update an event row in `events`.
    Never overwrite a non-NULL DB value with NULL input.
    """
    if not conn:
        return False
    
    url = dataset['url']
    event_id = dataset['event_id']
    event_date_str = dataset['date']
    event_date, _ = parse_event_date(event_date_str)
    location = dataset['location']
    title = dataset['title']

    cmd = """
        INSERT INTO events (
            event_id, event_url, title, event_datestr, location, date
        ) VALUES (
            %(event_id)s, %(event_url)s, %(title)s, %(event_datestr)s, %(location)s, %(date)s
        )
        ON DUPLICATE KEY UPDATE
            event_url    = IF(VALUES(event_url)    IS NULL, event_url,    VALUES(event_url)),
            title        = IF(VALUES(title)        IS NULL, title,        VALUES(title)),
            event_datestr= IF(VALUES(event_datestr)IS NULL, event_datestr,VALUES(event_datestr)),
            location     = IF(VALUES(location)     IS NULL, location,     VALUES(location)),
            date         = IF(VALUES(date)         IS NULL, date,         VALUES(date))
    """
    params = {
        "event_id": event_id,
        "event_url": url,
        "title": title,
        "event_datestr": event_date_str,
        "location": location,
        "date": event_date,
    }
    return run_query(conn, cmd, params)


def push_fighter(idx, careerstats, conn):
    """
    Push one fighter's stats into the fighters table.
    Never overwrite a non-NULL DB value with NULL input.
    """
    if not conn:
        return False

    # --- Match fighter name ---
    if not careerstats:
        return False
    
    fname = careerstats.get("fighter_name")
    match, score, fighter_eventset = idx.find(fname, threshold=0.82)
    
    if match:
        final_name = match
        nickname = fighter_eventset.get("nickname")
        img_link = fighter_eventset.get("img_link")
    else:
        final_name = fname
        nickname = None
        img_link = None
    final_name = normalize_name(final_name)

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

    # --- SQL upsert (NULL-safe updates everywhere) ---
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
        name        = IF(VALUES(name)        IS NULL, name,        VALUES(name)),
        nickname    = IF(VALUES(nickname)    IS NULL, nickname,    VALUES(nickname)),
        img_link    = IF(VALUES(img_link)    IS NULL, img_link,    VALUES(img_link)),
        height_in   = IF(VALUES(height_in)   IS NULL, height_in,   VALUES(height_in)),
        weight_lbs  = IF(VALUES(weight_lbs)  IS NULL, weight_lbs,  VALUES(weight_lbs)),
        reach_in    = IF(VALUES(reach_in)    IS NULL, reach_in,    VALUES(reach_in)),
        stance      = IF(VALUES(stance)      IS NULL, stance,      VALUES(stance)),
        dob         = IF(VALUES(dob)         IS NULL, dob,         VALUES(dob)),
        slpm        = IF(VALUES(slpm)        IS NULL, slpm,        VALUES(slpm)),
        str_acc     = IF(VALUES(str_acc)     IS NULL, str_acc,     VALUES(str_acc)),
        sapm        = IF(VALUES(sapm)        IS NULL, sapm,        VALUES(sapm)),
        str_def     = IF(VALUES(str_def)     IS NULL, str_def,     VALUES(str_def)),
        td_avg      = IF(VALUES(td_avg)      IS NULL, td_avg,      VALUES(td_avg)),
        td_acc      = IF(VALUES(td_acc)      IS NULL, td_acc,      VALUES(td_acc)),
        td_def      = IF(VALUES(td_def)      IS NULL, td_def,      VALUES(td_def)),
        sub_avg     = IF(VALUES(sub_avg)     IS NULL, sub_avg,     VALUES(sub_avg)),
        win         = IF(VALUES(win)         IS NULL, win,         VALUES(win)),
        loss        = IF(VALUES(loss)        IS NULL, loss,        VALUES(loss)),
        draw        = IF(VALUES(draw)        IS NULL, draw,        VALUES(draw))
    """
    return run_query(conn, cmd, fighter_data)

###### FUNCS NOT WORKED ON YET ######

def push_fights(idx, dataset, career_stats, conn):
    """
    Upsert a fight row into `fights`.
    Expects keys like:
      dataset['fight_id'], ['url'], ['date'], ['meta']{method, format, type, ref, time, weight_class}
      dataset['stats']['fighters'] -> [fighter1_name, fighter2_name]
      dataset['winner_loser'] -> {winner, loser}
    """
    if not dataset:
        return False
    
    if not conn:
        return False

    fighter1_name = career_stats.get('fighter_name')
    fighter1_id = career_stats.get('fighter_id')
    
    fighter2_name = dataset['ops_careerstats']['fighter_name']
    fighter2_id = dataset['ops_careerstats']['fighter_id']
    
    fname_winner = dataset['winner_loser']['winner']
    fname_loser = dataset['winner_loser']['loser']
    fighter1_name = normalize_name(fighter1_name)
    fighter2_name = normalize_name(fighter2_name)
    fname_winner = normalize_name(fname_winner)
    fname_loser = normalize_name(fname_loser)
    if fighter1_name == fname_winner:
        winner_id = fighter1_id
        loser_id = fighter2_id
    elif fighter2_name == fname_winner:
        winner_id = fighter2_id
        loser_id = fighter1_id
    elif fname_winner == 'draw':
        winner_id = 'drawornc'
        loser_id = 'drawornc'
    else:
        return False #Names dont match fake data lol
    
    fname1_match, score, fighter_eventset_1 = idx.find(fighter1_name, threshold=0.78)
    fname2_match, score, fighter_eventset_2 = idx.find(fighter2_name, threshold=0.78)


    # parse date -> DATE (you already have parse_event_date)
    fight_date = _to_sql_date(dataset.get('date'))
    event_date_real = None
    if fname1_match and fname2_match:
        event_date = parse_event_date(fighter_eventset_1.get('date'))
        if event_date[0] == fight_date:
            event_date_real = fighter_eventset_1.get('event_id')
    meta = dataset.get("meta")
    cmd = """
        INSERT INTO fights (
            fight_id, event_id, fighter1_id, fighter2_id, winner_id, loser_id,
            fighter1_name, fighter2_name, fight_date, fight_link, method,
            fight_format, fight_type, referee, end_time, weight_class
        ) VALUES (
            %(fight_id)s, %(event_id)s, %(fighter1_id)s, %(fighter2_id)s, %(winner_id)s, %(loser_id)s,
            %(fighter1_name)s, %(fighter2_name)s, %(fight_date)s, %(fight_link)s, %(method)s,
            %(fight_format)s, %(fight_type)s, %(referee)s, %(end_time)s, %(weight_class)s
        )
        ON DUPLICATE KEY UPDATE
            event_id      = VALUES(event_id),
            fighter1_id   = VALUES(fighter1_id),
            fighter2_id   = VALUES(fighter2_id),
            winner_id     = VALUES(winner_id),
            loser_id      = VALUES(loser_id),
            fighter1_name = VALUES(fighter1_name),
            fighter2_name = VALUES(fighter2_name),
            fight_date    = VALUES(fight_date),
            fight_link    = VALUES(fight_link),
            method        = VALUES(method),
            fight_format  = VALUES(fight_format),
            fight_type    = VALUES(fight_type),
            referee       = VALUES(referee),
            end_time      = VALUES(end_time),
            weight_class  = VALUES(weight_class)
    """
    params = {
        "fight_id":     dataset.get("fight_id"),
        "event_id":     event_date_real,
        "fighter1_id":  fighter1_id,
        "fighter2_id":  fighter2_id,
        "winner_id":    winner_id,
        "loser_id":     loser_id,
        "fighter1_name": fighter1_name,
        "fighter2_name": fighter2_name,
        "fight_date":    fight_date,
        "fight_link":    dataset.get("url"),
        "method":        meta.get("method"),
        "fight_format":  meta.get("format"),
        "fight_type":    meta.get("type"),
        "referee":       meta.get("ref"),
        "end_time":      meta.get("time"),
        "weight_class":  meta.get("weight_class"),
    }
    return run_query(conn, cmd, params)

def push_fights_upcoming(idx, dataset, conn):
    """
    Upsert a fight row into `fights`.
    Expects keys like:
      dataset['fight_id'], ['url'], ['date'], ['meta']{method, format, type, ref, time, weight_class}
      dataset['stats']['fighters'] -> [fighter1_name, fighter2_name]
      dataset['winner_loser'] -> {winner, loser}
    """
    fname_1 = normalize_name(dataset.get('fighter1_name'))
    fighter2_career_stats = dataset.get('fighter2_careerstats')
    fname_2 = normalize_name(fighter2_career_stats.get('fighter_name'))

    fname1_match, score, fighter_eventset_1 = idx.find(fname_1, threshold=0.78)
    fname2_match, score, fighter_eventset_2 = idx.find(fname_2, threshold=0.78)
    if not fname1_match or not fname2_match:
        return False
    
    if not conn:
        return False
    
    fight_type_pre = fighter_eventset_1.get('fight_card_type').lower()
    if 'main event' in fight_type_pre:
        fight_type = 'main'
    elif 'title' in fight_type_pre:
        fight_type = 'title'
    else: fight_type = None

    if fight_type:
        fight_format = 5
    else:
        fight_format = 3

    weight_class = fighter_eventset_1.get('weight_class')

    # parse date -> DATE (you already have parse_event_date)
    fight_date, _ = parse_event_date(fighter_eventset_1.get("date"))

    cmd = """
        INSERT INTO fights (
            fight_id, event_id, fighter1_id, fighter2_id, winner_id, loser_id,
            fighter1_name, fighter2_name, fight_date, fight_link, method,
            fight_format, fight_type, referee, end_time, weight_class
        ) VALUES (
            %(fight_id)s, %(event_id)s, %(fighter1_id)s, %(fighter2_id)s, %(winner_id)s, %(loser_id)s,
            %(fighter1_name)s, %(fighter2_name)s, %(fight_date)s, %(fight_link)s, %(method)s,
            %(fight_format)s, %(fight_type)s, %(referee)s, %(end_time)s, %(weight_class)s
        )
        ON DUPLICATE KEY UPDATE
            event_id      = IF(VALUES(event_id) IS NULL, event_id, VALUES(event_id)),
            fighter1_id   = IF(VALUES(fighter1_id) IS NULL, fighter1_id, VALUES(fighter1_id)),
            fighter2_id   = IF(VALUES(fighter2_id) IS NULL, fighter2_id, VALUES(fighter2_id)),
            winner_id     = IF(VALUES(winner_id) IS NULL, winner_id, VALUES(winner_id)),
            loser_id      = IF(VALUES(loser_id) IS NULL, loser_id, VALUES(loser_id)),
            fighter1_name = IF(VALUES(fighter1_name) IS NULL, fighter1_name, VALUES(fighter1_name)),
            fighter2_name = IF(VALUES(fighter2_name) IS NULL, fighter2_name, VALUES(fighter2_name)),
            fight_date    = IF(VALUES(fight_date) IS NULL, fight_date, VALUES(fight_date)),
            fight_link    = IF(VALUES(fight_link) IS NULL, fight_link, VALUES(fight_link)),
            method        = IF(VALUES(method) IS NULL, method, VALUES(method)),
            fight_format  = IF(VALUES(fight_format) IS NULL, fight_format, VALUES(fight_format)),
            fight_type    = IF(VALUES(fight_type) IS NULL, fight_type, VALUES(fight_type)),
            referee       = IF(VALUES(referee) IS NULL, referee, VALUES(referee)),
            end_time      = IF(VALUES(end_time) IS NULL, end_time, VALUES(end_time)),
            weight_class  = IF(VALUES(weight_class) IS NULL, weight_class, VALUES(weight_class))
        """
    params = {
        "fight_id":     dataset.get("fight_id"),
        "event_id":     fighter_eventset_1.get("event_id"),
        "fighter1_id":  dataset.get("fighter1_id"),
        "fighter2_id":  fighter2_career_stats.get("fighter_id"),
        "winner_id":    None,
        "loser_id":     None,
        "fighter1_name": fname_1,
        "fighter2_name": fname_2,
        "fight_date":    fight_date,
        "fight_link":    f'{ufcstats_fight_details_link}{dataset.get("fight_id")}',
        "method":        None,
        "fight_format":  fight_format,
        "fight_type":    fight_type,
        "referee":       None,
        "end_time":      None,          # fights.end_time is VARCHAR(50)
        "weight_class":  weight_class,
    }
    return run_query(conn, cmd, params)

# --- fight_totals ----------------------------------------------------------

def push_totals(dataset, careerstats, conn):
    """
    Upsert per-fighter totals into `fight_totals` (two rows, one per fighter).
    Uses composite PK (fight_id, fighter_id). If fighter_id is unknown, pass None
    (you may later backfill IDs); PK will still work if you have a surrogate or
    you ensure (fight_id, fighter_id) uniqueness.
    """
    if not conn:
        return False
    if not dataset:
        return False
    if not careerstats:
        return False

    fight_id = dataset.get("fight_id")
    fighters = dataset.get("stats", {}).get("fighters") or [None, None]
    totals = dataset.get("stats", {}).get("totals_total") or {}
    sig = totals.get("sig_str") or [{}, {}]
    total = totals.get("total_str") or [{}, {}]
    td = totals.get("td") or [{}, {}]
    sub_att = totals.get("sub_att") or [None, None]
    rev = totals.get("rev") or [None, None]
    ctrl = totals.get("ctrl") or [None, None]
    brk = totals.get("sig_breakdown") or {}
    head = brk.get("head") or [{}, {}]
    body = brk.get("body") or [{}, {}]
    leg = brk.get("leg") or [{}, {}]
    dist = brk.get("distance") or [{}, {}]
    clinch = brk.get("clinch") or [{}, {}]
    ground = brk.get("ground") or [{}, {}]
    kd = totals.get("kd") or [None, None]
    f1_name = normalize_name(dataset['ops_careerstats']['fighter_name'])
    f1_id = dataset['ops_careerstats']['fighter_id']
    f2_name = normalize_name(careerstats.get('fighter_name'))
    f2_id = careerstats.get('fighter_id')
    rows = []
    for i in range(2):
        fname = normalize_name(fighters[i])
        if compare_names(fname, f1_name) > .88:
            fid = f1_id
            fighter_nameZ = f1_name
        elif compare_names(fname, f2_name) > .88:
            fid = f2_id
            fighter_nameZ = f2_name
        else:
            print('no name to fid match')
            print(f'{fname} != {f1_name}, {f2_name}')
            return False # No real data
        rows.append({
            "fight_id": fight_id,
            "fighter_id": fid,
            "fighter_name": fighter_nameZ,
            "kd": kd[i] if i < len(kd) else None,
            "sig_str_landed": sig[i].get("landed") if i < len(sig) else None,
            "sig_str_attempts": sig[i].get("attempts") if i < len(sig) else None,
            "total_str_landed": total[i].get("landed") if i < len(total) else None,
            "total_str_attempts": total[i].get("attempts") if i < len(total) else None,
            "td_landed": td[i].get("landed") if i < len(td) else None,
            "td_attempts": td[i].get("attempts") if i < len(td) else None,
            "sub_att": sub_att[i] if i < len(sub_att) else None,
            "rev": rev[i] if i < len(rev) else None,
            "ctrl_time_s": _mmss_to_seconds(ctrl[i]) if i < len(ctrl) else None,
            "head_landed": head[i].get("landed") if i < len(head) else None,
            "head_attempts": head[i].get("attempts") if i < len(head) else None,
            "body_landed": body[i].get("landed") if i < len(body) else None,
            "body_attempts": body[i].get("attempts") if i < len(body) else None,
            "leg_landed": leg[i].get("landed") if i < len(leg) else None,
            "leg_attempts": leg[i].get("attempts") if i < len(leg) else None,
            "distance_landed": dist[i].get("landed") if i < len(dist) else None,
            "distance_attempts": dist[i].get("attempts") if i < len(dist) else None,
            "clinch_landed": clinch[i].get("landed") if i < len(clinch) else None,
            "clinch_attempts": clinch[i].get("attempts") if i < len(clinch) else None,
            "ground_landed": ground[i].get("landed") if i < len(ground) else None,
            "ground_attempts": ground[i].get("attempts") if i < len(ground) else None,
        })

    cmd = """
        INSERT INTO fight_totals (
            fight_id, fighter_id, fighter_name, kd,
            sig_str_landed, sig_str_attempts,
            total_str_landed, total_str_attempts,
            td_landed, td_attempts, sub_att, rev, ctrl_time_s,
            head_landed, head_attempts, body_landed, body_attempts,
            leg_landed, leg_attempts, distance_landed, distance_attempts,
            clinch_landed, clinch_attempts, ground_landed, ground_attempts
        ) VALUES (
            %(fight_id)s, %(fighter_id)s, %(fighter_name)s, %(kd)s,
            %(sig_str_landed)s, %(sig_str_attempts)s,
            %(total_str_landed)s, %(total_str_attempts)s,
            %(td_landed)s, %(td_attempts)s, %(sub_att)s, %(rev)s, %(ctrl_time_s)s,
            %(head_landed)s, %(head_attempts)s, %(body_landed)s, %(body_attempts)s,
            %(leg_landed)s, %(leg_attempts)s, %(distance_landed)s, %(distance_attempts)s,
            %(clinch_landed)s, %(clinch_attempts)s, %(ground_landed)s, %(ground_attempts)s
        )
        ON DUPLICATE KEY UPDATE
            fighter_name        = IF(VALUES(fighter_name) IS NULL, fighter_name, VALUES(fighter_name)),
            kd                  = IF(VALUES(kd) IS NULL, kd, VALUES(kd)),
            sig_str_landed      = IF(VALUES(sig_str_landed) IS NULL, sig_str_landed, VALUES(sig_str_landed)),
            sig_str_attempts    = IF(VALUES(sig_str_attempts) IS NULL, sig_str_attempts, VALUES(sig_str_attempts)),
            total_str_landed    = IF(VALUES(total_str_landed) IS NULL, total_str_landed, VALUES(total_str_landed)),
            total_str_attempts  = IF(VALUES(total_str_attempts) IS NULL, total_str_attempts, VALUES(total_str_attempts)),
            td_landed           = IF(VALUES(td_landed) IS NULL, td_landed, VALUES(td_landed)),
            td_attempts         = IF(VALUES(td_attempts) IS NULL, td_attempts, VALUES(td_attempts)),
            sub_att             = IF(VALUES(sub_att) IS NULL, sub_att, VALUES(sub_att)),
            rev                 = IF(VALUES(rev) IS NULL, rev, VALUES(rev)),
            ctrl_time_s         = IF(VALUES(ctrl_time_s) IS NULL, ctrl_time_s, VALUES(ctrl_time_s)),
            head_landed         = IF(VALUES(head_landed) IS NULL, head_landed, VALUES(head_landed)),
            head_attempts       = IF(VALUES(head_attempts) IS NULL, head_attempts, VALUES(head_attempts)),
            body_landed         = IF(VALUES(body_landed) IS NULL, body_landed, VALUES(body_landed)),
            body_attempts       = IF(VALUES(body_attempts) IS NULL, body_attempts, VALUES(body_attempts)),
            leg_landed          = IF(VALUES(leg_landed) IS NULL, leg_landed, VALUES(leg_landed)),
            leg_attempts        = IF(VALUES(leg_attempts) IS NULL, leg_attempts, VALUES(leg_attempts)),
            distance_landed     = IF(VALUES(distance_landed) IS NULL, distance_landed, VALUES(distance_landed)),
            distance_attempts   = IF(VALUES(distance_attempts) IS NULL, distance_attempts, VALUES(distance_attempts)),
            clinch_landed       = IF(VALUES(clinch_landed) IS NULL, clinch_landed, VALUES(clinch_landed)),
            clinch_attempts     = IF(VALUES(clinch_attempts) IS NULL, clinch_attempts, VALUES(clinch_attempts)),
            ground_landed       = IF(VALUES(ground_landed) IS NULL, ground_landed, VALUES(ground_landed)),
            ground_attempts     = IF(VALUES(ground_attempts) IS NULL, ground_attempts, VALUES(ground_attempts))
        """


    ok1 = run_query(conn, cmd, rows[0])
    ok2 = run_query(conn, cmd, rows[1])
    return bool(ok1 and ok2)

# --- fight_rounds ----------------------------------------------------------

def push_rounds(dataset, careerstats, conn):
    """
    Upsert per-fighter per-round rows into `fight_rounds`.
    Expects dataset['stats']['rounds'] mapping (round1..round5).
    Uses composite PK (fight_id, round_number, fighter_id).
    Skips inserting rows where all stat fields are NULL.
    """
    if not conn:
        return False
    if not dataset:
        return False
    if not careerstats:
        return False

    fight_id = dataset.get("fight_id")
    fighters = dataset.get("stats", {}).get("fighters") or [None, None]

    # fighter1 info from ops_careerstats, fighter2 info from careerstats
    f1_name = normalize_name(dataset['ops_careerstats']['fighter_name'])
    f1_id = dataset['ops_careerstats']['fighter_id']
    f2_name = normalize_name(careerstats.get('fighter_name'))
    f2_id = careerstats.get('fighter_id')

    rounds = dataset.get("stats", {}).get("rounds") or {}

    def _round_no(k: str):
        """convert 'round3' -> 3"""
        try:
            return int("".join(ch for ch in k if ch.isdigit()))
        except ValueError:
            return None

    def _is_empty_round(row: dict) -> bool:
        """Return True if all stat fields are None (ignoring PKs and fighter_name)."""
        ignore = {"fight_id", "round_number", "fighter_id", "fighter_name"}
        return all(v is None for k, v in row.items() if k not in ignore)


    cmd = """
        INSERT INTO fight_rounds (
            fight_id, round_number, fighter_id, fighter_name,
            kd, sig_str_landed, sig_str_attempts,
            total_str_landed, total_str_attempts,
            td_landed, td_attempts, sub_att, rev, ctrl_time_s,
            head_landed, head_attempts, body_landed, body_attempts,
            leg_landed, leg_attempts, distance_landed, distance_attempts,
            clinch_landed, clinch_attempts, ground_landed, ground_attempts
        ) VALUES (
            %(fight_id)s, %(round_number)s, %(fighter_id)s, %(fighter_name)s,
            %(kd)s, %(sig_str_landed)s, %(sig_str_attempts)s,
            %(total_str_landed)s, %(total_str_attempts)s,
            %(td_landed)s, %(td_attempts)s, %(sub_att)s, %(rev)s, %(ctrl_time_s)s,
            %(head_landed)s, %(head_attempts)s, %(body_landed)s, %(body_attempts)s,
            %(leg_landed)s, %(leg_attempts)s, %(distance_landed)s, %(distance_attempts)s,
            %(clinch_landed)s, %(clinch_attempts)s, %(ground_landed)s, %(ground_attempts)s
        )
        ON DUPLICATE KEY UPDATE
            fighter_name        = IF(VALUES(fighter_name) IS NULL, fighter_name, VALUES(fighter_name)),
            kd                  = IF(VALUES(kd) IS NULL, kd, VALUES(kd)),
            sig_str_landed      = IF(VALUES(sig_str_landed) IS NULL, sig_str_landed, VALUES(sig_str_landed)),
            sig_str_attempts    = IF(VALUES(sig_str_attempts) IS NULL, sig_str_attempts, VALUES(sig_str_attempts)),
            total_str_landed    = IF(VALUES(total_str_landed) IS NULL, total_str_landed, VALUES(total_str_landed)),
            total_str_attempts  = IF(VALUES(total_str_attempts) IS NULL, total_str_attempts, VALUES(total_str_attempts)),
            td_landed           = IF(VALUES(td_landed) IS NULL, td_landed, VALUES(td_landed)),
            td_attempts         = IF(VALUES(td_attempts) IS NULL, td_attempts, VALUES(td_attempts)),
            sub_att             = IF(VALUES(sub_att) IS NULL, sub_att, VALUES(sub_att)),
            rev                 = IF(VALUES(rev) IS NULL, rev, VALUES(rev)),
            ctrl_time_s         = IF(VALUES(ctrl_time_s) IS NULL, ctrl_time_s, VALUES(ctrl_time_s)),
            head_landed         = IF(VALUES(head_landed) IS NULL, head_landed, VALUES(head_landed)),
            head_attempts       = IF(VALUES(head_attempts) IS NULL, head_attempts, VALUES(head_attempts)),
            body_landed         = IF(VALUES(body_landed) IS NULL, body_landed, VALUES(body_landed)),
            body_attempts       = IF(VALUES(body_attempts) IS NULL, body_attempts, VALUES(body_attempts)),
            leg_landed          = IF(VALUES(leg_landed) IS NULL, leg_landed, VALUES(leg_landed)),
            leg_attempts        = IF(VALUES(leg_attempts) IS NULL, leg_attempts, VALUES(leg_attempts)),
            distance_landed     = IF(VALUES(distance_landed) IS NULL, distance_landed, VALUES(distance_landed)),
            distance_attempts   = IF(VALUES(distance_attempts) IS NULL, distance_attempts, VALUES(distance_attempts)),
            clinch_landed       = IF(VALUES(clinch_landed) IS NULL, clinch_landed, VALUES(clinch_landed)),
            clinch_attempts     = IF(VALUES(clinch_attempts) IS NULL, clinch_attempts, VALUES(clinch_attempts)),
            ground_landed       = IF(VALUES(ground_landed) IS NULL, ground_landed, VALUES(ground_landed)),
            ground_attempts     = IF(VALUES(ground_attempts) IS NULL, ground_attempts, VALUES(ground_attempts))
    """

    ok_all = True
    for rk, rdata in rounds.items():
        rnd = _round_no(rk)
        if not rnd:
            continue

        kd = rdata.get("kd") or [None, None]
        sig = rdata.get("sig_str") or [{}, {}]
        total = rdata.get("total_str") or [{}, {}]
        td = rdata.get("td") or [{}, {}]
        sub_att = rdata.get("sub_att") or [None, None]
        rev = rdata.get("rev") or [None, None]
        ctrl = rdata.get("ctrl") or [None, None]
        brk = rdata.get("sig_breakdown") or {}
        head = brk.get("head") or [{}, {}]
        body = brk.get("body") or [{}, {}]
        leg = brk.get("leg") or [{}, {}]
        dist = brk.get("distance") or [{}, {}]
        clinch = brk.get("clinch") or [{}, {}]
        ground = brk.get("ground") or [{}, {}]

        for i, raw_name in enumerate(fighters):
            fname = normalize_name(raw_name)
            if compare_names(fname, f1_name) > .88:
                fid = f1_id
                fighter_nameZ = f1_name
            elif compare_names(fname, f2_name) > .88:
                fid = f2_id
                fighter_nameZ = f2_name
            else:
                print(f"no name→id match for {fname} != {f1_name}, {f2_name}")
                return False

            row = {
                "fight_id": fight_id,
                "round_number": rnd,
                "fighter_id": fid,
                "fighter_name": fighter_nameZ,
                "kd": kd[i] if i < len(kd) else None,
                "sig_str_landed": sig[i].get("landed") if i < len(sig) else None,
                "sig_str_attempts": sig[i].get("attempts") if i < len(sig) else None,
                "total_str_landed": total[i].get("landed") if i < len(total) else None,
                "total_str_attempts": total[i].get("attempts") if i < len(total) else None,
                "td_landed": td[i].get("landed") if i < len(td) else None,
                "td_attempts": td[i].get("attempts") if i < len(td) else None,
                "sub_att": sub_att[i] if i < len(sub_att) else None,
                "rev": rev[i] if i < len(rev) else None,
                "ctrl_time_s": _mmss_to_seconds(ctrl[i]) if i < len(ctrl) else None,
                "head_landed": head[i].get("landed") if i < len(head) else None,
                "head_attempts": head[i].get("attempts") if i < len(head) else None,
                "body_landed": body[i].get("landed") if i < len(body) else None,
                "body_attempts": body[i].get("attempts") if i < len(body) else None,
                "leg_landed": leg[i].get("landed") if i < len(leg) else None,
                "leg_attempts": leg[i].get("attempts") if i < len(leg) else None,
                "distance_landed": dist[i].get("landed") if i < len(dist) else None,
                "distance_attempts": dist[i].get("attempts") if i < len(dist) else None,
                "clinch_landed": clinch[i].get("landed") if i < len(clinch) else None,
                "clinch_attempts": clinch[i].get("attempts") if i < len(clinch) else None,
                "ground_landed": ground[i].get("landed") if i < len(ground) else None,
                "ground_attempts": ground[i].get("attempts") if i < len(ground) else None,
            }

            if not _is_empty_round(row):
                ok = run_query(conn, cmd, row)
                ok_all = ok_all and bool(ok)

    return ok_all