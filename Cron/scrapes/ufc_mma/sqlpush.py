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
# --- helpers ---------------------------------------------------------------

def _mmss_to_seconds(s):
    """'MM:SS' -> int seconds. Returns None on falsy/invalid."""
    if not s or not isinstance(s, str) or ":" not in s:
        return None
    m, s = s.split(":")
    try:
        return int(m) * 60 + int(s)
    except ValueError:
        return None

# --- fights ----------------------------------------------------------------

def push_fights(dataset, conn):
    """
    Upsert a fight row into `fights`.
    Expects keys like:
      dataset['fight_id'], ['url'], ['date'], ['meta']{method, format, type, ref, time, weight_class}
      dataset['stats']['fighters'] -> [fighter1_name, fighter2_name]
      dataset['winner_loser'] -> {winner, loser}
    """
    if not conn:
        return False

    # parse date -> DATE (you already have parse_event_date)
    fight_date, _ = parse_event_date(dataset.get("date"))

    fighters = (dataset.get("stats", {}).get("fighters") or [None, None])
    f1_name = fighters[0] if len(fighters) > 0 else None
    f2_name = fighters[1] if len(fighters) > 1 else None

    meta = dataset.get("meta", {}) or {}
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
        # fill these if/when you resolve IDs upstream; otherwise None is fine
        "event_id":     dataset.get("event_id"),
        "fighter1_id":  dataset.get("fighter1_id"),
        "fighter2_id":  dataset.get("fighter2_id"),
        "winner_id":    dataset.get("winner_id"),
        "loser_id":     dataset.get("loser_id"),
        "fighter1_name": f1_name,
        "fighter2_name": f2_name,
        "fight_date":    fight_date,
        "fight_link":    dataset.get("url"),
        "method":        meta.get("method"),
        "fight_format":  meta.get("format"),
        "fight_type":    meta.get("type"),
        "referee":       meta.get("ref"),
        "end_time":      meta.get("time"),          # fights.end_time is VARCHAR(50)
        "weight_class":  meta.get("weight_class"),
    }
    return run_query(conn, cmd, params)

# --- fight_totals ----------------------------------------------------------

def push_totals(dataset, conn):
    """
    Upsert per-fighter totals into `fight_totals` (two rows, one per fighter).
    Uses composite PK (fight_id, fighter_id). If fighter_id is unknown, pass None
    (you may later backfill IDs); PK will still work if you have a surrogate or
    you ensure (fight_id, fighter_id) uniqueness.
    """
    if not conn:
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

    rows = []
    for i in range(2):
        rows.append({
            "fight_id": fight_id,
            "fighter_id": (dataset.get("fighter1_id") if i == 0 else dataset.get("fighter2_id")),
            "fighter_name": fighters[i] if i < len(fighters) else None,
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
            fighter_name        = VALUES(fighter_name),
            kd                  = VALUES(kd),
            sig_str_landed      = VALUES(sig_str_landed),
            sig_str_attempts    = VALUES(sig_str_attempts),
            total_str_landed    = VALUES(total_str_landed),
            total_str_attempts  = VALUES(total_str_attempts),
            td_landed           = VALUES(td_landed),
            td_attempts         = VALUES(td_attempts),
            sub_att             = VALUES(sub_att),
            rev                 = VALUES(rev),
            ctrl_time_s         = VALUES(ctrl_time_s),
            head_landed         = VALUES(head_landed),
            head_attempts       = VALUES(head_attempts),
            body_landed         = VALUES(body_landed),
            body_attempts       = VALUES(body_attempts),
            leg_landed          = VALUES(leg_landed),
            leg_attempts        = VALUES(leg_attempts),
            distance_landed     = VALUES(distance_landed),
            distance_attempts   = VALUES(distance_attempts),
            clinch_landed       = VALUES(clinch_landed),
            clinch_attempts     = VALUES(clinch_attempts),
            ground_landed       = VALUES(ground_landed),
            ground_attempts     = VALUES(ground_attempts)
    """

    ok1 = run_query(conn, cmd, rows[0])
    ok2 = run_query(conn, cmd, rows[1])
    return bool(ok1 and ok2)

# --- fight_rounds ----------------------------------------------------------

def push_rounds(dataset, conn):
    """
    Upsert per-fighter per-round rows into `fight_rounds`.
    Expects dataset['stats']['rounds'] mapping (round1..round5).
    """
    if not conn:
        return False

    fight_id = dataset.get("fight_id")
    fighters = dataset.get("stats", {}).get("fighters") or [None, None]
    f1_id = dataset.get("fighter1_id")
    f2_id = dataset.get("fighter2_id")

    rounds = dataset.get("stats", {}).get("rounds") or {}
    # Normalize keys to numeric round numbers
    def _round_no(k):
        # 'round3' -> 3
        try:
            return int("".join(ch for ch in k if ch.isdigit()))
        except ValueError:
            return None

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
            fighter_name        = VALUES(fighter_name),
            kd                  = VALUES(kd),
            sig_str_landed      = VALUES(sig_str_landed),
            sig_str_attempts    = VALUES(sig_str_attempts),
            total_str_landed    = VALUES(total_str_landed),
            total_str_attempts  = VALUES(total_str_attempts),
            td_landed           = VALUES(td_landed),
            td_attempts         = VALUES(td_attempts),
            sub_att             = VALUES(sub_att),
            rev                 = VALUES(rev),
            ctrl_time_s         = VALUES(ctrl_time_s),
            head_landed         = VALUES(head_landed),
            head_attempts       = VALUES(head_attempts),
            body_landed         = VALUES(body_landed),
            body_attempts       = VALUES(body_attempts),
            leg_landed          = VALUES(leg_landed),
            leg_attempts        = VALUES(leg_attempts),
            distance_landed     = VALUES(distance_landed),
            distance_attempts   = VALUES(distance_attempts),
            clinch_landed       = VALUES(clinch_landed),
            clinch_attempts     = VALUES(clinch_attempts),
            ground_landed       = VALUES(ground_landed),
            ground_attempts     = VALUES(ground_attempts)
    """

    ok_all = True
    for rk, rdata in rounds.items():
        rnd = _round_no(rk)
        if not rnd:  # skip malformed round keys
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

        rows = []
        # fighter A (index 0)
        rows.append({
            "fight_id": fight_id,
            "round_number": rnd,
            "fighter_id": f1_id,
            "fighter_name": fighters[0] if len(fighters) > 0 else None,
            "kd": kd[0] if len(kd) > 0 else None,
            "sig_str_landed": sig[0].get("landed") if len(sig) > 0 else None,
            "sig_str_attempts": sig[0].get("attempts") if len(sig) > 0 else None,
            "total_str_landed": total[0].get("landed") if len(total) > 0 else None,
            "total_str_attempts": total[0].get("attempts") if len(total) > 0 else None,
            "td_landed": td[0].get("landed") if len(td) > 0 else None,
            "td_attempts": td[0].get("attempts") if len(td) > 0 else None,
            "sub_att": sub_att[0] if len(sub_att) > 0 else None,
            "rev": rev[0] if len(rev) > 0 else None,
            "ctrl_time_s": _mmss_to_seconds(ctrl[0]) if len(ctrl) > 0 else None,
            "head_landed": head[0].get("landed") if len(head) > 0 else None,
            "head_attempts": head[0].get("attempts") if len(head) > 0 else None,
            "body_landed": body[0].get("landed") if len(body) > 0 else None,
            "body_attempts": body[0].get("attempts") if len(body) > 0 else None,
            "leg_landed": leg[0].get("landed") if len(leg) > 0 else None,
            "leg_attempts": leg[0].get("attempts") if len(leg) > 0 else None,
            "distance_landed": dist[0].get("landed") if len(dist) > 0 else None,
            "distance_attempts": dist[0].get("attempts") if len(dist) > 0 else None,
            "clinch_landed": clinch[0].get("landed") if len(clinch) > 0 else None,
            "clinch_attempts": clinch[0].get("attempts") if len(clinch) > 0 else None,
            "ground_landed": ground[0].get("landed") if len(ground) > 0 else None,
            "ground_attempts": ground[0].get("attempts") if len(ground) > 0 else None,
        })
        # fighter B (index 1)
        rows.append({
            "fight_id": fight_id,
            "round_number": rnd,
            "fighter_id": f2_id,
            "fighter_name": fighters[1] if len(fighters) > 1 else None,
            "kd": kd[1] if len(kd) > 1 else None,
            "sig_str_landed": sig[1].get("landed") if len(sig) > 1 else None,
            "sig_str_attempts": sig[1].get("attempts") if len(sig) > 1 else None,
            "total_str_landed": total[1].get("landed") if len(total) > 1 else None,
            "total_str_attempts": total[1].get("attempts") if len(total) > 1 else None,
            "td_landed": td[1].get("landed") if len(td) > 1 else None,
            "td_attempts": td[1].get("attempts") if len(td) > 1 else None,
            "sub_att": sub_att[1] if len(sub_att) > 1 else None,
            "rev": rev[1] if len(rev) > 1 else None,
            "ctrl_time_s": _mmss_to_seconds(ctrl[1]) if len(ctrl) > 1 else None,
            "head_landed": head[1].get("landed") if len(head) > 1 else None,
            "head_attempts": head[1].get("attempts") if len(head) > 1 else None,
            "body_landed": body[1].get("landed") if len(body) > 1 else None,
            "body_attempts": body[1].get("attempts") if len(body) > 1 else None,
            "leg_landed": leg[1].get("landed") if len(leg) > 1 else None,
            "leg_attempts": leg[1].get("attempts") if len(leg) > 1 else None,
            "distance_landed": dist[1].get("landed") if len(dist) > 1 else None,
            "distance_attempts": dist[1].get("attempts") if len(dist) > 1 else None,
            "clinch_landed": clinch[1].get("landed") if len(clinch) > 1 else None,
            "clinch_attempts": clinch[1].get("attempts") if len(clinch) > 1 else None,
            "ground_landed": ground[1].get("landed") if len(ground) > 1 else None,
            "ground_attempts": ground[1].get("attempts") if len(ground) > 1 else None,
        })

        okA = run_query(conn, cmd, rows[0])
        okB = run_query(conn, cmd, rows[1])
        ok_all = ok_all and bool(okA and okB)

    return ok_all