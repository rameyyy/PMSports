"""
Daily cron job: compute season-to-date model performance stats and insert into model_performance table.
Runs for a given date (defaults to yesterday). Called daily at 8am.

Usage:
    python update_model_performance.py              # runs for yesterday
    python update_model_performance.py 2026-02-08   # runs for specific date
"""
import csv
import os
import sys
from datetime import datetime, timedelta
from scrapes.sqlconn import create_connection, fetch, execute_query

# Load team name mappings
MAPPINGS_CSV = os.path.join(os.path.dirname(__file__), 'bookmaker', 'team_mappings.csv')
mappings = {}
with open(MAPPINGS_CSV, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row.get('from_odds_team_name') and row.get('my_team_name'):
            mappings[row['from_odds_team_name']] = row['my_team_name']


def run(target_date):
    print(f"[+] Computing model performance for date={target_date} (season-to-date)")

    conn = create_connection()
    if not conn:
        print("[-] Failed to connect")
        return False

    # Completed moneyline games up to target_date
    ml_game_filter = """
        SELECT DISTINCT game_id FROM moneyline
        WHERE season = 2026
          AND winning_team IS NOT NULL
          AND game_date <= %s
    """

    # ============================================================
    # 1. BOOKMAKER ML ACCURACY
    # ============================================================
    ml_query = """
        SELECT o.bookmaker, o.home_team, o.away_team, o.ml_home, o.ml_away,
               g.team_1, g.team_2, g.team_1_score, g.team_2_score
        FROM odds o
        JOIN games g ON o.game_id = g.game_id
        WHERE g.team_1_score IS NOT NULL AND g.team_2_score IS NOT NULL
          AND g.team_1_score != g.team_2_score
          AND o.ml_home IS NOT NULL AND o.ml_away IS NOT NULL
          AND g.season = 2026
          AND o.game_id IN (
              SELECT DISTINCT game_id FROM moneyline
              WHERE season = 2026 AND winning_team IS NOT NULL AND game_date <= %s
          )
    """
    ml_rows = fetch(conn, ml_query, (target_date,))

    book_ml_stats = {}
    for r in ml_rows:
        book = r['bookmaker']
        if book not in book_ml_stats:
            book_ml_stats[book] = {'correct': 0, 'total': 0}

        ml_home = float(r['ml_home'])
        ml_away = float(r['ml_away'])
        home_mapped = mappings.get(r['home_team'], r['home_team'])
        away_mapped = mappings.get(r['away_team'], r['away_team'])
        actual_winner = r['team_1'] if r['team_1_score'] > r['team_2_score'] else r['team_2']

        if ml_home == ml_away:
            book_pick = None
        elif ml_home < ml_away:
            book_pick = home_mapped
        else:
            book_pick = away_mapped

        book_ml_stats[book]['total'] += 1
        if book_pick and book_pick == actual_winner:
            book_ml_stats[book]['correct'] += 1

    # ============================================================
    # 2. ALGOPICKS ML ACCURACY
    # ============================================================
    algopicks_ml_query = """
        SELECT team_1, team_2, gbm_prob_team_1, gbm_prob_team_2, winning_team
        FROM moneyline
        WHERE season = 2026
          AND winning_team IS NOT NULL
          AND gbm_prob_team_1 IS NOT NULL
          AND gbm_prob_team_2 IS NOT NULL
          AND game_date <= %s
    """
    algopicks_ml_rows = fetch(conn, algopicks_ml_query, (target_date,))

    algo_ml_correct = 0
    algo_ml_total = 0
    for r in algopicks_ml_rows:
        p1 = float(r['gbm_prob_team_1'])
        p2 = float(r['gbm_prob_team_2'])
        if p1 == p2:
            predicted = None
        elif p1 > p2:
            predicted = r['team_1']
        else:
            predicted = r['team_2']

        algo_ml_total += 1
        if predicted and predicted == r['winning_team']:
            algo_ml_correct += 1

    # ============================================================
    # 3. BOOKMAKER O/U LINE MAE
    # ============================================================
    ou_query = """
        SELECT o.bookmaker, o.over_point,
               g.team_1_score, g.team_2_score
        FROM odds o
        JOIN games g ON o.game_id = g.game_id
        WHERE g.team_1_score IS NOT NULL AND g.team_2_score IS NOT NULL
          AND o.over_point IS NOT NULL
          AND g.season = 2026
          AND o.game_id IN (
              SELECT DISTINCT game_id FROM moneyline
              WHERE season = 2026 AND winning_team IS NOT NULL AND game_date <= %s
          )
    """
    ou_rows = fetch(conn, ou_query, (target_date,))

    book_ou_mae = {}
    for r in ou_rows:
        book = r['bookmaker']
        if book not in book_ou_mae:
            book_ou_mae[book] = {'total_err': 0.0, 'count': 0}
        actual_total = r['team_1_score'] + r['team_2_score']
        err = abs(float(r['over_point']) - actual_total)
        book_ou_mae[book]['total_err'] += err
        book_ou_mae[book]['count'] += 1

    # ============================================================
    # 4. ALGOPICKS O/U MAE (lgb_pred)
    # ============================================================
    algo_ou_query = """
        SELECT ou.lgb_pred, g.team_1_score, g.team_2_score
        FROM overunder ou
        JOIN games g ON ou.game_id = g.game_id
        WHERE g.team_1_score IS NOT NULL AND g.team_2_score IS NOT NULL
          AND ou.lgb_pred IS NOT NULL
          AND g.season = 2026
          AND ou.game_id IN (
              SELECT DISTINCT game_id FROM moneyline
              WHERE season = 2026 AND winning_team IS NOT NULL AND game_date <= %s
          )
    """
    algo_ou_rows = fetch(conn, algo_ou_query, (target_date,))

    algo_ou_total_err = 0.0
    algo_ou_count = 0
    for r in algo_ou_rows:
        actual_total = r['team_1_score'] + r['team_2_score']
        err = abs(float(r['lgb_pred']) - actual_total)
        algo_ou_total_err += err
        algo_ou_count += 1

    # ============================================================
    # 5. ALGOPICKS O/U ACCURACY PER BOOK (lgb_pred vs book's line)
    # ============================================================
    my_ou_query = """
        SELECT o.bookmaker, o.over_point, ou.lgb_pred,
               g.team_1_score, g.team_2_score
        FROM odds o
        JOIN games g ON o.game_id = g.game_id
        JOIN overunder ou ON o.game_id = ou.game_id
        WHERE g.team_1_score IS NOT NULL AND g.team_2_score IS NOT NULL
          AND o.over_point IS NOT NULL
          AND ou.lgb_pred IS NOT NULL
          AND g.season = 2026
          AND o.game_id IN (
              SELECT DISTINCT game_id FROM moneyline
              WHERE season = 2026 AND winning_team IS NOT NULL AND game_date <= %s
          )
    """
    my_ou_rows = fetch(conn, my_ou_query, (target_date,))

    my_ou_stats = {}
    for r in my_ou_rows:
        book = r['bookmaker']
        if book not in my_ou_stats:
            my_ou_stats[book] = {
                'over_correct': 0, 'over_total': 0,
                'under_correct': 0, 'under_total': 0
            }

        lgb = float(r['lgb_pred'])
        line = float(r['over_point'])
        actual_total = r['team_1_score'] + r['team_2_score']

        if lgb == line:
            continue

        if lgb > line:
            my_ou_stats[book]['over_total'] += 1
            if actual_total > line:
                my_ou_stats[book]['over_correct'] += 1
        else:
            my_ou_stats[book]['under_total'] += 1
            if actual_total < line:
                my_ou_stats[book]['under_correct'] += 1

    # ============================================================
    # INSERT ROWS
    # ============================================================

    # Clear existing rows for this date
    execute_query(connection=conn, query="DELETE FROM model_performance WHERE date = %s", params=(target_date,))

    insert_query = """
        INSERT INTO model_performance
            (date, book, ml_right, ml_total, ou_mae, ou_games,
             ap_ou_right, ap_ou_total, ap_ou_acc, ap_over_acc, ap_under_acc)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """

    all_books = set(book_ml_stats.keys()) | set(book_ou_mae.keys()) | set(my_ou_stats.keys())

    for book in sorted(all_books):
        ml = book_ml_stats.get(book, {'correct': 0, 'total': 0})
        ou = book_ou_mae.get(book, {'total_err': 0, 'count': 0})
        ap = my_ou_stats.get(book, {'over_correct': 0, 'over_total': 0, 'under_correct': 0, 'under_total': 0})

        ou_mae_val = round(ou['total_err'] / ou['count'], 2) if ou['count'] > 0 else None

        ap_total = ap['over_total'] + ap['under_total']
        ap_right = ap['over_correct'] + ap['under_correct']
        ap_ou_acc = round((ap_right / ap_total) * 100, 2) if ap_total > 0 else None
        ap_over_acc = round((ap['over_correct'] / ap['over_total']) * 100, 2) if ap['over_total'] > 0 else None
        ap_under_acc = round((ap['under_correct'] / ap['under_total']) * 100, 2) if ap['under_total'] > 0 else None

        params = (
            target_date, book,
            ml['correct'], ml['total'],
            ou_mae_val, ou['count'],
            ap_right, ap_total,
            ap_ou_acc, ap_over_acc, ap_under_acc
        )
        execute_query(connection=conn, query=insert_query, params=params)

    # AlgoPicks row
    algo_ou_mae_val = round(algo_ou_total_err / algo_ou_count, 2) if algo_ou_count > 0 else None
    algo_params = (
        target_date, 'AlgoPicks',
        algo_ml_correct, algo_ml_total,
        algo_ou_mae_val, algo_ou_count,
        None, None, None, None, None  # ap_* columns are NULL for AlgoPicks
    )
    execute_query(connection=conn, query=insert_query, params=algo_params)

    conn.close()

    # Print summary
    print(f"\n[+] Inserted {len(all_books) + 1} rows into model_performance for {target_date}")
    print(f"    Books: {', '.join(sorted(all_books))}")
    print(f"    AlgoPicks ML: {algo_ml_correct}/{algo_ml_total} ({(algo_ml_correct/algo_ml_total)*100:.2f}%)" if algo_ml_total > 0 else "")
    print(f"    AlgoPicks O/U MAE: {algo_ou_mae_val}")
    return True


if __name__ == "__main__":
    if len(sys.argv) > 1:
        target_date = sys.argv[1]
    else:
        yesterday = datetime.now()
        target_date = yesterday.strftime('%Y-%m-%d')

    run(target_date)
