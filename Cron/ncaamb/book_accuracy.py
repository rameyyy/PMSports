"""Book ML accuracy with 50/50 lines counting as 0.5 correct"""
import csv
import os
from math import ceil
from scrapes.sqlconn import create_connection, fetch

MAPPINGS_CSV = os.path.join(os.path.dirname(__file__), 'bookmaker', 'team_mappings.csv')
mappings = {}
with open(MAPPINGS_CSV, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row.get('from_odds_team_name') and row.get('my_team_name'):
            mappings[row['from_odds_team_name']] = row['my_team_name']

conn = create_connection()
query = """
    SELECT o.bookmaker, o.home_team, o.away_team, o.ml_home, o.ml_away,
           g.team_1, g.team_2, g.team_1_score, g.team_2_score
    FROM odds o
    JOIN games g ON o.game_id = g.game_id
    WHERE g.team_1_score IS NOT NULL AND g.team_2_score IS NOT NULL
      AND g.team_1_score != g.team_2_score
      AND o.ml_home IS NOT NULL AND o.ml_away IS NOT NULL
      AND g.season = 2026
      AND o.game_id IN (SELECT DISTINCT game_id FROM moneyline WHERE season = 2026 AND winning_team IS NOT NULL)
"""
rows = fetch(conn, query)
conn.close()

stats = {}
for r in rows:
    book = r['bookmaker']
    if book not in stats:
        stats[book] = {'correct': 0.0, 'total': 0}

    ml_home = float(r['ml_home'])
    ml_away = float(r['ml_away'])
    home_mapped = mappings.get(r['home_team'], r['home_team'])
    away_mapped = mappings.get(r['away_team'], r['away_team'])
    actual_winner = r['team_1'] if r['team_1_score'] > r['team_2_score'] else r['team_2']

    stats[book]['total'] += 1

    if ml_home == ml_away:
        stats[book]['correct'] += 0.5
    elif ml_home < ml_away:
        if home_mapped == actual_winner:
            stats[book]['correct'] += 1
    else:
        if away_mapped == actual_winner:
            stats[book]['correct'] += 1

sorted_books = sorted(stats.items(), key=lambda x: ceil(x[1]['correct']) / x[1]['total'] if x[1]['total'] > 0 else 0, reverse=True)

print(f"\n{'Bookmaker':<20} {'Correct':>10} {'Total':>8} {'Accuracy':>10}")
print("-" * 52)
for book, s in sorted_books:
    rounded = ceil(s['correct'])
    acc = (rounded / s['total']) * 100 if s['total'] > 0 else 0
    print(f"{book:<20} {s['correct']:>10.1f} -> {rounded:<4} {s['total']:>4} {acc:>9.2f}%")
