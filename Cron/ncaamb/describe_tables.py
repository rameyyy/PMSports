#!/usr/bin/env python3
"""
Describe all tables in the ncaamb database
"""
import mysql.connector
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv('.env')

def describe_table(cursor, table_name):
    """Describe a table's schema and show sample data"""
    print("\n" + "="*100)
    print(f"TABLE: {table_name}")
    print("="*100)

    # Get column info
    cursor.execute(f"DESCRIBE {table_name}")
    columns = cursor.fetchall()

    print("\nCOLUMNS:")
    print(f"{'Field':<30} {'Type':<20} {'Null':<5} {'Key':<5} {'Default':<15} {'Extra':<15}")
    print("-"*100)
    for col in columns:
        print(f"{str(col[0]):<30} {str(col[1]):<20} {str(col[2]):<5} {str(col[3]):<5} {str(col[4]):<15} {str(col[5]):<15}")

    # Get row count
    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
    count = cursor.fetchone()[0]
    print(f"\nTOTAL ROWS: {count:,}")

    # Show sample data (first 3 rows)
    if count > 0:
        cursor.execute(f"SELECT * FROM {table_name} LIMIT 3")
        sample_rows = cursor.fetchall()
        column_names = [desc[0] for desc in cursor.description]

        print(f"\nSAMPLE DATA (first 3 rows):")
        print("-"*100)
        for i, row in enumerate(sample_rows, 1):
            print(f"\nRow {i}:")
            for col_name, value in zip(column_names, row):
                # Truncate long values
                str_value = str(value)
                if len(str_value) > 50:
                    str_value = str_value[:47] + "..."
                print(f"  {col_name:<30} {str_value}")

try:
    conn = mysql.connector.connect(
        host=os.getenv("DB_HOST"),
        port=int(os.getenv("DB_PORT")),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("NCAAMB_DB"),
    )

    cursor = conn.cursor()

    tables = ['games', 'leaderboard', 'player_stats', 'teams']

    for table in tables:
        describe_table(cursor, table)

    cursor.close()
    conn.close()

    print("\n" + "="*100)
    print("DESCRIPTION COMPLETE")
    print("="*100)

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
