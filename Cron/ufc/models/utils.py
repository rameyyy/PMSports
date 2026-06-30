import mysql.connector
from mysql.connector import Error
from dotenv import load_dotenv
import os
import time

# load environment variables from .env
load_dotenv()

def create_connection(max_retries: int = 5, base_wait: int = 5):
    """Connect to MySQL, retrying transient network errors (e.g. errno 113,
    'no route to host') with exponential backoff. Raises if all attempts fail
    so callers fail loudly instead of crashing on a None connection."""
    last_err = None
    for attempt in range(max_retries):
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
            last_err = e
            wait = base_wait * (2 ** attempt)
            print(f"❌ DB connect failed (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                print(f"  retrying in {wait}s...")
                time.sleep(wait)
    raise ConnectionError(f"Could not connect to MySQL after {max_retries} attempts: {last_err}")


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
