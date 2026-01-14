import mysql.connector
import os
from dotenv import load_dotenv

load_dotenv()

def get_db_connection(database='ufc'):
    """
    Create and return a database connection

    Args:
        database (str): Which database to connect to ('ufc' or 'ncaamb')
    """
    try:
        if database == 'ncaamb':
            conn = mysql.connector.connect(
                host=os.getenv("NCAAMB_DB_HOST"),
                port=int(os.getenv("NCAAMB_DB_PORT")),
                user=os.getenv("NCAAMB_DB_USER"),
                password=os.getenv("NCAAMB_DB_PASSWORD"),
                database=os.getenv("NCAAMB_DB")
            )
        else:  # default to UFC
            conn = mysql.connector.connect(
                host=os.getenv("DB_HOST"),
                port=int(os.getenv("DB_PORT")),
                user=os.getenv("DB_USER"),
                password=os.getenv("DB_PASSWORD"),
                database=os.getenv("UFC_DB")
            )

        if conn.is_connected():
            return conn
    except mysql.connector.Error as e:
        print(f"❌ Database connection error ({database}): {e}")
        return None

def execute_query(query, params=None, fetch_one=False, database='ufc'):
    """
    Execute a query and return results

    Args:
        query (str): SQL query to execute
        params (tuple): Query parameters
        fetch_one (bool): Whether to fetch one result or all
        database (str): Which database to query ('ufc' or 'ncaamb')
    """
    conn = get_db_connection(database=database)
    if not conn:
        return None

    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute(query, params or ())

        if fetch_one:
            result = cursor.fetchone()
        else:
            result = cursor.fetchall()

        cursor.close()
        conn.close()
        return result
    except mysql.connector.Error as err:
        print(f"❌ Query execution error ({database}): {err}")
        return None