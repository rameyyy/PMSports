import mysql.connector
import os
from dotenv import load_dotenv

load_dotenv()

def get_db_connection():
    """Create and return a database connection"""
    try:
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
        print(f"❌ Database connection error: {e}")
        return None

def execute_query(query, params=None, fetch_one=False):
    """Execute a query and return results"""
    conn = get_db_connection()
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
        print(f"❌ Query execution error: {err}")
        return None