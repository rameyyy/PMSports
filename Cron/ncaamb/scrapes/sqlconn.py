import mysql.connector
from mysql.connector import Error
from dotenv import load_dotenv
import os
import pandas as pd
from sqlalchemy import create_engine
from typing import Optional, Union

# Load environment variables from ncaamb/.env
ncaamb_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
env_path = os.path.join(ncaamb_dir, '.env')
load_dotenv(env_path)

def create_connection():
    """Create and return a MySQL database connection"""
    try:
        conn = mysql.connector.connect(
            host=os.getenv("DB_HOST"),
            port=int(os.getenv("DB_PORT")),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            database=os.getenv("NCAAMB_DB"),
        )
        if conn.is_connected():
            return conn
    except mysql.connector.Error as e:
        print(f"❌ Error: {e}")
        return None


def fetch(connection, query, params=None):
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


def execute_query(connection=None, query: Optional[str] = None, params=None,
                  df: Optional[pd.DataFrame] = None, table_name: Optional[str] = None,
                  if_exists: str = 'append'):
    """
    Execute a query or insert a pandas DataFrame into the database.

    Args:
        connection: MySQL connection object (required if using query, optional if using df)
        query: SQL query string to execute
        params: Parameters for the query
        df: pandas DataFrame to insert
        table_name: Target table name (required if df is provided)
        if_exists: How to behave if table exists ('fail', 'replace', 'append')

    Returns:
        bool: True if successful, False otherwise
    """
    # If DataFrame is provided, use SQLAlchemy to push data
    if df is not None:
        if table_name is None:
            print("❌ Error: table_name must be provided when using DataFrame")
            return False

        try:
            # Create SQLAlchemy engine
            db_host = os.getenv("DB_HOST")
            db_port = os.getenv("DB_PORT")
            db_user = os.getenv("DB_USER")
            db_password = os.getenv("DB_PASSWORD")
            db_name = os.getenv("NCAAMB_DB")

            engine = create_engine(
                f"mysql+mysqlconnector://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
            )

            # Push DataFrame to database
            df.to_sql(name=table_name, con=engine, if_exists=if_exists, index=False)
            engine.dispose()
            print(f"✅ Successfully inserted {len(df)} rows into {table_name}")
            return True

        except Exception as e:
            print(f"❌ Error inserting DataFrame: {e}")
            return False

    # Otherwise, execute the query using the connection
    if query is None or connection is None:
        print("❌ Error: query and connection must be provided if not using DataFrame")
        return False

    cursor = connection.cursor()
    try:
        cursor.execute(query, params or ())
        connection.commit()
        return True
    except Error as e:
        if e.errno == 1062:  # duplicate entry
            return True
        print(f"❌ Error running query: {e}")
        return False
    finally:
        cursor.close()
