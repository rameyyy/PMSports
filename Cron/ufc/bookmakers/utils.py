import mysql.connector
from mysql.connector import Error
from dotenv import load_dotenv
import os
from typing import List

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
        
def get_model_accuracies_batched(model: str, probs: List[float], window: float = 0.01) -> dict:
    """
    Get accuracies for multiple probability values in one query.
    Returns dict: {prob: {"correct": int, "prob_range": int}}
    """
    # Build CASE statements for each probability band
    cases = []
    for i, prob in enumerate(probs):
        lower = prob - window
        upper = prob + window
        cases.append(f"""
            SUM(CASE 
                WHEN {model}_f1_prob BETWEEN {lower} AND {upper} 
                     AND {model}_correct IS NOT NULL 
                THEN 1 ELSE 0 
            END) AS band_{i}_total,
            SUM(CASE 
                WHEN {model}_f1_prob BETWEEN {lower} AND {upper} 
                     AND {model}_correct = 1 
                THEN 1 ELSE 0 
            END) AS band_{i}_correct
        """)
    
    query = f"""
    SELECT {','.join(cases)}
    FROM ufc.predictions
    WHERE {model}_f1_prob IS NOT NULL;
    """
    
    conn = create_connection()
    df = fetch_query(conn, query)
    
    if not df:
        return {prob: {"correct": 0, "prob_range": 0} for prob in probs}
    
    result = {}
    for i, prob in enumerate(probs):
        total = int(df[0].get(f"band_{i}_total", 0))
        correct = int(df[0].get(f"band_{i}_correct", 0))
        result[prob] = {"correct": correct, "prob_range": total}
    
    return result