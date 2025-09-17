import json
import mysql.connector
from mysql.connector import Error

def load_config():
    """Load database configuration from config.json"""
    try:
        with open('config.json', 'r') as file:
            config = json.load(file)
            return config['mysql']
    except FileNotFoundError:
        print("Error: config.json file not found")
        return None
    except KeyError:
        print("Error: 'mysql' configuration not found in config.json")
        return None
    except json.JSONDecodeError:
        print("Error: Invalid JSON format in config.json")
        return None

def connect_to_mysql():
    """Connect to MySQL database using configuration from config.json"""
    import pdb; pdb.set_trace()
    config = load_config()
    if not config:
        return None

    try:
        connection = mysql.connector.connect(
            host=config['host'],
            port=config['port'],
            database=config['database'],
            user=config['username'],
            password=config['password']
        )

        if connection.is_connected():
            db_info = connection.get_server_info()
            print(f"Successfully connected to MySQL Server version {db_info}")

            cursor = connection.cursor()
            cursor.execute("SELECT DATABASE();")
            database_name = cursor.fetchone()
            print(f"Connected to database: {database_name[0]}")

            return connection

    except Error as e:
        print(f"Error while connecting to MySQL: {e}")
        return None

def main():
    """Main function to test the database connection"""
    connection = connect_to_mysql()

    if connection:
        # Example query
        try:
            cursor = connection.cursor()
            cursor.execute("SHOW TABLES;")
            tables = cursor.fetchall()
            print(f"Tables in database: {[table[0] for table in tables]}")

        except Error as e:
            print(f"Error executing query: {e}")

        finally:
            if connection.is_connected():
                cursor.close()
                connection.close()
                print("MySQL connection is closed")

if __name__ == "__main__":
    main()