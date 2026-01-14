"""
Database Schema Documentation Script

This script connects to the ncaamb database and prints detailed information
about every table, including:
- Column names, types, nullability, keys, and defaults
- Indexes
- Foreign key relationships

Use this to help Claude understand the database structure.
"""

import mysql.connector
from dotenv import load_dotenv
import os

# Load environment variables from ncaamb/.env
ncaamb_dir = os.path.dirname(os.path.abspath(__file__))
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
        print(f"Error: {e}")
        return None


def get_all_tables(conn):
    """Get list of all tables in the database"""
    cursor = conn.cursor()
    cursor.execute("SHOW TABLES")
    tables = [table[0] for table in cursor.fetchall()]
    cursor.close()
    return tables


def describe_table(conn, table_name):
    """Get detailed information about a table's structure"""
    cursor = conn.cursor(dictionary=True)

    # Get column information
    cursor.execute(f"DESCRIBE {table_name}")
    columns = cursor.fetchall()

    # Get table indexes
    cursor.execute(f"SHOW INDEX FROM {table_name}")
    indexes = cursor.fetchall()

    # Get foreign key information
    cursor.execute(f"""
        SELECT
            COLUMN_NAME,
            REFERENCED_TABLE_NAME,
            REFERENCED_COLUMN_NAME
        FROM
            INFORMATION_SCHEMA.KEY_COLUMN_USAGE
        WHERE
            TABLE_SCHEMA = %s
            AND TABLE_NAME = %s
            AND REFERENCED_TABLE_NAME IS NOT NULL
    """, (os.getenv("NCAAMB_DB"), table_name))
    foreign_keys = cursor.fetchall()

    # Get row count
    cursor.execute(f"SELECT COUNT(*) as count FROM {table_name}")
    row_count = cursor.fetchone()['count']

    cursor.close()

    return {
        'columns': columns,
        'indexes': indexes,
        'foreign_keys': foreign_keys,
        'row_count': row_count
    }


def format_column_info(columns):
    """Format column information into a readable table"""
    if not columns:
        return "No columns"

    # Calculate column widths
    widths = {
        'Field': max(len('Field'), max(len(str(col['Field'])) for col in columns)),
        'Type': max(len('Type'), max(len(str(col['Type'])) for col in columns)),
        'Null': max(len('Null'), max(len(str(col['Null'])) for col in columns)),
        'Key': max(len('Key'), max(len(str(col['Key'])) for col in columns)),
        'Default': max(len('Default'), max(len(str(col['Default'] or '')) for col in columns)),
        'Extra': max(len('Extra'), max(len(str(col['Extra'])) for col in columns))
    }

    # Create separator
    separator = '+' + '+'.join('-' * (widths[h] + 2) for h in ['Field', 'Type', 'Null', 'Key', 'Default', 'Extra']) + '+'

    # Create header
    header = '|' + '|'.join(f" {h:{widths[h]}} " for h in ['Field', 'Type', 'Null', 'Key', 'Default', 'Extra']) + '|'

    # Create rows
    result = [separator, header, separator]
    for col in columns:
        row = '|' + '|'.join([
            f" {str(col['Field']):{widths['Field']}} ",
            f" {str(col['Type']):{widths['Type']}} ",
            f" {str(col['Null']):{widths['Null']}} ",
            f" {str(col['Key']):{widths['Key']}} ",
            f" {str(col['Default'] or ''):{widths['Default']}} ",
            f" {str(col['Extra']):{widths['Extra']}} "
        ]) + '|'
        result.append(row)

    result.append(separator)
    return '\n'.join(result)


def format_indexes(indexes):
    """Format index information"""
    if not indexes:
        return "No indexes"

    # Group indexes by name
    index_dict = {}
    for idx in indexes:
        idx_name = idx['Key_name']
        if idx_name not in index_dict:
            index_dict[idx_name] = {
                'unique': idx['Non_unique'] == 0,
                'columns': []
            }
        index_dict[idx_name]['columns'].append(idx['Column_name'])

    result = []
    for idx_name, idx_info in index_dict.items():
        idx_type = "UNIQUE" if idx_info['unique'] else "INDEX"
        columns = ', '.join(idx_info['columns'])
        result.append(f"  {idx_type}: {idx_name} ({columns})")

    return '\n'.join(result)


def format_foreign_keys(foreign_keys):
    """Format foreign key information"""
    if not foreign_keys:
        return "No foreign keys"

    result = []
    for fk in foreign_keys:
        result.append(
            f"  {fk['COLUMN_NAME']} -> "
            f"{fk['REFERENCED_TABLE_NAME']}.{fk['REFERENCED_COLUMN_NAME']}"
        )

    return '\n'.join(result)


def main():
    """Main function to describe all tables in the database"""
    conn = create_connection()
    if not conn:
        print("Could not connect to database")
        return

    db_name = os.getenv("NCAAMB_DB")
    print("=" * 80)
    print(f"DATABASE: {db_name}")
    print("=" * 80)
    print()

    tables = get_all_tables(conn)
    print(f"Found {len(tables)} tables in database\n")

    for table in sorted(tables):
        print("=" * 80)
        print(f"TABLE: {table}")
        print("=" * 80)

        info = describe_table(conn, table)

        print(f"\nRow count: {info['row_count']:,}")
        print("\nColumns:")
        print(format_column_info(info['columns']))

        print("\nIndexes:")
        print(format_indexes(info['indexes']))

        print("\nForeign Keys:")
        print(format_foreign_keys(info['foreign_keys']))

        print("\n")

    conn.close()
    print("=" * 80)
    print("Database schema description complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
