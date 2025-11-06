import duckdb

duckdb.sql("""
    SELECT to_json(STRUCT_PACK(*)) AS row_json
    FROM 'raw_df.parquet'
    LIMIT 1
""").show()
