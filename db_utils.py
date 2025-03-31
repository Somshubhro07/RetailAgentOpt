# db_utils.py
import sqlite3
import pandas as pd
import os

# --- Configuration ---
DATA_DIR = 'data'
DB_FILENAME = 'inventory.db'
DB_PATH = os.path.join(DATA_DIR, DB_FILENAME)

def create_connection():
    """ Creates and returns a database connection. """
    conn = None
    try:
        # check_same_thread=False might be needed if multiple agents access DB concurrently,
        # but start without it for simplicity unless threading issues arise.
        conn = sqlite3.connect(DB_PATH)
        # Return rows as dictionaries or Row objects for easier access by column name
        conn.row_factory = sqlite3.Row
        print("DB connection created.")
        return conn
    except sqlite3.Error as e:
        print(f"Error connecting to database: {e}")
        return None

# --- Query Functions ---
# Add functions here to perform common queries needed by agents

def get_inventory_levels(conn, product_id=None, store_id=None):
    """ Fetches inventory levels, optionally filtered by product_id and/or store_id (as INTEGERs). """
    try:
        cursor = conn.cursor()
        # Ensure table/column names match cleaned names used during creation
        query = "SELECT * FROM inventory_levels"
        params = []
        conditions = []

        # Match INTEGER columns with integer parameters
        if product_id:
            conditions.append("product_id = ?")
            params.append(int(product_id)) # Cast parameter to int
        if store_id:
            conditions.append("store_id = ?")
            params.append(int(store_id)) # Cast parameter to int

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        print(f"Executing query: {query} with params: {params}")
        cursor.execute(query, params)
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    except sqlite3.Error as e:
        print(f"Error fetching inventory levels: {e}")
        return []
    except (ValueError, TypeError) as e: # Catch error if product_id/store_id cannot be cast to int
        print(f"Error converting ID to integer for inventory query: {e}")
        return []


def get_demand_forecast(conn, product_id=None, store_id=None, date=None):
    """ Fetches demand forecasts, optionally filtered (IDs as INTEGERs). """
    try:
        cursor = conn.cursor()
        query = "SELECT * FROM demand_forecasts"
        params = []
        conditions = []

        # Match INTEGER columns with integer parameters
        if product_id:
            conditions.append("product_id = ?")
            params.append(int(product_id)) # Cast parameter to int
        if store_id:
            conditions.append("store_id = ?")
            params.append(int(store_id)) # Cast parameter to int
        if date:
             conditions.append("date = ?") # Assuming date is TEXT
             params.append(date)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY date DESC"

        print(f"Executing query: {query} with params: {params}")
        cursor.execute(query, params)
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    except sqlite3.Error as e:
        print(f"Error fetching demand forecasts: {e}")
        return []
    except (ValueError, TypeError) as e: # Catch error if product_id/store_id cannot be cast to int
        print(f"Error converting ID to integer for demand query: {e}")
        return []
# --- Update/Insert Functions --- (Example - Adapt as needed) ---

def update_stock_level(conn, product_id, store_id, new_stock_level):
    """ Updates the stock level for a specific product and store (IDs as INTEGERs). """
    query = """
        UPDATE inventory_levels
        SET stock_levels = ?
        WHERE product_id = ? AND store_id = ?;
    """
    try:
        # Cast IDs to int for the WHERE clause
        params = (new_stock_level, int(product_id), int(store_id))
        cursor = conn.cursor()
        print(f"Executing update: Set stock to {new_stock_level} for Product {product_id}, Store {store_id}")
        cursor.execute(query, params)
        conn.commit()
        print(f"Rows affected: {cursor.rowcount}")
        return cursor.rowcount > 0
    except sqlite3.Error as e:
        print(f"Error updating stock level: {e}")
        return False
    except (ValueError, TypeError) as e:
        print(f"Error converting ID to integer for stock update: {e}")
        return False


def get_generic_query(conn, table_name, filters=None, order_by=None):
    """ Fetches data from a table with optional filters (handling potential INTEGER IDs). """
    try:
        cursor = conn.cursor()
        query = f"SELECT * FROM {table_name}"
        params = []
        conditions = []

        if filters:
            for key, value in filters.items():
                 # Check if the key is likely an ID column needing integer conversion
                 # Adapt this list if other ID columns are INTEGER
                 if key in ['product_id', 'store_id']:
                     conditions.append(f"{key} = ?")
                     params.append(int(value)) # Cast ID values to int
                 else:
                     # Assume other filter columns are TEXT or REAL for now
                     conditions.append(f"{key} = ?")
                     params.append(value) # Pass other values as is (usually string)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        if order_by:
             query += f" ORDER BY {order_by}"

        print(f"Executing generic query: {query} with params: {params}")
        cursor.execute(query, params)
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    except sqlite3.Error as e:
        print(f"Error during generic query on table {table_name}: {e}")
        return []
    except (ValueError, TypeError) as e: # Catch error if ID cannot be cast to int
        print(f"Error converting filter value to integer for generic query (key: {key}): {e}")
        return []