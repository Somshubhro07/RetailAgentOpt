import pandas as pd
import sqlite3
import os
import re # Import regular expression module for advanced cleaning

# --- Configuration ---
DATA_DIR = 'data'
DB_FILENAME = 'inventory.db'
DB_PATH = os.path.join(DATA_DIR, DB_FILENAME)

# <<< --- ACTION REQUIRED: Verify file names and types --- >>>
# Make sure these filenames match yours and specify 'csv' or 'excel'
DATA_FILES = {
    'demand': {'filename': 'demand_forecasting.csv', 'type': 'csv', 'table_name': 'demand_forecasts'},
    'inventory': {'filename': 'inventory_monitoring.csv', 'type': 'csv', 'table_name': 'inventory_levels'},
    'pricing': {'filename': 'pricing_optimization.csv', 'type': 'csv', 'table_name': 'pricing_data'}
}

# --- Database Functions ---

def create_connection(db_path):
    """ Creates and returns a database connection. """
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        print(f"SQLite connection established to '{db_path}' (Version: {sqlite3.sqlite_version})")
        return conn
    except sqlite3.Error as e:
        print(f"Error connecting to database: {e}")
        return None

def close_connection(conn):
    """ Closes the database connection. """
    if conn:
        conn.close()
        print("SQLite connection closed.")

def create_table(conn, table_name, columns_sql):
    """ Creates a table with the specified name and column definitions. """
    # Construct the full CREATE TABLE statement
    create_table_sql = f"CREATE TABLE IF NOT EXISTS {table_name} ({columns_sql});"
    try:
        cursor = conn.cursor()
        print(f"\nExecuting SQL: {create_table_sql.strip()}")
        cursor.execute(create_table_sql)
        print(f"Table '{table_name}' created successfully (or already exists).")
    except sqlite3.Error as e:
        print(f"Error creating table '{table_name}': {e}")

def load_data_to_db(conn, df, table_name):
    """ Loads data from a Pandas DataFrame into an SQLite table. Replaces existing table. """
    try:
        print(f"\nLoading data into table '{table_name}'...")
        # Ensure DataFrame column names match the SQL table definition (case-insensitive isn't guaranteed)
        # The cleaning step should handle this, but it's good practice.
        df.to_sql(table_name, conn, if_exists='replace', index=False)

        cursor = conn.cursor()
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        count = cursor.fetchone()[0]
        print(f"Successfully loaded {count} rows into '{table_name}'.")
        return True
    except Exception as e:
        print(f"Error loading data into table '{table_name}': {e}")
        return False

# --- Data Loading and Cleaning Function ---

def clean_col_names(df):
    """ Cleans DataFrame column names: lowercase, replaces non-alphanumeric with underscore. """
    new_columns = []
    for col in df.columns:
        # Convert to lowercase
        cleaned_col = col.lower()
        # Replace sequences of non-alphanumeric characters with a single underscore
        cleaned_col = re.sub(r'[^a-z0-9_]+', '_', cleaned_col, flags=re.IGNORECASE)
        # Remove leading/trailing underscores that might result
        cleaned_col = cleaned_col.strip('_')
        # Ensure column name isn't empty after cleaning, use a default if it is
        if not cleaned_col:
            cleaned_col = 'unnamed_col'
        new_columns.append(cleaned_col)

    df.columns = new_columns
    print("Cleaned column names:", df.columns.tolist())
    return df


def load_and_clean_dataframe(file_path, file_type='csv'):
    """ Loads data, cleans column names. """
    print(f"\nLoading DataFrame from: {file_path} (Type: {file_type})")
    try:
        if file_type == 'csv':
            df = pd.read_csv(file_path)
        elif file_type == 'excel':
            df = pd.read_excel(file_path) # Requires 'pip install openpyxl'
        else:
            print(f"Error: Unsupported file type '{file_type}'.")
            return None

        print("Original columns found:", df.columns.tolist())
        df = clean_col_names(df) # Clean column names right after loading

        print("Data sample (first 3 rows):")
        print(df.head(3))
        print("Data types inferred by pandas:")
        print(df.info())
        return df
    except FileNotFoundError:
        print(f"Error: File not found at '{file_path}'.")
        return None
    except Exception as e:
        print(f"Error reading file '{file_path}': {e}")
        return None

# --- Date Handling Function ---

def convert_date_columns(df, date_cols_to_check):
    """ Converts specified columns to datetime if they exist in the DataFrame. """
    print(f"\nAttempting to convert date columns: {date_cols_to_check}")
    for col in date_cols_to_check:
        if col in df.columns:
            print(f"Converting column '{col}' to datetime...")
            # errors='coerce' will turn unparseable dates into NaT (Not a Time)
            # You might need a specific format string if pandas can't infer it:
            # format='%Y-%m-%d' or format='%d/%m/%Y' etc.
            original_dtype = df[col].dtype
            df[col] = pd.to_datetime(df[col], errors='coerce')
            # Check if conversion actually changed the type (useful for debugging)
            if df[col].dtype != original_dtype:
                 print(f"Column '{col}' successfully converted to {df[col].dtype}.")
            else:
                 print(f"Column '{col}' conversion attempted, but dtype remained {df[col].dtype}. Check data format.")
            # Optional: Format date to string for SQLite consistency if preferred
            # df[col] = df[col].dt.strftime('%Y-%m-%d') # Store as YYYY-MM-DD strings
        else:
            print(f"Column '{col}' not found in DataFrame, skipping conversion.")
    return df

# --- Define Table Schemas (Using Cleaned Column Names) ---
# These functions return the SQL string defining columns for CREATE TABLE
# Review the inferred types (TEXT, INTEGER, REAL)

def get_demand_schema():
    return """
        product_id TEXT,
        date TEXT,
        store_id TEXT,
        sales_quantity INTEGER,
        price REAL,
        promotions TEXT,
        seasonality_factors REAL,
        external_factors REAL,
        demand_trend REAL,
        customer_segments TEXT,
        PRIMARY KEY (product_id, date, store_id)
    """

def get_inventory_schema():
    return """
        product_id TEXT,
        store_id TEXT,
        stock_levels INTEGER,
        supplier_lead_time_days INTEGER,
        stockout_frequency REAL,
        reorder_point INTEGER,
        expiry_date TEXT,            -- This will be converted by convert_date_columns if present
        warehouse_capacity INTEGER,
        order_fulfillment_time_days INTEGER,
        PRIMARY KEY (product_id, store_id) -- Assumes snapshot, add date/timestamp if needed
    """

def get_pricing_schema():
    return """
        product_id TEXT,
        store_id TEXT,
        price REAL,
        competitor_prices REAL,
        discounts REAL,
        sales_volume INTEGER,
        customer_reviews TEXT,
        return_rate_ REAL,           -- Cleaned name from 'Return Rate (%)'
        storage_cost REAL,
        elasticity_index REAL,
        PRIMARY KEY (product_id, store_id) -- Assumes current, add date if needed
    """

# Map keys to schema functions
SCHEMA_FUNCTIONS = {
    'demand': get_demand_schema,
    'inventory': get_inventory_schema,
    'pricing': get_pricing_schema
}

# Specify which date columns to *look for* in each file type (use cleaned names)
DATE_COLUMNS_TO_CHECK = {
    'demand': ['date'],
    'inventory': ['expiry_date'],
    'pricing': [] # No date columns specified in the user's list for this file
}

# --- Main Execution Logic ---

if __name__ == "__main__":
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"Created directory: {DATA_DIR}")

    conn = create_connection(DB_PATH)

    if conn:
        # Process each file configuration
        for key, info in DATA_FILES.items():
            file_path = os.path.join(DATA_DIR, info['filename'])
            table_name = info['table_name']

            # 1. Load and clean data
            df = load_and_clean_dataframe(file_path, info['type'])
            if df is None:
                print(f"Stopping script due to failure loading '{key}'.")
                break # Stop processing if a file fails to load

            # 2. Convert relevant date columns (if they exist)
            df = convert_date_columns(df, DATE_COLUMNS_TO_CHECK.get(key, []))

            # 3. Get table schema SQL
            schema_func = SCHEMA_FUNCTIONS.get(key)
            if schema_func:
                columns_sql = schema_func()
                # 4. Create table
                create_table(conn, table_name, columns_sql)
                # 5. Load data into table
                if not load_data_to_db(conn, df, table_name):
                    print(f"Stopping script due to failure loading data into table '{table_name}'.")
                    break # Stop if loading data fails
            else:
                print(f"Warning: Schema function not defined for key '{key}'. Skipping table creation.")

        close_connection(conn)
    else:
        print("Failed to establish database connection. Exiting.")