import sqlite3
import os
import sys

# --- Configuration ---
# Make sure this matches the path used in your CommandHistory class
DB_PATH = os.path.expanduser("~/npcsh_history.db")
TABLE_NAME = "conversation_history"
COLUMN_NAME = "team"
COLUMN_TYPE = "TEXT" # Or whatever type you intend (TEXT is usually safe for this)
# --- End Configuration ---

def add_column_if_not_exists(db_path, table_name, column_name, column_type):
    """Adds a column to a table if it doesn't already exist."""
    conn = None
    try:
        print(f"Connecting to database: {db_path}")
        if not os.path.exists(db_path):
            print(f"Error: Database file not found at {db_path}", file=sys.stderr)
            return False

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Check if the column already exists
        cursor.execute(f"PRAGMA table_info({table_name});")
        columns = [info[1] for info in cursor.fetchall()] # Column names are at index 1

        if column_name in columns:
            print(f"Column '{column_name}' already exists in table '{table_name}'. No migration needed.")
            return True
        else:
            print(f"Column '{column_name}' not found in '{table_name}'. Attempting to add...")
            # Add the column. Using TEXT is generally safe.
            # You could add DEFAULT NULL if desired, but SQLite often handles this.
            alter_sql = f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type};"
            print(f"Executing: {alter_sql}")
            cursor.execute(alter_sql)
            conn.commit()
            print(f"Successfully added column '{column_name}' to table '{table_name}'.")
            return True

    except sqlite3.Error as e:
        print(f"An SQLite error occurred: {e}", file=sys.stderr)
        # Check specifically for table not found error
        if f"no such table: {table_name}" in str(e):
             print(f"Error: The table '{table_name}' does not exist in the database.", file=sys.stderr)
        return False
    except Exception as e:
         print(f"An unexpected error occurred: {e}", file=sys.stderr)
         return False
    finally:
        if conn:
            conn.close()
            print("Database connection closed.")

# --- Main execution ---
if __name__ == "__main__":
    print("Starting database schema migration...")
    success = add_column_if_not_exists(DB_PATH, TABLE_NAME, COLUMN_NAME, COLUMN_TYPE)

    if success:
        # Verify
        print("\nVerifying schema...")
        try:
            conn_verify = sqlite3.connect(DB_PATH)
            cursor_verify = conn_verify.cursor()
            cursor_verify.execute(f"PRAGMA table_info({TABLE_NAME});")
            columns_after = [info[1] for info in cursor_verify.fetchall()]
            conn_verify.close()
            print(f"Columns in '{TABLE_NAME}' after migration: {columns_after}")
            if COLUMN_NAME in columns_after:
                 print("Verification successful: Column found.")
            else:
                 print("Verification failed: Column NOT found after migration attempt.", file=sys.stderr)
        except Exception as e:
            print(f"Verification failed with error: {e}", file=sys.stderr)

        print("\nMigration process finished.")
    else:
        print("\nMigration process failed.", file=sys.stderr)
        sys.exit(1) # Exit with error code if migration failed