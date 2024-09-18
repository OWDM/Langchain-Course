import sqlite3

db = 'D:\\owd1\\Documents\\GitHub-REPO\\Langchain-Course\\db.sqlite'

# Function to check database connectivity
def check_db_connection(db_path):
    try:
        conn = sqlite3.connect(db_path)
        print("Successfully connected to the database.")
        conn.close()
    except sqlite3.Error as e:
        print(f"Error connecting to the database: {e}")

# Check if the connection to the database is successful
check_db_connection(db)

# Establish the connection
try:
    conn = sqlite3.connect(db)
    print("Connection established.")
except sqlite3.Error as e:
    print(f"Failed to connect to the database: {e}")

# Function to run a SQLite query
def run_sqlite_query(query):
    try:
        c = conn.cursor()
        c.execute(query)
        result = c.fetchall()
        c.close()
        return result
    except sqlite3.Error as e:
        print(f"SQLite error: {e}")

# Example: Check if the table 'users' exists
tables = run_sqlite_query("SELECT name FROM sqlite_master WHERE type='table';")
print("Available tables:", tables)