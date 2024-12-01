# tests/test_db_connection.py
from src.database import init_db

def test_connection():
    try:
        session = init_db()
        print("SQLite database connection successful!")
        print("Database location: face_recognition.db")
        return True
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return False

if __name__ == "__main__":
    test_connection()