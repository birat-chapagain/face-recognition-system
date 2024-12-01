# test_db.py
from src.database import init_db

try:
    db_session = init_db()
    print("Database connection successful!")
    print("Tables created!")
except Exception as e:
    print(f"Error: {e}")