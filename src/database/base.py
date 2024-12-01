import os
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker, scoped_session
from sqlalchemy.pool import QueuePool

# Get database path from environment variable or use default
DB_PATH = os.getenv('FACE_RECOGNITION_DB', 'face_recognition.db')
DATABASE_URL = f'sqlite:///{DB_PATH}'

# Create engine with connection pooling
engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=5,
    max_overflow=10,
    pool_timeout=30,
    pool_recycle=1800,
    connect_args={'check_same_thread': False}
)

# Create session factory
session_factory = sessionmaker(bind=engine)
Session = scoped_session(session_factory)

# Create base class for declarative models
Base = declarative_base()
