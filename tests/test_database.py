# tests/test_database.py
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import numpy as np
from datetime import datetime
from src.database import Base, KnownFace, DetectionLog

@pytest.fixture
def db_session():
    # Use SQLite in-memory database for testing
    engine = create_engine('sqlite:///:memory:')
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()

def test_create_known_face(db_session):
    # Create dummy face encoding
    face_encoding = np.random.rand(128).tobytes()
    
    # Create new face entry
    new_face = KnownFace(
        name="Test Person",
        face_encoding=face_encoding
    )
    
    db_session.add(new_face)
    db_session.commit()
    
    # Query and verify
    saved_face = db_session.query(KnownFace).first()
    assert saved_face.name == "Test Person"
    assert saved_face.face_encoding == face_encoding
    assert isinstance(saved_face.created_at, datetime)

def test_create_detection_log(db_session):
    # Create known face first
    face = KnownFace(
        name="Test Person",
        face_encoding=np.random.rand(128).tobytes()
    )
    db_session.add(face)
    db_session.commit()
    
    # Create detection log
    log = DetectionLog(
        face_id=face.id,
        confidence=0.95
    )
    db_session.add(log)
    db_session.commit()
    
    # Query and verify
    saved_log = db_session.query(DetectionLog).first()
    assert saved_log.face_id == face.id
    assert saved_log.confidence == 0.95
    assert isinstance(saved_log.detected_at, datetime)

def test_face_without_name(db_session):
    with pytest.raises(Exception):
        face = KnownFace(
            face_encoding=np.random.rand(128).tobytes()
        )
        db_session.add(face)
        db_session.commit()