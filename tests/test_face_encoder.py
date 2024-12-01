# tests/test_face_encoder.py
import pytest
import cv2
import numpy as np
import mediapipe as mp
from src.face_encoder import FaceEncoder
from src.database import KnownFace, DetectionLog

@pytest.fixture
def db_session():
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from src.database import Base
    
    engine = create_engine('sqlite:///:memory:')
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    return Session()

@pytest.fixture
def encoder(db_session):
    return FaceEncoder(db_session)

@pytest.fixture
def sample_frame():
    # Create frame with simple shape
    frame = np.zeros((300, 300, 3), dtype=np.uint8)
    # Draw a circle to simulate a face
    cv2.circle(frame, (150, 150), 50, (255, 255, 255), -1)
    return frame

@pytest.fixture
def sample_encoding():
    # MediaPipe face mesh has 468 landmarks with x,y,z coordinates
    return np.random.rand(468 * 3)

def test_encode_face(encoder, sample_frame):
    encoding = encoder.encode_face(sample_frame)
    if encoding is not None:
        assert isinstance(encoding, np.ndarray)
        assert encoding.shape[0] == 468 * 3  # MediaPipe landmarks

def test_save_face(encoder, db_session, sample_encoding):
    # Add new face to database
    result = encoder.add_face("Test Person", sample_encoding)
    assert result == True
    
    # Verify face was added
    faces = db_session.query(KnownFace).all()
    assert len(faces) == 1
    assert faces[0].name == "Test Person"
    assert np.array_equal(
        np.frombuffer(faces[0].face_encoding, dtype=np.float64),
        sample_encoding
    )

def test_compare_faces(encoder, sample_encoding):
    # Add known face
    encoder.add_face("Known Person", sample_encoding)
    
    # Test with same encoding
    name, confidence = encoder.compare_faces(sample_encoding)
    assert name == "Known Person"
    assert confidence > 0.9

def test_unknown_face(encoder, sample_encoding):
    # Test with different encoding
    unknown_encoding = np.random.rand(468 * 3)
    name, confidence = encoder.compare_faces(unknown_encoding)
    assert name is None
    assert confidence < 0.6

def test_draw_face(encoder, sample_frame):
    # Process frame
    rgb_frame = cv2.cvtColor(sample_frame, cv2.COLOR_BGR2RGB)
    results = encoder.face_mesh.process(rgb_frame)
    
    if results.multi_face_landmarks:
        # Test drawing
        drawn_frame = encoder.draw_face(sample_frame, results.multi_face_landmarks[0])
        assert drawn_frame is not None
        assert drawn_frame.shape == sample_frame.shape