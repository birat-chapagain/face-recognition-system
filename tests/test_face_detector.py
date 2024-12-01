# tests/test_face_detector.py
import pytest
import cv2
import numpy as np
from src.face_detector import FaceDetector
from src.config import FaceDetectionConfig

@pytest.fixture
def face_detector():
    return FaceDetector()

@pytest.fixture
def sample_frame():
    # Create a blank frame
    return np.zeros((480, 640, 3), dtype=np.uint8)

@pytest.fixture
def test_image():
    # Load test image with a known face
    test_image_path = "tests/test_data/test_face.jpeg"
    frame = cv2.imread(test_image_path)
    if frame is None:
        pytest.skip("Test image not found at: " + test_image_path)
    return frame

def test_face_detector_initialization(face_detector):
    assert face_detector.face_cascade is not None
    assert not face_detector.face_cascade.empty()

def test_detect_faces_empty_frame(face_detector, sample_frame):
    faces = face_detector.detect_faces(sample_frame)
    assert len(faces) == 0
    assert isinstance(faces, tuple) or isinstance(faces, np.ndarray)

def test_draw_faces_empty_frame(face_detector, sample_frame):
    faces = []
    processed_frame = face_detector.draw_faces(sample_frame.copy(), faces)
    assert processed_frame.shape == sample_frame.shape
    assert isinstance(processed_frame, np.ndarray)

def test_detect_faces_with_test_image(face_detector, test_image):
    faces = face_detector.detect_faces(test_image)
    assert len(faces) > 0
    # Verify face coordinates
    for (x, y, w, h) in faces:
        assert x >= 0 and y >= 0
        assert w > 0 and h > 0
        assert x + w <= test_image.shape[1]
        assert y + h <= test_image.shape[0]

@pytest.mark.parametrize("invalid_input", [
    None,
    np.zeros((480, 640)), # 2D array instead of 3D
    np.zeros((480, 640, 4)) # 4 channels instead of 3
])
def test_detect_faces_invalid_input(face_detector, invalid_input):
    with pytest.raises(Exception):
        face_detector.detect_faces(invalid_input)