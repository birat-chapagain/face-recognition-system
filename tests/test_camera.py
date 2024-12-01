# tests/test_camera.py

import pytest
import cv2
import numpy as np
from unittest.mock import Mock, patch
from src.camera import CameraManager, CameraError
from src.config import CameraConfig

@pytest.fixture
def mock_camera():
    with patch('cv2.VideoCapture') as mock_cap:
        # Mock successful camera initialization
        mock_cap.return_value.isOpened.return_value = True
        mock_cap.return_value.read.return_value = (True, np.zeros((480, 640, 3)))
        mock_cap.return_value.get.return_value = 640  # Mock resolution
        yield mock_cap

def test_camera_initialization(mock_camera):
    cam = CameraManager()
    assert cam.cap.isOpened()

def test_camera_initialization_failure():
    with patch('cv2.VideoCapture') as mock_cap:
        mock_cap.return_value.isOpened.return_value = False
        with pytest.raises(CameraError):
            CameraManager()

def test_frame_reading(mock_camera):
    cam = CameraManager()
    ret, frame = cam.read_frame()
    assert ret
    assert isinstance(frame, np.ndarray)
    assert frame.shape == (480, 640, 3)

def test_fps_calculation(mock_camera):
    cam = CameraManager()
    cam.read_frame()
    assert cam.fps >= 0

def test_camera_release(mock_camera):
    cam = CameraManager()
    cam.release()
    assert mock_camera.return_value.release.called

@pytest.mark.parametrize("resolution", [
    (640, 480),
    (1280, 720),
    (1920, 1080)
])
def test_camera_resolution(resolution, mock_camera):
    with patch('src.config.CameraConfig') as mock_config:
        mock_config.FRAME_WIDTH = resolution[0]
        mock_config.FRAME_HEIGHT = resolution[1]
        cam = CameraManager()
        assert mock_camera.return_value.set.called