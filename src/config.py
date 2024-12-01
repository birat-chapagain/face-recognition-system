# src/config.py
import os
import logging
import cv2
from dotenv import load_dotenv

load_dotenv()

class CameraConfig:
    CAMERA_ID = 0  # Default webcam
    FRAME_WIDTH = 640
    FRAME_HEIGHT = 480
    TARGET_FPS = 30
    WINDOW_NAME = "Facial Recognition"

class FaceDetectionConfig:
    CASCADE_PATH = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
    SCALE_FACTOR = 1.1
    MIN_NEIGHBORS = 5
    MIN_SIZE = (30, 30)
    RECTANGLE_COLOR = (0, 255, 0)  # BGR Green
    RECTANGLE_THICKNESS = 2

class DatabaseConfig:
    DB_PATH = 'face_recognition.db'
    
    @staticmethod
    def get_connection_string():
        return f'sqlite:///{DatabaseConfig.DB_PATH}'

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )