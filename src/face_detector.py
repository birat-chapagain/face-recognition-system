# src/face_detector.py
import cv2
import logging
from src.config import FaceDetectionConfig

class FaceDetector:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        try:
            self.face_cascade = cv2.CascadeClassifier(FaceDetectionConfig.CASCADE_PATH)
            if self.face_cascade.empty():
                raise Exception("Failed to load cascade classifier")
            self.logger.info("Face detector initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize face detector: {str(e)}")
            raise

    def detect_faces(self, frame):
        """
        Detect faces in the given frame
        Returns: List of (x, y, w, h) tuples for detected faces
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=FaceDetectionConfig.SCALE_FACTOR,
            minNeighbors=FaceDetectionConfig.MIN_NEIGHBORS,
            minSize=FaceDetectionConfig.MIN_SIZE
        )
        return faces

    def draw_faces(self, frame, faces):
        """Draw rectangles around detected faces"""
        for (x, y, w, h) in faces:
            cv2.rectangle(
                frame, 
                (x, y), 
                (x+w, y+h), 
                FaceDetectionConfig.RECTANGLE_COLOR,
                FaceDetectionConfig.RECTANGLE_THICKNESS
            )
        return frame