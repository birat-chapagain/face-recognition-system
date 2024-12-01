# src/camera.py

import cv2
import time
import logging
from src.config import CameraConfig
from src.face_detector import FaceDetector

class CameraError(Exception):
    """Base exception for camera-related errors"""
    pass

class CameraManager:
    """
    Manages webcam operations including frame capture and face detection.
    
    Attributes:
        cap: OpenCV VideoCapture object
        face_detector: FaceDetector instance
        fps: Current frames per second
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        try:
            self.cap = cv2.VideoCapture(CameraConfig.CAMERA_ID)
            if not self.cap.isOpened():
                raise CameraError("Failed to open camera")
                
            self._setup_camera()
            self.face_detector = FaceDetector()
            self.last_frame_time = time.time()
            self.fps = 0
            
        except Exception as e:
            self.logger.error(f"Camera initialization failed: {str(e)}")
            raise CameraError(f"Camera initialization failed: {str(e)}")
            
    def _setup_camera(self):
        """Configure camera properties"""
        try:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CameraConfig.FRAME_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CameraConfig.FRAME_HEIGHT)
            actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            
            if actual_width != CameraConfig.FRAME_WIDTH or actual_height != CameraConfig.FRAME_HEIGHT:
                self.logger.warning(f"Camera resolution mismatch. Requested: {CameraConfig.FRAME_WIDTH}x{CameraConfig.FRAME_HEIGHT}, Got: {actual_width}x{actual_height}")
                
        except Exception as e:
            raise CameraError(f"Failed to setup camera properties: {str(e)}")

    def calculate_fps(self):
        current_time = time.time()
        time_diff = current_time - self.last_frame_time
        self.fps = 1 / time_diff if time_diff > 0 else 0
        self.last_frame_time = current_time
        
        if self.fps < CameraConfig.TARGET_FPS * 0.8:  # Alert if FPS drops below 80% of target
            self.logger.warning(f"Low FPS detected: {self.fps:.2f}")

    def read_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.logger.error("Failed to grab frame")
            return False, None
        self.calculate_fps()
        return ret, frame

    def process_frame(self, frame):
        faces = self.face_detector.detect_faces(frame)
        frame = self.face_detector.draw_faces(frame, faces)
        return frame, len(faces)

    def display_frame(self, frame, face_count):
        fps_text = f"FPS: {self.fps:.2f} Faces: {face_count}"
        cv2.putText(frame, fps_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow(CameraConfig.WINDOW_NAME, frame)

    def run(self):
        self.logger.info("Starting camera feed")
        try:
            while True:
                ret, frame = self.read_frame()
                if not ret:
                    break

                frame, face_count = self.process_frame(frame)
                self.display_frame(frame, face_count)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.logger.info("User requested quit")
                    break
        except Exception as e:
            self.logger.error(f"Error during camera operation: {str(e)}")
        finally:
            self.release()

    def release(self):
        self.logger.info("Releasing camera resources")
        self.cap.release()
        cv2.destroyAllWindows()

def main():
    camera_manager = CameraManager()
    try:
        camera_manager.run()
    except KeyboardInterrupt:
        camera_manager.release()

if __name__ == "__main__":
    main()