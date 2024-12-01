# src/face_encoder.py
import face_recognition
import cv2
import numpy as np
import logging
from database import Person, FaceEncoding, DetectionLog, DatabaseManager, Session
from sqlalchemy import desc

class FaceEncoder:
    def __init__(self, db_session):
        self.db_session = db_session
        self.logger = logging.getLogger(__name__)
        self.known_face_encodings = []
        self.known_face_ids = []
        self.known_face_names = []
        self.last_face_location = None
        self.load_known_faces()
        
    def load_known_faces(self):
        """Load known faces from database"""
        try:
            # Get all active face encodings
            face_encodings = self.db_session.query(FaceEncoding)\
                .join(Person)\
                .filter(FaceEncoding.is_active == True)\
                .filter(Person.is_active == True)\
                .all()
                
            self.known_face_encodings = []
            self.known_face_ids = []
            self.known_face_names = []
            
            for face_encoding in face_encodings:
                encoding = np.frombuffer(face_encoding.encoding_data, dtype=np.float32)
                if encoding.shape[0] == 128:
                    self.known_face_encodings.append(encoding)
                    self.known_face_ids.append(face_encoding.person_id)
                    self.known_face_names.append(face_encoding.person.name)
                else:
                    self.logger.warning(f"Incorrect encoding shape for person {face_encoding.person.name}, skipping")
        except Exception as e:
            self.logger.error(f"Failed to load known faces: {str(e)}")
            
    def encode_face(self, frame):
        """Convert frame to face encoding using face_recognition library"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Use HOG-based model for CPU
        face_locations = face_recognition.face_locations(rgb_frame, model='hog')
        if face_locations:
            self.last_face_location = face_locations[0]  # Store for later use
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            if face_encodings:
                return face_encodings[0]  # 128-dimensional encoding
        return None
            
    def compare_faces(self, encoding):
        """Compare face encoding with known faces and return (name, confidence, person_id)"""
        if not self.known_face_encodings or encoding is None:
            return None

        if len(self.known_face_encodings) > 0:
            # Convert list to numpy array
            known_encodings = np.array(self.known_face_encodings)
            # Calculate distances
            distances = face_recognition.face_distance(known_encodings, encoding)
            
            if len(distances) > 0:
                min_distance = np.min(distances)
                best_match_index = np.argmin(distances)
                confidence = 1 - min_distance
                
                if confidence >= 0.6:
                    return (
                        self.known_face_names[best_match_index],
                        confidence,
                        self.known_face_ids[best_match_index]
                    )
        
        return None
        
    def log_detection(self, person_id, confidence, frame_quality, detection_environment, location_data):
        """Log a face detection event"""
        try:
            # Get the most recent face encoding for this person
            face_encoding = self.db_session.query(FaceEncoding)\
                .filter_by(person_id=person_id)\
                .filter_by(is_active=True)\
                .order_by(desc(FaceEncoding.created_at))\
                .first()
                
            if face_encoding:
                detection_log = DetectionLog(
                    face_encoding_id=face_encoding.id,
                    confidence=confidence,
                    frame_quality=frame_quality,
                    detection_environment=detection_environment,
                    location_data=location_data
                )
                self.db_session.add(detection_log)
                self.db_session.commit()
                return True
        except Exception as e:
            self.logger.error(f"Failed to log detection: {str(e)}")
            self.db_session.rollback()
        return False

# In your main application or a separate script
def re_encode_known_faces():
    db_session = Session()
    face_encoder = FaceEncoder(db_session)
    # Clear existing known faces
    db_session.query(FaceEncoding).delete()
    db_session.commit()

    # Re-add known faces
    images_and_names = [("person1.jpg", "Person 1"), ("person2.jpg", "Person 2")]
    for image_path, name in images_and_names:
        image = cv2.imread(image_path)
        encoding = face_encoder.encode_face(image)
        if encoding is not None:
            person = Person(name=name)
            db_session.add(person)
            db_session.commit()
            face_encoding = FaceEncoding(person_id=person.id, encoding_data=encoding.tobytes())
            db_session.add(face_encoding)
            db_session.commit()
            print(f"Added {name} with encoding size {encoding.shape}")
        else:
            print(f"Face not detected in {image_path}")