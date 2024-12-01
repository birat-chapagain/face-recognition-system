# utilities/re_encode_faces.py

import sys
import os
import cv2
from src.face_encoder import FaceEncoder
from src.database import init_db

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def re_encode_face(name, image_path):
    db_session = init_db()
    face_encoder = FaceEncoder(db_session)
    
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image for {name} from {image_path}.")
        return
    
    # Encode the face
    encoding = face_encoder.encode_face(image)
    if encoding is not None and encoding.shape[0] == 128:
        # Add to the database
        success = face_encoder.add_face(name, encoding)
        if success:
            print(f"Successfully re-encoded and added {name}.")
        else:
            print(f"Failed to add {name} to the database.")
    else:
        print(f"Failed to encode face for {name}.")
    
    db_session.close()

if __name__ == "__main__":
    # Provide the names and paths to the images
    faces_to_encode = [
        {'name': '111', 'image_path': 'path/to/111_image.jpg'},
        {'name': 'birtc', 'image_path': 'path/to/birtc_image.jpg'},
        {'name': 'Sande', 'image_path': 'path/to/Sande_image.jpg'},
    ]
    
    for face in faces_to_encode:
        re_encode_face(face['name'], face['image_path'])

# src/face_encoder.py

def add_face(self, name, encoding):
    """Add new face to the database."""
    if encoding.shape[0] != 128:
        self.logger.error(f"Invalid encoding shape: {encoding.shape}")
        return False
    try:
        face = KnownFace(
            name=name,
            face_encoding=encoding.tobytes()
        )
        self.db_session.add(face)
        self.db_session.commit()
        self.logger.info(f"Added {name} to the database.")
        return True
    except Exception as e:
        self.logger.error(f"Failed to add {name}: {str(e)}")
        self.db_session.rollback()
        return False