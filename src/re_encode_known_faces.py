# re_encode_known_faces.py
import os
import cv2
from src.face_encoder import FaceEncoder
from src.database import init_db

def re_encode_known_faces():
    db_session = init_db()
    face_encoder = FaceEncoder(db_session)
    
    # Clear existing known faces
    db_session.query(KnownFace).delete()
    db_session.commit()
    
    # Directory containing known face images
    known_faces_dir = os.path.join(os.getcwd(), 'known_faces')
    images_and_names = []
    
    # Iterate through the known_faces directory and collect image paths and names
    for filename in os.listdir(known_faces_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            name = os.path.splitext(filename)[0]  # Filename without extension as name
            image_path = os.path.join(known_faces_dir, filename)
            images_and_names.append((image_path, name))
    
    # Re-add known faces
    for image_path, name in images_and_names:
        image = cv2.imread(image_path)
        encoding = face_encoder.encode_face(image)
        if encoding is not None:
            face_encoder.add_face(name, encoding)
            print(f"Added {name} with encoding size {encoding.shape}")
        else:
            print(f"Face not detected in {image_path}")

if __name__ == "__main__":
    re_encode_known_faces()