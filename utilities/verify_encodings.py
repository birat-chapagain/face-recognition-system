# utilities/verify_encodings.py
import sys
import os
import numpy as np
from src.database import KnownFace, init_db

# Add the project root to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def verify_encodings():
    db_session = init_db()
    known_faces = db_session.query(KnownFace).all()
    for face in known_faces:
        encoding = np.frombuffer(face.face_encoding, dtype=np.float32)
        if encoding.shape[0] != 128:
            print(f"Encoding for {face.name} has incorrect shape: {encoding.shape}")
        else:
            print(f"Encoding for {face.name} is valid.")

if __name__ == "__main__":
    verify_encodings()