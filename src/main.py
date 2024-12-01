# src/main.py
import cv2
import time
import numpy as np
from face_encoder import FaceEncoder
from database import DatabaseManager, Session
import logging
import json

def resize_frame(frame, scale=0.5):
    return cv2.resize(frame, None, fx=scale, fy=scale)

def calculate_frame_quality(frame):
    """Calculate frame quality based on brightness and blur"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    blur = cv2.Laplacian(gray, cv2.CV_64F).var()
    return (brightness / 255.0 * 0.5) + (min(blur / 1000.0, 0.5))

def get_face_location(frame, face):
    """Get face location in relative coordinates"""
    height, width = frame.shape[:2]
    x, y, w, h = face
    return {
        'x': float(x) / width,
        'y': float(y) / height,
        'width': float(w) / width,
        'height': float(h) / height
    }

def main():
    # Initialize
    db_session = Session()
    db_manager = DatabaseManager(db_session)
    face_encoder = FaceEncoder(db_session)
    cap = cv2.VideoCapture(0)
    
    # State variables
    adding_new_face = False
    temp_name = ""
    temp_group = ""
    input_mode = "name"  # Can be "name" or "group"
    unrecognized_count = 0
    
    print("Controls:")
    print("'q' - Quit")
    print("'a' - Start adding new face")
    print("'n' - Skip adding this face")
    print("'g' - Add to group (when adding face)")
    print("'Enter' - Save face or confirm group")
    print("'Tab' - Switch between name and group input")
    print("'backspace' - Delete last character")
    
    # FPS calculation
    prev_frame_time = 0
    new_frame_time = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        small_frame = resize_frame(frame, scale=0.5)
        frame_quality = calculate_frame_quality(small_frame)

        # Calculate FPS
        new_frame_time = time.time()
        fps = 1/(new_frame_time-prev_frame_time) if prev_frame_time > 0 else 0
        prev_frame_time = new_frame_time

        # Get face encoding
        encoding = face_encoder.encode_face(small_frame)
        
        if encoding is not None:
            if adding_new_face:
                # Display instructions and current input
                cv2.putText(frame, f"Mode: {'Name' if input_mode == 'name' else 'Group'}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Name: {temp_name}", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, (0, 255, 0), 2)
                if temp_group:
                    cv2.putText(frame, f"Group: {temp_group}", 
                               (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.7, (0, 255, 0), 2)
                cv2.putText(frame, "Press Enter to save, Tab to switch input", 
                           (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, (0, 255, 0), 2)
            else:
                # Recognition mode
                person_data = face_encoder.compare_faces(encoding)
                
                if person_data:
                    name, confidence, person_id = person_data
                    unrecognized_count = 0
                    
                    # Get person statistics
                    stats = db_manager.get_person_statistics(person_id)
                    
                    # Display recognition results
                    cv2.putText(small_frame, f"{name}: {confidence:.2f}", 
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                              1, (0, 255, 0), 2)
                    if stats:
                        cv2.putText(small_frame, f"Seen {stats['total_detections']} times", 
                                  (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                                  0.7, (0, 255, 0), 2)
                        
                    # Log detection
                    face_location = get_face_location(small_frame, face_encoder.last_face_location)
                    detection_env = {
                        'lighting': frame_quality,
                        'face_size': face_location['width'] * frame_quality['height']
                    }
                    
                    face_encoder.log_detection(
                        person_id,
                        confidence,
                        frame_quality,
                        json.dumps(detection_env),
                        json.dumps(face_location)
                    )
                else:
                    unrecognized_count += 1
                    cv2.putText(small_frame, "Unknown Face Detected", 
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                              1, (0, 0, 255), 2)
                    
                    if unrecognized_count >= 30:
                        cv2.putText(small_frame, "Press 'a' to add or 'n' to skip", 
                                  (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                                  1, (0, 0, 255), 2)

        # Draw FPS and frame quality
        cv2.putText(small_frame, f"FPS: {int(fps)} Quality: {frame_quality:.2f}", 
                   (10, small_frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (0, 255, 0), 2)

        cv2.imshow('Face Recognition', small_frame)
        
        # Key handling
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('a') and not adding_new_face:
            adding_new_face = True
            temp_name = ""
            temp_group = ""
            input_mode = "name"
            print("Adding a new face. Enter name:")
        elif key == ord('n'):
            adding_new_face = False
            temp_name = ""
            temp_group = ""
            unrecognized_count = 0
        elif key == 9:  # Tab key
            input_mode = "group" if input_mode == "name" else "name"
            print(f"Switched to {input_mode} input")
        elif adding_new_face:
            if key == 13:  # Enter key
                if input_mode == "name" and temp_name:
                    if not temp_group:
                        input_mode = "group"
                        print("Enter group (or press Enter to skip):")
                    else:
                        # Save face with group
                        try:
                            person = db_manager.create_person(
                                temp_name,
                                groups=[temp_group] if temp_group else None
                            )
                            
                            # Add face encoding
                            db_manager.add_face_encoding(
                                person.id,
                                encoding,
                                quality_score=frame_quality
                            )
                            
                            # Save reference image
                            _, buffer = cv2.imencode('.jpg', small_frame)
                            db_manager.add_reference_image(
                                person.id,
                                buffer.tobytes(),
                                'front',
                                {'quality': frame_quality}
                            )
                            
                            print(f"Added face for: {temp_name}" + 
                                  (f" in group: {temp_group}" if temp_group else ""))
                        except Exception as e:
                            print(f"Failed to add face: {str(e)}")
                        
                        adding_new_face = False
                        temp_name = ""
                        temp_group = ""
            elif key == 8:  # Backspace
                if input_mode == "name":
                    temp_name = temp_name[:-1]
                    print(f"Current name: {temp_name}", end='\r')
                else:
                    temp_group = temp_group[:-1]
                    print(f"Current group: {temp_group}", end='\r')
            elif 32 <= key <= 126:  # Printable characters
                if input_mode == "name":
                    temp_name += chr(key)
                    print(f"Current name: {temp_name}", end='\r')
                else:
                    temp_group += chr(key)
                    print(f"Current group: {temp_group}", end='\r')

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
