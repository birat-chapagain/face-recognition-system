# test_mediapipe.py
import cv2
import mediapipe as mp
import numpy as np

def test_face_detection():
    # Initialize MediaPipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    )
    
    # Initialize drawing utils
    mp_drawing = mp.solutions.drawing_utils
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
    
    # Start webcam
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue
            
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = face_mesh.process(rgb_frame)
        
        # Draw landmarks
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=drawing_spec)
                
                # Get face encoding
                encoding = np.array([[lm.x, lm.y, lm.z] for lm in face_landmarks.landmark]).flatten()
                print(f"Encoding shape: {encoding.shape}")  # Should be (1404,)
        
        # Display frame
        cv2.imshow('MediaPipe Face Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def test_simple_face_detection():
    # Initialize MediaPipe Face Detection (simpler than Face Mesh)
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils
    face_detection = mp_face_detection.FaceDetection(
        model_selection=0,
        min_detection_confidence=0.5
    )

    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process image
        results = face_detection.process(image_rgb)

        # Draw detections
        if results.detections:
            for detection in results.detections:
                mp_drawing.draw_detection(image, detection)
                
        # Display
        cv2.imshow('MediaPipe Face Detection Test', image)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_face_detection()
    test_simple_face_detection()