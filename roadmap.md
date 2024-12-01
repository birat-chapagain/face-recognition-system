<!-- filepath: d:/Semester 1/Computer Vision/face_recognition_env/readme.md -->

# Face Recognition Project

This project is part of the Computer Vision course and focuses on implementing face recognition using Python.

## Project Progress

### Week 1: Project Setup and Environment Configuration

- [x] **Set Up Development Environment**
  - [x] Install Python 3.x and create a virtual environment
  - [x] Install required libraries:
    - [x] OpenCV: `pip install opencv-python`
    - [x] Face recognition library: `pip install face_recognition`
    - [x] PostgreSQL adapter: `pip install psycopg2`
    - [ ] OpenVINO toolkit (prepare for future optimization)
  - [x] Ensure PostgreSQL and pgAdmin are installed and configured

- [x] **Webcam Test Script**
  - [x] Write a Python script to capture video from the webcam using OpenCV
  - [x] Display the video feed in a window
  - [x] Close the window when 'q' is pressed
  - [x] Verify that the webcam operates at 30 FPS

### Week 2: Implement Face Detection

- [ ] **Integrate Face Detection**
  - [ ] Use OpenCV's Haar Cascades for face detection
    - [ ] Load the pre-trained Haar Cascade classifier
  - [ ] Modify the webcam script to detect faces in real-time
  - [ ] Draw bounding boxes around detected faces
  - [ ] Optimize detection to maintain 30 FPS:
    - [ ] Adjust parameters like scale factor and minimum neighbors
    - [ ] Reduce frame size or convert frames to grayscale if necessary

### Week 3: Database Integration

- [ ] **Design Database Schema**
  - [ ] Create a PostgreSQL database for the application
  - [ ] Create tables:
    - [ ] `known_faces` (id, name, face_encoding)
    - [ ] `access_logs` (id, name, timestamp)
  - [ ] Define primary keys and appropriate data types

- [ ] **Implement Database Functions**
  - [ ] Write Python functions to:
    - [ ] Connect to PostgreSQL using `psycopg2`
    - [ ] Add new face data to `known_faces`
    - [ ] Retrieve face encodings from the database
    - [ ] Insert recognition events into `access_logs`

### Week 4: Implement Face Recognition

- [ ] **Integrate Face Recognition Library**
  - [ ] Use `face_recognition` to compute face encodings from detected faces
  - [ ] Compare live face encodings with known encodings from the database
    - [ ] Implement a confidence threshold of 70%
    - [ ] Use `face_recognition.compare_faces()` and `face_recognition.face_distance()`

- [ ] **Handle Unrecognized Faces**
  - [ ] Label unrecognized faces as 'New Person'
  - [ ] Optionally allow adding new faces to the database
    - [ ] Implement functionality to capture and store new faces with names

### Week 5: Optimize Performance with OpenVINO

- [ ] **Set Up OpenVINO Toolkit**
  - [ ] Install and configure OpenVINO on the system
  - [ ] Ensure compatibility with hardware (CPU/GPU)

- [ ] **Optimize Inference Computations**
  - [ ] Convert face detection and recognition models to OpenVINO format
  - [ ] Update the code to use OpenVINO's inference engine
  - [ ] Test performance improvements to maintain 30 FPS

### Week 6: Develop User Interface and Logging

- [ ] **Enhance Real-Time Display**
  - [ ] Overlay names and confidence scores on the video feed
  - [ ] Ensure text and bounding boxes are clearly visible

- [ ] **Implement Logging**
  - [ ] Record each recognition event with a timestamp in `access_logs`
  - [ ] Include both recognized and unrecognized faces

- [ ] **User Interaction**
  - [ ] Decide on interface type (command-line or GUI)
  - [ ] Implement controls:
    - [ ] Press 'a' to add a new person's face
    - [ ] Press 'q' to quit the application

### Week 7: Testing, Refinement, and Documentation

- [ ] **Testing**
  - [ ] Test the application under various conditions:
    - [ ] Different lighting and backgrounds
    - [ ] Multiple faces in the frame
    - [ ] Faces with various orientations and expressions
  - [ ] Verify recognition accuracy and system stability

- [ ] **Refinement**
  - [ ] Fine-tune parameters for face detection and recognition
  - [ ] Optimize database queries and data handling
  - [ ] Address any bugs or performance issues

- [ ] **Documentation**
  - [ ] Write setup and usage instructions
  - [ ] Comment code for readability and maintainability
  - [ ] Provide guidelines for scaling up

## Summary of Progress

- Completed environment setup and installed necessary libraries.
- Created and tested the webcam script with OpenCV.
- Reviewed the `progress.py` module from the `rich` library for progress bar implementation.