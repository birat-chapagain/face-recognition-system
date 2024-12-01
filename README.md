# Face Recognition System

A comprehensive face recognition system with advanced features including group management, quality assessment, and detailed tracking.

## Features
- Real-time face detection and recognition
- Advanced database management with SQLAlchemy
- Group-based person organization
- Multiple face encodings per person
- Frame quality assessment
- Detection environment tracking
- Comprehensive statistics and logging

## Technical Stack
- Python 3.9 (recommended)
- OpenCV for video capture
- face_recognition library for face detection and encoding
- SQLAlchemy for database management
- SQLite for data storage

## Project Structure
```
src/
├── database/           # Database related modules
│   ├── __init__.py
│   ├── base.py        # Database connection setup
│   ├── models.py      # SQLAlchemy models
│   └── manager.py     # Database operations
├── main.py            # Main application entry
├── face_encoder.py    # Face encoding and recognition
└── utilities/         # Helper utilities

tests/                 # Unit tests
```

## Setup
1. Create a Python virtual environment (Python 3.9 recommended):
   ```bash
   python -m venv face_recognition_env
   ```

2. Activate the environment:
   - Windows: `face_recognition_env\Scripts\activate`
   - Unix/MacOS: `source face_recognition_env/bin/activate`

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
Run the main application:
```bash
python src/main.py
```

### Controls:
- 'q' - Quit
- 'a' - Start adding new face
- 'n' - Skip adding this face
- 'g' - Add to group (when adding face)
- 'Enter' - Save face or confirm group
- 'Tab' - Switch between name and group input
- 'backspace' - Delete last character

## Database Structure
- **Person**: Stores individual details
- **Group**: Allows person categorization
- **FaceEncoding**: Manages face encodings
- **ReferenceImage**: Stores reference images
- **DetectionLog**: Tracks detection events

## Features in Detail
1. **Face Management**:
   - Multiple encodings per person
   - Quality assessment for each encoding
   - Group-based organization

2. **Detection Tracking**:
   - Confidence scoring
   - Frame quality assessment
   - Environment data logging
   - Location tracking

3. **Data Management**:
   - Soft delete support
   - Comprehensive statistics
   - Performance optimization
   - Connection pooling

## Contributing
1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License
[MIT License](LICENSE)