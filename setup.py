# setup.py
from setuptools import setup, find_packages

setup(
    name="face_recognition_system",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'opencv-python',
        'face-recognition',
        'psycopg2-binary',
        'python-dotenv',
        'pytest'
    ]
)