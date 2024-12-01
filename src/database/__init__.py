from .base import Base, Session, engine
from .models import Person, Group, FaceEncoding, ReferenceImage, DetectionLog
from .manager import DatabaseManager

def init_db():
    Base.metadata.create_all(engine)

__all__ = [
    'Base', 'Session', 'engine',
    'Person', 'Group', 'FaceEncoding', 'ReferenceImage', 'DetectionLog',
    'DatabaseManager',
    'init_db'
]
