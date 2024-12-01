import datetime
import json
from sqlalchemy import Column, Integer, String, DateTime, Boolean, ForeignKey, Table, Float, LargeBinary, Index
from sqlalchemy.orm import relationship
from .base import Base

# Association table for many-to-many relationship between persons and groups
person_groups = Table(
    'person_groups',
    Base.metadata,
    Column('person_id', Integer, ForeignKey('persons.id'), primary_key=True),
    Column('group_id', Integer, ForeignKey('groups.id'), primary_key=True)
)

class Person(Base):
    __tablename__ = 'persons'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    last_updated_at = Column(DateTime, onupdate=datetime.datetime.utcnow)
    is_active = Column(Boolean, default=True)
    notes = Column(String(500))
    person_metadata = Column(String(1000))
    
    # Relationships
    face_encodings = relationship("FaceEncoding", back_populates="person", cascade="all, delete-orphan")
    groups = relationship("Group", secondary=person_groups, back_populates="persons")
    reference_images = relationship("ReferenceImage", back_populates="person", cascade="all, delete-orphan")
    detection_logs = relationship("DetectionLog", back_populates="person")
    
    __table_args__ = (Index('idx_person_name_active', 'name', 'is_active'),)

    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'is_active': self.is_active,
            'notes': self.notes,
            'metadata': json.loads(self.person_metadata) if self.person_metadata else {},
            'groups': [group.name for group in self.groups],
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'last_updated_at': self.last_updated_at.isoformat() if self.last_updated_at else None
        }

class Group(Base):
    __tablename__ = 'groups'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False, unique=True)
    description = Column(String(500))
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    # Relationships
    persons = relationship("Person", secondary=person_groups, back_populates="groups")

    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

class FaceEncoding(Base):
    __tablename__ = 'face_encodings'
    
    id = Column(Integer, primary_key=True)
    person_id = Column(Integer, ForeignKey('persons.id'), nullable=False)
    encoding = Column(String(2000), nullable=False)  # Stored as JSON string
    quality_score = Column(Float)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    # Relationships
    person = relationship("Person", back_populates="face_encodings")
    
    __table_args__ = (Index('idx_person_encoding', 'person_id'),)

    def to_dict(self):
        return {
            'id': self.id,
            'person_id': self.person_id,
            'quality_score': self.quality_score,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

class ReferenceImage(Base):
    __tablename__ = 'reference_images'
    
    id = Column(Integer, primary_key=True)
    person_id = Column(Integer, ForeignKey('persons.id'), nullable=False)
    image_data = Column(LargeBinary, nullable=False)
    image_type = Column(String(50))
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    image_metadata = Column(String(500))
    
    # Relationships
    person = relationship("Person", back_populates="reference_images")

    def to_dict(self):
        return {
            'id': self.id,
            'person_id': self.person_id,
            'image_type': self.image_type,
            'metadata': json.loads(self.image_metadata) if self.image_metadata else {},
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

class DetectionLog(Base):
    __tablename__ = 'detection_logs'
    
    id = Column(Integer, primary_key=True)
    person_id = Column(Integer, ForeignKey('persons.id'), nullable=False)
    confidence = Column(Float, nullable=False)
    frame_quality = Column(Float)
    location_x = Column(Integer)
    location_y = Column(Integer)
    detection_time = Column(DateTime, default=datetime.datetime.utcnow)
    environment_data = Column(String(500))  # JSON field for environment data
    
    # Relationships
    person = relationship("Person", back_populates="detection_logs")
    
    __table_args__ = (
        Index('idx_detection_person_time', 'person_id', 'detection_time'),
    )

    def to_dict(self):
        return {
            'id': self.id,
            'person_id': self.person_id,
            'confidence': self.confidence,
            'frame_quality': self.frame_quality,
            'location': {'x': self.location_x, 'y': self.location_y},
            'detection_time': self.detection_time.isoformat() if self.detection_time else None,
            'environment': json.loads(self.environment_data) if self.environment_data else {}
        }
