import json
from contextlib import contextmanager
from sqlalchemy.exc import SQLAlchemyError
from .base import Session
from .models import Person, Group, FaceEncoding, ReferenceImage, DetectionLog

class DatabaseManager:
    def __init__(self):
        self.session = Session()

    def __del__(self):
        self.session.close()

    @contextmanager
    def session_scope(self):
        """Provide a transactional scope around a series of operations."""
        try:
            yield self.session
            self.session.commit()
        except Exception as e:
            self.session.rollback()
            raise e

    def create_person(self, name, groups=None, notes=None, metadata=None):
        """Create a new person with optional group assignments"""
        try:
            with self.session_scope() as session:
                person = Person(
                    name=name,
                    notes=notes,
                    person_metadata=json.dumps(metadata) if metadata else None
                )
                if groups:
                    for group_name in groups:
                        group = self.get_or_create_group(group_name)
                        person.groups.append(group)
                
                session.add(person)
                return person
        except SQLAlchemyError as e:
            raise Exception(f"Error creating person: {str(e)}")

    def get_or_create_group(self, name, description=None):
        """Get existing group or create new one"""
        with self.session_scope() as session:
            group = session.query(Group).filter_by(name=name).first()
            if not group:
                group = Group(name=name, description=description)
                session.add(group)
            return group

    def add_face_encoding(self, person_id, encoding, quality_score=None):
        """Add a face encoding for a person"""
        try:
            with self.session_scope() as session:
                face_encoding = FaceEncoding(
                    person_id=person_id,
                    encoding=json.dumps(encoding.tolist()),
                    quality_score=quality_score
                )
                session.add(face_encoding)
                return face_encoding
        except SQLAlchemyError as e:
            raise Exception(f"Error adding face encoding: {str(e)}")

    def add_reference_image(self, person_id, image_data, image_type='front', metadata=None):
        """Add a reference image for a person"""
        try:
            with self.session_scope() as session:
                ref_image = ReferenceImage(
                    person_id=person_id,
                    image_data=image_data,
                    image_type=image_type,
                    image_metadata=json.dumps(metadata) if metadata else None
                )
                session.add(ref_image)
                return ref_image
        except SQLAlchemyError as e:
            raise Exception(f"Error adding reference image: {str(e)}")

    def log_detection(self, person_id, confidence, frame_quality=None,
                     location_x=None, location_y=None, environment_data=None):
        """Log a face detection event"""
        try:
            with self.session_scope() as session:
                log = DetectionLog(
                    person_id=person_id,
                    confidence=confidence,
                    frame_quality=frame_quality,
                    location_x=location_x,
                    location_y=location_y,
                    environment_data=json.dumps(environment_data) if environment_data else None
                )
                session.add(log)
                return log
        except SQLAlchemyError as e:
            raise Exception(f"Error logging detection: {str(e)}")

    def get_person_by_id(self, person_id):
        """Get person by ID"""
        return self.session.query(Person).filter_by(id=person_id, is_active=True).first()

    def get_person_by_name(self, name):
        """Get person by name"""
        return self.session.query(Person).filter_by(name=name, is_active=True).first()

    def get_all_persons(self, include_inactive=False):
        """Get all persons"""
        query = self.session.query(Person)
        if not include_inactive:
            query = query.filter_by(is_active=True)
        return query.all()

    def get_persons_by_group(self, group_name):
        """Get all persons in a group"""
        return (self.session.query(Person)
                .join(Person.groups)
                .filter(Group.name == group_name, Person.is_active == True)
                .all())

    def get_face_encodings(self, person_id):
        """Get all face encodings for a person"""
        return (self.session.query(FaceEncoding)
                .filter_by(person_id=person_id)
                .all())

    def get_detection_stats(self, person_id):
        """Get detection statistics for a person"""
        stats = self.session.query(
            DetectionLog.person_id,
            func.count(DetectionLog.id).label('detection_count'),
            func.avg(DetectionLog.confidence).label('avg_confidence'),
            func.avg(DetectionLog.frame_quality).label('avg_frame_quality')
        ).filter_by(person_id=person_id).group_by(DetectionLog.person_id).first()
        
        return {
            'detection_count': stats.detection_count if stats else 0,
            'avg_confidence': float(stats.avg_confidence) if stats and stats.avg_confidence else 0.0,
            'avg_frame_quality': float(stats.avg_frame_quality) if stats and stats.avg_frame_quality else 0.0
        }

    def soft_delete_person(self, person_id):
        """Soft delete a person"""
        try:
            with self.session_scope() as session:
                person = session.query(Person).filter_by(id=person_id).first()
                if person:
                    person.is_active = False
                    return True
                return False
        except SQLAlchemyError as e:
            raise Exception(f"Error soft deleting person: {str(e)}")

    def update_person(self, person_id, name=None, notes=None, metadata=None, groups=None):
        """Update person information"""
        try:
            with self.session_scope() as session:
                person = session.query(Person).filter_by(id=person_id).first()
                if not person:
                    return None

                if name:
                    person.name = name
                if notes is not None:
                    person.notes = notes
                if metadata is not None:
                    person.person_metadata = json.dumps(metadata)
                if groups is not None:
                    person.groups = [self.get_or_create_group(g) for g in groups]

                return person
        except SQLAlchemyError as e:
            raise Exception(f"Error updating person: {str(e)}")
