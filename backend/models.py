from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship
from backend.database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(80), unique=True, index=True, nullable=False)
    email = Column(String(120), unique=True, index=True, nullable=False)
    hashed_password = Column(String(256), nullable=False)
    role = Column(String(20), default="patient") # 'patient' or 'clinician'
    full_name = Column(String(100), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    patient_profile = relationship("PatientProfile", foreign_keys="[PatientProfile.user_id]", back_populates="user", uselist=False)

class PatientProfile(Base):
    __tablename__ = "patient_profiles"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True, nullable=False)
    condition = Column(String(50), default="parkinsons") # 'parkinsons' or 'stroke'
    
    # Baseline & targets
    baseline_cadence = Column(Float, default=44.0)
    target_cadence = Column(Float, default=54.0)
    
    # Clinician-Set Safety Envelope (Human-In-The-Loop)
    min_safe_bpm = Column(Float, default=45.0)
    max_safe_bpm = Column(Float, default=72.0)
    max_duration_mins = Column(Integer, default=15)
    freezing_tolerance = Column(Integer, default=2)

    assigned_clinician_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", foreign_keys=[user_id], back_populates="patient_profile")
    sessions = relationship("TherapySession", back_populates="patient", cascade="all, delete-orphan")
    assessments = relationship("BaselineAssessment", back_populates="patient", cascade="all, delete-orphan")

class BaselineAssessment(Base):
    __tablename__ = "baseline_assessments"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patient_profiles.id"), nullable=False)
    assessment_type = Column(String(50), default="gait") # 'gait', 'tapping', 'speech', 'balance'
    measured_value = Column(Float, nullable=False)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    patient = relationship("PatientProfile", back_populates="assessments")

class TherapySession(Base):
    __tablename__ = "therapy_sessions"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patient_profiles.id"), nullable=False)
    session_type = Column(String(50), default="gait") # 'gait', 'tapping', 'speech', 'balance'
    status = Column(String(20), default="in_progress") # 'in_progress', 'completed', 'paused'
    
    start_time = Column(DateTime, default=datetime.utcnow)
    end_time = Column(DateTime, nullable=True)
    
    initial_bpm = Column(Float, default=50.0)
    final_bpm = Column(Float, default=50.0)
    target_bpm = Column(Float, default=54.0)
    
    duration_seconds = Column(Integer, default=0)
    avg_sync_score = Column(Float, default=0.0)
    total_steps = Column(Integer, default=0)
    freezing_events_count = Column(Integer, default=0)
    
    ai_summary = Column(Text, nullable=True)
    notes = Column(Text, nullable=True)

    patient = relationship("PatientProfile", back_populates="sessions")
    movement_events = relationship("MovementEvent", back_populates="session", cascade="all, delete-orphan")
    adaptation_events = relationship("AdaptationEvent", back_populates="session", cascade="all, delete-orphan")

class MovementEvent(Base):
    __tablename__ = "movement_events"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("therapy_sessions.id"), nullable=False)
    timestamp = Column(Float, nullable=False) # Session timestamp in seconds
    event_type = Column(String(20), default="STEP") # 'STEP', 'TAP', 'VOICE'
    side = Column(String(10), nullable=True) # 'LEFT', 'RIGHT'
    confidence = Column(Float, default=1.0)
    matched_beat_timestamp = Column(Float, nullable=True)
    timing_error_ms = Column(Float, nullable=True)
    sync_score = Column(Float, nullable=True)
    phase = Column(String(20), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    session = relationship("TherapySession", back_populates="movement_events")

class AdaptationEvent(Base):
    __tablename__ = "adaptation_events"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("therapy_sessions.id"), nullable=False)
    timestamp = Column(Float, nullable=False) # Session timestamp in seconds
    previous_bpm = Column(Float, nullable=False)
    new_bpm = Column(Float, nullable=False)
    trigger_reason = Column(String(255), nullable=False)
    sync_score = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    session = relationship("TherapySession", back_populates="adaptation_events")
