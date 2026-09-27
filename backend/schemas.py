from typing import Optional, List
from pydantic import BaseModel, Field
from datetime import datetime

# --- Auth ---
class LoginRequest(BaseModel):
    username: str
    password: str

class UserSchema(BaseModel):
    id: int
    username: str
    email: str
    role: str
    full_name: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True

class AuthResponse(BaseModel):
    token: str
    user: UserSchema
    patient_id: Optional[int] = None

# --- Patient & Profile ---
class PatientProfileSchema(BaseModel):
    id: int
    user_id: int
    condition: str
    baseline_cadence: float
    target_cadence: float
    min_safe_bpm: float
    max_safe_bpm: float
    max_duration_mins: int
    freezing_tolerance: int
    assigned_clinician_id: Optional[int] = None
    full_name: Optional[str] = None

    class Config:
        from_attributes = True

class SafetyEnvelopeUpdate(BaseModel):
    min_safe_bpm: Optional[float] = None
    max_safe_bpm: Optional[float] = None
    max_duration_mins: Optional[int] = None
    freezing_tolerance: Optional[int] = None

class BaselineSchema(BaseModel):
    baseline_cadence: float
    target_cadence: float
    recommended_start_bpm: float

# --- Sessions ---
class SessionCreateRequest(BaseModel):
    patient_id: int
    session_type: str = "gait"
    initial_bpm: float = 50.0
    target_bpm: float = 54.0

class MovementEventItem(BaseModel):
    timestamp: float
    type: str = "STEP"
    side: Optional[str] = "LEFT"
    confidence: float = 0.95
    matched_beat_timestamp: Optional[float] = None
    timing_error_ms: Optional[float] = None
    sync_score: Optional[float] = None
    phase: Optional[str] = None

class SessionEventsPush(BaseModel):
    events: List[MovementEventItem]

# --- Telemetry ---
class PoseTelemetry(BaseModel):
    confidence: float = 0.0
    tracking_state: str = "LOST"

class MovementTelemetry(BaseModel):
    state: str = "STATIONARY"
    confidence: float = 0.0
    quality: int = 85

class GaitTelemetry(BaseModel):
    cadence_spm: Optional[float] = None
    left_steps: int = 0
    right_steps: int = 0
    balance: int = 100

class SyncTelemetry(BaseModel):
    target_bpm: int = 60
    score: Optional[float] = None
    timing_error_ms: Optional[float] = None
    valid: bool = False
    phase: Optional[str] = None

class AudioTelemetry(BaseModel):
    level: float = 0.0
    activity: bool = False
    confidence: float = 0.8

class PerformanceTelemetry(BaseModel):
    camera_fps: float = 0.0
    pose_fps: float = 0.0
    inference_latency_ms: float = 0.0
    dropped_frames: int = 0

class SessionTelemetryPush(BaseModel):
    session_id: Optional[int] = None
    timestamp: float
    pose: PoseTelemetry = PoseTelemetry()
    movement: MovementTelemetry = MovementTelemetry()
    gait: GaitTelemetry = GaitTelemetry()
    sync: SyncTelemetry = SyncTelemetry()
    audio: AudioTelemetry = AudioTelemetry()
    performance: PerformanceTelemetry = PerformanceTelemetry()

class SessionCompleteRequest(BaseModel):
    final_bpm: float
    duration_seconds: int
    avg_sync_score: float
    total_steps: int
    freezing_events_count: int = 0
    notes: Optional[str] = None

class TherapySessionSchema(BaseModel):
    id: int
    patient_id: int
    session_type: str
    status: str
    start_time: datetime
    end_time: Optional[datetime] = None
    initial_bpm: float
    final_bpm: float
    target_bpm: float
    duration_seconds: int
    avg_sync_score: float
    total_steps: int
    freezing_events_count: int
    ai_summary: Optional[str] = None
    notes: Optional[str] = None

    class Config:
        from_attributes = True

# --- AI & Summaries (Strictly Bounded) ---
class SessionSummaryRequest(BaseModel):
    session_id: int
    patient_name: str = "Patient"
    duration_seconds: int
    start_bpm: float
    final_bpm: float
    avg_sync_score: float
    total_steps: int
    freezing_events: int = 0
    cadence_improvement: float = 0.0

class SessionSummaryResponse(BaseModel):
    summary: str
    clinician_notes: str
    encouraging_cue: str

class RhythmConfigRequest(BaseModel):
    preferred_style: str = "acoustic" # 'acoustic', 'ambient', 'metronome', 'drum'
    target_bpm: float = 54.0
    patient_condition: str = "parkinsons"

class RhythmConfigResponse(BaseModel):
    style: str
    bpm: float
    accent_pattern: str
    texture: str
    therapeutic_focus: str

# --- Adaptation & ML Recommendation ---
class AdaptationRecommendationRequest(BaseModel):
    patient_id: int
    recent_sync_scores: List[float] = []
    current_bpm: float
    baseline_cadence: float

class AdaptationRecommendationResponse(BaseModel):
    recommended_bpm: float
    recommended_difficulty: str
    target_cadence: float
    rationale: str
