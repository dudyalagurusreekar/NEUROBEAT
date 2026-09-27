from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from backend.database import get_db
from backend.models import TherapySession, MovementEvent, AdaptationEvent, PatientProfile
from backend.schemas import (
    SessionCreateRequest, SessionEventsPush, SessionTelemetryPush,
    SessionCompleteRequest, TherapySessionSchema
)

router = APIRouter(prefix="/sessions", tags=["sessions"])

@router.post("", response_model=TherapySessionSchema)
def create_session(req: SessionCreateRequest, db: Session = Depends(get_db)):
    patient = db.query(PatientProfile).filter(PatientProfile.id == req.patient_id).first()
    if not patient:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": True, "code": "PATIENT_NOT_FOUND", "message": "Patient not found"}
        )

    session = TherapySession(
        patient_id=req.patient_id,
        session_type=req.session_type,
        status="in_progress",
        start_time=datetime.utcnow(),
        initial_bpm=req.initial_bpm,
        final_bpm=req.initial_bpm,
        target_bpm=req.target_bpm
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    return TherapySessionSchema.from_orm(session)

@router.post("/{session_id}/events")
def push_session_events(session_id: int, push_data: SessionEventsPush, db: Session = Depends(get_db)):
    session = db.query(TherapySession).filter(TherapySession.id == session_id).first()
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": True, "code": "SESSION_NOT_FOUND", "message": f"Session with ID {session_id} not found"}
        )

    valid_sync_scores = []
    for ev in push_data.events:
        db_event = MovementEvent(
            session_id=session_id,
            timestamp=ev.timestamp,
            event_type=ev.type,
            side=ev.side,
            confidence=ev.confidence,
            matched_beat_timestamp=ev.matched_beat_timestamp,
            timing_error_ms=ev.timing_error_ms,
            sync_score=ev.sync_score,
            phase=ev.phase
        )
        db.add(db_event)
        if ev.sync_score is not None:
            valid_sync_scores.append(ev.sync_score)

    session.total_steps += len(push_data.events)
    if valid_sync_scores:
        for score in valid_sync_scores:
            if session.avg_sync_score == 0:
                session.avg_sync_score = float(score)
            else:
                session.avg_sync_score = round((session.avg_sync_score * 0.85) + (score * 0.15), 1)

    db.commit()

    return {"success": True, "events_recorded": len(push_data.events), "total_steps": session.total_steps}

@router.post("/{session_id}/telemetry")
def push_session_telemetry(session_id: int, telemetry: SessionTelemetryPush, db: Session = Depends(get_db)):
    session = db.query(TherapySession).filter(TherapySession.id == session_id).first()
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": True, "code": "SESSION_NOT_FOUND", "message": f"Session with ID {session_id} not found"}
        )

    # Incrementally update session aggregate telemetry
    new_total = (telemetry.gait.left_steps or 0) + (telemetry.gait.right_steps or 0)
    if new_total > session.total_steps:
        session.total_steps = new_total

    # Only aggregate valid synchronization events, never idle camera frames
    if getattr(telemetry.sync, 'valid', False) and telemetry.sync.score is not None:
        if session.avg_sync_score == 0:
            session.avg_sync_score = float(telemetry.sync.score)
        else:
            session.avg_sync_score = round((session.avg_sync_score * 0.85) + (telemetry.sync.score * 0.15), 1)

    db.commit()
    return {"success": True, "session_id": session_id, "timestamp": telemetry.timestamp}

@router.post("/{session_id}/complete", response_model=TherapySessionSchema)
def complete_session(session_id: int, req: SessionCompleteRequest, db: Session = Depends(get_db)):
    session = db.query(TherapySession).filter(TherapySession.id == session_id).first()
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": True, "code": "SESSION_NOT_FOUND", "message": f"Session with ID {session_id} not found"}
        )

    session.status = "completed"
    session.end_time = datetime.utcnow()
    session.final_bpm = req.final_bpm
    session.duration_seconds = req.duration_seconds
    session.avg_sync_score = req.avg_sync_score
    session.total_steps = req.total_steps
    session.freezing_events_count = req.freezing_events_count
    session.notes = req.notes

    # Simple rule-based default summary if none yet
    mins = req.duration_seconds // 60
    secs = req.duration_seconds % 60
    session.ai_summary = (
        f"Session completed in {mins}m {secs}s. "
        f"Gait cadence progressed from {session.initial_bpm:.0f} to {req.final_bpm:.0f} BPM "
        f"with an average synchronization accuracy of {req.avg_sync_score:.0f}%. "
        f"{req.freezing_events_count} freezing episodes detected."
    )

    db.commit()
    db.refresh(session)
    return TherapySessionSchema.from_orm(session)

@router.get("/{session_id}", response_model=TherapySessionSchema)
def get_session_detail(session_id: int, db: Session = Depends(get_db)):
    session = db.query(TherapySession).filter(TherapySession.id == session_id).first()
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": True, "code": "SESSION_NOT_FOUND", "message": f"Session with ID {session_id} not found"}
        )
    return TherapySessionSchema.from_orm(session)
