"""
NURO-BEATS Longitudinal Intelligence & Historical Analysis Layer
services/historical_analysis.py

Implements 9-feature longitudinal clinical intelligence without external vector databases:
1. Structured Report Storage (SOAP + bullet points)
2. Session History Retrieval
3. Deterministic Trend Analysis (improving, declining, stable)
4. Advisory Recommended BPM
5. Contextual Accuracy Comparison
6. Longitudinal AI Report Context Builder
7. In-Memory Prototype Demo Mode
8. Zero External Vector DB (Pure SQL / SQLAlchemy)
9. Non-Invasive API Integration helpers
"""

import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from app import db
from models import TherapySession, ClinicalReport, PatientProfile

logger = logging.getLogger("HistoricalAnalysis")


def get_patient_history(
    patient_id: int,
    activity_type: Optional[str] = None,
    limit: int = 10
) -> List[Dict[str, Any]]:
    """
    Feature 2: Retrieve the patient's recent completed sessions.
    Filtered by activity_type if provided.
    """
    query = TherapySession.query.filter_by(
        patient_id=patient_id,
        completed=True
    )
    if activity_type:
        query = query.filter_by(session_type=activity_type)

    sessions = query.order_by(TherapySession.end_time.desc()).limit(limit).all()

    history = []
    for s in sessions:
        history.append({
            "session_id": s.id,
            "activity_type": s.session_type,
            "duration_seconds": s.duration_seconds or 0,
            "initial_bpm": s.initial_bpm or 60.0,
            "final_bpm": s.final_bpm or s.initial_bpm or 60.0,
            "target_bpm": s.target_bpm or 60.0,
            "accuracy_score": round(float(s.accuracy_score or 0.0), 1),
            "date": s.end_time.strftime("%Y-%m-%d %H:%M") if s.end_time else None
        })
    return history


def compute_accuracy_trend(history: List[Dict[str, Any]]) -> str:
    """
    Feature 3: Deterministic Trend Analysis.
    Compares recent session accuracies against older baseline:
    - 'improving' if recent average > baseline average + 2.0%
    - 'declining' if recent average < baseline average - 2.0%
    - 'stable' otherwise
    """
    if len(history) < 2:
        return "stable"

    accuracies = [h["accuracy_score"] for h in history if h.get("accuracy_score") is not None]
    if len(accuracies) < 2:
        return "stable"

    # Reverse to chronological order (oldest to newest)
    chronological = list(reversed(accuracies))
    midpoint = len(chronological) // 2

    baseline_avg = sum(chronological[:midpoint]) / max(1, len(chronological[:midpoint]))
    recent_avg = sum(chronological[midpoint:]) / max(1, len(chronological[midpoint:]))

    diff = recent_avg - baseline_avg
    if diff > 2.0:
        return "improving"
    elif diff < -2.0:
        return "declining"
    return "stable"


def calculate_advisory_bpm(
    current_bpm: float,
    accuracy_score: float,
    trend: str,
    min_bpm: float = 40.0,
    max_bpm: float = 140.0
) -> float:
    """
    Feature 4: Advisory Recommended BPM for next session.
    Safe, advisory pacing recommendation:
    - High accuracy (>= 80%) & improving/stable: +2 to +3 BPM
    - Low accuracy (< 60%): -2 to -4 BPM for consolidation
    - Clamped strictly within [min_bpm, max_bpm]
    """
    base_bpm = float(current_bpm)

    if accuracy_score >= 85.0 and trend == "improving":
        delta = 3.0
    elif accuracy_score >= 80.0:
        delta = 2.0
    elif accuracy_score < 60.0:
        delta = -3.0
    elif accuracy_score < 70.0 and trend == "declining":
        delta = -2.0
    else:
        delta = 0.0

    advisory = base_bpm + delta
    return max(min_bpm, min(max_bpm, advisory))


def calculate_accuracy_delta(
    current_accuracy: float,
    history: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Feature 5: Contextual Accuracy Comparison.
    Computes difference between current accuracy and historical average.
    """
    if not history:
        return {
            "historical_average": current_accuracy,
            "delta": 0.0,
            "text": "First session on record"
        }

    valid_scores = [h["accuracy_score"] for h in history if h.get("accuracy_score") is not None]
    if not valid_scores:
        avg_score = current_accuracy
    else:
        avg_score = sum(valid_scores) / len(valid_scores)

    delta = current_accuracy - avg_score
    sign = "+" if delta >= 0 else ""
    return {
        "historical_average": round(avg_score, 1),
        "delta": round(delta, 1),
        "text": f"{sign}{delta:.1f}% vs historical average ({avg_score:.1f}%)"
    }


def format_historical_context_for_prompt(
    patient_id: int,
    activity_type: Optional[str] = None
) -> str:
    """
    Feature 6: Longitudinal AI Report Enrichment.
    Creates a concise contextual prompt injection for Gemini reporting.
    """
    history = get_patient_history(patient_id, activity_type, limit=5)
    if not history:
        return "No prior session history available for this patient. Treat as initial baseline."

    trend = compute_accuracy_trend(history)
    accuracies = [f"{h['accuracy_score']}%" for h in reversed(history)]

    context = (
        f"PATIENT RECENT HISTORY ({len(history)} sessions):\n"
        f"- Recent Accuracies (chronological): {' -> '.join(accuracies)}\n"
        f"- Overall Longitudinal Trend: {trend.upper()}\n"
        f"- Last Session Target: {history[0]['final_bpm']} BPM with {history[0]['accuracy_score']}% accuracy\n"
        f"Ensure clinical report reflects whether patient is showing entrainment progress over time."
    )
    return context


def save_or_update_clinical_report(
    session_id: int,
    report_dict: Dict[str, Any]
) -> ClinicalReport:
    """
    Feature 1: Structured Report Storage with Idempotency.
    Updates existing row if session already has a report, otherwise creates a new one.
    """
    ts = db.session.get(TherapySession, session_id) if hasattr(db.session, 'get') else TherapySession.query.get(session_id)
    if not ts:
        raise ValueError(f"TherapySession #{session_id} not found.")

    report = ClinicalReport.query.filter_by(session_id=session_id).first()
    if not report:
        report = ClinicalReport(
            session_id=session_id,
            patient_id=ts.patient_id
        )
        db.session.add(report)

    # Populate quantitative session data
    report.activity_type = ts.session_type
    report.duration_seconds = ts.duration_seconds or 0
    report.initial_bpm = ts.initial_bpm
    report.final_bpm = ts.final_bpm or ts.initial_bpm
    report.target_bpm = ts.target_bpm
    report.accuracy_score = ts.accuracy_score or 0.0
    report.movement_count = int(report_dict.get("movement_count", ts.total_steps or 0))

    # Structured clinical bullet points
    def _to_json_or_text(val):
        if isinstance(val, (list, dict)):
            return json.dumps(val)
        return str(val) if val is not None else ""

    report.summary = _to_json_or_text(report_dict.get("summary", ""))
    report.what_you_did = _to_json_or_text(report_dict.get("what_you_did", []))
    report.performance_observations = _to_json_or_text(report_dict.get("performance_observations", []))
    report.what_to_improve = _to_json_or_text(report_dict.get("what_to_improve", []))
    report.recommendations = _to_json_or_text(report_dict.get("recommendations", []))

    # SOAP documentation
    soap = report_dict.get("soap") or {}
    report.soap_subjective = str(soap.get("subjective", ""))
    report.soap_objective = str(soap.get("objective", f"Completed {report.duration_seconds}s at {report.final_bpm} BPM with {report.accuracy_score}% accuracy."))
    report.soap_assessment = str(soap.get("assessment", ""))
    report.soap_plan = str(soap.get("plan", ""))

    report.ai_model = str(report_dict.get("ai_model", "gemini-2.5-flash"))
    report.created_at = datetime.utcnow()

    db.session.commit()
    return report


def get_mock_trajectory_for_demo() -> List[Dict[str, Any]]:
    """
    Feature 7: Prototype Demo Mode.
    Returns 4 synthetic sessions showing positive trajectory (68% -> 80% accuracy)
    with zero database writes.
    """
    return [
        {"session_id": 901, "activity_type": "gait_trainer", "accuracy_score": 68.0, "final_bpm": 58.0, "date": "2026-09-20"},
        {"session_id": 902, "activity_type": "gait_trainer", "accuracy_score": 72.0, "final_bpm": 60.0, "date": "2026-09-22"},
        {"session_id": 903, "activity_type": "gait_trainer", "accuracy_score": 77.0, "final_bpm": 62.0, "date": "2026-09-24"},
        {"session_id": 904, "activity_type": "gait_trainer", "accuracy_score": 81.5, "final_bpm": 64.0, "date": "2026-09-26"}
    ]
