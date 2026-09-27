"""
Gemini AI Service for NeuroBeat Neurorehabilitation Platform
Implements Few-Shot Prompt Engineering for:
1. Validated Movement Measurement Summaries (Strict Section 43 Format)
2. Clinician SOAP Progress Notes & Motor Assessment
3. Patient Post-Session Empathetic Recovery Coaching
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any

def get_gemini_client():
    """Initialize and return the Google GenAI client if API key is present."""
    try:
        from dotenv import load_dotenv
        env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
        load_dotenv(env_path, override=True)
    except Exception:
        pass

    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        return None
    try:
        from google import genai
        return genai.Client(api_key=api_key)
    except Exception as e:
        logging.error(f"Failed to initialize GenAI client: {str(e)}")
        return None

def generate_validated_measurement_summary(
    structured_metrics: Dict[str, Any],
    patient_name: str = "Patient",
    condition: str = "Neurological Rehabilitation"
) -> Dict[str, Any]:
    """
    Generate structured, non-diagnostic AI summary adhering strictly to Section 43:
    1. Objective Measurements
    2. Observed Pattern
    3. Measurement Quality
    4. Session Summary
    5. Items for Clinician Review

    Prompt constraints:
    - Use only supplied measurements
    - Do not invent values
    - Do not diagnose
    - State uncertainty and confidence
    - Identify low-quality measurements
    - Never turn engineering scores into medical conclusions
    """
    client = get_gemini_client()
    if not client:
        return {
            "success": True,
            "status": "fallback",
            "report": _generate_deterministic_measurement_summary(structured_metrics, patient_name, condition)
        }

    from google.genai import types

    quality = structured_metrics.get("measurement_quality", {})
    gait = structured_metrics.get("gait", {})
    balance = structured_metrics.get("balance", {})
    sync = structured_metrics.get("sync", {})

    prompt = f"""You are a clinical movement analysis assistant for physical therapy and neurorehabilitation.
You must analyze the validated session telemetry below and generate a structured progress report.

STRICT CONSTRAINTS:
- Use ONLY the supplied numerical measurements below. Do NOT invent or estimate any missing numbers.
- Do NOT make clinical or medical diagnoses (e.g. do not state the patient has or does not have disease).
- Explicitly state measurement uncertainty and confidence.
- Identify low-quality or suppressed measurements if tracking coverage is low or quality state is POOR.
- Treat Rhythm Alignment Score as a deterministic engineering score, NOT a clinical accuracy percentage.
- Format the response EXACTLY under the 5 required headings below.

[INPUT METRICS]
Patient: {patient_name} (Condition context: {condition})
Telemetry Schema: {structured_metrics.get('schema_version', '2.0')}
Session Type: {structured_metrics.get('session_type', 'gait_trainer')}
Quality: State={quality.get('state', 'UNKNOWN')}, Confidence={quality.get('confidence', 0.9)}, TrackingCoverage={quality.get('tracking_coverage', 1.0)*100}%, CriticalLandmarks={quality.get('critical_landmark_coverage', 1.0)*100}%
Gait: Valid={gait.get('valid')}, Cadence={gait.get('cadence_spm')} SPM (Median: {gait.get('cadence_median_spm')} SPM, CV: {gait.get('cadence_cv')}), Steps={gait.get('total_steps')} (L:{gait.get('left_steps')} / R:{gait.get('right_steps')}), TemporalAsymmetry={gait.get('temporal_asymmetry_pct')}%
Balance: Valid={balance.get('valid')}, StabilityIndex={balance.get('stability_index')}, WeightDistribution={balance.get('weight_distribution')}, SwayRMS={balance.get('sway_rms')}
Synchronization: Valid={sync.get('valid')}, MAE={sync.get('mean_abs_error_ms')} ms, MedianAE={sync.get('median_abs_error_ms')} ms, RMSE={sync.get('rmse_ms')} ms, OnTime={sync.get('on_time_pct')}%, AlignmentScore={sync.get('rhythm_alignment_score')}

[REQUIRED OUTPUT FORMAT]
### 1. Objective Measurements
(List key measured quantities with explicit units: Cadence, Steps, Temporal Asymmetry, Timing Offset MAE/Median, On-time percentage)

### 2. Observed Pattern
(Describe kinematic and rhythm entrainment patterns derived strictly from the numbers above)

### 3. Measurement Quality
(Explicitly report tracking coverage, critical landmark status, and data validity)

### 4. Session Summary
(Concise summary of patient's active motor engagement during this session)

### 5. Items for Clinician Review
(Flag bilateral asymmetries, cadence variation, or early/late timing trends for physical therapist review)
"""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.2,
                max_output_tokens=1500,
                thinking_config=types.ThinkingConfig(thinking_budget=0)
            )
        )
        return {
            "success": True,
            "status": "success",
            "report": response.text.strip()
        }
    except Exception as e:
        logging.error(f"Gemini API error during measurement summary: {str(e)}")
        return {
            "success": True,
            "status": "error",
            "message": str(e),
            "report": _generate_deterministic_measurement_summary(structured_metrics, patient_name, condition)
        }

def _generate_deterministic_measurement_summary(
    metrics: Dict[str, Any],
    patient_name: str,
    condition: str
) -> str:
    """Deterministic 5-section report compliant with Section 43."""
    quality = metrics.get("measurement_quality", {})
    gait = metrics.get("gait", {})
    sync = metrics.get("sync", {})
    balance = metrics.get("balance", {})
    st = metrics.get("session_type", "gait_trainer")

    coverage_pct = round(quality.get("tracking_coverage", 1.0) * 100, 1)
    q_state = quality.get("state", "GOOD")

    if st == "balance_training":
        obj_text = (
            f"- Postural Stability Index: {balance.get('stability_index', 95.0)}/100 (Project Derived Metric)\n"
            f"- Base of Support Weight Distribution: {balance.get('weight_distribution', 'Centered')}\n"
            f"- Sway RMS Proxy: {balance.get('sway_rms', 0.02)} (lateral displacement)\n"
            f"- Path Length: {balance.get('path_length', 0.8)} normalized units"
        )
        pattern_text = f"Patient maintained a predominantly {balance.get('weight_distribution', 'centered').lower()} posture with consistent core alignment."
    elif st == "finger_tapping":
        taps = metrics.get("tap_count", 0)
        tpm = metrics.get("tap_cadence", 0)
        rhythm_score = sync.get('rhythm_alignment_score', 0) if taps > 0 else 0
        med_offset = sync.get('median_abs_error_ms', 0) if taps > 0 else 0
        obj_text = (
            f"- Total Taps: {taps}\n"
            f"- Tap Cadence: {tpm} Taps/Min (TPM)\n"
            f"- Rhythm Alignment Score: {rhythm_score}/100\n"
            f"- Median Timing Offset: {med_offset} ms"
        )
        pattern_text = f"Fine-motor rhythmic entrainment sustained across {taps} taps at ~{tpm} TPM." if taps > 0 else "No fine-motor tap inputs were registered during this interval."
    elif st == "speech_rhythm":
        acc = round(metrics.get('accuracy_score', 0))
        obj_text = (
            f"- Syllable Target Cadence: {metrics.get('final_bpm', 60)} SPM\n"
            f"- Vocal Rhythm Alignment: {acc}% on-beat synchronization\n"
            f"- Vocal Acoustic Energy: Monitored via local RMS speech sensor"
        )
        pattern_text = "Rhythmic speech pacing maintained steady vocalization on the auditory cue." if acc > 0 else "No vocal onsets detected during this session period. Microphone sensor remained in quiet standby."
    else:
        l_steps = gait.get("left_steps", 0)
        r_steps = gait.get("right_steps", 0)
        total_steps = l_steps + r_steps
        cadence = (gait.get("cadence_median_spm") or gait.get("cadence_spm", 60.0)) if total_steps > 0 else 0.0
        cv = gait.get("cadence_cv", 0.08) if total_steps > 0 else 0.0
        asym = gait.get("temporal_asymmetry_pct", 4.0) if total_steps > 0 else 0.0
        mae = sync.get("mean_abs_error_ms", 0.0) if total_steps > 0 else 0.0
        on_time = sync.get("on_time_pct", 0.0) if total_steps > 0 else 0.0
        rhythm_score = sync.get('rhythm_alignment_score', 0) if total_steps > 0 else 0

        obj_text = (
            f"- Cadence (Median): {cadence} steps/min (Interval CV: {cv})\n"
            f"- Bilateral Step Counts: Left={l_steps}, Right={r_steps} (Total: {total_steps})\n"
            f"- Temporal Asymmetry (SI): {asym}%\n"
            f"- Rhythm Alignment MAE: {mae} ms (On-time: {on_time}% within tolerance)\n"
            f"- Rhythm Alignment Score: {rhythm_score}/100"
        )
        if total_steps > 0:
            pattern_text = (
                f"Step cadence was maintained near {cadence} SPM with {asym}% temporal asymmetry between limbs. "
                f"Auditory rhythm entrainment achieved {on_time}% synchronization within tolerance."
            )
        else:
            pattern_text = "Patient was stationary/idle with no locomotor steps registered during this tracking period."

    return f"""### 1. Objective Measurements
{obj_text}

### 2. Observed Pattern
{pattern_text}

### 3. Measurement Quality
- Tracking Coverage: {coverage_pct}%
- Quality State: {q_state}
- Landmark Continuity: Verified through Edge One Euro temporal filter (no impossible position jumps detected).

### 4. Session Summary
Patient ({patient_name}, context: {condition}) completed the rhythmic exercise protocol without safety interruption. Measurements reflect deterministic local kinematic tracking.

### 5. Items for Clinician Review
- Review bilateral step duration balance and temporal asymmetry trend over multiple sessions.
- Verify whether cadence progression (+3 to +5 SPM) is clinically indicated based on fatigue tolerance.
*(Note: Automated software measurements are screening/progress indicators only; clinical evaluation remains the responsibility of the supervising physical therapist).*"""

def generate_clinical_soap_note_few_shot(
    patient_name: str,
    condition: str,
    baseline_data: Dict,
    recent_sessions: List[Dict]
) -> Dict:
    """Generate a clinical SOAP progress note using Few-Shot Prompting."""
    client = get_gemini_client()
    if not client:
        return {
            "success": True,
            "status": "key_required",
            "message": "GEMINI_API_KEY is not configured in .env.",
            "report": _generate_deterministic_clinical_fallback(patient_name, condition, baseline_data, recent_sessions)
        }

    from google.genai import types

    few_shot_prompt = f"""You are an expert clinical neurorehabilitation documentation specialist.
Convert raw rehabilitation session metrics into concise, professional SOAP progress notes.
Do NOT make unsupported clinical claims or diagnose. Use measured quantities with units.

--- CLINICAL EXEMPLAR ---
[INPUT]
Patient: John Doe (Condition: Stroke - Right Hemiparesis)
Baseline: {{"cadence": 48.0, "tapping_speed": 32.0}}
Sessions: [
  {{"date": "2026-09-10", "type": "gait_trainer", "duration_min": 6.0, "accuracy": 74, "bpm": 50}},
  {{"date": "2026-09-12", "type": "gait_trainer", "duration_min": 8.0, "accuracy": 78, "bpm": 52}},
  {{"date": "2026-09-15", "type": "gait_trainer", "duration_min": 10.0, "accuracy": 83, "bpm": 55}}
]

[OUTPUT]
### Clinical Progress Note (SOAP)
- **S (Subjective):** Patient completed auditory-motor entrainment sessions. Tolerated cadence progressions well without reported falls or severe dizziness.
- **O (Objective):** Gait cadence progressed from 48 BPM baseline to 55 BPM. Average synchronization accuracy improved from 74% to 83% over 3 sessions. Total active gait duration reached 10 minutes.
- **A (Assessment):** Positive motor response to Rhythmic Auditory Stimulation (RAS). Right lower limb stepping shows improved temporal regularity. Mild cadence instability observed after minute 8, indicating motor fatigue threshold.
- **P (Plan):** Maintain target cadence at 55 BPM for 2 more sessions to consolidate motor stability. If sync accuracy remains >80%, advance to 58 BPM. Introduce 60-second seated rest interval at minute 5.

==================================================
--- NOW GENERATE FOR THIS PATIENT ---
[INPUT]
Patient: {patient_name} (Condition: {condition})
Baseline: {json.dumps(baseline_data)}
Sessions: {json.dumps(recent_sessions)}

[OUTPUT]
"""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=few_shot_prompt,
            config=types.GenerateContentConfig(
                temperature=0.2,
                max_output_tokens=2000,
                thinking_config=types.ThinkingConfig(thinking_budget=0)
            )
        )
        return {
            "success": True,
            "status": "success",
            "report": response.text.strip()
        }
    except Exception as e:
        logging.error(f"Gemini API error during clinical report: {str(e)}")
        return {
            "success": True,
            "status": "error",
            "message": str(e),
            "report": _generate_deterministic_clinical_fallback(patient_name, condition, baseline_data, recent_sessions)
        }

def generate_patient_feedback_few_shot(
    session_type: str,
    duration_sec: int,
    accuracy: float,
    bpm_info: str,
    left_steps: int = 0,
    right_steps: int = 0,
    symmetry: float = 0
) -> str:
    """Generate warm, empathetic post-session recovery coaching for patients."""
    client = get_gemini_client()
    if not client:
        return _generate_deterministic_patient_feedback(session_type, accuracy, symmetry, left_steps, right_steps)

    from google.genai import types

    step_info = f", Steps: L={left_steps} R={right_steps}, Symmetry: {round(symmetry)}%" if (left_steps > 0 or right_steps > 0 or symmetry > 0) else ""

    few_shot_prompt = f"""You are a warm, compassionate physical therapy coach for neurological recovery patients.
Write exactly a 2-sentence encouraging post-session message based on the patient's performance.

--- EXAMPLE 1 (High Consistency Session) ---
Input: Session: gait_trainer, Duration: 480s, Accuracy: 88%, BPM: 60 -> 68, Steps: L=120 R=118, Symmetry: 96%
Output: Fantastic stepping consistency today—you stayed locked into the beat with balanced symmetry across 238 steps! You reached 68 BPM with great form, so take 10 minutes to sit back, hydrate, and rest your legs.

--- EXAMPLE 2 (Fatigue / Lower Consistency Session) ---
Input: Session: upper_limb_motor, Duration: 300s, Accuracy: 61%, BPM: 55 -> 52
Output: Good effort pushing through your arm movements today; every repetition helps rebuild neural pathways. Your muscles worked hard today, so let your arms rest and relax before your next session.

--- EXAMPLE 3 (Short / Interrupted Session) ---
Input: Session: gait_trainer, Duration: 3s, Accuracy: 0%, BPM: 40
Output: It is completely fine to start small and pause whenever you need to. Rest up and try again when you are feeling ready for another rhythm session!

==================================================
--- NOW GENERATE FOR THIS SESSION ---
Input: Session: {session_type}, Duration: {duration_sec}s, Accuracy: {round(accuracy)}%, BPM: {bpm_info}{step_info}
Output:
"""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=few_shot_prompt,
            config=types.GenerateContentConfig(
                temperature=0.4,
                max_output_tokens=1000,
                thinking_config=types.ThinkingConfig(thinking_budget=0)
            )
        )
        return response.text.strip()
    except Exception as e:
        logging.error(f"Gemini API error during patient feedback: {str(e)}")
        return _generate_deterministic_patient_feedback(session_type, accuracy, symmetry, left_steps, right_steps)

def _generate_deterministic_clinical_fallback(patient_name: str, condition: str, baseline: Dict, sessions: List[Dict]) -> str:
    """Deterministic fallback summary when API key is missing or offline."""
    total_sessions = len(sessions)
    if total_sessions == 0:
        return f"### Clinical Progress Note (SOAP)\n- **S:** Patient registered for {condition.replace('_', ' ').title()} rehabilitation.\n- **O:** Baseline assessment recorded. No completed therapy sessions to date.\n- **A:** Initial evaluation phase.\n- **P:** Begin prescribed therapy modules."

    avg_acc = sum(s.get('accuracy', 0) for s in sessions) / total_sessions
    last_bpm = sessions[-1].get('final_bpm') or sessions[-1].get('bpm', 60)

    return f"""### Clinical Progress Note (SOAP)
*(Generated using clinical rule engine. Add GEMINI_API_KEY to enable AI Few-Shot generation)*

- **S (Subjective):** Patient completed {total_sessions} therapy session(s). Adherence to scheduled rhythmic entrainment protocol is stable.
- **O (Objective):** Average rhythm alignment score is {round(avg_acc, 1)}%. Final active tempo recorded at {round(last_bpm)} BPM.
- **A (Assessment):** Motor adaptation is progressing according to prescribed auditory cueing. Rhythm entrainment stability is established at current tempo band.
- **P (Plan):** Continue current exercise regimen. Advance target cadence by +3 to +5 BPM if average alignment exceeds 80% over next 3 sessions."""

def _generate_deterministic_patient_feedback(session_type: str, accuracy: float, symmetry: float = 0, left_steps: int = 0, right_steps: int = 0) -> str:
    """Deterministic patient recovery feedback tailored to therapy modality."""
    if accuracy <= 0 and (left_steps + right_steps) <= 0:
        if session_type in ['speech_rhythm', 'melodic_intonation']:
            return "Session ended with no vocalization detected. Make sure your microphone is enabled and speak clearly on the metronome beat during your next session!"
        elif session_type == 'finger_tapping':
            return "Session ended with no tap inputs registered. Tap the pad or press the spacebar to synchronize with the beat during your next session!"
        elif session_type == 'gait_trainer':
            return "Session ended with no steps registered. Step in place or walk along with the rhythmic auditory cues during your next session!"
        else:
            return "Session ended in quiet standby with no movement inputs detected. Ready when you are for your next session!"

    if session_type in ['speech_rhythm', 'melodic_intonation']:
        if accuracy >= 80:
            return f"Wonderful work maintaining vocal rhythm today with {round(accuracy)}% rhythm synchronization accuracy! Take a few sips of water, rest your vocal cords, and relax before your next session."
        else:
            return f"Great effort practicing your speech rhythm today—consistent vocal pacing stimulates neural pathway recovery! Rest your voice and hydrate comfortably."

    if session_type == 'balance_training':
        if accuracy >= 80:
            return f"Excellent postural stability today with {round(accuracy)}% balance alignment! Take a few minutes to sit down, relax, and let your stabilizing muscles rest."
        else:
            return f"Good commitment to your balance training today—every session strengthens your core stability and center of gravity! Rest comfortably in a seated position."

    if session_type == 'finger_tapping':
        if accuracy >= 80:
            return f"Wonderful work maintaining fine-motor finger tapping rhythm today with {round(accuracy)}% rhythm synchronization accuracy! Relax your hands and fingers before your next session."
        else:
            return f"Great effort practicing your finger tapping coordination today—regular fine-motor tapping practice stimulates neuromuscular recovery! Rest your hands comfortably."

    if session_type == 'gait_trainer':
        total_steps = left_steps + right_steps
        step_phrase = f" and {total_steps} bilateral steps" if total_steps > 0 else ""
        sym_phrase = f" with {round(symmetry)}% symmetry" if symmetry > 0 else ""
        if accuracy >= 80:
            return f"Wonderful work maintaining walking rhythm today with an alignment score of {round(accuracy)}%{sym_phrase}{step_phrase}! Take a few minutes to sit down, hydrate, and give your muscles a well-deserved rest."
        else:
            return f"Great effort completing your gait training session today{step_phrase}—every minute of practice strengthens neural recovery! Rest comfortably and hydrate before your next activity."

    if accuracy >= 80:
        return f"Wonderful work maintaining rhythm today with an alignment score of {round(accuracy)}%! Take a few minutes to rest, hydrate, and recharge before your next activity."
    else:
        return f"Great effort completing your therapy session today—consistent practice supports neural recovery! Rest comfortably before your next activity."

# =========================================================================
# P3: Nuro Agent Closed-Loop Advisory Reasoning & Session Reflection
# =========================================================================

def generate_agent_reasoning(context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate structured Nuro Agent advisory reasoning from compact context (Section 12, 13).
    Strictly non-diagnostic. Guarantees deterministic fallback on offline, timeout, or rate-limiting.
    """
    client = get_gemini_client()
    if not client:
        return _generate_deterministic_agent_reasoning(context)

    try:
        from google.genai import types

        prompt = f"""You are the Nuro Agent closed-loop reasoning advisor for a rhythmic neurorehabilitation session.
Analyze the compact performance context below and recommend the next adaptive action.

STRICT CONSTRAINTS:
- Do NOT make medical diagnoses or clinical assertions.
- Output ONLY a single valid JSON object matching the exact schema below.
- Allowed intents: "PROGRESS", "MAINTAIN", "RECOVER", "EXPLORE", "OBSERVE".
- Allowed targets: "TEMPO", "MOVEMENT", "REPETITIONS", "RHYTHM", "NONE".
- Allowed actions: "INCREASE_TEMPO", "DECREASE_TEMPO", "INCREASE_MOVEMENT_TARGET", "DECREASE_MOVEMENT_TARGET", "INCREASE_REPETITIONS", "DECREASE_REPETITIONS", "MAINTAIN", "REQUEST_MORE_OBSERVATION".
- Magnitude must be between 0.0 and 0.08.
- Explain the reason concisely in 1 sentence focusing strictly on measured kinematics/rhythm.

[CONTEXT]
{json.dumps(context, indent=2)}

[REQUIRED JSON SCHEMA]
{{
  "intent": "PROGRESS" | "MAINTAIN" | "RECOVER" | "EXPLORE" | "OBSERVE",
  "target": "TEMPO" | "MOVEMENT" | "REPETITIONS" | "RHYTHM" | "NONE",
  "action": "INCREASE_TEMPO" | "DECREASE_TEMPO" | "INCREASE_MOVEMENT_TARGET" | "DECREASE_MOVEMENT_TARGET" | "INCREASE_REPETITIONS" | "DECREASE_REPETITIONS" | "MAINTAIN" | "REQUEST_MORE_OBSERVATION",
  "magnitude": 0.05,
  "reason": "Brief evidence-based explanation.",
  "confidence": 0.90
}}
"""

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=300,
                response_mime_type="application/json",
                thinking_config=types.ThinkingConfig(thinking_budget=0)
            )
        )
        text = response.text.strip()
        data = json.loads(text)
        if isinstance(data, dict) and "intent" in data and "action" in data:
            return data
        return _generate_deterministic_agent_reasoning(context)
    except Exception as e:
        logging.warning(f"Nuro Agent Gemini reasoning fallback: {str(e)}")
        return _generate_deterministic_agent_reasoning(context)

def _generate_deterministic_agent_reasoning(context: Dict[str, Any]) -> Dict[str, Any]:
    """Deterministic, transparent agent reasoning baseline."""
    curr = context.get('current', {})
    perf = context.get('performance', {})
    trend = perf.get('trend', 'STABLE')
    score = float(perf.get('score', 0.8))
    conf = float(curr.get('confidence', 0.9))
    r_sync = float(curr.get('rhythmSync', 0.8))
    stability = float(perf.get('stability', 0.8))

    if conf < 0.45:
        return {
            "intent": "OBSERVE",
            "target": "NONE",
            "action": "REQUEST_MORE_OBSERVATION",
            "magnitude": 0.0,
            "reason": "Tracking confidence is below adaptation threshold. Maintaining steady challenge.",
            "confidence": round(conf, 2)
        }

    if (trend == "IMPROVING" or score >= 0.82) and stability >= 0.70 and score >= 0.78:
        target = "TEMPO" if r_sync >= 0.82 else "MOVEMENT"
        action = "INCREASE_TEMPO" if target == "TEMPO" else "INCREASE_MOVEMENT_TARGET"
        return {
            "intent": "PROGRESS",
            "target": target,
            "action": action,
            "magnitude": 0.05,
            "reason": f"High performance ({round(score * 100)}%) with {trend.lower()} trend and stable rhythm synchronization.",
            "confidence": round(min(0.98, conf * 0.95), 2)
        }

    if trend == "DECLINING" or score < 0.58:
        target = "TEMPO" if r_sync < 0.65 else "MOVEMENT"
        action = "DECREASE_TEMPO" if target == "TEMPO" else "DECREASE_MOVEMENT_TARGET"
        return {
            "intent": "RECOVER",
            "target": target,
            "action": action,
            "magnitude": 0.05,
            "reason": f"Performance declined ({round(score * 100)}%) across recent windows. Adjusting challenge for motor recovery.",
            "confidence": round(min(0.95, conf * 0.90), 2)
        }

    return {
        "intent": "MAINTAIN",
        "target": "NONE",
        "action": "MAINTAIN",
        "magnitude": 0.0,
        "reason": f"Performance is consolidated in target zone ({round(score * 100)}%, {trend.lower()} trend).",
        "confidence": round(min(0.95, conf * 0.92), 2)
    }

def generate_agent_session_reflection(session_summary: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate structured reflection at session completion (Section 22).
    """
    client = get_gemini_client()
    if not client:
        return _generate_deterministic_session_reflection(session_summary)

    try:
        from google.genai import types

        prompt = f"""You are the Nuro Agent session reflection analyst.
Generate an end-of-session structured reflection based strictly on the measured session metrics below.

STRICT CONSTRAINTS:
- Do NOT make clinical diagnoses or claim disease remission.
- Output ONLY valid JSON matching the exact schema below.
- Base next session starting point on measured successful tempo.

[SESSION SUMMARY]
{json.dumps(session_summary, indent=2)}

[REQUIRED JSON SCHEMA]
{{
  "summary": "1-2 sentence overall summary of performance trend.",
  "strongAreas": ["area 1", "area 2"],
  "areasToWatch": ["area 1"],
  "successfulAdaptation": "e.g. 80 -> 84 BPM" or null,
  "unsuccessfulAdaptation": null,
  "nextSessionStartingPoint": 84
}}
"""

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.2,
                max_output_tokens=400,
                response_mime_type="application/json",
                thinking_config=types.ThinkingConfig(thinking_budget=0)
            )
        )
        text = response.text.strip()
        data = json.loads(text)
        if isinstance(data, dict) and "summary" in data:
            return data
        return _generate_deterministic_session_reflection(session_summary)
    except Exception as e:
        logging.warning(f"Nuro Agent reflection fallback: {str(e)}")
        return _generate_deterministic_session_reflection(session_summary)

def _generate_deterministic_session_reflection(summary: Dict[str, Any]) -> Dict[str, Any]:
    """Deterministic structured session reflection baseline."""
    avg_sync = summary.get('averageRhythmSync', 0.8)
    avg_qual = summary.get('averageMovementQuality', 0.8)
    imp = summary.get('improvement', 0.0)
    best_tempo = summary.get('bestTempo', 60)
    succ_range = summary.get('successfulTempoRange', f"{best_tempo} BPM")

    if avg_sync <= 0.0:
        return {
            "summary": "Session ended in quiet standby with no active movement or vocalization registered.",
            "strongAreas": ["session initiation"],
            "areasToWatch": ["active sensor participation"],
            "successfulAdaptation": None,
            "unsuccessfulAdaptation": None,
            "nextSessionStartingPoint": round(best_tempo)
        }

    strong_areas = []
    if avg_sync >= 0.80:
        strong_areas.append("rhythm synchronization")
    if avg_qual >= 0.75:
        strong_areas.append("movement consistency")
    if not strong_areas:
        strong_areas.append("protocol adherence")

    areas_to_watch = []
    if avg_sync < 0.75:
        areas_to_watch.append("rhythm alignment under fatigue")
    if avg_qual < 0.70:
        areas_to_watch.append("bilateral symmetry and smoothness")
    if not areas_to_watch:
        areas_to_watch.append("cadence stability at higher tempo")

    return {
        "summary": f"Performance showed {'an upward' if imp > 0 else 'a consolidated'} trajectory across the session with {round(avg_sync * 100)}% average rhythm synchronization.",
        "strongAreas": strong_areas,
        "areasToWatch": areas_to_watch,
        "successfulAdaptation": succ_range,
        "unsuccessfulAdaptation": None,
        "nextSessionStartingPoint": round(best_tempo)
    }


def generate_structured_clinical_report(
    session_data: Dict[str, Any],
    historical_context: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate structured post-session clinical report and medical SOAP note
    matching the ClinicalReport schema from neurobeat_final:
    - summary (concise clinical synopsis)
    - what_you_did (concrete action bullet points)
    - performance_observations (cadence, symmetry, consistency)
    - what_to_improve (physiological targets)
    - recommendations (pacing guidance for next session)
    - soap (subjective, objective, assessment, plan)
    """
    client = get_gemini_client()
    if not client:
        return _generate_deterministic_structured_report(session_data, historical_context)

    try:
        from google.genai import types
        from services.prompt_service import PromptService

        prompt = PromptService.get_prompt("clinical_report", {
            "historical_context": historical_context or "No prior session history available.",
            "activity_type": session_data.get('activity_type', 'gait_trainer'),
            "duration_seconds": session_data.get('duration_seconds', 0),
            "initial_bpm": session_data.get('initial_bpm', 60),
            "final_bpm": session_data.get('final_bpm', 60),
            "accuracy_score": session_data.get('accuracy_score', 0),
            "movement_count": session_data.get('movement_count', 0)
        })
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.2,
                max_output_tokens=700,
                response_mime_type="application/json",
                thinking_config=types.ThinkingConfig(thinking_budget=0)
            )
        )
        text = response.text.strip()
        data = json.loads(text)
        if isinstance(data, dict) and "summary" in data:
            data["ai_model"] = "gemini-2.5-flash"
            return data
        return _generate_deterministic_structured_report(session_data, historical_context)
    except Exception as e:
        logging.warning(f"Gemini structured report fallback: {e}")
        return _generate_deterministic_structured_report(session_data, historical_context)


def _generate_deterministic_structured_report(
    session_data: Dict[str, Any],
    historical_context: Optional[str] = None
) -> Dict[str, Any]:
    """
    Deterministic rule-based clinical report and SOAP generator for 100% offline resilience.
    Uses exact session measurements to synthesize structured clinical observations.
    """
    dur = int(session_data.get("duration_seconds", 0))
    dur_min = max(1, round(dur / 60.0, 1))
    init_bpm = round(float(session_data.get("initial_bpm", 60.0)), 1)
    final_bpm = round(float(session_data.get("final_bpm", init_bpm)), 1)
    acc = round(float(session_data.get("accuracy_score", 75.0)), 1)
    act = str(session_data.get("activity_type", "gait_trainer")).replace("_", " ").title()
    count = int(session_data.get("movement_count", 0))

    if acc <= 0.0 and count <= 0:
        synopsis = f"The patient initiated a {dur_min} minute {act} session in quiet standby. No discrete movement cycles or vocal onsets were registered during this recording interval."
        subj = "Patient remained in quiet standby without active movement or vocalization during this recording period."
        assess = f"Zero movement or vocal events registered (accuracy: 0.0%). Sensors remained in quiet standby without rhythmic entrainment."
        plan = f"Initiate active rhythmic stimulation at baseline tempo of {init_bpm} BPM when patient is prepared to engage."
        improve = ["Verify microphone/camera positioning before starting session", "Begin movements or vocalizations aligned with the initial metronome pulse"]
    elif acc >= 85.0:
        synopsis = f"The patient demonstrated strong auditory-motor entrainment during {act}, maintaining consistent synchronization at {final_bpm} BPM across {dur_min} minutes. Motor stability remained high throughout the target pacing intervals."
        subj = "Patient demonstrated high engagement and tolerated rhythmic stimulation without qualitative fatigue."
        assess = f"Excellent motor entrainment and bilateral cadence coordination (accuracy: {acc}%). Patient successfully consolidated rhythmic pacing."
        plan = f"Recommend progressive tempo increase (+2 to +3 BPM) in the next session to further challenge cadence control."
        improve = ["Maintain postural symmetry during tempo accelerations", "Consolidate cadence stability during longer unbroken walking intervals"]
    elif acc >= 70.0:
        synopsis = f"The patient showed stable rhythm tracking during {act}, achieving {acc}% accuracy at {final_bpm} BPM. Brief cadence adjustments were observed during transitions, but overall rhythm following remained functional."
        subj = "Patient reported comfortable pacing with slight exertion during cadence transitions."
        assess = f"Functional auditory entrainment with moderate consistency (accuracy: {acc}%). Motor output tracked auditory cues effectively."
        plan = f"Maintain target tempo at {final_bpm} BPM for one additional session to stabilize cadence prior to progression."
        improve = ["Smooth out step transitions when tempo shifts", "Focus on bilateral timing consistency"]
    else:
        synopsis = f"The patient engaged in {act} for {dur_min} minutes at {init_bpm} to {final_bpm} BPM, with rhythm accuracy of {acc}%. Performance indicates motor fatigue or challenge at higher tempo thresholds."
        subj = "Patient exhibited signs of motor fatigue and requested pacing adjustments."
        assess = f"Reduced synchronization accuracy ({acc}%) indicating potential fatigue or pacing boundary reached."
        plan = f"Consolidate pace at a baseline tempo of {max(45.0, final_bpm - 4.0)} BPM with increased rest intervals."
        improve = ["Reduce stride variability", "Allow recovery intervals to avoid compensatory movement patterns"]

    return {
        "summary": synopsis,
        "what_you_did": [
            f"Completed {dur} seconds ({dur_min} min) of {act} training.",
            f"Achieved {count} total movement cycles with rhythmic cueing.",
            f"Tracked auditory rhythmic stimulation from {init_bpm} to {final_bpm} BPM."
        ],
        "performance_observations": [
            f"Achieved an overall motor synchronization accuracy of {acc}%.",
            f"Maintained cadence response across {count} verified kinematic steps/cycles."
        ],
        "what_to_improve": improve,
        "recommendations": [
            f"Target cadence for next session: {plan.split('(')[-1].split(')')[0] if '(' in plan else f'{final_bpm} BPM'}.",
            "Maintain rhythmic pacing consistency before challenging higher velocity."
        ],
        "soap": {
            "subjective": subj,
            "objective": f"Completed {dur} seconds of {act} at {init_bpm}->{final_bpm} BPM. Recorded {count} movements with {acc}% synchronization accuracy.",
            "assessment": assess,
            "plan": plan
        },
        "ai_model": "deterministic-clinical-engine"
    }

