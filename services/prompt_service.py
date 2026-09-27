"""
NURO-BEATS Prompt Configuration and Orchestration Layer
services/prompt_service.py

Centralized prompt registry supporting:
1. session_planner: Therapy session target BPM, progression, and safety envelope
2. session_report: Post-session patient feedback and recovery coaching
3. clinical_report: Medical SOAP progress notes and 4 structured observation cards
4. progress_interpretation: Longitudinal clinical trend evaluation
5. rhythm_audio_generation: Hugging Face MusicGen neural conditioning

Enforces strict clinical guidelines:
- Use only real supplied data
- Never fabricate measurements or diagnoses
- Never claim clinical efficacy without medical supervision
"""

import json
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger("PromptService")


DEFAULT_PROMPT_TEMPLATES = {
    "session_planner": """You are a senior physical therapist and neurorehabilitation clinician planning a rhythmic auditory stimulation session.
Condition: {condition}
Baseline Cadence: {baseline_cadence} BPM
Target Cadence: {target_cadence} BPM
Session Type: {session_type}
Recent Accuracy Trend: {trend}
Last Session Cadence: {last_cadence} BPM with {last_accuracy}% accuracy

STRICT CLINICAL RULES:
1. Recommend an initial pacing BPM and progression schedule based strictly on the patient's verified metrics.
2. If recent accuracy was < 70%, recommend cadence consolidation (maintain or -2 BPM).
3. If recent accuracy was >= 80%, recommend gentle progression (+2 to +3 BPM).
4. Never exceed safety bounds [40.0, 140.0] BPM.
5. Output structured JSON with keys: "starting_bpm", "target_bpm", "warmup_seconds", "pacing_strategy", "clinical_rationale".""",

    "session_report": """You are an empathetic, encouraging physical therapy coach for NeuroBeat neuro-rehabilitation.
Provide warm, encouraging, and specific feedback for a patient completing a session.

Session Details:
- Activity: {session_type}
- Duration: {duration_seconds} seconds
- Accuracy / Rhythm Sync: {accuracy_score}%
- Cadence Transition: {tempo_progression}
- Gait / Motor Metrics: Left Steps={left_steps}, Right Steps={right_steps}, Symmetry={symmetry}%

CONSTRAINTS:
- Keep feedback concise, uplifting, and actionable (2-3 sentences max).
- Reference their actual numbers (e.g. cadence or accuracy) accurately.
- Never give pharmaceutical advice or formal medical diagnoses.
- Celebrate motor consistency and rhythmic entrainment effort.""",

    "clinical_report": """You are a physical medicine and neurorehabilitation clinician generating an official clinical session report.
Patient History:
{historical_context}

Current Session Telemetry:
- Activity Type: {activity_type}
- Duration: {duration_seconds} seconds
- Cadence: Initial={initial_bpm} BPM, Final={final_bpm} BPM
- Rhythm Synchronization Score: {accuracy_score}%
- Total Movements / Steps: {movement_count}

Generate a comprehensive clinical report formatted STRICTLY as JSON with these keys:
{{
  "summary": "1-2 sentence executive clinical summary citing exact telemetry",
  "what_you_did": [
    "Completed X seconds of rhythmic auditory stimulation",
    "Tracked audio cues from X to Y BPM",
    "Completed N total movement cycles"
  ],
  "performance_observations": [
    "Observation on cadence stability and auditory-motor entrainment consistency",
    "Observation on motor rhythm maintenance across tempo changes"
  ],
  "what_to_improve": [
    "Kinematic focus area for next session (e.g. step cadence consistency, phase lag reduction)"
  ],
  "recommendations": [
    "Specific next session starting cadence recommendation",
    "Target practice tempo and pacing advice"
  ],
  "soap": {{
    "subjective": "Patient engagement and exertion tolerance based on duration and completed effort",
    "objective": "Verified telemetry: duration, initial/final BPM, accuracy %, movement count",
    "assessment": "Clinical assessment of auditory-motor entrainment, fatigue indicators, and cadence stability",
    "plan": "Specific next session cadence targets, resting intervals, and exercise focus"
  }}
}}

STRICT RULES:
- Never diagnose medical conditions or prescribe medications.
- Base all statements strictly on the supplied quantitative measurements.
- Return ONLY valid JSON.""",

    "progress_interpretation": """You are a clinical biomechanist evaluating longitudinal progress in neurorehabilitation.
Patient Baseline Cadence: {baseline_cadence} SPM
Target Cadence: {target_cadence} SPM
Total Completed Sessions: {total_sessions}
Average Accuracy: {avg_accuracy}%
Longitudinal Trend: {trend}
Recent Accuracies: {recent_accuracies}

Provide a clinical progress evaluation covering:
1. Cadence Progression: Rate of entrainment adaptability toward target.
2. Synchronization Consistency: Variance and stability across sessions.
3. Fatigue / Tolerance: Signs of plateau or sustained endurance.
4. Next Milestone Recommendation: Suggested tempo milestones and focus areas.
Ensure all metrics match the supplied data exactly.""",

    "rhythm_audio_generation": """{prompt}, steady {bpm} BPM rhythmic walking cue, therapeutic auditory entrainment, 44.1kHz stereo high fidelity"""
}


class PromptService:
    """Centralized prompt registry and templating engine."""
    
    _templates: Dict[str, str] = dict(DEFAULT_PROMPT_TEMPLATES)
    _descriptions: Dict[str, str] = {
        "session_planner": "Generates therapy session pacing goals, BPM bounds, and safety parameters.",
        "session_report": "Empathetic post-session patient reflection and recovery coaching.",
        "clinical_report": "Medical SOAP documentation and 4 structured clinical observation cards.",
        "progress_interpretation": "Longitudinal clinical trend evaluation across multiple sessions.",
        "rhythm_audio_generation": "Hugging Face MusicGen conditioning prompt for therapeutic beat generation."
    }

    @classmethod
    def get_template(cls, prompt_type: str) -> str:
        """Retrieve the raw template string for a prompt type."""
        return cls._templates.get(prompt_type, DEFAULT_PROMPT_TEMPLATES.get(prompt_type, ""))

    @classmethod
    def set_template(cls, prompt_type: str, template: str) -> bool:
        """Update a prompt template at runtime."""
        if not prompt_type or not isinstance(template, str) or not template.strip():
            return False
        cls._templates[prompt_type] = template.strip()
        logger.info(f"[PromptService] Updated template for '{prompt_type}' ({len(template)} chars).")
        return True

    @classmethod
    def get_prompt(cls, prompt_type: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Render a prompt template with the provided context dictionary."""
        template = cls.get_template(prompt_type)
        if not template:
            return ""
        if not context:
            return template

        # Safe formatting with fallback for missing keys
        class SafeDict(dict):
            def __missing__(self, key):
                return f"{{{key}}}"

        try:
            return template.format_map(SafeDict(context))
        except Exception as e:
            logger.warning(f"[PromptService] Error formatting prompt '{prompt_type}': {e}. Returning raw template.")
            return template

    @classmethod
    def get_all_prompts(cls) -> Dict[str, Any]:
        """Return all prompt templates with metadata."""
        return {
            k: {
                "template": cls._templates.get(k, ""),
                "description": cls._descriptions.get(k, ""),
                "is_customized": cls._templates.get(k) != DEFAULT_PROMPT_TEMPLATES.get(k)
            }
            for k in DEFAULT_PROMPT_TEMPLATES
        }

    @classmethod
    def reset_to_default(cls, prompt_type: Optional[str] = None):
        """Reset one or all templates to built-in defaults."""
        if prompt_type:
            if prompt_type in DEFAULT_PROMPT_TEMPLATES:
                cls._templates[prompt_type] = DEFAULT_PROMPT_TEMPLATES[prompt_type]
        else:
            cls._templates = dict(DEFAULT_PROMPT_TEMPLATES)
