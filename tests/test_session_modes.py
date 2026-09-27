import unittest
from session_modes import (
    SESSION_MODES,
    UNKNOWN_SESSION_MODE,
    normalize_session_type,
    get_session_mode,
    is_camera_required,
    is_microphone_required
)
from services.gemini_service import _generate_deterministic_patient_feedback

class TestSessionModes(unittest.TestCase):
    def test_camera_requirements(self):
        """Camera must ONLY be required for gait_trainer, balance_training, and upper_limb_motor."""
        self.assertTrue(is_camera_required("gait_trainer"))
        self.assertTrue(is_camera_required("balance_training"))
        self.assertTrue(is_camera_required("upper_limb_motor"))
        
        # Non-camera modes MUST NOT require camera
        self.assertFalse(is_camera_required("finger_tapping"))
        self.assertFalse(is_camera_required("speech_rhythm"))
        self.assertFalse(is_camera_required("melodic_intonation"))
        self.assertFalse(is_camera_required("cognitive_rhythm"))
        self.assertFalse(is_camera_required("unknown"))

    def test_microphone_requirements(self):
        """Microphone must only be required for speech rhythm and melodic intonation."""
        self.assertTrue(is_microphone_required("speech_rhythm"))
        self.assertTrue(is_microphone_required("melodic_intonation"))
        
        self.assertFalse(is_microphone_required("gait_trainer"))
        self.assertFalse(is_microphone_required("balance_training"))
        self.assertFalse(is_microphone_required("finger_tapping"))

    def test_normalization_aliases(self):
        """Normalization maps common aliases accurately."""
        self.assertEqual(normalize_session_type("gait"), "gait_trainer")
        self.assertEqual(normalize_session_type("walking"), "gait_trainer")
        self.assertEqual(normalize_session_type("balance"), "balance_training")
        self.assertEqual(normalize_session_type("posture"), "balance_training")
        self.assertEqual(normalize_session_type("tapping"), "finger_tapping")
        self.assertEqual(normalize_session_type("finger-tapping"), "finger_tapping")
        self.assertEqual(normalize_session_type("speech"), "speech_rhythm")
        self.assertEqual(normalize_session_type("vocal"), "speech_rhythm")

    def test_unknown_normalization_safety(self):
        """Unknown or invalid types must normalize to 'unknown' and never fallback to gait_trainer."""
        self.assertEqual(normalize_session_type("random_invalid_mode"), "unknown")
        self.assertEqual(normalize_session_type(""), "unknown")
        self.assertEqual(normalize_session_type(None), "unknown")
        
        mode = get_session_mode("random_invalid_mode")
        self.assertEqual(mode["key"], "unknown")
        self.assertNotEqual(mode["title"], "Gait Trainer")
        self.assertFalse(mode["camera_required"])
        self.assertFalse(is_camera_required("random_invalid_mode"))

    def test_mode_separation_titles_and_metrics(self):
        """Each of the 4 modes has distinct titles, subtitles, instructions and metrics."""
        gait = get_session_mode("gait_trainer")
        balance = get_session_mode("balance_training")
        tapping = get_session_mode("finger_tapping")
        speech = get_session_mode("speech_rhythm")

        self.assertEqual(gait["title"], "Gait Trainer")
        self.assertEqual(balance["title"], "Balance Training")
        self.assertEqual(tapping["title"], "Finger Tapping")
        self.assertEqual(speech["title"], "Speech Rhythm")

        # Instructions must be modality-specific
        self.assertIn("Walk in place", gait["instructions"])
        self.assertIn("postural stability", balance["instructions"])
        self.assertIn("tap pad", tapping["instructions"])
        self.assertIn("syllables rhythmically", speech["instructions"])

        # Metric labels must be distinct
        self.assertEqual(gait["primary_metric_label"], "Steps (L/R)")
        self.assertEqual(balance["primary_metric_label"], "Posture Sway")
        self.assertEqual(tapping["primary_metric_label"], "Total Taps")
        self.assertEqual(speech["primary_metric_label"], "Pacing (SPM)")

    def test_deterministic_patient_feedback_mode_separation(self):
        """Feedback generation must not show steps or gait symmetry in non-gait sessions."""
        speech_fb = _generate_deterministic_patient_feedback("speech_rhythm", accuracy=88.0)
        self.assertIn("vocal rhythm", speech_fb.lower())
        self.assertNotIn("step", speech_fb.lower())
        self.assertNotIn("symmetry", speech_fb.lower())

        balance_fb = _generate_deterministic_patient_feedback("balance_training", accuracy=92.0)
        self.assertIn("postural stability", balance_fb.lower())
        self.assertNotIn("step", balance_fb.lower())
        self.assertNotIn("symmetry", balance_fb.lower())

        tapping_fb = _generate_deterministic_patient_feedback("finger_tapping", accuracy=85.0)
        self.assertIn("finger tapping", tapping_fb.lower())
        self.assertNotIn("step", tapping_fb.lower())
        self.assertNotIn("gait symmetry", tapping_fb.lower())

        gait_fb = _generate_deterministic_patient_feedback("gait_trainer", accuracy=85.0, symmetry=94.0, left_steps=12, right_steps=12)
        self.assertIn("steps", gait_fb.lower())
        self.assertIn("symmetry", gait_fb.lower())

    def test_deterministic_patient_feedback_zero_input(self):
        """Zero-input sessions must truthfully state no vocalization, taps, or steps detected."""
        speech_zero = _generate_deterministic_patient_feedback("speech_rhythm", accuracy=0.0)
        self.assertIn("no vocalization detected", speech_zero.lower())
        self.assertIn("microphone", speech_zero.lower())
        self.assertNotIn("wonderful", speech_zero.lower())

        tapping_zero = _generate_deterministic_patient_feedback("finger_tapping", accuracy=0.0)
        self.assertIn("no tap inputs registered", tapping_zero.lower())
        self.assertNotIn("wonderful", tapping_zero.lower())

        gait_zero = _generate_deterministic_patient_feedback("gait_trainer", accuracy=0.0, left_steps=0, right_steps=0)
        self.assertIn("no steps registered", gait_zero.lower())
        self.assertNotIn("wonderful", gait_zero.lower())

    def test_deterministic_structured_report_zero_input(self):
        """Clinical report synopsis and assessment must indicate quiet standby when zero input is registered."""
        from services.gemini_service import _generate_deterministic_structured_report
        zero_data = {
            "activity_type": "speech_rhythm",
            "duration_seconds": 60,
            "initial_bpm": 60,
            "final_bpm": 60,
            "accuracy_score": 0.0,
            "movement_count": 0
        }
        report = _generate_deterministic_structured_report(zero_data)
        self.assertIn("quiet standby", report["summary"].lower())
        self.assertIn("zero movement or vocal events registered", report["soap"]["assessment"].lower())

    def test_deterministic_session_reflection_zero_input(self):
        """Agent reflection must acknowledge quiet standby without claiming phantom adaptation or upward trajectory."""
        from services.gemini_service import _generate_deterministic_session_reflection
        summary = {
            "averageRhythmSync": 0.0,
            "averageMovementQuality": 0.0,
            "improvement": 0.0,
            "bestTempo": 60,
            "successfulTempoRange": "60 BPM"
        }
        reflection = _generate_deterministic_session_reflection(summary)
        self.assertIn("quiet standby", reflection["summary"].lower())
        self.assertIsNone(reflection["successfulAdaptation"])

if __name__ == '__main__':
    unittest.main()

