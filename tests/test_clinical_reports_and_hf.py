"""
Test Suite for Clinical Reports, Longitudinal Historical Intelligence,
and Hugging Face Dual-Engine Audio Generation.
"""

import unittest
import json
import os
from datetime import datetime
from app import app, db
import routes
from models import User, PatientProfile, TherapySession, ClinicalReport
from services.historical_analysis import (
    get_patient_history,
    compute_accuracy_trend,
    calculate_advisory_bpm,
    calculate_accuracy_delta,
    format_historical_context_for_prompt,
    save_or_update_clinical_report,
    get_mock_trajectory_for_demo
)
from services.gemini_service import (
    generate_structured_clinical_report,
    _generate_deterministic_structured_report
)
from beat_generator import BeatGenerator


class TestClinicalReportsAndHF(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app.config['TESTING'] = True
        app.config['WTF_CSRF_ENABLED'] = False
        cls.app = app
        cls.client = app.test_client()
        cls.ctx = app.app_context()
        cls.ctx.push()

        user = User.query.filter_by(username='clinical_tester').first()
        if not user:
            user = User(
                username='clinical_tester',
                email='clinical_tester@example.com',
                user_type='patient',
                first_name='Eleanor',
                last_name='Vance'
            )
            user.set_password('SecurePassword123!')
            db.session.add(user)
            db.session.commit()

        patient = PatientProfile.query.filter_by(user_id=user.id).first()
        if not patient:
            patient = PatientProfile(
                user_id=user.id,
                condition='parkinsons',
                baseline_cadence=62.0,
                target_cadence=74.0
            )
            db.session.add(patient)
            db.session.commit()

        # Create or fetch completed session
        sess = TherapySession.query.filter_by(patient_id=patient.id, completed=True).first()
        if not sess:
            sess = TherapySession(
                patient_id=patient.id,
                session_type="gait_trainer",
                initial_bpm=62.0,
                target_bpm=74.0,
                final_bpm=68.0,
                accuracy_score=86.5,
                duration_seconds=120,
                completed=True
            )
            db.session.add(sess)
            db.session.commit()

        cls.user = user
        cls.patient = patient
        cls.session = sess

    @classmethod
    def tearDownClass(cls):
        try:
            ClinicalReport.query.filter_by(patient_id=cls.patient.id).delete()
            TherapySession.query.filter_by(patient_id=cls.patient.id).delete()
            PatientProfile.query.filter_by(id=cls.patient.id).delete()
            User.query.filter_by(id=cls.user.id).delete()
            db.session.commit()
        except Exception:
            db.session.rollback()
        cls.ctx.pop()

    def test_clinical_report_model_and_storage(self):
        """Test ClinicalReport database model persistence and idempotency."""
        report_data = {
            "summary": "Patient Eleanor completed 120s gait training with 86.5% accuracy.",
            "what_you_did": ["Achieved 84 steps", "Maintained rhythmic entrainment"],
            "performance_observations": ["Cadence rose smoothly from 62 to 68 BPM"],
            "what_to_improve": ["Left toe-off latency was slightly delayed"],
            "recommendations": ["Progress starting BPM to 65 next session"],
            "soap": {
                "subjective": "Patient reported feeling steady.",
                "objective": "Completed 120s at 68 BPM with 86.5% accuracy.",
                "assessment": "Consistent motor entrainment observed.",
                "plan": "Advance cadence challenge."
            },
            "movement_count": 84,
            "ai_model": "gemini-2.5-flash"
        }

        # 1. Save new report
        report = save_or_update_clinical_report(self.session.id, report_data)
        self.assertIsNotNone(report.id)
        self.assertEqual(report.session_id, self.session.id)
        self.assertEqual(report.patient_id, self.patient.id)
        self.assertEqual(report.accuracy_score, 86.5)
        self.assertEqual(report.movement_count, 84)

        # Verify to_dict
        d = report.to_dict()
        self.assertEqual(d["session_id"], self.session.id)
        self.assertIn("Achieved 84 steps", d["what_you_did"])
        self.assertEqual(d["soap_subjective"], "Patient reported feeling steady.")

        # 2. Update existing report (Idempotency)
        report_data["summary"] = "Updated summary text."
        updated_report = save_or_update_clinical_report(self.session.id, report_data)
        self.assertEqual(updated_report.id, report.id)
        self.assertEqual(updated_report.summary, "Updated summary text.")

    def test_historical_analysis_layer(self):
        """Test all 9 features of longitudinal historical analysis."""
        # Feature 2: Get patient history
        hist = get_patient_history(self.patient.id, "gait_trainer")
        self.assertGreaterEqual(len(hist), 1)
        self.assertEqual(hist[0]["session_id"], self.session.id)

        # Feature 3: Trend calculation
        synthetic_history = [
            {"accuracy_score": 85.0},
            {"accuracy_score": 82.0},
            {"accuracy_score": 75.0},
            {"accuracy_score": 72.0}
        ]
        trend = compute_accuracy_trend(synthetic_history)
        self.assertEqual(trend, "improving")

        synthetic_declining = [
            {"accuracy_score": 60.0},
            {"accuracy_score": 65.0},
            {"accuracy_score": 78.0},
            {"accuracy_score": 80.0}
        ]
        trend_dec = compute_accuracy_trend(synthetic_declining)
        self.assertEqual(trend_dec, "declining")

        # Feature 4: Advisory BPM calculation
        adv_bpm = calculate_advisory_bpm(62.0, 88.0, "improving")
        self.assertEqual(adv_bpm, 65.0)

        adv_clamped = calculate_advisory_bpm(39.0, 50.0, "declining", min_bpm=40.0)
        self.assertEqual(adv_clamped, 40.0)

        # Feature 5: Accuracy delta
        delta_info = calculate_accuracy_delta(86.5, hist)
        self.assertIn("delta", delta_info)
        self.assertIn("historical_average", delta_info)

        # Feature 6: Format context for prompt
        ctx = format_historical_context_for_prompt(self.patient.id, "gait_trainer")
        self.assertIn("PATIENT RECENT HISTORY", ctx)

        # Feature 7: Prototype demo mock
        demo_traj = get_mock_trajectory_for_demo()
        self.assertEqual(len(demo_traj), 4)
        self.assertEqual(demo_traj[0]["session_id"], 901)

    def test_deterministic_gemini_fallback(self):
        """Test Gemini deterministic report fallback when offline."""
        data = {
            "activity_type": "gait_trainer",
            "duration_seconds": 180,
            "initial_bpm": 60.0,
            "final_bpm": 66.0,
            "accuracy_score": 84.0,
            "movement_count": 120
        }
        report = _generate_deterministic_structured_report(data)
        self.assertIn("summary", report)
        self.assertIn("what_you_did", report)
        self.assertIn("soap", report)
        self.assertIn("Patient", report["soap"]["subjective"])
        self.assertIn("84.0%", report["soap"]["objective"])

    def test_beat_generator_dual_engine(self):
        """Test BeatGenerator Hugging Face router & local acoustic fallback."""
        bg = BeatGenerator()
        conn = bg.check_connection()
        self.assertIn("endpoint", conn)
        self.assertIn("router.huggingface.co", conn["endpoint"])

        # Generate beat (should smoothly succeed with studio acoustic synthesis)
        res = bg.generate_beat_detailed(bpm=72.0, duration=3, session_type="gait_trainer")
        self.assertTrue(res["success"])
        self.assertIsNotNone(res["audio_url"])
        self.assertIn(res["engine_used"], ["procedural_acoustic_synth", "huggingface_router"])
        self.assertEqual(res["bpm"], 72.0)

        # Verify generated WAV file exists on disk and has valid RIFF header
        clean_url = res["audio_url"].split("?")[0].lstrip("/")
        wav_path = os.path.join(app.root_path, clean_url)
        self.assertTrue(os.path.exists(wav_path))
        with open(wav_path, "rb") as f:
            header = f.read(4)
            self.assertEqual(header, b"RIFF")

    def test_api_endpoints(self):
        """Test API endpoints for HF status, beat generation, trends, and clinical reports."""
        with self.client.session_transaction() as sess:
            sess["user_id"] = self.user.id
            sess["user_type"] = "patient"

        # 1. GET /api/hf/status
        resp = self.client.get("/api/hf/status")
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.data)
        self.assertTrue(data["success"])
        self.assertIn("router.huggingface.co", data["status"]["endpoint"])

        # 2. POST /api/beat/generate_ai
        resp = self.client.post("/api/beat/generate_ai", json={
            "bpm": 64.0,
            "duration": 2,
            "session_type": "gait_trainer"
        })
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.data)
        self.assertTrue(data["success"])
        self.assertIsNotNone(data["audio_url"])

        # 3. GET /api/patient/<id>/historical-trends
        resp = self.client.get(f"/api/patient/{self.patient.id}/historical-trends?activity_type=gait_trainer")
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.data)
        self.assertTrue(data["success"])
        self.assertEqual(data["patient_id"], self.patient.id)
        self.assertIn("trend", data)
        self.assertIn("advisory_bpm", data)

        # 4. GET /api/session/<id>/clinical-report
        resp = self.client.get(f"/api/session/{self.session.id}/clinical-report")
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.data)
        self.assertTrue(data["success"])
        self.assertIsNotNone(data["clinical_report"])
        self.assertIn("soap_objective", data["clinical_report"])


if __name__ == "__main__":
    unittest.main()
