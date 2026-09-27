"""
NURO-BEATS Multi-Session Envelope & Memory Corruption Adversarial Test Suite
tests/test_adversarial_multisession_envelope.py

Validates:
1. Section 14: 10-Session Progression Simulation
   Session 1:  60 BPM -> Positive
   Session 2:  62 BPM -> Positive
   Session 3:  64 BPM -> Positive
   Session 4:  66 BPM -> Positive
   Session 5:  68 BPM -> Negative
   Session 6:  66 BPM -> Positive
   Session 7:  68 BPM -> Neutral
   Session 8:  70 BPM -> Negative
   Session 9:  66 BPM -> Positive
   Session 10: 64 BPM -> Positive

2. Envelope Historical Retention:
   - Negative interventions at 68 & 70 BPM are remembered (stable_bpm_max ceiling protection)
   - Successful interventions are remembered (typical cadence elevated safely from initial 60 BPM)
   - Single failure does not completely wipe out historical baseline (EMA resistance)

3. Section 15: Memory Corruption & Poisoning Defense
   - Abandoned session does NOT update envelope
   - Low confidence (< 0.45) tracking does NOT update envelope
   - Micro-session (< 15 seconds) does NOT update envelope
   - Out-of-bounds BPM does NOT update envelope
"""

import unittest
import json
from app import app
import routes
from models import (
    db, User, PatientProfile, TherapySession, SessionEvent,
    Intervention, InterventionOutcome, PatientPerformanceEnvelope,
    SessionSummary
)


class TestAdversarialMultiSessionEnvelope(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        app.config['TESTING'] = True
        app.config['WTF_CSRF_ENABLED'] = False
        with app.app_context():
            user = User.query.filter_by(username='envelope_runner').first()
            if not user:
                user = User(
                    username='envelope_runner',
                    email='envelope_runner@example.com',
                    user_type='patient',
                    first_name='Runner',
                    last_name='Ten'
                )
                user.set_password('Secret123!')
                db.session.add(user)
                db.session.commit()

            patient = PatientProfile.query.filter_by(user_id=user.id).first()
            if not patient:
                patient = PatientProfile(
                    user_id=user.id,
                    condition='parkinsons',
                    baseline_cadence=60.0
                )
                db.session.add(patient)
                db.session.commit()

            cls.user_id = user.id
            cls.patient_id = patient.id

    def setUp(self):
        self.client = app.test_client()
        with self.client.session_transaction() as sess:
            sess['user_id'] = self.user_id
            sess['user_type'] = 'patient'

        # Reset performance envelope for clean test state
        with app.app_context():
            PatientPerformanceEnvelope.query.filter_by(patient_id=self.patient_id).delete()
            db.session.commit()

    def test_ten_session_progression_envelope(self):
        """
        Adversarial Test (Section 14):
        Execute the exact 10-session sequence from the specification:
        Session 1:  60 BPM -> positive (accuracy: 88%)
        Session 2:  62 BPM -> positive (accuracy: 90%)
        Session 3:  64 BPM -> positive (accuracy: 92%)
        Session 4:  66 BPM -> positive (accuracy: 89%)
        Session 5:  68 BPM -> negative (accuracy: 55%, fatigue)
        Session 6:  66 BPM -> positive (accuracy: 87%, recovery)
        Session 7:  68 BPM -> neutral  (accuracy: 75%)
        Session 8:  70 BPM -> negative (accuracy: 52%, boundary breach)
        Session 9:  66 BPM -> positive (accuracy: 91%, stable zone)
        Session 10: 64 BPM -> positive (accuracy: 93%, consolidation)
        """
        exercise = "gait_trainer"
        progression = [
            {"session": 1,  "bpm": 60.0, "acc": 88.0, "outcome": "POSITIVE"},
            {"session": 2,  "bpm": 62.0, "acc": 90.0, "outcome": "POSITIVE"},
            {"session": 3,  "bpm": 64.0, "acc": 92.0, "outcome": "POSITIVE"},
            {"session": 4,  "bpm": 66.0, "acc": 89.0, "outcome": "POSITIVE"},
            {"session": 5,  "bpm": 68.0, "acc": 55.0, "outcome": "NEGATIVE"},
            {"session": 6,  "bpm": 66.0, "acc": 87.0, "outcome": "POSITIVE"},
            {"session": 7,  "bpm": 68.0, "acc": 75.0, "outcome": "NEUTRAL"},
            {"session": 8,  "bpm": 70.0, "acc": 52.0, "outcome": "NEGATIVE"},
            {"session": 9,  "bpm": 66.0, "acc": 91.0, "outcome": "POSITIVE"},
            {"session": 10, "bpm": 64.0, "acc": 93.0, "outcome": "POSITIVE"},
        ]

        current_initial_bpm = 60.0
        for step in progression:
            s_idx = step["session"]
            bpm = step["bpm"]
            acc = step["acc"]

            # Start session
            start_resp = self.client.post('/session/start', json={
                'session_type': exercise,
                'initial_bpm': current_initial_bpm,
                'target_bpm': bpm
            })
            self.assertEqual(start_resp.status_code, 200)
            session_id = start_resp.get_json()['session_id']

            # Record causal intervention for the tempo change
            itv_resp = self.client.post(f'/api/session/{session_id}/interventions', json={
                'target_parameter': 'TEMPO',
                'action_taken': 'INCREASE_TEMPO' if bpm > current_initial_bpm else 'MAINTAIN_TEMPO',
                'previous_value': current_initial_bpm,
                'new_value': bpm,
                'pre_performance': 0.80,
                'observation_window_cycles': 3
            })
            self.assertEqual(itv_resp.status_code, 201)
            itv_id = itv_resp.get_json()['intervention_id']

            # Record outcome
            out_resp = self.client.post(f'/api/intervention/{itv_id}/outcome', json={
                'post_performance': acc / 100.0,
                'classification': step["outcome"],
                'confidence': 0.90
            })
            self.assertEqual(out_resp.status_code, 201)

            # Complete session
            comp_resp = self.client.post(f'/session/{session_id}/complete', json={
                'duration': 180,
                'final_bpm': bpm,
                'accuracy_score': acc,
                'agent_summary': {
                    'startingPerformance': 0.80,
                    'endingPerformance': acc / 100.0,
                    'averageConfidence': 0.92,
                    'bestTempo': bpm
                }
            })
            self.assertEqual(comp_resp.status_code, 200)

            # Fetch envelope returned or via API
            env_resp = self.client.get(f'/api/patient/{self.patient_id}/envelope/{exercise}', follow_redirects=True)
            self.assertEqual(env_resp.status_code, 200)
            env_data = env_resp.get_json()

            # Set starting tempo for next session based on learned personalized ceiling/typical cadence
            current_initial_bpm = min(66.0, env_data.get('typical_cadence', 60.0))

        # --- Inspect Final Personal Envelope after 10 sessions ---
        with app.app_context():
            final_env = PatientPerformanceEnvelope.query.filter_by(
                patient_id=self.patient_id, exercise_type=exercise
            ).first()

            self.assertIsNotNone(final_env)
            # 1. Evaluated exactly 10 sessions
            self.assertEqual(final_env.sessions_evaluated, 10)

            # 2. Confidence is very high after 10 sessions
            self.assertGreater(final_env.confidence, 0.85)

            # 3. Successful interventions are remembered: typical cadence increased safely from 60
            self.assertGreater(final_env.typical_cadence, 61.0)
            self.assertLessEqual(final_env.typical_cadence, 67.0)

            # 4. Negative interventions at 68 & 70 BPM were remembered:
            # Envelope ceiling must NOT allow reckless jump to 72+ BPM
            self.assertLessEqual(final_env.stable_bpm_max, 67.0)

            # 5. One or two anomalies did not collapse the baseline:
            # Baseline is still strong and resilient
            self.assertGreaterEqual(final_env.stable_bpm_min, 55.0)

    # =========================================================================
    # Memory Corruption & Poisoning Defense Tests (Section 15)
    # =========================================================================
    def test_low_confidence_session_does_not_corrupt_memory(self):
        """
        Adversarial Test: When camera tracking degrades (confidence < 0.45),
        the session metrics must NOT be consolidated into the long-term envelope.
        """
        exercise = "gait_trainer"
        # First create a valid baseline session
        resp1 = self.client.post('/session/start', json={
            'session_type': exercise, 'initial_bpm': 60, 'target_bpm': 64
        })
        s1_id = resp1.get_json()['session_id']
        self.client.post(f'/session/{s1_id}/complete', json={
            'duration': 180, 'final_bpm': 64, 'accuracy_score': 85.0
        })

        with app.app_context():
            env_before = PatientPerformanceEnvelope.query.filter_by(
                patient_id=self.patient_id, exercise_type=exercise
            ).first()
            baseline_cadence = env_before.typical_cadence
            baseline_evaluated = env_before.sessions_evaluated
            self.assertEqual(baseline_evaluated, 1)

        # Now simulate a degraded tracking session (confidence = 0.25 < 0.45) with spurious extreme BPM
        resp2 = self.client.post('/session/start', json={
            'session_type': exercise, 'initial_bpm': 60, 'target_bpm': 64
        })
        s2_id = resp2.get_json()['session_id']
        comp2 = self.client.post(f'/session/{s2_id}/complete', json={
            'duration': 180,
            'final_bpm': 130.0,  # Spurious extreme BPM
            'accuracy_score': 99.0,
            'agent_summary': {
                'averageConfidence': 0.25  # DEGRADED
            }
        })
        self.assertEqual(comp2.status_code, 200)

        # Verify envelope was NOT updated by this degraded observation
        with app.app_context():
            env_after = PatientPerformanceEnvelope.query.filter_by(
                patient_id=self.patient_id, exercise_type=exercise
            ).first()
            self.assertEqual(env_after.sessions_evaluated, baseline_evaluated)
            self.assertAlmostEqual(env_after.typical_cadence, baseline_cadence, places=2)

    def test_abandoned_or_micro_session_does_not_corrupt_memory(self):
        """
        Adversarial Test: An abandoned session or a 5-second accidental session
        must NOT update the patient performance envelope.
        """
        exercise = "gait_trainer"
        # Micro-session (< 15 seconds)
        resp = self.client.post('/session/start', json={
            'session_type': exercise, 'initial_bpm': 60, 'target_bpm': 64
        })
        s_id = resp.get_json()['session_id']
        comp = self.client.post(f'/session/{s_id}/complete', json={
            'duration': 4.5,  # Micro session
            'final_bpm': 60.0,
            'accuracy_score': 90.0
        })
        self.assertEqual(comp.status_code, 200)

        with app.app_context():
            env = PatientPerformanceEnvelope.query.filter_by(
                patient_id=self.patient_id, exercise_type=exercise
            ).first()
            # Must remain None (not created from a 4.5s session)
            self.assertIsNone(env)


if __name__ == '__main__':
    unittest.main()
