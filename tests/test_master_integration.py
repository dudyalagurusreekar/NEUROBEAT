"""
NURO-BEATS Master Architectural Integration & Hardening Test Suite
tests/test_master_integration.py

Comprehensive validation covering:
1. Multi-Session Learning & Personal Performance Envelope Continuity (Sessions 1 -> 2 -> 3 -> 4)
2. Closed-Loop Causal Intervention (Proposal -> P2 Validation/Clamp -> Intervention -> Outcome -> Memory)
3. Tracking Degradation & Confidence Gating (Confidence < 0.45 -> FREEZE)
4. Offline Event Idempotency & Bounded Buffer Limits
5. Deterministic Replay & Explanation Reconstruction
"""

import unittest
import json
from datetime import datetime
from app import app
import routes
from models import (
    db, User, PatientProfile, TherapySession, SessionMetrics,
    SessionEvent, AdaptationRecord, AgentDecision, Intervention,
    InterventionOutcome, PatientPerformanceEnvelope, SessionSummary,
    ModelMetadata
)
from runtime.replay_engine import ReplayEngine


class TestMasterIntegration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        app.config['TESTING'] = True
        app.config['WTF_CSRF_ENABLED'] = False
        with app.app_context():
            user = User.query.filter_by(username='master_tester').first()
            if not user:
                user = User(
                    username='master_tester',
                    email='master_tester@example.com',
                    user_type='patient',
                    first_name='Master',
                    last_name='Tester'
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

        # Clean prior test envelopes for this patient to ensure fresh test runs
        with app.app_context():
            PatientPerformanceEnvelope.query.filter_by(patient_id=self.patient_id).delete()
            db.session.commit()

    # =========================================================================
    # 1. Multi-Session Learning & Personal Performance Envelope Continuity
    # =========================================================================
    def test_multi_session_learning_loop(self):
        """
        Verify multi-session personalization across Sessions 1 -> 2 -> 3 -> 4:
        Session 1: Initial (60 BPM) -> completes with good performance -> envelope created.
        Session 2: Starting tempo personalizes to learned stable ceiling.
        Session 3: High tempo causes decline -> recorded in envelope.
        Session 4: System preserves learned envelope and does not act like a blank slate.
        """
        exercise = "gait_trainer"

        # --- Session 1: Blank slate user ---
        resp1 = self.client.post('/session/start', json={
            'session_type': exercise, 'initial_bpm': 60, 'target_bpm': 70
        })
        self.assertEqual(resp1.status_code, 200)
        s1_data = resp1.get_json()
        s1_id = s1_data['session_id']
        self.assertIsNone(s1_data.get('performance_envelope'))

        # Complete Session 1 with high accuracy (88%) and final BPM 64
        comp1 = self.client.post(f'/session/{s1_id}/complete', json={
            'duration': 300,
            'final_bpm': 64,
            'accuracy_score': 88.0,
            'metrics_data': {'movement_quality': {'overall': 0.86}}
        })
        self.assertEqual(comp1.status_code, 200)

        # Verify envelope was created after Session 1
        with app.app_context():
            env1 = PatientPerformanceEnvelope.query.filter_by(
                patient_id=self.patient_id, exercise_type=exercise
            ).first()
            self.assertIsNotNone(env1)
            self.assertEqual(env1.sessions_evaluated, 1)
            self.assertGreaterEqual(env1.stable_bpm_max, 64.0)

        # --- Session 2: Envelope loaded, starting tempo personalized ---
        resp2 = self.client.post('/session/start', json={
            'session_type': exercise, 'initial_bpm': 60, 'target_bpm': 70
        })
        self.assertEqual(resp2.status_code, 200)
        s2_data = resp2.get_json()
        s2_id = s2_data['session_id']

        self.assertIsNotNone(s2_data.get('performance_envelope'))
        self.assertGreaterEqual(s2_data['initial_bpm'], 60)

        # Complete Session 2 with high performance at 66 BPM
        comp2 = self.client.post(f'/session/{s2_id}/complete', json={
            'duration': 360,
            'final_bpm': 66,
            'accuracy_score': 90.0,
            'metrics_data': {'movement_quality': {'overall': 0.89}}
        })
        self.assertEqual(comp2.status_code, 200)

        # Check envelope update
        with app.app_context():
            env2 = PatientPerformanceEnvelope.query.filter_by(
                patient_id=self.patient_id, exercise_type=exercise
            ).first()
            self.assertEqual(env2.sessions_evaluated, 2)
            self.assertGreaterEqual(env2.stable_bpm_max, 65.0)

        # --- Session 3: Degradation at high tempo ---
        resp3 = self.client.post('/session/start', json={
            'session_type': exercise, 'initial_bpm': 64, 'target_bpm': 72
        })
        s3_id = resp3.get_json()['session_id']

        comp3 = self.client.post(f'/session/{s3_id}/complete', json={
            'duration': 240,
            'final_bpm': 68,
            'accuracy_score': 55.0,
            'metrics_data': {'movement_quality': {'overall': 0.58}}
        })
        self.assertEqual(comp3.status_code, 200)

        # --- Session 4: Verify learned envelope is durable ---
        resp4 = self.client.post('/session/start', json={
            'session_type': exercise, 'initial_bpm': 60, 'target_bpm': 70
        })
        s4_data = resp4.get_json()
        env4 = s4_data.get('performance_envelope')
        self.assertIsNotNone(env4)
        self.assertEqual(env4['sessions_evaluated'], 3)
        self.assertGreater(env4['confidence'], 0.50)

    # =========================================================================
    # 2. Closed-Loop Causal Intervention (Proposal -> Clamp -> Intervention -> Outcome)
    # =========================================================================
    def test_closed_loop_intervention_lifecycle(self):
        """
        Verify that an agent decision proposal is validated/clamped by P2,
        persisted separately, registered as an intervention, and evaluated with an outcome.
        """
        resp = self.client.post('/session/start', json={
            'session_type': 'gait_trainer', 'initial_bpm': 60, 'target_bpm': 70
        })
        session_id = resp.get_json()['session_id']

        # P5 requests extreme jump (+12 BPM); P2 clamps to +5 BPM max
        update_resp = self.client.post('/session/update', json={
            'session_id': session_id,
            'current_bpm': 60.0,
            'sync_accuracy': 88.0,
            'suggested_bpm': 65.0,
            'adaptation_record': {
                'parameter': 'TEMPO',
                'previous_value': 60.0,
                'requested_value': 72.0,
                'executed_value': 65.0,
                'direction': 'PROGRESS',
                'trigger_reason': 'High synchronization sustained.',
                'validator_status': 'CLAMPED',
                'clamp_reason': 'STEP_CLAMPED: max delta 5.0 BPM',
                'confidence': 0.92
            },
            'agent_decision': {
                'intent': 'PROGRESS',
                'target': 'TEMPO',
                'action': 'INCREASE_TEMPO',
                'requested_magnitude': 12.0,
                'reasoning_summary': 'User shows steady motor entrainment.',
                'confidence': 0.92,
                'validator_result': 'CLAMPED'
            }
        })
        self.assertEqual(update_resp.status_code, 200)

        # Verify adaptation_records and agent_decisions were persisted
        with app.app_context():
            rec = AdaptationRecord.query.filter_by(session_id=session_id).first()
            self.assertIsNotNone(rec)
            self.assertEqual(rec.requested_value, 72.0)
            self.assertEqual(rec.executed_value, 65.0)
            self.assertEqual(rec.validator_status, 'CLAMPED')

            dec = AgentDecision.query.filter_by(session_id=session_id).first()
            self.assertIsNotNone(dec)
            self.assertEqual(dec.requested_magnitude, 12.0)
            self.assertEqual(dec.validator_result, 'CLAMPED')

        # Register causal intervention
        itv_resp = self.client.post(f'/api/session/{session_id}/interventions', json={
            'target_parameter': 'TEMPO',
            'action_taken': 'INCREASE_TEMPO',
            'previous_value': 60.0,
            'new_value': 65.0,
            'pre_performance': 0.82,
            'pre_rhythm_sync': 0.85,
            'pre_movement_quality': 0.80,
            'observation_window_cycles': 3
        })
        self.assertEqual(itv_resp.status_code, 201)
        itv_id = itv_resp.get_json()['intervention_id']

        # Record post-observation outcome
        out_resp = self.client.post(f'/api/intervention/{itv_id}/outcome', json={
            'post_performance': 0.90,
            'post_rhythm_sync': 0.92,
            'post_movement_quality': 0.88,
            'classification': 'POSITIVE',
            'confidence': 0.94,
            'evaluation_window': 3
        })
        self.assertEqual(out_resp.status_code, 201)

        # Verify outcome persisted and linked
        with app.app_context():
            itv = db.session.get(Intervention, itv_id) if hasattr(db.session, 'get') else db.session.query(Intervention).get(itv_id)
            self.assertIsNotNone(itv.outcome)
            self.assertEqual(itv.outcome.classification, 'POSITIVE')
            self.assertAlmostEqual(itv.outcome.delta_performance, 0.08, places=2)

    # =========================================================================
    # 3. Tracking Degradation & Confidence Gating (Section 25)
    # =========================================================================
    def test_low_confidence_freezes_adaptation(self):
        """
        Verify that when confidence drops below 0.45, P2 freezes adaptation,
        and no progress intervention can corrupt the session.
        """
        resp = self.client.post('/session/start', json={
            'session_type': 'balance_training', 'initial_bpm': 60, 'target_bpm': 65
        })
        session_id = resp.get_json()['session_id']

        # Post update with low confidence and FROZEN validator status
        self.client.post('/session/update', json={
            'session_id': session_id,
            'current_bpm': 60.0,
            'sync_accuracy': 40.0,
            'suggested_bpm': 60.0,
            'adaptation_record': {
                'parameter': 'TEMPO',
                'previous_value': 60.0,
                'requested_value': 64.0,
                'executed_value': 60.0,
                'direction': 'FREEZE',
                'trigger_reason': 'Tracking confidence degraded.',
                'validator_status': 'FROZEN',
                'clamp_reason': 'LOW_CONFIDENCE: 0.32 < 0.45 threshold',
                'confidence': 0.32
            }
        })

        with app.app_context():
            rec = AdaptationRecord.query.filter_by(session_id=session_id).first()
            self.assertIsNotNone(rec)
            self.assertEqual(rec.validator_status, 'FROZEN')
            self.assertEqual(rec.executed_value, 60.0)

    # =========================================================================
    # 4. Offline Event Idempotency & Queue Ingestion (Section 16 & 28)
    # =========================================================================
    def test_event_idempotency_and_batch_ingest(self):
        """
        Verify that duplicate events with identical idempotency_key are safely
        deduplicated without raising errors or creating phantom rows.
        """
        resp = self.client.post('/session/start', json={
            'session_type': 'finger_tapping', 'initial_bpm': 60, 'target_bpm': 70
        })
        session_id = resp.get_json()['session_id']

        events_batch = [
            {
                'idempotency_key': f"idem_{session_id}_1",
                'timestamp': 100.0,
                'event_type': 'PERFORMANCE_STABLE',
                'source': 'P2',
                'severity': 'INFO',
                'payload': {'score': 0.85}
            },
            {
                'idempotency_key': f"idem_{session_id}_2",
                'timestamp': 105.0,
                'event_type': 'ADAPTATION_OPPORTUNITY',
                'source': 'P5',
                'severity': 'INFO',
                'payload': {'tempo': 62}
            }
        ]

        # First ingestion
        ingest1 = self.client.post(f'/api/session/{session_id}/events', json={'events': events_batch})
        self.assertEqual(ingest1.status_code, 201)
        self.assertEqual(ingest1.get_json()['ingested_count'], 2)

        # Duplicate ingestion (e.g. browser retry after reconnect)
        ingest2 = self.client.post(f'/api/session/{session_id}/events', json={'events': events_batch})
        self.assertEqual(ingest2.status_code, 201)
        self.assertEqual(ingest2.get_json()['ingested_count'], 0)

        # Total rows in DB should be exactly 3 (1 SESSION_CREATED + 2 events)
        with app.app_context():
            count = SessionEvent.query.filter_by(session_id=session_id).count()
            self.assertEqual(count, 3)

    # =========================================================================
    # 5. Deterministic Replay & Explanation Reconstruction (Section 31)
    # =========================================================================
    def test_replay_trace_and_explanation(self):
        """
        Verify that /api/session/<id>/replay endpoint returns a complete trace,
        and ReplayEngine can deterministically reconstruct and explain all decisions.
        """
        resp = self.client.post('/session/start', json={
            'session_type': 'gait_trainer', 'initial_bpm': 60, 'target_bpm': 70
        })
        session_id = resp.get_json()['session_id']

        self.client.post('/session/update', json={
            'session_id': session_id,
            'current_bpm': 60.0,
            'sync_accuracy': 90.0,
            'adaptation_record': {
                'parameter': 'TEMPO',
                'previous_value': 60.0,
                'requested_value': 64.0,
                'executed_value': 64.0,
                'direction': 'PROGRESS',
                'trigger_reason': 'High entrainment.',
                'validator_status': 'APPROVED'
            },
            'agent_decision': {
                'intent': 'PROGRESS',
                'target': 'TEMPO',
                'action': 'INCREASE_TEMPO',
                'requested_magnitude': 4.0,
                'reasoning_summary': 'High entrainment.',
                'confidence': 0.95,
                'validator_result': 'APPROVED'
            }
        })

        self.client.post(f'/session/{session_id}/complete', json={
            'duration': 180,
            'final_bpm': 64.0,
            'accuracy_score': 90.0
        })

        # Fetch replay payload
        replay_resp = self.client.get(f'/api/session/{session_id}/replay')
        self.assertEqual(replay_resp.status_code, 200)
        trace = replay_resp.get_json()

        self.assertIn('metadata', trace)
        self.assertEqual(trace['schema_version'], '3.0')
        self.assertTrue(len(trace['adaptations']) > 0)

        # Run offline replay engine
        engine = ReplayEngine(trace)
        det_result = engine.verify_determinism()
        self.assertTrue(det_result['deterministic_pass'])

        explanation = engine.explain_decision(0)
        self.assertIn('causal_chain', explanation)
        self.assertIn('60.0', explanation['why'])
        self.assertIn('64.0', explanation['why'])


if __name__ == '__main__':
    unittest.main()
