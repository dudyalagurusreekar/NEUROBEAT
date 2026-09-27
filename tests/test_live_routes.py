import unittest
from app import app
import routes
from models import db, User, PatientProfile, TherapySession

class TestLiveRoutes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with app.app_context():
            user = User.query.filter_by(username='patient_test').first()
            if not user:
                user = User(
                    username='patient_test',
                    email='patient_test@example.com',
                    user_type='patient',
                    first_name='Test',
                    last_name='Patient'
                )
                user.set_password('password123')
                db.session.add(user)
                db.session.commit()
                profile = PatientProfile(user_id=user.id, condition='parkinsons', baseline_cadence=60)
                db.session.add(profile)
                db.session.commit()
            elif not user.patient_profile:
                profile = PatientProfile(user_id=user.id, condition='parkinsons', baseline_cadence=60)
                db.session.add(profile)
                db.session.commit()
            cls.user_id = user.id

    def setUp(self):
        self.client = app.test_client()
        with self.client.session_transaction() as sess:
            sess['user_id'] = self.user_id
            sess['user_type'] = 'patient'

    def test_gait_trainer_view(self):
        resp = self.client.post('/session/start', json={'session_type': 'gait_trainer', 'initial_bpm': 60, 'target_bpm': 70})
        self.assertEqual(resp.status_code, 200)
        sid = resp.get_json()['session_id']

        view_resp = self.client.get(f'/session/{sid}')
        self.assertEqual(view_resp.status_code, 200)
        html = view_resp.data.decode('utf-8')

        self.assertIn('<title>Gait Trainer - NeuroBeat</title>', html)
        self.assertIn('id="cameraStage"', html)
        self.assertIn('id="cameraFeed"', html)
        self.assertIn('id="cameraStatusBadge"', html)
        self.assertIn('Left Leg:', html)
        self.assertIn('Symmetry:', html)

    def test_balance_training_view(self):
        resp = self.client.post('/session/start', json={'session_type': 'balance_training', 'initial_bpm': 60, 'target_bpm': 70})
        self.assertEqual(resp.status_code, 200)
        sid = resp.get_json()['session_id']

        view_resp = self.client.get(f'/session/{sid}')
        self.assertEqual(view_resp.status_code, 200)
        html = view_resp.data.decode('utf-8')

        self.assertIn('<title>Balance Training - NeuroBeat</title>', html)
        self.assertIn('id="cameraStage"', html)
        self.assertIn('id="cameraFeed"', html)
        self.assertIn('id="cameraStatusBadge"', html)
        self.assertIn('Posture &amp; Balance View', html)
        self.assertIn('Left Weight: Centered', html)
        self.assertIn('Balance Stability:', html)
        # Gait leg stepping labels must NOT appear in balance mode
        self.assertNotIn('Left Leg: <span id="leftLegState">', html)

    def test_finger_tapping_view(self):
        resp = self.client.post('/session/start', json={'session_type': 'finger_tapping', 'initial_bpm': 60, 'target_bpm': 70})
        self.assertEqual(resp.status_code, 200)
        sid = resp.get_json()['session_id']

        view_resp = self.client.get(f'/session/{sid}')
        self.assertEqual(view_resp.status_code, 200)
        html = view_resp.data.decode('utf-8')

        self.assertIn('<title>Finger Tapping - NeuroBeat</title>', html)
        self.assertIn('class="tap-pad-wrapper"', html)
        self.assertIn('id="fingerTapBtn"', html)
        self.assertIn('Fine-Motor Rhythm Tap Pad', html)
        self.assertIn('Finger Tapping Ready', html)

        # Camera DOM elements must NOT be rendered in finger tapping mode
        self.assertNotIn('id="cameraStage"', html)
        self.assertNotIn('id="cameraFeed"', html)
        self.assertNotIn('id="cameraStatusBadge"', html)
        self.assertNotIn('id="cameraBox"', html)

    def test_speech_rhythm_view(self):
        resp = self.client.post('/session/start', json={'session_type': 'speech_rhythm', 'initial_bpm': 60, 'target_bpm': 70})
        self.assertEqual(resp.status_code, 200)
        sid = resp.get_json()['session_id']

        view_resp = self.client.get(f'/session/{sid}')
        self.assertEqual(view_resp.status_code, 200)
        html = view_resp.data.decode('utf-8')

        self.assertIn('<title>Speech Rhythm - NeuroBeat</title>', html)
        self.assertIn('Vocal Rhythm', html)
        self.assertIn('TA &mdash; TA &mdash; TA &mdash; TA', html)
        self.assertIn('Microphone Ready', html)

        # Camera DOM elements must NOT be rendered in speech rhythm mode
        self.assertNotIn('id="cameraStage"', html)
        self.assertNotIn('id="cameraFeed"', html)
        self.assertNotIn('id="cameraStatusBadge"', html)
        self.assertNotIn('id="cameraBox"', html)

    def test_unknown_session_view(self):
        resp = self.client.post('/session/start', json={'session_type': 'random_invalid_type', 'initial_bpm': 60, 'target_bpm': 70})
        self.assertEqual(resp.status_code, 200)
        sid = resp.get_json()['session_id']

        view_resp = self.client.get(f'/session/{sid}')
        self.assertEqual(view_resp.status_code, 200)
        html = view_resp.data.decode('utf-8')

        # Document and card title must be 'Therapy Session', never falling through to 'Gait Trainer'
        self.assertIn('<title>Therapy Session - NeuroBeat</title>', html)
        self.assertIn('Therapy Session', html)
        self.assertNotIn('<title>Gait Trainer - NeuroBeat</title>', html)
        self.assertNotIn('id="cameraStage"', html)
        self.assertNotIn('id="cameraFeed"', html)
        self.assertNotIn('id="cameraStatusBadge"', html)

    def test_speech_rhythm_completion(self):
        resp = self.client.post('/session/start', json={'session_type': 'speech_rhythm', 'initial_bpm': 60, 'target_bpm': 70})
        self.assertEqual(resp.status_code, 200)
        sid = resp.get_json()['session_id']

        # Complete with verified vocal metrics
        payload = {
            'duration': 60,
            'final_bpm': 62,
            'accuracy_score': 84.5,
            'vocal_count': 24,
            'vocal_cadence': 60,
            'notes': 'Completed Speech Rhythm session with rhythmic syllable pacing and vocal synchronization. Syllables: 24.',
            'metrics_data': {
                'schema_version': '2.0',
                'session_id': sid,
                'session_type': 'speech_rhythm',
                'movement': {'vocal_count': 24, 'vocal_cadence_spm': 60},
                'sync': {'valid': True, 'rhythm_alignment_score': 84.5}
            }
        }
        comp_resp = self.client.post(f'/session/{sid}/complete', json=payload)
        self.assertEqual(comp_resp.status_code, 200)
        self.assertTrue(comp_resp.get_json()['success'])

        # Verify DB session record
        with app.app_context():
            session = TherapySession.query.get(sid)
            self.assertTrue(session.completed)
            self.assertEqual(session.accuracy_score, 84.5)
            self.assertIn('Syllables: 24', session.notes)

if __name__ == '__main__':
    unittest.main()
