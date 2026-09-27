from flask import render_template, request, redirect, url_for, session, flash, jsonify
from werkzeug.exceptions import HTTPException
from app import app, db
from models import (
    User, PatientProfile, ClinicianProfile, TherapySession, SessionMetrics, BaselineAssessment,
    SchemaVersion, ModelMetadata, SessionEvent, AdaptationRecord, AgentDecision,
    Intervention, InterventionOutcome, PatientPerformanceEnvelope, SessionSummary,
    ClinicalReport
)
from datetime import datetime, timedelta
import json
import logging
from session_modes import get_session_mode, normalize_session_type, is_camera_required

@app.route('/')
def index():
    """Landing page - login/register interface"""
    if 'user_id' in session:
        user = User.query.get(session['user_id'])
        if user:
            if user.user_type == 'patient':
                return redirect(url_for('patient_dashboard'))
            else:
                return redirect(url_for('clinician_dashboard'))
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    """User registration for clinicians"""
    if request.method == 'POST':
        is_ajax = (
            request.headers.get('X-Requested-With') == 'XMLHttpRequest' or
            'application/json' in request.headers.get('Accept', '') or
            request.is_json
        )
        try:
            username = request.form.get('username') or (request.json.get('username') if request.is_json else None)
            email = request.form.get('email') or (request.json.get('email') if request.is_json else None)
            password = request.form.get('password') or (request.json.get('password') if request.is_json else None)
            user_type = request.form.get('user_type', 'clinician')
            first_name = request.form.get('first_name') or (request.json.get('first_name') if request.is_json else None)
            last_name = request.form.get('last_name') or (request.json.get('last_name') if request.is_json else None)

            if not username or not email or not password or not first_name or not last_name:
                err_msg = 'Please fill out all required fields.'
                if is_ajax:
                    return jsonify({'success': False, 'error': err_msg}), 400
                flash(err_msg, 'error')
                return redirect(url_for('index'))

            # Check if username already exists
            if User.query.filter_by(username=username).first():
                err_msg = f'Username "{username}" is already taken. Please choose another username.'
                if is_ajax:
                    return jsonify({'success': False, 'error': err_msg}), 400
                flash(err_msg, 'error')
                return redirect(url_for('index'))

            # Check if email already exists
            if User.query.filter_by(email=email).first():
                err_msg = f'An account with email "{email}" already exists.'
                if is_ajax:
                    return jsonify({'success': False, 'error': err_msg}), 400
                flash(err_msg, 'error')
                return redirect(url_for('index'))

            # Only allow clinician registration through this form
            if user_type != 'clinician':
                err_msg = 'Only clinicians can register through this form.'
                if is_ajax:
                    return jsonify({'success': False, 'error': err_msg}), 400
                flash(err_msg, 'error')
                return redirect(url_for('index'))

            profession = (request.form.get('profession') or (request.json.get('profession') if request.is_json else '')).strip()
            if not profession:
                err_msg = 'Please select your profession.'
                if is_ajax:
                    return jsonify({'success': False, 'error': err_msg}), 400
                flash(err_msg, 'error')
                return redirect(url_for('index'))

            # Create new user
            user = User(
                username=username,
                email=email,
                user_type=user_type,
                first_name=first_name,
                last_name=last_name
            )
            user.set_password(password)
            db.session.add(user)
            db.session.flush()  # Get user.id

            license_number = (request.form.get('license_number') or (request.json.get('license_number') if request.is_json else '')).strip()
            specialization = (request.form.get('specialization') or (request.json.get('specialization') if request.is_json else '')).strip()
            clinician_profile = ClinicianProfile(
                user_id=user.id,
                profession=profession,
                license_number=license_number,
                specialization=specialization
            )
            db.session.add(clinician_profile)
            db.session.commit()

            # Auto-login the user after successful registration
            session['user_id'] = user.id
            session['user_type'] = user.user_type
            flash(f'Welcome to NeuroBeat, {user.first_name}!', 'success')

            if is_ajax:
                return jsonify({'success': True, 'redirect': url_for('clinician_dashboard')}), 200

            return redirect(url_for('clinician_dashboard'))

        except Exception as e:
            db.session.rollback()
            logging.error(f"Registration error: {str(e)}", exc_info=True)
            err_msg = 'Registration failed. Please check your details and try again.'
            if is_ajax:
                return jsonify({'success': False, 'error': err_msg}), 500
            flash(err_msg, 'error')
            return redirect(url_for('index'))

    return redirect(url_for('index'))

@app.route('/login', methods=['POST'])
def login():
    """User login"""
    username = request.form['username']
    password = request.form['password']

    user = User.query.filter_by(username=username).first()

    if user and user.check_password(password):
        session['user_id'] = user.id
        session['user_type'] = user.user_type
        flash(f'Welcome back, {user.first_name}!', 'success')

        if user.user_type == 'patient':
            return redirect(url_for('patient_dashboard'))
        else:
            return redirect(url_for('clinician_dashboard'))
    else:
        flash('Invalid username or password', 'error')
        return redirect(url_for('index'))

@app.route('/logout')
def logout():
    """User logout"""
    session.clear()
    flash('You have been logged out successfully.', 'info')
    return redirect(url_for('index'))

@app.route('/patient/dashboard')
def patient_dashboard():
    """Patient dashboard - main interface for patients"""
    if 'user_id' not in session or session.get('user_type') != 'patient':
        flash('Please log in as a patient to access this page.', 'error')
        return redirect(url_for('index'))

    user = User.query.get(session['user_id'])
    patient_profile = user.patient_profile

    if not patient_profile:
        flash('Patient profile not found.', 'error')
        return redirect(url_for('index'))

    # Get recent sessions
    recent_sessions = TherapySession.query.filter_by(
        patient_id=patient_profile.id
    ).order_by(TherapySession.start_time.desc()).limit(5).all()

    # Calculate progress metrics
    total_sessions = TherapySession.query.filter_by(
        patient_id=patient_profile.id, 
        completed=True
    ).count()

    avg_accuracy = db.session.query(db.func.avg(TherapySession.accuracy_score)).filter_by(
        patient_id=patient_profile.id,
        completed=True
    ).scalar() or 0

    return render_template('patient_dashboard.html', 
                         user=user, 
                         patient_profile=patient_profile,
                         recent_sessions=recent_sessions,
                         total_sessions=total_sessions,
                         avg_accuracy=round(avg_accuracy, 1))

@app.route('/api/patient/recent-sessions')
def api_patient_recent_sessions():
    """API endpoint for live real-time dashboard session updates"""
    if 'user_id' not in session or session.get('user_type') != 'patient':
        return jsonify({'error': 'Unauthorized'}), 401

    user = User.query.get(session['user_id'])
    if not user or not user.patient_profile:
        return jsonify({'error': 'Patient profile not found'}), 404

    patient_profile = user.patient_profile
    recent = TherapySession.query.filter_by(
        patient_id=patient_profile.id
    ).order_by(TherapySession.start_time.desc()).limit(6).all()

    total_sessions = TherapySession.query.filter_by(
        patient_id=patient_profile.id, 
        completed=True
    ).count()

    avg_accuracy = db.session.query(db.func.avg(TherapySession.accuracy_score)).filter_by(
        patient_id=patient_profile.id,
        completed=True
    ).scalar() or 0

    sessions_data = []
    for s in recent:
        if s.completed and s.duration_seconds is not None:
            if s.duration_seconds < 60:
                duration_str = f"{s.duration_seconds}s"
            else:
                m = s.duration_seconds // 60
                sec = s.duration_seconds % 60
                duration_str = f"{m}m {sec}s" if sec > 0 else f"{m} min"
        elif s.completed:
            duration_str = "< 1 min"
        else:
            duration_str = "In Progress"

        sessions_data.append({
            'id': s.id,
            'date': s.start_time.strftime('%m/%d/%Y %I:%M %p') if s.start_time else 'Recently',
            'type': s.session_type.replace('_', ' ').title(),
            'completed': s.completed,
            'duration': duration_str,
            'accuracy': round(s.accuracy_score) if s.accuracy_score is not None else None,
            'bpm_range': f"{round(s.initial_bpm)} - {round(s.final_bpm or s.target_bpm or s.initial_bpm)}"
        })

    return jsonify({
        'total_sessions': total_sessions,
        'avg_accuracy': round(avg_accuracy, 1),
        'sessions': sessions_data
    })

@app.route('/clinician/dashboard')
def clinician_dashboard():
    """Clinician dashboard - patient management interface"""
    if 'user_id' not in session or session.get('user_type') != 'clinician':
        flash('Please log in as a clinician to access this page.', 'error')
        return redirect(url_for('index'))

    user = User.query.get(session['user_id'])
    clinician_profile = user.clinician_profile

    if not clinician_profile:
        flash('Clinician profile not found.', 'error')
        return redirect(url_for('index'))

    # Get assigned patients
    patients = PatientProfile.query.filter_by(assigned_clinician_id=user.id).all()

    # Get unassigned patients
    unassigned_patients = PatientProfile.query.filter_by(assigned_clinician_id=None).all()

    return render_template('clinician_dashboard.html',
                         user=user,
                         clinician_profile=clinician_profile,
                         patients=patients,
                         unassigned_patients=unassigned_patients)

@app.route('/session/start', methods=['POST'])
def start_session():
    """Start a new therapy session"""
    if 'user_id' not in session or session.get('user_type') != 'patient':
        return jsonify({'error': 'Unauthorized: Please log in as a patient'}), 401

    try:
        from beat_generator import BeatGenerator

        user = User.query.get(session['user_id'])
        if not user:
            return jsonify({'error': 'User not found'}), 401

        patient_profile = user.patient_profile
        if not patient_profile:
            patient_profile = PatientProfile(
                user_id=user.id,
                condition='parkinsons'
            )
            db.session.add(patient_profile)
            db.session.commit()

        data = request.get_json(silent=True) or request.form.to_dict() or {}
        raw_session_type = data.get('session_type', 'gait_trainer') or 'gait_trainer'
        session_type = normalize_session_type(raw_session_type)

        try:
            initial_bpm = float(data.get('initial_bpm', 60))
        except (ValueError, TypeError):
            initial_bpm = 60.0

        try:
            target_bpm = float(data.get('target_bpm', 70))
        except (ValueError, TypeError):
            target_bpm = 70.0

        # Generate beats for stroke patients
        beat_url = None
        if patient_profile.condition == 'stroke':
            try:
                beat_generator = BeatGenerator()
                patient_condition = {
                    'affected_side': patient_profile.stroke_affected_side,
                    'severity': patient_profile.stroke_severity,
                    'aphasia_type': patient_profile.aphasia_type,
                    'dysarthria_severity': patient_profile.dysarthria_severity,
                    'motor_impairment': patient_profile.motor_impairment_level,
                    'cognitive_status': patient_profile.cognitive_status,
                    'emotional_status': patient_profile.emotional_status,
                    'preferred_genre': patient_profile.preferred_music_genre,
                    'preferred_sound': patient_profile.preferred_beat_sound or 'metronome'
                }
                beat_url = beat_generator.generate_stroke_therapy_beat(session_type, int(initial_bpm), patient_condition)
                optimal_bpm = beat_generator.get_optimal_bpm_for_stroke_therapy(session_type, patient_condition)
                if abs(initial_bpm - optimal_bpm) > 10:
                    initial_bpm = float(optimal_bpm)
            except Exception as bg_err:
                logging.warning(f"Beat generator fallback: {str(bg_err)}")
                beat_url = f"local_audio:metronome:{int(initial_bpm)}"

        # Safe parsing of cognitive load level
        raw_cog = data.get('cognitive_load_level')
        try:
            cog_level = int(raw_cog) if raw_cog is not None else 1
        except (ValueError, TypeError):
            cog_level = 1

        # Multi-Session Learning: Load Personal Performance Envelope
        envelope = PatientPerformanceEnvelope.query.filter_by(
            patient_id=patient_profile.id,
            exercise_type=session_type
        ).first()

        envelope_data = None
        if envelope and envelope.sessions_evaluated > 0:
            envelope_data = {
                'stable_bpm_min': envelope.stable_bpm_min,
                'stable_bpm_max': envelope.stable_bpm_max,
                'typical_cadence': envelope.typical_cadence,
                'cadence_cv': envelope.cadence_cv,
                'phase_stability_mean': envelope.phase_stability_mean,
                'adaptation_tolerance': envelope.adaptation_tolerance,
                'confidence': envelope.confidence,
                'sessions_evaluated': envelope.sessions_evaluated
            }
            # Personalize starting tempo from learned history if not explicitly requested
            if 'initial_bpm' not in data and envelope.typical_cadence:
                initial_bpm = float(round(envelope.typical_cadence))
                target_bpm = float(round(min(envelope.stable_bpm_max + 4.0, initial_bpm + 6.0)))

        # Create new session
        therapy_session = TherapySession(
            patient_id=patient_profile.id,
            session_type=session_type,
            initial_bpm=initial_bpm,
            target_bpm=target_bpm,
            start_time=datetime.utcnow(),
            generated_beat_url=beat_url,
            affected_limb=data.get('affected_limb'),
            cognitive_load_level=cog_level
        )

        db.session.add(therapy_session)
        db.session.commit()

        # Emit initial session lifecycle event
        init_event = SessionEvent(
            session_id=therapy_session.id,
            timestamp=0.0,
            event_type='LIFECYCLE',
            source='P0',
            severity='INFO',
            payload=json.dumps({
                'status': 'SESSION_CREATED',
                'initial_bpm': initial_bpm,
                'target_bpm': target_bpm,
                'has_learned_envelope': envelope_data is not None
            }),
            idempotency_key=f"init_{therapy_session.id}_{int(datetime.utcnow().timestamp())}"
        )
        db.session.add(init_event)
        db.session.commit()

        return jsonify({
            'session_id': therapy_session.id,
            'initial_bpm': initial_bpm,
            'target_bpm': target_bpm,
            'session_type': session_type,
            'beat_url': beat_url,
            'stroke_specific': patient_profile.condition == 'stroke',
            'performance_envelope': envelope_data
        })

    except Exception as e:
        db.session.rollback()
        logging.error(f"Error starting session: {str(e)}", exc_info=True)
        return jsonify({'error': 'Failed to start session', 'details': str(e)}), 500

@app.route('/session/<int:session_id>')
def session_view(session_id):
    """Session interface for therapy"""
    if 'user_id' not in session or session.get('user_type') != 'patient':
        flash('Please log in as a patient to access this page.', 'error')
        return redirect(url_for('index'))

    therapy_session = TherapySession.query.get_or_404(session_id)
    user = User.query.get(session['user_id'])

    # Verify session belongs to current patient
    if therapy_session.patient.user_id != user.id:
        flash('Unauthorized access to session.', 'error')
        return redirect(url_for('patient_dashboard'))

    mode_config = get_session_mode(therapy_session.session_type)
    envelope = PatientPerformanceEnvelope.query.filter_by(
        patient_id=therapy_session.patient_id,
        exercise_type=therapy_session.session_type
    ).first()
    envelope_dict = envelope.to_dict() if envelope else None

    return render_template(
        'session.html',
        therapy_session=therapy_session,
        mode_config=mode_config,
        performance_envelope=envelope_dict
    )

def _can_access_patient(user, patient):
    """Centralized authorization check for patient access"""
    if not user or not patient:
        return False
    if user.user_type == 'patient':
        return patient.user_id == user.id
    if user.user_type == 'clinician':
        if patient.assigned_clinician_id is not None and patient.assigned_clinician_id != user.id:
            return False
        return True
    return False

def _can_access_session(user, therapy_session):
    """Centralized authorization check for session access"""
    if not user or not therapy_session:
        return False
    return _can_access_patient(user, therapy_session.patient)

def _execute_session_update(session_id, data):
    """Authoritative handler for session update with ownership and P2/P5 persistence"""
    if 'user_id' not in session or session.get('user_type') != 'patient':
        return jsonify({'error': 'Unauthorized'}), 401

    user = User.query.get(session['user_id'])
    if not user:
        return jsonify({'error': 'Unauthorized'}), 401

    therapy_session = TherapySession.query.get_or_404(session_id)
    if not _can_access_session(user, therapy_session):
        return jsonify({'error': 'Forbidden: Session does not belong to user'}), 403

    current_bpm = float(data.get('current_bpm', therapy_session.initial_bpm))
    sync_accuracy = float(data.get('sync_accuracy', 0))

    # Add session metric
    metric = SessionMetrics(
        session_id=session_id,
        current_bpm=current_bpm,
        sync_accuracy=sync_accuracy,
        timestamp=datetime.utcnow()
    )
    db.session.add(metric)

    # Calculate BPM adjustment based on adaptive engine recommendation or accuracy
    suggested_bpm = data.get('suggested_bpm')
    if suggested_bpm is not None:
        try:
            adjustment_bpm = max(40.0, min(140.0, float(suggested_bpm)))
        except (ValueError, TypeError):
            adjustment_bpm = current_bpm
    else:
        adjustment_bpm = current_bpm
        if sync_accuracy > 0:
            if sync_accuracy < 70:
                adjustment_bpm = max(current_bpm - 2, 40)
            elif sync_accuracy > 90:
                adjustment_bpm = min(current_bpm + 1, 120)

    if adjustment_bpm != current_bpm:
        metric.adjustment_made = True

    # Hardened Persistence: Ingest Adaptation Record if provided
    ad_record_data = data.get('adaptation_record') or data.get('adaptation_decision')
    if ad_record_data and isinstance(ad_record_data, dict):
        ar = AdaptationRecord(
            session_id=session_id,
            parameter=ad_record_data.get('parameter', 'TEMPO'),
            previous_value=float(ad_record_data.get('previous_value', current_bpm)),
            requested_value=float(ad_record_data.get('requested_value', current_bpm)),
            executed_value=float(ad_record_data.get('executed_value', current_bpm)),
            direction=ad_record_data.get('direction', 'MAINTAIN'),
            trigger_reason=ad_record_data.get('trigger_reason', 'Automated adaptation'),
            validator_status=ad_record_data.get('validator_status', 'APPROVED'),
            clamp_reason=ad_record_data.get('clamp_reason'),
            confidence=float(ad_record_data.get('confidence', 1.0))
        )
        db.session.add(ar)

    # Hardened Persistence: Ingest Agent Decision if provided
    agent_dec_data = data.get('agent_decision')
    if agent_dec_data and isinstance(agent_dec_data, dict) and agent_dec_data.get('action'):
        ag = AgentDecision(
            session_id=session_id,
            intent=agent_dec_data.get('intent', 'MAINTAIN'),
            target=agent_dec_data.get('target', 'TEMPO'),
            action=agent_dec_data.get('action', 'MAINTAIN'),
            requested_magnitude=float(agent_dec_data.get('magnitude') or agent_dec_data.get('requested_magnitude') or 0.0),
            reasoning_summary=agent_dec_data.get('reason') or agent_dec_data.get('reasoning_summary') or '',
            confidence=float(agent_dec_data.get('confidence', 0.9)),
            validator_result=agent_dec_data.get('validator_result', 'APPROVED')
        )
        db.session.add(ag)

    db.session.commit()

    return jsonify({
        'adjusted_bpm': adjustment_bpm,
        'sync_accuracy': sync_accuracy
    })

@app.route('/session/update', methods=['POST'])
def update_session():
    """Update session metrics in real-time with P2/P5 persistence"""
    try:
        data = request.get_json(silent=True) or {}
        session_id = int(data.get('session_id'))
        return _execute_session_update(session_id, data)
    except Exception as e:
        db.session.rollback()
        logging.error(f"Error updating session: {str(e)}")
        return jsonify({'error': 'Failed to update session'}), 500

@app.route('/session/<int:session_id>/update', methods=['POST'])
def update_session_legacy(session_id):
    """Compatibility route: delegates to unified authoritative _execute_session_update"""
    try:
        data = request.get_json(silent=True) or {}
        return _execute_session_update(session_id, data)
    except Exception as e:
        db.session.rollback()
        logging.error(f"Error updating session: {str(e)}")
        return jsonify({'error': 'Failed to update session'}), 500

@app.route('/session/<int:session_id>/complete', methods=['POST'])
def complete_session(session_id):
    """Complete a therapy session with full closed-loop persistence and memory consolidation"""
    if 'user_id' not in session or session.get('user_type') != 'patient':
        return jsonify({'error': 'Unauthorized'}), 401

    try:
        therapy_session = TherapySession.query.get_or_404(session_id)
        user = User.query.get(session['user_id'])

        # Verify session belongs to current patient
        if not user or not _can_access_session(user, therapy_session):
            return jsonify({'error': 'Forbidden: Access denied'}), 403

        # Update session completion data
        data = request.get_json(silent=True) or request.form.to_dict() or {}
        therapy_session.end_time = datetime.utcnow()
        therapy_session.completed = True
        duration = int(data.get('duration', 0))
        final_bpm = float(data.get('final_bpm', therapy_session.initial_bpm))
        accuracy_score = float(data.get('accuracy_score', 0))
        left_steps = int(data.get('left_steps', 0))
        right_steps = int(data.get('right_steps', 0))
        gait_symmetry = float(data.get('gait_symmetry', 0))

        therapy_session.duration_seconds = duration
        therapy_session.final_bpm = final_bpm
        therapy_session.accuracy_score = accuracy_score
        therapy_session.notes = data.get('notes', '')

        tap_count = int(data.get('tap_count', 0))
        tap_cadence = float(data.get('tap_cadence', 0))
        vocal_count = int(data.get('vocal_count', 0))
        vocal_cadence = float(data.get('vocal_cadence', 0))
        posture_stability = float(data.get('posture_stability', 100 if therapy_session.session_type == 'balance_training' else 0))

        metrics_dict = data.get('metrics_data', {})
        if not isinstance(metrics_dict, dict):
            metrics_dict = {}
        metrics_dict['left_steps'] = left_steps
        metrics_dict['right_steps'] = right_steps
        metrics_dict['total_steps'] = left_steps + right_steps
        metrics_dict['gait_symmetry'] = gait_symmetry
        metrics_dict['tap_count'] = tap_count
        metrics_dict['tap_cadence'] = tap_cadence
        metrics_dict['vocal_count'] = vocal_count
        metrics_dict['vocal_cadence'] = vocal_cadence
        metrics_dict['posture_stability'] = posture_stability

        # Generate Nuro Agent session reflection
        from services.gemini_service import generate_patient_feedback_few_shot, generate_agent_session_reflection
        incoming_summary = data.get('agent_summary') or metrics_dict.get('agent_summary') or data.get('summary')
        
        mq_candidate = None
        if isinstance(incoming_summary, dict) and incoming_summary.get('averageMovementQuality') is not None:
            mq_candidate = incoming_summary.get('averageMovementQuality')
        elif metrics_dict.get('movement_quality') is not None:
            mq_candidate = metrics_dict.get('movement_quality')
        elif data.get('movement_quality') is not None:
            mq_candidate = data.get('movement_quality')

        if isinstance(mq_candidate, dict):
            raw_mov_qual = float(mq_candidate.get('overall', 0.85))
        elif mq_candidate is not None:
            try:
                raw_mov_qual = float(mq_candidate)
            except (ValueError, TypeError):
                raw_mov_qual = 0.85
        else:
            raw_mov_qual = 0.85

        acc_ratio = (accuracy_score / 100.0) if accuracy_score is not None else 0.0
        agent_summary = incoming_summary or {
            'sessionId': session_id,
            'duration': duration,
            'endingPerformance': acc_ratio,
            'averageMovementQuality': raw_mov_qual,
            'averageRhythmSync': acc_ratio,
            'bestTempo': final_bpm
        }
        agent_reflection = generate_agent_session_reflection(agent_summary)
        metrics_dict['agent_reflection'] = agent_reflection
        therapy_session.set_metrics(metrics_dict)

        # 1. Relational Persistence: SessionSummary
        sess_summary = SessionSummary.query.filter_by(session_id=session_id).first()
        if not sess_summary:
            sess_summary = SessionSummary(session_id=session_id)
            db.session.add(sess_summary)
        
        sess_summary.starting_performance = float(agent_summary.get('startingPerformance') if agent_summary.get('startingPerformance') is not None else acc_ratio)
        sess_summary.ending_performance = float(agent_summary.get('endingPerformance') if agent_summary.get('endingPerformance') is not None else acc_ratio)
        sess_summary.improvement = float(agent_summary.get('improvement') or 0.0)
        sess_summary.average_movement_quality = float(agent_summary.get('averageMovementQuality') if agent_summary.get('averageMovementQuality') is not None else raw_mov_qual)
        sess_summary.average_rhythm_sync = float(agent_summary.get('averageRhythmSync') if agent_summary.get('averageRhythmSync') is not None else acc_ratio)
        raw_conf = agent_summary.get('averageConfidence')
        sess_summary.average_confidence = float(raw_conf) if raw_conf is not None else 0.9
        sess_summary.best_tempo = float(agent_summary.get('bestTempo') or final_bpm)
        sess_summary.successful_tempo_range = str(agent_summary.get('successfulTempoRange') or f"{round(final_bpm)} BPM")
        sess_summary.successful_adaptations = int(agent_summary.get('successfulAdaptations') or 0)
        sess_summary.unsuccessful_adaptations = int(agent_summary.get('unsuccessfulAdaptations') or 0)
        sess_summary.performance_trend = str(agent_summary.get('performanceTrend') or 'STABLE')
        sess_summary.agent_reflection_json = json.dumps(agent_reflection)

        # 2. Multi-Session Memory: Update PatientPerformanceEnvelope
        st = therapy_session.session_type
        envelope = PatientPerformanceEnvelope.query.filter_by(
            patient_id=therapy_session.patient_id,
            exercise_type=st
        ).first()

        # Section 15: Memory Protection Guard against corrupted/incomplete/degraded evidence
        is_valid_observation = (
            data.get('status') != 'ABANDONED' and
            not data.get('abandoned', False) and
            duration >= 15.0 and
            0.0 <= accuracy_score <= 100.0 and
            40.0 <= final_bpm <= 140.0 and
            sess_summary.average_confidence >= 0.45
        )

        if is_valid_observation:
            if not envelope:
                envelope = PatientPerformanceEnvelope(
                    patient_id=therapy_session.patient_id,
                    exercise_type=st,
                    stable_bpm_min=min(therapy_session.initial_bpm, final_bpm),
                    stable_bpm_max=max(therapy_session.initial_bpm, final_bpm) if accuracy_score >= 75 else therapy_session.initial_bpm,
                    typical_cadence=final_bpm if accuracy_score >= 75 else therapy_session.initial_bpm,
                    sessions_evaluated=1,
                    confidence=min(1.0, (accuracy_score / 100.0))
                )
                db.session.add(envelope)
            else:
                alpha = 0.15
                envelope.typical_cadence = (1.0 - alpha) * (envelope.typical_cadence or final_bpm) + alpha * final_bpm
                if accuracy_score >= 80:
                    envelope.stable_bpm_max = max(envelope.stable_bpm_max, final_bpm)
                elif accuracy_score <= 60:
                    # Degradation at high tempo means upper ceiling must be respected
                    if final_bpm >= (envelope.typical_cadence or 60.0):
                        envelope.stable_bpm_max = min(envelope.stable_bpm_max, max(final_bpm - 1.0, envelope.stable_bpm_min))
                    else:
                        envelope.stable_bpm_min = max(envelope.stable_bpm_min, min(final_bpm + 1.0, envelope.stable_bpm_max))

                envelope.sessions_evaluated += 1
                envelope.confidence = min(0.98, envelope.confidence + 0.05)
                envelope.updated_at = datetime.utcnow()

        # 3. Emit session completion lifecycle event
        comp_event = SessionEvent(
            session_id=session_id,
            timestamp=float(duration),
            event_type='LIFECYCLE',
            source='P5',
            severity='INFO',
            payload=json.dumps({
                'status': 'SESSION_COMPLETED',
                'duration': duration,
                'final_bpm': final_bpm,
                'accuracy_score': accuracy_score,
                'improvement': sess_summary.improvement
            }),
            idempotency_key=f"complete_{session_id}_{int(datetime.utcnow().timestamp())}"
        )
        db.session.add(comp_event)

        db.session.commit()

        # Generate personalized recovery coaching feedback via Gemini Few-Shot
        feedback = generate_patient_feedback_few_shot(
            therapy_session.session_type,
            duration,
            accuracy_score,
            f"{round(therapy_session.initial_bpm)} -> {round(final_bpm)}",
            left_steps=left_steps,
            right_steps=right_steps,
            symmetry=gait_symmetry
        )

        # Generate and persist Structured Clinical Report (Gemini + Longitudinal Context)
        clinical_report_dict = None
        try:
            from services.historical_analysis import format_historical_context_for_prompt, save_or_update_clinical_report
            from services.gemini_service import generate_structured_clinical_report

            hist_ctx = format_historical_context_for_prompt(therapy_session.patient_id, therapy_session.session_type)
            clinical_data = {
                'activity_type': therapy_session.session_type,
                'duration_seconds': duration,
                'initial_bpm': therapy_session.initial_bpm,
                'final_bpm': final_bpm,
                'accuracy_score': accuracy_score,
                'movement_count': max(left_steps + right_steps, tap_count, vocal_count),
                'vocal_count': vocal_count,
                'vocal_cadence': vocal_cadence
            }
            clinical_report_json = generate_structured_clinical_report(clinical_data, hist_ctx)
            saved_report = save_or_update_clinical_report(session_id, clinical_report_json)
            if saved_report:
                clinical_report_dict = saved_report.to_dict()
        except Exception as e:
            logging.error(f"Error generating clinical report in complete_session: {e}")

        return jsonify({
            'success': True,
            'feedback': feedback,
            'agent_reflection': agent_reflection,
            'clinical_report': clinical_report_dict,
            'session_summary': {
                'best_tempo': sess_summary.best_tempo,
                'improvement': sess_summary.improvement,
                'successful_tempo_range': sess_summary.successful_tempo_range
            },
            'performance_envelope': {
                'stable_bpm_min': envelope.stable_bpm_min,
                'stable_bpm_max': envelope.stable_bpm_max,
                'typical_cadence': envelope.typical_cadence,
                'sessions_evaluated': envelope.sessions_evaluated
            } if envelope else None,
            'left_steps': left_steps,
            'right_steps': right_steps,
            'gait_symmetry': gait_symmetry,
            'tap_count': tap_count,
            'tap_cadence': tap_cadence,
            'vocal_count': vocal_count,
            'vocal_cadence': vocal_cadence,
            'posture_stability': posture_stability
        })

    except HTTPException as he:
        db.session.rollback()
        return jsonify({'error': he.description}), he.code
    except Exception as e:
        db.session.rollback()
        logging.error(f"Error completing session: {str(e)}", exc_info=True)
        return jsonify({'error': 'Failed to complete session'}), 500

@app.route('/api/session/<int:session_id>/events', methods=['POST'])
def push_session_events(session_id):
    """Batch-ingest idempotent session events without blocking the video frame loop"""
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401

    try:
        user = User.query.get(session['user_id'])
        therapy_session = TherapySession.query.get_or_404(session_id)
        if not user or not _can_access_session(user, therapy_session):
            return jsonify({'error': 'Forbidden: Access denied'}), 403

        data = request.get_json(silent=True) or {}
        raw_events = data.get('events', [])
        if isinstance(data, list):
            raw_events = data

        ingested = 0
        skipped = 0
        for ev in raw_events:
            if not isinstance(ev, dict):
                continue
            key = ev.get('idempotency_key')
            if key and SessionEvent.query.filter_by(idempotency_key=key).first():
                skipped += 1
                continue

            payload_val = ev.get('payload')
            if isinstance(payload_val, (dict, list)):
                payload_str = json.dumps(payload_val)
            elif isinstance(payload_val, str):
                payload_str = payload_val
            else:
                payload_str = json.dumps({})

            event_row = SessionEvent(
                session_id=session_id,
                timestamp=float(ev.get('timestamp', 0.0)),
                event_type=str(ev.get('event_type', 'GENERAL')),
                source=str(ev.get('source', 'P0')),
                severity=str(ev.get('severity', 'INFO')),
                payload=payload_str,
                idempotency_key=key
            )
            db.session.add(event_row)
            ingested += 1

        db.session.commit()
        return jsonify({
            'success': True,
            'session_id': session_id,
            'ingested_count': ingested,
            'ingested_events': ingested,
            'skipped_duplicate_events': skipped
        }), 201
    except HTTPException as he:
        db.session.rollback()
        return jsonify({'error': he.description}), he.code
    except Exception as e:
        db.session.rollback()
        logging.error(f"Error pushing session events: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/session/<int:session_id>/interventions', methods=['POST'])
def create_intervention(session_id):
    """Register a causal intervention before entering the observation window"""
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401

    try:
        user = User.query.get(session['user_id'])
        therapy_session = TherapySession.query.get_or_404(session_id)
        if not user or not _can_access_session(user, therapy_session):
            return jsonify({'error': 'Forbidden: Access denied'}), 403

        data = request.get_json(silent=True) or {}

        intervention = Intervention(
            session_id=session_id,
            target_parameter=data.get('target_parameter', 'TEMPO'),
            action_taken=data.get('action_taken', 'INCREASE_TEMPO'),
            previous_value=float(data.get('previous_value', therapy_session.initial_bpm)),
            new_value=float(data.get('new_value', therapy_session.initial_bpm)),
            pre_performance=float(data.get('pre_performance', 0.8)),
            pre_rhythm_sync=float(data.get('pre_rhythm_sync', 0.8)),
            pre_movement_quality=float(data.get('pre_movement_quality', 0.8)),
            observation_window_cycles=int(data.get('observation_window_cycles', 3)),
            status='OBSERVING'
        )
        db.session.add(intervention)
        db.session.commit()

        return jsonify({
            'success': True,
            'intervention_id': intervention.id,
            'status': 'OBSERVING'
        }), 201
    except HTTPException as he:
        db.session.rollback()
        return jsonify({'error': he.description}), he.code
    except Exception as e:
        db.session.rollback()
        logging.error(f"Error creating intervention: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/intervention/<int:intervention_id>/outcome', methods=['POST'])
def record_intervention_outcome(intervention_id):
    """Record evaluated empirical outcome for a causal intervention"""
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401

    try:
        user = User.query.get(session['user_id'])
        intervention = Intervention.query.get_or_404(intervention_id)
        if not user or not _can_access_session(user, intervention.session):
            return jsonify({'error': 'Forbidden: Access denied'}), 403

        data = request.get_json(silent=True) or {}

        outcome = InterventionOutcome.query.filter_by(intervention_id=intervention_id).first()
        if not outcome:
            outcome = InterventionOutcome(intervention_id=intervention_id)
            db.session.add(outcome)

        post_p = float(data.get('post_performance', 0.8))
        post_r = float(data.get('post_rhythm_sync', 0.8))
        post_m = float(data.get('post_movement_quality', 0.8))

        delta_p = data.get('delta_performance')
        if delta_p is None:
            delta_p = post_p - (intervention.pre_performance or 0.8)

        delta_r = data.get('delta_rhythm_sync')
        if delta_r is None:
            delta_r = post_r - (intervention.pre_rhythm_sync or 0.8)

        delta_m = data.get('delta_movement_quality')
        if delta_m is None:
            delta_m = post_m - (intervention.pre_movement_quality or 0.8)

        # Section 20: ResponseScore = 0.50 * delta_p + 0.25 * delta_r + 0.25 * delta_m
        resp_score = data.get('response_score')
        if resp_score is None:
            resp_score = 0.50 * float(delta_p) + 0.25 * float(delta_r) + 0.25 * float(delta_m)

        outcome.post_performance = post_p
        outcome.post_rhythm_sync = post_r
        outcome.post_movement_quality = post_m
        outcome.delta_performance = float(delta_p)
        outcome.delta_rhythm_sync = float(delta_r)
        outcome.delta_movement_quality = float(delta_m)
        outcome.response_score = float(resp_score)
        outcome.classification = str(data.get('classification', 'POSITIVE' if delta_p > 0.03 else ('NEGATIVE' if delta_p < -0.03 else 'NEUTRAL')))
        outcome.confidence = float(data.get('confidence', 0.9))
        outcome.evaluation_window_cycles = int(data.get('evaluation_window_cycles') or data.get('evaluation_window') or 3)

        intervention.status = 'EVALUATED'
        db.session.commit()

        return jsonify({
            'success': True,
            'intervention_id': intervention_id,
            'classification': outcome.classification,
            'delta_performance': outcome.delta_performance
        }), 201
    except HTTPException as he:
        db.session.rollback()
        return jsonify({'error': he.description}), he.code
    except Exception as e:
        db.session.rollback()
        logging.error(f"Error recording intervention outcome: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/patient/<int:patient_id>/envelope', defaults={'exercise_type': 'gait_trainer'}, methods=['GET'])
@app.route('/api/patient/<int:patient_id>/envelope/<exercise_type>', methods=['GET'])
def get_patient_envelope(patient_id, exercise_type='gait_trainer'):
    """Retrieve personal performance envelope for exercise personalization"""
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401

    user = User.query.get(session['user_id'])
    patient = PatientProfile.query.get_or_404(patient_id)
    if not user or not _can_access_patient(user, patient):
        return jsonify({'error': 'Forbidden: Access denied'}), 403

    envelope = PatientPerformanceEnvelope.query.filter_by(
        patient_id=patient_id,
        exercise_type=exercise_type
    ).first()

    if not envelope:
        return jsonify({
            'patient_id': patient_id,
            'exercise_type': exercise_type,
            'has_envelope': False,
            'stable_bpm_min': 45.0,
            'stable_bpm_max': 72.0,
            'typical_cadence': 60.0,
            'adaptation_tolerance': 0.5,
            'sessions_evaluated': 0
        })

    return jsonify({
        'patient_id': patient_id,
        'exercise_type': exercise_type,
        'has_envelope': True,
        'stable_bpm_min': envelope.stable_bpm_min,
        'stable_bpm_max': envelope.stable_bpm_max,
        'typical_cadence': envelope.typical_cadence,
        'cadence_cv': envelope.cadence_cv,
        'phase_stability_mean': envelope.phase_stability_mean,
        'adaptation_tolerance': envelope.adaptation_tolerance,
        'confidence': envelope.confidence,
        'sessions_evaluated': envelope.sessions_evaluated,
        'updated_at': envelope.updated_at.isoformat() if envelope.updated_at else None
    })

@app.route('/api/session/<int:session_id>/replay', methods=['GET'])
def get_session_replay(session_id):
    """Retrieve full chronological trace for deterministic offline replay"""
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401

    user = User.query.get(session['user_id'])
    therapy_session = TherapySession.query.get_or_404(session_id)
    if not user or not _can_access_session(user, therapy_session):
        return jsonify({'error': 'Forbidden: Access denied'}), 403

    events = SessionEvent.query.filter_by(session_id=session_id).order_by(SessionEvent.timestamp.asc()).all()
    adaptations = AdaptationRecord.query.filter_by(session_id=session_id).order_by(AdaptationRecord.timestamp.asc()).all()
    decisions = AgentDecision.query.filter_by(session_id=session_id).order_by(AgentDecision.timestamp.asc()).all()
    interventions = Intervention.query.filter_by(session_id=session_id).order_by(Intervention.timestamp.asc()).all()

    replay_payload = {
        'session_id': session_id,
        'patient_id': therapy_session.patient_id,
        'session_type': therapy_session.session_type,
        'initial_bpm': therapy_session.initial_bpm,
        'final_bpm': therapy_session.final_bpm,
        'duration_seconds': therapy_session.duration_seconds,
        'events': [{
            'id': e.id, 'timestamp': e.timestamp, 'event_type': e.event_type,
            'source': e.source, 'severity': e.severity, 'payload': e.get_payload()
        } for e in events],
        'adaptations': [{
            'id': a.id, 'timestamp': a.timestamp.isoformat(), 'parameter': a.parameter,
            'previous_value': a.previous_value, 'requested_value': a.requested_value,
            'executed_value': a.executed_value, 'direction': a.direction,
            'trigger_reason': a.trigger_reason, 'validator_status': a.validator_status,
            'clamp_reason': a.clamp_reason
        } for a in adaptations],
        'agent_decisions': [{
            'id': d.id, 'timestamp': d.timestamp.isoformat(), 'intent': d.intent,
            'target': d.target, 'action': d.action, 'magnitude': d.requested_magnitude,
            'reason': d.reasoning_summary, 'confidence': d.confidence,
            'validator_result': d.validator_result
        } for d in decisions],
        'interventions': [{
            'id': i.id, 'timestamp': i.timestamp.isoformat(), 'parameter': i.target_parameter,
            'action': i.action_taken, 'before_bpm': i.previous_value, 'after_bpm': i.new_value,
            'outcome': {
                'classification': i.outcome.classification,
                'delta_performance': i.outcome.delta_performance,
                'response_score': i.outcome.response_score
            } if i.outcome else None
        } for i in interventions],
        'summary': {
            'starting_performance': therapy_session.summary.starting_performance,
            'ending_performance': therapy_session.summary.ending_performance,
            'improvement': therapy_session.summary.improvement,
            'best_tempo': therapy_session.summary.best_tempo,
            'successful_tempo_range': therapy_session.summary.successful_tempo_range
        } if therapy_session.summary else None,
        'metadata': {
            m.component: {'version': m.model_version, 'hash': m.model_hash, 'schema_version': m.schema_version}
            for m in ModelMetadata.query.all()
        },
        'schema_version': '3.0'
    }
    return jsonify(replay_payload)

@app.route('/api/motion/telemetry', methods=['POST'])
def motion_telemetry():
    """Real-time movement analysis endpoint for live HUD & backend consumption (Schema v2.0)"""
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401

    try:
        data = request.get_json() or {}
        from services.measurement_service import measurement_service
        feature_vector = measurement_service.extract_feature_vector(data)
        session_id = data.get('session_id') or feature_vector.get('session_id')

        return jsonify({
            'status': 'received',
            'schema_version': '2.0',
            'session_id': session_id,
            'feature_vector': feature_vector
        })
    except Exception as e:
        logging.error(f"Error in motion telemetry: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/session/<int:session_id>/measurement-summary', methods=['GET'])
def session_measurement_summary(session_id):
    """Retrieve 5-section validated measurement summary for a session conforming to Section 43"""
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401

    user = User.query.get(session['user_id'])
    therapy_session = TherapySession.query.get_or_404(session_id)
    if not user or not _can_access_session(user, therapy_session):
        return jsonify({'error': 'Forbidden: Access denied'}), 403

    from services.measurement_service import measurement_service
    from services.gemini_service import generate_validated_measurement_summary

    raw_metrics = therapy_session.get_metrics()
    feature_vector = measurement_service.extract_feature_vector(raw_metrics)
    patient_name = f"{therapy_session.patient.user.first_name} {therapy_session.patient.user.last_name}"
    condition = therapy_session.patient.condition or "Neurological Rehabilitation"

    report_data = generate_validated_measurement_summary(feature_vector, patient_name, condition)
    return jsonify({
        'session_id': session_id,
        'schema_version': '2.0',
        'feature_vector': feature_vector,
        'report': report_data.get('report')
    })

@app.route('/api/nuro-agent/reason', methods=['POST'])
def nuro_agent_reason():
    """Advisory reasoning endpoint for Nuro Agent with deterministic fallback"""
    try:
        data = request.get_json(silent=True) or {}
        context = data.get('context', {})
        from services.gemini_service import generate_agent_reasoning
        decision = generate_agent_reasoning(context)
        return jsonify({
            'success': True,
            'decision': decision
        })
    except Exception as e:
        logging.error(f"Error in nuro_agent_reason: {e}")
        from services.gemini_service import _generate_deterministic_agent_reasoning
        return jsonify({
            'success': True,
            'decision': _generate_deterministic_agent_reasoning({})
        })

@app.route('/api/session/<int:session_id>/agent-summary', methods=['GET'])
def session_agent_summary(session_id):
    """Retrieve structured agent session memory, experiments, and reflection"""
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401

    user = User.query.get(session['user_id'])
    therapy_session = TherapySession.query.get_or_404(session_id)
    if not user or not _can_access_session(user, therapy_session):
        return jsonify({'error': 'Forbidden: Access denied'}), 403

    metrics = therapy_session.get_metrics()
    return jsonify({
        'session_id': session_id,
        'agent_summary': metrics.get('agent_summary'),
        'agent_state': metrics.get('agent_state'),
        'agent_reflection': metrics.get('agent_reflection'),
        'adaptation_decision': metrics.get('adaptation_decision')
    })

@app.route('/api/clinician/ai-report/<int:patient_id>', methods=['POST'])
def generate_ai_report(patient_id):
    """Generate clinical progress note and SOAP assessment via Gemini Few-Shot prompting"""
    if 'user_id' not in session or session.get('user_type') != 'clinician':
        return jsonify({'error': 'Unauthorized: Clinician access required'}), 401

    try:
        user = User.query.get(session['user_id'])
        patient = PatientProfile.query.get_or_404(patient_id)
        if not user or not _can_access_patient(user, patient):
            return jsonify({'error': 'Forbidden: Clinician not authorized for patient'}), 403

        from services.gemini_service import generate_clinical_soap_note_few_shot

        # Gather baseline metrics
        baseline = {
            'cadence': patient.baseline_cadence,
            'tapping_speed': patient.baseline_tapping_speed,
            'speech_rate': patient.baseline_speech_rate
        }

        # Gather last 8 completed sessions
        completed_sessions = [s for s in patient.sessions if s.completed]
        recent_sessions = [{
            'date': s.start_time.strftime('%Y-%m-%d') if s.start_time else 'N/A',
            'type': s.session_type,
            'duration_min': round((s.duration_seconds or 0) / 60, 1),
            'accuracy': round(s.accuracy_score or 0),
            'bpm': round(s.final_bpm or s.initial_bpm or 60)
        } for s in completed_sessions[-8:]]

        report_result = generate_clinical_soap_note_few_shot(
            f"{patient.user.first_name} {patient.user.last_name}",
            patient.condition,
            baseline,
            recent_sessions
        )

        return jsonify(report_result)

    except Exception as e:
        logging.error(f"Error generating AI clinical report: {str(e)}", exc_info=True)
        return jsonify({'error': 'Failed to generate report', 'details': str(e)}), 500

@app.route('/baseline/assessment', methods=['GET', 'POST'])
def baseline_assessment():
    """Baseline assessment interface - clinicians only"""
    if 'user_id' not in session or session.get('user_type') != 'clinician':
        flash('Please log in as a clinician to access this page.', 'error')
        return redirect(url_for('index'))

    user = User.query.get(session['user_id'])

    if request.method == 'POST':
        try:
            assessment_type = request.form['assessment_type']
            measured_value = float(request.form['measured_value'])
            notes = request.form.get('notes', '')

            # Clinician assessing a patient
            patient_id = int(request.form['patient_id'])
            assessed_by = user.id

            assessment = BaselineAssessment(
                patient_id=patient_id,
                assessment_type=assessment_type,
                measured_value=measured_value,
                notes=notes,
                assessed_by=assessed_by
            )

            db.session.add(assessment)

            # Update patient profile with baseline values
            patient_profile = PatientProfile.query.get(patient_id)
            if assessment_type == 'gait':
                patient_profile.baseline_cadence = measured_value
                patient_profile.target_cadence = measured_value * 1.1  # 10% improvement target
            elif assessment_type == 'tapping':
                patient_profile.baseline_tapping_speed = measured_value
            elif assessment_type == 'speech':
                patient_profile.baseline_speech_rate = measured_value
                patient_profile.target_speech_rate = measured_value * 1.15  # 15% improvement target
            elif assessment_type == 'balance':
                # Store balance score in notes for now, can add dedicated column later
                assessment.notes = f"Berg Balance Scale Score: {measured_value}/56. " + (notes or "")
            elif assessment_type == 'coordination':
                # Store coordination time in notes for now
                assessment.notes = f"Finger-to-Nose Time: {measured_value}s per repetition. " + (notes or "")
            elif assessment_type == 'cognitive':
                # Store cognitive score in notes for now
                assessment.notes = f"MoCA Score: {measured_value}/30. " + (notes or "")

            db.session.commit()
            flash('Baseline assessment recorded successfully!', 'success')

            return redirect(url_for('clinician_dashboard'))

        except Exception as e:
            db.session.rollback()
            logging.error(f"Error recording assessment: {str(e)}")
            flash('Failed to record assessment. Please try again.', 'error')

    # For GET request or form errors - only clinicians can access
    patients = PatientProfile.query.filter_by(assigned_clinician_id=user.id).all()
    return render_template('baseline_assessment.html', patients=patients, user=user)

@app.route('/progress/<int:patient_id>')
def progress_view(patient_id):
    """Progress visualization page"""
    if 'user_id' not in session:
        flash('Please log in to access this page.', 'error')
        return redirect(url_for('index'))

    user = User.query.get(session['user_id'])
    patient_profile = PatientProfile.query.get_or_404(patient_id)

    # Check authorization
    if user.user_type == 'patient' and patient_profile.user_id != user.id:
        flash('Unauthorized access.', 'error')
        return redirect(url_for('patient_dashboard'))
    elif user.user_type == 'clinician' and patient_profile.assigned_clinician_id != user.id:
        flash('Unauthorized access.', 'error')
        return redirect(url_for('clinician_dashboard'))

    # Get session data for charts
    sessions = TherapySession.query.filter_by(
        patient_id=patient_id,
        completed=True
    ).order_by(TherapySession.start_time.asc()).all()

    return render_template('progress.html', 
                         patient_profile=patient_profile,
                         sessions=sessions,
                         user=user)

@app.route('/api/progress/<int:patient_id>')
def progress_data(patient_id):
    """API endpoint for progress chart data"""
    from flask import session as flask_session
    
    if 'user_id' not in flask_session:
        return jsonify({'error': 'Unauthorized'}), 401

    user = User.query.get(flask_session['user_id'])
    patient_profile = PatientProfile.query.get_or_404(patient_id)

    # Check authorization
    if user.user_type == 'patient' and patient_profile.user_id != user.id:
        return jsonify({'error': 'Unauthorized'}), 401
    elif user.user_type == 'clinician' and patient_profile.assigned_clinician_id != user.id:
        return jsonify({'error': 'Unauthorized'}), 401

    # Get therapy_sessions for the last 30 days
    thirty_days_ago = datetime.utcnow() - timedelta(days=30)
    therapy_sessions = TherapySession.query.filter(
        TherapySession.patient_id == patient_id,
        TherapySession.completed == True,
        TherapySession.start_time >= thirty_days_ago
    ).order_by(TherapySession.start_time.asc()).all()

    # Prepare chart data
    dates = []
    accuracy_scores = []
    bpm_values = []

    for therapy_session in therapy_sessions:
        dates.append(therapy_session.start_time.strftime('%Y-%m-%d'))
        accuracy_scores.append(therapy_session.accuracy_score or 0)
        bpm_values.append(therapy_session.final_bpm or therapy_session.initial_bpm)

    return jsonify({
        'dates': dates,
        'accuracy_scores': accuracy_scores,
        'bpm_values': bpm_values,
        'baseline_cadence': patient_profile.baseline_cadence,
        'target_cadence': patient_profile.target_cadence
    })

@app.route('/create_patient', methods=['POST'])
def create_patient():
    """Create a new patient account by clinician"""
    if 'user_id' not in session or session.get('user_type') != 'clinician':
        flash('Please log in as a clinician to access this page.', 'error')
        return redirect(url_for('index'))

    try:
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        first_name = request.form['first_name']
        last_name = request.form['last_name']
        condition = request.form['condition']

        # Check if user already exists
        if User.query.filter_by(username=username).first():
            flash('Username already exists', 'error')
            return redirect(url_for('clinician_dashboard'))

        if User.query.filter_by(email=email).first():
            flash('Email already exists', 'error')
            return redirect(url_for('clinician_dashboard'))

        # Create new patient user
        user = User(
            username=username,
            email=email,
            user_type='patient',
            first_name=first_name,
            last_name=last_name
        )
        user.set_password(password)
        db.session.add(user)
        db.session.flush()  # Get the user ID

        # Create patient profile with stroke-specific fields
        patient_profile = PatientProfile(
            user_id=user.id,
            condition=condition,
            assigned_clinician_id=session['user_id']
        )

        # Add stroke-specific fields if condition is stroke
        if condition == 'stroke':
            patient_profile.stroke_affected_side = request.form.get('stroke_affected_side')
            patient_profile.stroke_severity = request.form.get('stroke_severity')
            patient_profile.aphasia_type = request.form.get('aphasia_type')
            patient_profile.dysarthria_severity = request.form.get('dysarthria_severity')
            patient_profile.motor_impairment_level = request.form.get('motor_impairment_level')
            patient_profile.cognitive_status = request.form.get('cognitive_status')
            patient_profile.emotional_status = request.form.get('emotional_status')
            patient_profile.preferred_music_genre = request.form.get('preferred_music_genre')
            patient_profile.preferred_beat_sound = request.form.get('preferred_beat_sound', 'metronome')
        
        # Set initial baseline values from clinician input or defaults if not provided
        # This part needs to be integrated with the UI to allow clinicians to set these.
        # For now, assuming these might be set directly if available in the form or defaults.
        # The core change is that the patient cannot set these directly in a patient-initiated flow.
        if condition == 'stroke': # Example: setting baseline for stroke patients
            patient_profile.baseline_cadence = float(request.form.get('baseline_cadence', 120)) # Default cadence
            patient_profile.target_cadence = float(request.form.get('target_cadence', patient_profile.baseline_cadence * 1.1)) # Default target
            patient_profile.baseline_tapping_speed = float(request.form.get('baseline_tapping_speed', 5)) # Default tapping speed
            patient_profile.baseline_speech_rate = float(request.form.get('baseline_speech_rate', 150)) # Default speech rate
            patient_profile.target_speech_rate = float(request.form.get('target_speech_rate', patient_profile.baseline_speech_rate * 1.15)) # Default target speech rate
            # Add other baseline fields as needed, ensuring they are set by the clinician during patient creation or later.


        db.session.add(patient_profile)
        db.session.commit()
        flash(f'Patient {first_name} {last_name} created successfully and assigned to you!', 'success')

    except Exception as e:
        db.session.rollback()
        logging.error(f"Error creating patient: {str(e)}")
        flash('Failed to create patient. Please try again.', 'error')

    return redirect(url_for('clinician_dashboard'))

@app.route('/assign_patient', methods=['POST'])
def assign_patient():
    """Assign patient to clinician"""
    if 'user_id' not in session or session.get('user_type') != 'clinician':
        return jsonify({'error': 'Unauthorized'}), 401

    try:
        patient_id = int(request.json.get('patient_id'))
        clinician_id = session['user_id']

        patient_profile = PatientProfile.query.get_or_404(patient_id)
        patient_profile.assigned_clinician_id = clinician_id

        db.session.commit()

        return jsonify({'success': True})

    except Exception as e:
        logging.error(f"Error assigning patient: {str(e)}")
        return jsonify({'error': 'Failed to assign patient'}), 500

@app.route('/patient/<int:patient_id>/details')
def patient_details(patient_id):
    """Detailed patient view for clinicians"""
    if 'user_id' not in session or session.get('user_type') != 'clinician':
        flash('Please log in as a clinician to access this page.', 'error')
        return redirect(url_for('index'))

    user = User.query.get(session['user_id'])
    patient_profile = PatientProfile.query.get_or_404(patient_id)

    # Check if patient is assigned to this clinician
    if patient_profile.assigned_clinician_id != user.id:
        flash('Unauthorized access to patient details.', 'error')
        return redirect(url_for('clinician_dashboard'))

    # Get patient's recent sessions
    recent_sessions = TherapySession.query.filter_by(
        patient_id=patient_id
    ).order_by(TherapySession.start_time.desc()).limit(10).all()

    # Get baseline assessments
    assessments = BaselineAssessment.query.filter_by(
        patient_id=patient_id
    ).order_by(BaselineAssessment.created_at.desc()).all()

    # Calculate progress statistics
    total_sessions = TherapySession.query.filter_by(
        patient_id=patient_id, 
        completed=True
    ).count()

    avg_accuracy = db.session.query(db.func.avg(TherapySession.accuracy_score)).filter_by(
        patient_id=patient_id,
        completed=True
    ).scalar() or 0

    return render_template('patient_details.html',
                         patient_profile=patient_profile,
                         recent_sessions=recent_sessions,
                         assessments=assessments,
                         total_sessions=total_sessions,
                         avg_accuracy=round(avg_accuracy, 1),
                         user=user)

@app.route('/patient/<int:patient_id>/credentials')
def patient_credentials(patient_id):
    """View patient login credentials"""
    if 'user_id' not in session or session.get('user_type') != 'clinician':
        flash('Please log in as a clinician to access this page.', 'error')
        return redirect(url_for('index'))

    user = User.query.get(session['user_id'])
    patient_profile = PatientProfile.query.get_or_404(patient_id)

    # Check if patient is assigned to this clinician
    if patient_profile.assigned_clinician_id != user.id:
        flash('Unauthorized access to patient credentials.', 'error')
        return redirect(url_for('clinician_dashboard'))

    patient_user = patient_profile.user
    return render_template('patient_credentials.html',
                         patient_profile=patient_profile,
                         patient_user=patient_user,
                         user=user)


# ==============================================================================
# HUGGING FACE INFERENCE ROUTER & DUAL-ENGINE AUDIO GENERATION API
# ==============================================================================

@app.route('/api/hf/status', methods=['GET'])
def hf_status():
    """Check Hugging Face router connection & authentication status"""
    from beat_generator import BeatGenerator
    bg = BeatGenerator()
    status_info = bg.check_connection()
    return jsonify({
        'success': True,
        'status': status_info
    })


@app.route('/api/hf/token', methods=['POST'])
def hf_update_token():
    """Test and validate Hugging Face API token"""
    data = request.get_json(silent=True) or {}
    token = data.get('token', '').strip()
    from beat_generator import BeatGenerator
    bg = BeatGenerator(api_token=token if token else None)
    status_info = bg.check_connection()
    return jsonify({
        'success': status_info.get('authenticated', False),
        'status': status_info
    })


@app.route('/api/beat/generate_ai', methods=['POST'])
def generate_ai_beat():
    """Generate audio rhythm track using Hugging Face router with studio acoustic fallback"""
    data = request.get_json(silent=True) or {}
    bpm = float(data.get('bpm', 100))
    prompt = data.get('prompt')
    session_type = data.get('session_type', 'rhythmic_walking')
    duration = int(data.get('duration', 10))

    from beat_generator import BeatGenerator
    bg = BeatGenerator()
    result = bg.generate_beat_detailed(
        bpm=bpm,
        prompt=prompt,
        session_type=session_type,
        duration=duration
    )
    return jsonify({
        'success': result.get('success', False),
        'audio_url': result.get('audio_url'),
        'engine_used': result.get('engine_used'),
        'bpm': result.get('bpm'),
        'details': result
    })


# ==============================================================================
# LONGITUDINAL CLINICAL INTELLIGENCE & STRUCTURED REPORTING API
# ==============================================================================

@app.route('/api/patient/<int:patient_id>/historical-trends', methods=['GET'])
def patient_historical_trends(patient_id):
    """Retrieve 9-feature longitudinal intelligence trends for a patient"""
    if 'user_id' in session:
        user = db.session.get(User, session['user_id']) if hasattr(db.session, 'get') else User.query.get(session['user_id'])
        patient_profile = db.session.get(PatientProfile, patient_id) if hasattr(db.session, 'get') else PatientProfile.query.get(patient_id)
        if patient_profile and not _can_access_patient(user, patient_profile):
            return jsonify({'success': False, 'error': 'Unauthorized'}), 403

    activity_type = request.args.get('activity_type')
    from services.historical_analysis import (
        get_patient_history,
        compute_accuracy_trend,
        calculate_advisory_bpm,
        calculate_accuracy_delta
    )
    history = get_patient_history(patient_id, activity_type, limit=20)
    trend_info = compute_accuracy_trend(history)
    latest_acc = history[0]['accuracy_score'] if history else 0.0
    latest_bpm = history[0]['final_bpm'] if history else 60.0
    advisory_bpm = calculate_advisory_bpm(latest_bpm, latest_acc, trend_info)
    delta_info = calculate_accuracy_delta(latest_acc, history)

    return jsonify({
        'success': True,
        'patient_id': patient_id,
        'activity_type': activity_type,
        'history': history,
        'trend': trend_info,
        'advisory_bpm': advisory_bpm,
        'delta': delta_info
    })


@app.route('/api/session/<int:session_id>/clinical-report', methods=['GET'])
def get_session_clinical_report(session_id):
    """Retrieve structured clinical report and SOAP documentation for a session"""
    therapy_session = db.session.get(TherapySession, session_id) if hasattr(db.session, 'get') else TherapySession.query.get(session_id)
    if not therapy_session:
        return jsonify({'success': False, 'error': 'Session not found'}), 404

    if 'user_id' in session:
        user = db.session.get(User, session['user_id']) if hasattr(db.session, 'get') else User.query.get(session['user_id'])
        if user and not _can_access_session(user, therapy_session):
            return jsonify({'success': False, 'error': 'Unauthorized'}), 403

    report = ClinicalReport.query.filter_by(session_id=session_id).first()
    if not report:
        from services.historical_analysis import format_historical_context_for_prompt, save_or_update_clinical_report
        from services.gemini_service import generate_structured_clinical_report

        hist_ctx = format_historical_context_for_prompt(therapy_session.patient_id, therapy_session.session_type)
        clinical_data = {
            'activity_type': therapy_session.session_type,
            'duration_seconds': therapy_session.duration_seconds or 0,
            'initial_bpm': therapy_session.initial_bpm or 60.0,
            'final_bpm': therapy_session.final_bpm or therapy_session.initial_bpm or 60.0,
            'accuracy_score': therapy_session.accuracy_score or 0.0,
            'movement_count': therapy_session.total_steps or 0
        }
        report_json = generate_structured_clinical_report(clinical_data, hist_ctx)
        report = save_or_update_clinical_report(session_id, report_json)

    return jsonify({
        'success': True,
        'session_id': session_id,
        'clinical_report': report.to_dict() if report else None
    })