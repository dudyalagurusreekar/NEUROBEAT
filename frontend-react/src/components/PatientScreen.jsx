import React, { useState, useEffect, useRef } from 'react';
import { 
  Play, Pause, RotateCcw, Volume2, VolumeX, Mic, MicOff, 
  Camera, Activity, ShieldCheck, Heart, Sparkles, AlertCircle,
  Coffee, Compass, CheckCircle2, Sliders, Music, Info, Video,
  CheckCircle, ArrowRight, Zap, RefreshCw, Award
} from 'lucide-react';
import { soundEngine } from '../services/soundEngine';
import { NuroMotion, NuroSync, AdaptationEngine } from '../services/nuroMotion';
import { sessionsAPI, aiAPI } from '../services/api';
import { NuroMotionPanel } from '../features/neuromotion';

export default function PatientScreen({ clinicianSettings }) {
  // Therapy mode
  const [selectedMode] = useState('gait');
  
  // Playback & tempo state
  const [isPlaying, setIsPlaying] = useState(false);
  const [bpm, setBpm] = useState(52);
  const [soundType, setSoundType] = useState('bell');
  
  // Session tracking
  const [sessionId, setSessionId] = useState(null);
  const [duration, setDuration] = useState(0);
  const [syncAccuracy, setSyncAccuracy] = useState(null);
  const [timingErrorMs, setTimingErrorMs] = useState(null);
  const [totalSteps, setTotalSteps] = useState(0);
  const [_lastStepSide, setLastStepSide] = useState('LEFT');
  const [freezingCount, setFreezingCount] = useState(0);
  const [voiceCue] = useState('Nice and steady — match your steps to the gentle pulse.');
  
  // Adaptation Alert Banner
  const [adaptationNotice, setAdaptationNotice] = useState(null);

  // Sensing source: 'camera' vs 'demo'
  const [sensingSource, setSensingSource] = useState('demo'); // 'demo' or 'camera'
  
  // Calming 30-Second Rest Modal
  const [isResting, setIsResting] = useState(false);
  const [restCountdown, setRestCountdown] = useState(30);
  
  // AI Session Summary Modal
  const [showSummaryModal, setShowSummaryModal] = useState(false);
  const [isGeneratingSummary, setIsGeneratingSummary] = useState(false);
  const [aiSummary, setAiSummary] = useState(null);

  // Visual pulse state
  const [pulseActive, setPulseActive] = useState(false);

  // Refs for audio, engines, and video
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const sessionIdRef = useRef(null);
  const timerRef = useRef(null);
  const durationTimerRef = useRef(null);
  const beatCountRef = useRef(0);
  const restAmbientRef = useRef(null);
  const nuroMotionRef = useRef(null);
  const nuroSyncRef = useRef(null);
  const adaptationEngineRef = useRef(null);
  const syncHistoryRef = useRef([]);

  // Real-time Pose & Telemetry Tracking States
  const [landmarksData, setLandmarksData] = useState(null);
  const [trackingState, setTrackingState] = useState('LOST');
  const [poseConfidence, setPoseConfidence] = useState(0);
  const [cadenceSpm, setCadenceSpm] = useState('--');
  const [balanceScore, setBalanceScore] = useState(100);
  const [qualityScore, setQualityScore] = useState(85);
  const [lastStepEvent, setLastStepEvent] = useState(null);
  const [audioState, setAudioState] = useState({ level: 0, activity: false });
  const [motionError, setMotionError] = useState(null);
  const [framing, setFraming] = useState(null);
  const [diagnostics, setDiagnostics] = useState({
    modelName: 'MediaPipe Pose Landmarker Full',
    cameraFps: 0,
    poseFps: 0,
    latencyMs: 0,
    droppedFrames: 0,
  });

  // Sync sessionIdRef
  useEffect(() => {
    sessionIdRef.current = sessionId;
  }, [sessionId]);

  // Initialize Engines
  useEffect(() => {
    nuroSyncRef.current = new NuroSync(null, bpm);
    adaptationEngineRef.current = new AdaptationEngine(
      clinicianSettings?.minBpm || 45,
      clinicianSettings?.maxBpm || 72
    );

    nuroMotionRef.current = new NuroMotion((stepEvent) => {
      handleStepDetected(stepEvent);
    });

    if (nuroMotionRef.current.core) {
      nuroMotionRef.current.core.onPoseUpdate((data) => {
        setLandmarksData(data);
        setTrackingState(data.quality.state);
        setPoseConfidence(data.quality.confidence);
        if (data.quality?.framing) {
          setFraming(data.quality.framing);
        }
      });

      nuroMotionRef.current.core.onMetrics((data) => {
        setCadenceSpm(data.cadence.displaySpm);
        setBalanceScore(data.balance.balanceScore);
        setQualityScore(data.quality.qualityScore);
        if (data.sync && typeof data.sync.syncScore === 'number') {
          setSyncAccuracy(data.sync.syncScore);
          setTimingErrorMs(data.sync.timingErrorMs ?? 0);
        }
        setAudioState({
          level: data.audio?.level ?? 0,
          activity: data.audio?.activity ?? false,
        });
        setDiagnostics(data.diagnostics);
      });

      nuroMotionRef.current.core.onTelemetry((packet) => {
        if (sessionIdRef.current) {
          sessionsAPI.pushTelemetry(sessionIdRef.current, packet).catch(() => {});
        }
      });

      nuroMotionRef.current.core.onError((err) => {
        setMotionError(err?.message || 'Camera or model error');
      });
    }

    return () => {
      if (nuroMotionRef.current) nuroMotionRef.current.stop();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Update Adaptation Engine Safety Bounds
  useEffect(() => {
    if (adaptationEngineRef.current) {
      adaptationEngineRef.current.updateSafetyLimits(
        clinicianSettings?.minBpm || 45,
        clinicianSettings?.maxBpm || 72
      );
    }
  }, [clinicianSettings]);

  // Handle incoming step from NuroMotion
  const handleStepDetected = (stepEvent) => {
    setTotalSteps(prev => prev + 1);
    setLastStepSide(stepEvent.side);
    setLastStepEvent(stepEvent);

    // Evaluate sync with NuroSync
    if (nuroSyncRef.current) {
      const evaluation = nuroSyncRef.current.evaluateStep(stepEvent.timestamp);
      setTimingErrorMs(evaluation.timingErrorMs);
      setSyncAccuracy(evaluation.syncScore);
      syncHistoryRef.current.push(evaluation.syncScore);
      if (syncHistoryRef.current.length > 20) syncHistoryRef.current.shift();

      // Check adaptation
      if (adaptationEngineRef.current && isPlaying) {
        const result = adaptationEngineRef.current.evaluateAdaptation(bpm, syncHistoryRef.current);
        if (result.adapted) {
          setBpm(result.newBpm);
          setAdaptationNotice(result.reason);
          setTimeout(() => setAdaptationNotice(null), 6000);

          if (result.isFreezing) {
            setFreezingCount(prev => prev + 1);
            soundEngine.speakCue("Freezing detected. Easing rhythm for safety.");
          } else if (result.newBpm > bpm) {
            soundEngine.speakCue("Cadence steady. Gently lifting tempo.");
          } else {
            soundEngine.speakCue("Easing pace for joint protection.");
          }
        }
      }
    }
  };

  // Switch Sensing Source (Live Camera vs Demo Mode)
  const toggleSensingSource = (mode) => {
    setSensingSource(mode);
    if (!isPlaying) return;

    if (mode === 'camera') {
      if (videoRef.current) {
        nuroMotionRef.current.startCamera(videoRef.current);
      }
    } else {
      nuroMotionRef.current.startDemoMode(bpm);
    }
  };

  // Main Rhythm Loop
  useEffect(() => {
    if (!isPlaying || isResting) {
      if (timerRef.current) clearInterval(timerRef.current);
      return;
    }

    const intervalMs = (60 / bpm) * 1000;

    timerRef.current = setInterval(() => {
      beatCountRef.current += 1;
      const isDownbeat = beatCountRef.current % 4 === 1;
      const nowSec = performance.now() / 1000;

      // Record beat in NuroSync and NuroMotion
      if (nuroSyncRef.current) {
        nuroSyncRef.current.recordBeat(nowSec);
      }
      if (nuroMotionRef.current) {
        nuroMotionRef.current.registerBeat(nowSec);
      }

      // Play audio pulse
      soundEngine.playBeat(soundType, isDownbeat);

      // Visual pulse
      setPulseActive(true);
      setTimeout(() => setPulseActive(false), 240);
    }, intervalMs);

    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [isPlaying, bpm, soundType, isResting]);

  // Duration timer
  useEffect(() => {
    if (isPlaying && !isResting) {
      durationTimerRef.current = setInterval(() => {
        setDuration(prev => prev + 1);
      }, 1000);
    } else {
      if (durationTimerRef.current) clearInterval(durationTimerRef.current);
    }
    return () => {
      if (durationTimerRef.current) clearInterval(durationTimerRef.current);
    };
  }, [isPlaying, isResting]);

  // Synchronize target tempo across rhythm engines
  useEffect(() => {
    if (nuroSyncRef.current && typeof nuroSyncRef.current.setBpm === 'function') {
      nuroSyncRef.current.setBpm(bpm);
    }
    if (nuroMotionRef.current && typeof nuroMotionRef.current.setTargetBpm === 'function') {
      nuroMotionRef.current.setTargetBpm(bpm);
    }
  }, [bpm]);

  // Start Session handler
  const handleStartSession = async () => {
    setIsPlaying(true);
    soundEngine.speakCue("Beginning gait session. Follow the steady rhythm.");

    // Start motion tracker
    if (sensingSource === 'camera' && videoRef.current) {
      nuroMotionRef.current.startCamera(videoRef.current);
    } else {
      nuroMotionRef.current.startDemoMode(bpm);
    }

    // Register session in FastAPI backend
    try {
      const resp = await sessionsAPI.createSession({
        patient_id: 1,
        session_type: selectedMode,
        initial_bpm: bpm,
        target_bpm: clinicianSettings?.maxBpm || 54
      });
      setSessionId(resp.data.id);
    } catch (e) {
      console.warn("Backend session record skipped (offline mode):", e);
    }
  };

  // Pause Session
  const handlePauseSession = () => {
    setIsPlaying(false);
    if (nuroMotionRef.current) nuroMotionRef.current.stop();
  };

  // Complete Session & Trigger AI Summary
  const handleCompleteSession = async () => {
    setIsPlaying(false);
    if (nuroMotionRef.current) nuroMotionRef.current.stop();

    setShowSummaryModal(true);
    setIsGeneratingSummary(true);

    const avgScore = nuroSyncRef.current ? nuroSyncRef.current.getAverageSync() : syncAccuracy;

    // Call FastAPI GenAI endpoint
    try {
      const summaryResp = await aiAPI.generateSessionSummary({
        session_id: sessionId || 1,
        patient_name: "Arthur Pendelton",
        duration_seconds: duration,
        start_bpm: 50.0,
        final_bpm: bpm,
        avg_sync_score: avgScore,
        total_steps: totalSteps,
        freezing_events: freezingCount
      });
      setAiSummary(summaryResp.data);

      // Play audio cue
      soundEngine.speakCue(summaryResp.data.encouraging_cue || "Session complete. Excellent work!");

      // Finalize session in backend
      if (sessionId) {
        await sessionsAPI.completeSession(sessionId, {
          final_bpm: bpm,
          duration_seconds: duration,
          avg_sync_score: avgScore,
          total_steps: totalSteps,
          freezing_events_count: freezingCount
        });
      }
    } catch (err) {
      console.warn("AI summary fallback:", err);
      const fallbackSummary = {
        summary: `Arthur completed ${Math.floor(duration / 60)}m ${duration % 60}s of gait training. Cadence progressed from 50 to ${bpm} BPM with ${avgScore}% rhythmic entrainment.`,
        clinician_notes: `Patient demonstrated steady auditory-motor entrainment. ${freezingCount} freezing events observed.`,
        encouraging_cue: "Wonderful consistency! You are making measurable progress with every session."
      };
      setAiSummary(fallbackSummary);
      soundEngine.speakCue(fallbackSummary.encouraging_cue);
    } finally {
      setIsGeneratingSummary(false);
    }
  };

  // Breather modal logic
  const startRest = () => {
    handlePauseSession();
    setIsResting(true);
    setRestCountdown(30);
    restAmbientRef.current = soundEngine.playCalmAmbientWave();
    soundEngine.speakCue("Taking a gentle 30 second breather.");
  };

  const endRest = () => {
    if (restAmbientRef.current) restAmbientRef.current.stop();
    setIsResting(false);
    setRestCountdown(30);
    soundEngine.speakCue("Breather complete. Press resume whenever ready.");
  };

  const formatTime = (secs) => {
    const m = Math.floor(secs / 60);
    const s = secs % 60;
    return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
  };

  return (
    <div style={{ maxWidth: '860px', margin: '0 auto', padding: '16px 20px 60px' }}>
      
      {/* Patient Welcome Header */}
      <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        marginBottom: '20px',
        flexWrap: 'wrap',
        gap: '12px'
      }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '4px' }}>
            <span style={{
              display: 'inline-block',
              width: '10px',
              height: '10px',
              borderRadius: '50%',
              backgroundColor: '#3D9970',
              boxShadow: '0 0 10px rgba(61, 153, 112, 0.4)'
            }} />
            <span style={{ fontSize: '13px', fontWeight: 700, color: 'var(--primary)', letterSpacing: '0.04em', textTransform: 'uppercase' }}>
              Companion Active • Dr. Sharma's Safety Envelope: {clinicianSettings?.minBpm || 45}–{clinicianSettings?.maxBpm || 72} BPM
            </span>
          </div>
          <h1 style={{ fontSize: '26px', color: 'var(--text-main)', fontWeight: 700 }}>
            Welcome back, Arthur
          </h1>
          <p style={{ fontSize: '15px', color: 'var(--text-muted)' }}>
            Parkinson's Gait Training Protocol • Baseline: 44 SPM • Target: 54 SPM
          </p>
        </div>

        {/* Streak & Consistency Badge */}
        <div style={{
          backgroundColor: 'var(--coral-light)',
          border: '1px solid #F5D3C7',
          padding: '8px 16px',
          borderRadius: 'var(--radius-full)',
          display: 'flex',
          alignItems: 'center',
          gap: '8px'
        }}>
          <Sparkles size={18} color="var(--coral-accent)" />
          <span style={{ fontSize: '14px', fontWeight: 700, color: '#C86E53' }}>
            14-Day Consistency Streak
          </span>
        </div>
      </div>

      {/* ADAPTATION ALERT BANNER (Triggers visibly when BPM shifts) */}
      {adaptationNotice && (
        <div style={{
          backgroundColor: '#E8F8F5',
          border: '1.5px solid #2ECC71',
          borderRadius: 'var(--radius-md)',
          padding: '14px 20px',
          marginBottom: '20px',
          display: 'flex',
          alignItems: 'center',
          gap: '12px',
          boxShadow: '0 4px 14px rgba(46, 204, 113, 0.2)',
          animation: 'gentleFadeIn 0.3s ease'
        }}>
          <Zap size={22} color="#27AE60" />
          <div style={{ flex: 1 }}>
            <span style={{ fontSize: '12px', fontWeight: 800, textTransform: 'uppercase', color: '#27AE60', letterSpacing: '0.04em' }}>
              Real-Time Rhythm Adaptation
            </span>
            <p style={{ fontSize: '15px', fontWeight: 700, color: 'var(--text-main)', margin: 0 }}>
              {adaptationNotice}
            </p>
          </div>
        </div>
      )}

      {/* Real-Time NuroMotion Tracking Panel (MediaPipe Pose Landmarker Full) */}
      <NuroMotionPanel
        videoRef={videoRef}
        canvasRef={canvasRef}
        landmarksData={landmarksData}
        isRunning={isPlaying}
        isPaused={!isPlaying && duration > 0}
        isDemoMode={sensingSource === 'demo'}
        trackingState={trackingState}
        confidence={poseConfidence}
        framing={framing}
        metrics={{

          cadenceSpm,
          leftSteps: Math.round(totalSteps / 2),
          rightSteps: Math.floor(totalSteps / 2),
          totalSteps,
          balanceScore,
          qualityScore,
          syncScore: syncAccuracy,
          timingErrorMs,
        }}
        audioState={audioState}
        diagnostics={diagnostics}
        lastStep={lastStepEvent}
        errorMessage={motionError}
        onStartCamera={() => {
          toggleSensingSource('camera');
          if (!isPlaying) handleStartSession();
        }}
        onStartDemo={() => {
          toggleSensingSource('demo');
          if (!isPlaying) handleStartSession();
        }}
        onPause={handlePauseSession}
        onResume={handleStartSession}
        onStop={handleCompleteSession}
        onReset={() => {
          setTotalSteps(0);
          setDuration(0);
          if (nuroMotionRef.current && nuroMotionRef.current.core) {
            nuroMotionRef.current.core.reset();
          }
        }}
        targetBpm={bpm}
      />

      {/* MAIN RHYTHM HERO CARD */}
      <div style={{
        backgroundColor: 'var(--bg-surface)',
        borderRadius: 'var(--radius-lg)',
        padding: '36px 24px',
        boxShadow: 'var(--shadow-md)',
        border: '1px solid var(--border-soft)',
        textAlign: 'center',
        position: 'relative',
        overflow: 'hidden',
        marginBottom: '24px'
      }}>
        
        {/* Upper Live Telemetry Row */}
        <div style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          maxWidth: '540px',
          margin: '0 auto 28px',
          flexWrap: 'wrap',
          gap: '12px'
        }}>
          <div style={{ textAlign: 'left' }}>
            <span style={{ fontSize: '13px', color: 'var(--text-muted)', display: 'block' }}>Session Timer</span>
            <span style={{ fontSize: '22px', fontWeight: 800, color: 'var(--text-main)' }}>
              {formatTime(duration)}
            </span>
          </div>

          <div style={{
            display: 'inline-flex',
            alignItems: 'center',
            gap: '8px',
            backgroundColor: 'var(--bg-accent-soft)',
            padding: '8px 18px',
            borderRadius: 'var(--radius-full)',
            border: '1px solid rgba(74, 140, 140, 0.2)'
          }}>
            <CheckCircle2 size={18} color="var(--primary)" />
            <span style={{ fontSize: '15px', fontWeight: 800, color: 'var(--primary)' }}>
              {syncAccuracy !== null ? `${syncAccuracy}% Rhythm Sync` : 'Awaiting Steps'}
            </span>
          </div>

          <div style={{ textAlign: 'right' }}>
            <span style={{ fontSize: '13px', color: 'var(--text-muted)', display: 'block' }}>Timing Error</span>
            <span style={{ fontSize: '18px', fontWeight: 700, color: syncAccuracy !== null ? (timingErrorMs !== null && timingErrorMs < 45 ? '#27AE60' : 'var(--coral-accent)') : 'var(--text-muted)' }}>
              {syncAccuracy !== null && timingErrorMs !== null ? `±${timingErrorMs}ms` : '--'}
            </span>
          </div>
        </div>

        {/* CENTRAL CALM BREATHING / PULSE ORB */}
        <div style={{ position: 'relative', width: '220px', height: '220px', margin: '0 auto 28px' }}>
          
          {/* Outer Breathing Wave Halo */}
          <div style={{
            position: 'absolute',
            inset: 0,
            borderRadius: '50%',
            backgroundColor: isPlaying ? 'rgba(74, 140, 140, 0.15)' : 'rgba(74, 140, 140, 0.05)',
            transform: pulseActive ? 'scale(1.22)' : 'scale(1)',
            opacity: pulseActive ? 0.8 : 0.35,
            transition: 'all 0.22s cubic-bezier(0.16, 1, 0.3, 1)',
            pointerEvents: 'none'
          }} />

          {/* Secondary Soft Coral Ring for Warmth */}
          <div style={{
            position: 'absolute',
            inset: '14px',
            borderRadius: '50%',
            backgroundColor: pulseActive ? 'rgba(232, 150, 122, 0.2)' : 'rgba(232, 150, 122, 0.06)',
            transform: pulseActive ? 'scale(1.12)' : 'scale(1)',
            transition: 'all 0.22s cubic-bezier(0.16, 1, 0.3, 1)',
            pointerEvents: 'none'
          }} />

          {/* Main Interactive Orb */}
          <div 
            style={{
              position: 'absolute',
              inset: '28px',
              borderRadius: '50%',
              background: isPlaying 
                ? 'linear-gradient(135deg, #5B9A8B 0%, #4A8C8C 100%)' 
                : 'linear-gradient(135deg, #EEF3F0 0%, #E3ECE7 100%)',
              display: 'flex',
              flexDirection: 'column',
              justifyContent: 'center',
              alignItems: 'center',
              boxShadow: isPlaying ? '0 10px 30px rgba(74, 140, 140, 0.35)' : 'var(--shadow-sm)',
              transform: pulseActive ? 'scale(0.97)' : 'scale(1)',
              transition: 'transform 0.18s ease-out, background 0.3s ease',
              userSelect: 'none'
            }}
          >
            <span style={{
              fontSize: '46px',
              fontWeight: 800,
              lineHeight: 1,
              color: isPlaying ? '#FFFFFF' : 'var(--text-main)',
              letterSpacing: '-0.03em'
            }}>
              {bpm}
            </span>
            <span style={{
              fontSize: '13px',
              fontWeight: 600,
              color: isPlaying ? 'rgba(255, 255, 255, 0.9)' : 'var(--text-muted)',
              textTransform: 'uppercase',
              letterSpacing: '0.06em',
              marginTop: '4px'
            }}>
              BPM / Cadence
            </span>
            <span style={{
              fontSize: '11px',
              color: isPlaying ? 'rgba(255, 255, 255, 0.75)' : 'var(--text-light)',
              marginTop: '2px'
            }}>
              Target: {clinicianSettings?.maxBpm || 54} BPM
            </span>
          </div>
        </div>

        {/* Live Reassuring Voice Cue Banner */}
        <div style={{
          backgroundColor: 'var(--bg-subtle)',
          borderRadius: 'var(--radius-md)',
          padding: '12px 18px',
          maxWidth: '520px',
          margin: '0 auto 28px',
          display: 'flex',
          alignItems: 'center',
          gap: '12px',
          border: '1px solid var(--border-soft)'
        }}>
          <div style={{
            backgroundColor: '#FFFFFF',
            borderRadius: '50%',
            padding: '8px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            boxShadow: 'var(--shadow-sm)'
          }}>
            <Heart size={18} color="var(--coral-accent)" />
          </div>
          <div style={{ textAlign: 'left', flex: 1 }}>
            <span style={{ fontSize: '11px', textTransform: 'uppercase', fontWeight: 700, color: 'var(--text-light)', letterSpacing: '0.05em' }}>
              Companion Voice Cue
            </span>
            <p style={{ fontSize: '15px', fontWeight: 600, color: 'var(--text-main)', margin: 0 }}>
              "{voiceCue}"
            </p>
          </div>
        </div>

        {/* Primary Controls (Accessible Big Buttons) */}
        <div style={{
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
          gap: '16px',
          flexWrap: 'wrap'
        }}>
          
          {/* Breather / Take a Break Button */}
          <button
            onClick={startRest}
            style={{
              backgroundColor: 'var(--coral-light)',
              color: '#B55A3F',
              border: '1px solid #F5D3C7',
              height: '56px',
              padding: '0 20px',
              borderRadius: 'var(--radius-full)',
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              fontWeight: 700,
              fontSize: '15px',
              boxShadow: 'var(--shadow-sm)'
            }}
          >
            <Coffee size={20} color="var(--coral-accent)" />
            Need a Breather?
          </button>

          {/* Big Start / Pause Button */}
          <button
            onClick={() => {
              if (isPlaying) {
                handlePauseSession();
              } else {
                handleStartSession();
              }
            }}
            style={{
              backgroundColor: isPlaying ? 'var(--coral-accent)' : 'var(--primary)',
              color: '#FFFFFF',
              height: '62px',
              padding: '0 38px',
              borderRadius: 'var(--radius-full)',
              display: 'flex',
              alignItems: 'center',
              gap: '12px',
              fontWeight: 800,
              fontSize: '18px',
              boxShadow: isPlaying ? '0 8px 24px rgba(232, 150, 122, 0.4)' : '0 8px 24px rgba(74, 140, 140, 0.35)'
            }}
          >
            {isPlaying ? (
              <>
                <Pause size={24} fill="#FFFFFF" />
                Pause Session
              </>
            ) : (
              <>
                <Play size={24} fill="#FFFFFF" />
                Begin Session
              </>
            )}
          </button>

          {/* Complete Session Button */}
          <button
            onClick={handleCompleteSession}
            style={{
              backgroundColor: '#3D9970',
              color: '#FFFFFF',
              height: '56px',
              padding: '0 22px',
              borderRadius: 'var(--radius-full)',
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              fontWeight: 700,
              fontSize: '15px',
              boxShadow: '0 4px 14px rgba(61, 153, 112, 0.3)'
            }}
          >
            <CheckCircle size={20} />
            Complete & Summary
          </button>
        </div>

        {/* Timbre sound selections */}
        <div style={{
          marginTop: '28px',
          paddingTop: '20px',
          borderTop: '1px solid var(--border-soft)',
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
          gap: '12px',
          flexWrap: 'wrap'
        }}>
          <span style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: '6px' }}>
            <Music size={16} /> Beat Timbre:
          </span>
          {[
            { id: 'bell', label: 'Healing Bell' },
            { id: 'piano', label: 'Warm Piano' },
            { id: 'drum', label: 'Soft Drum' },
            { id: 'wood', label: 'Forest Wood' },
            { id: 'metronome', label: 'Gentle Click' }
          ].map((inst) => (
            <button
              key={inst.id}
              onClick={() => {
                setSoundType(inst.id);
                soundEngine.playBeat(inst.id, true);
              }}
              style={{
                backgroundColor: soundType === inst.id ? 'var(--primary-light)' : 'transparent',
                color: soundType === inst.id ? 'var(--primary)' : 'var(--text-muted)',
                border: soundType === inst.id ? '1px solid var(--primary)' : '1px solid transparent',
                borderRadius: 'var(--radius-full)',
                padding: '6px 14px',
                fontSize: '13px',
                fontWeight: 600
              }}
            >
              {inst.label}
            </button>
          ))}
        </div>
      </div>

      {/* 30-SECOND CALM REST MODAL */}
      {isResting && (
        <div style={{
          position: 'fixed',
          inset: 0,
          backgroundColor: 'rgba(45, 52, 54, 0.75)',
          backdropFilter: 'blur(8px)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          zIndex: 9999,
          padding: '20px'
        }}>
          <div style={{
            backgroundColor: '#FFFFFF',
            borderRadius: 'var(--radius-lg)',
            padding: '40px 32px',
            maxWidth: '460px',
            width: '100%',
            textAlign: 'center',
            boxShadow: 'var(--shadow-lg)'
          }}>
            <div style={{
              width: '64px',
              height: '64px',
              borderRadius: '50%',
              backgroundColor: 'var(--coral-light)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              margin: '0 auto 20px'
            }}>
              <Coffee size={32} color="var(--coral-accent)" />
            </div>

            <h3 style={{ fontSize: '24px', fontWeight: 700, color: 'var(--text-main)', marginBottom: '8px' }}>
              Gentle Rest Break
            </h3>
            <p style={{ fontSize: '15px', color: 'var(--text-muted)', marginBottom: '24px' }}>
              Resting your muscles and soothing the nervous system with calming 432Hz ambient sound.
            </p>

            <div style={{ fontSize: '48px', fontWeight: 800, color: 'var(--coral-accent)', marginBottom: '28px' }}>
              {restCountdown}s
            </div>

            <button
              onClick={endRest}
              style={{
                backgroundColor: 'var(--primary)',
                color: '#FFFFFF',
                width: '100%',
                height: '52px',
                borderRadius: 'var(--radius-full)',
                fontWeight: 700,
                fontSize: '16px'
              }}
            >
              I'm Ready to Resume
            </button>
          </div>
        </div>
      )}

      {/* AI SESSION SUMMARY MODAL (GenAI Bounded Integration) */}
      {showSummaryModal && (
        <div style={{
          position: 'fixed',
          inset: 0,
          backgroundColor: 'rgba(45, 52, 54, 0.75)',
          backdropFilter: 'blur(8px)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          zIndex: 9999,
          padding: '20px'
        }}>
          <div style={{
            backgroundColor: '#FFFFFF',
            borderRadius: 'var(--radius-lg)',
            padding: '36px 32px',
            maxWidth: '560px',
            width: '100%',
            boxShadow: 'var(--shadow-lg)',
            maxHeight: '90vh',
            overflowY: 'auto'
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '16px' }}>
              <div style={{
                width: '40px',
                height: '40px',
                borderRadius: '10px',
                backgroundColor: 'var(--primary-light)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center'
              }}>
                <Award size={22} color="var(--primary)" />
              </div>
              <div>
                <h3 style={{ fontSize: '20px', fontWeight: 700, color: 'var(--text-main)', margin: 0 }}>
                  Session Summary & Insights
                </h3>
                <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
                  Generated from Measured Telemetry • Google Gemini AI
                </span>
              </div>
            </div>

            {isGeneratingSummary ? (
              <div style={{ textAlign: 'center', padding: '40px 0' }}>
                <RefreshCw size={36} color="var(--primary)" style={{ animation: 'spin 1.5s linear infinite' }} />
                <p style={{ marginTop: '16px', color: 'var(--text-muted)', fontSize: '15px' }}>
                  Synthesizing measured gait telemetry into plain language...
                </p>
              </div>
            ) : aiSummary ? (
              <div>
                {/* Metrics Highlight Card */}
                <div style={{
                  display: 'grid',
                  gridTemplateColumns: 'repeat(3, 1fr)',
                  gap: '10px',
                  backgroundColor: 'var(--bg-subtle)',
                  borderRadius: 'var(--radius-md)',
                  padding: '16px',
                  textAlign: 'center',
                  marginBottom: '20px'
                }}>
                  <div>
                    <span style={{ fontSize: '11px', color: 'var(--text-muted)', display: 'block' }}>Duration</span>
                    <strong style={{ fontSize: '16px', color: 'var(--text-main)' }}>{formatTime(duration)}</strong>
                  </div>
                  <div>
                    <span style={{ fontSize: '11px', color: 'var(--text-muted)', display: 'block' }}>Avg Sync</span>
                    <strong style={{ fontSize: '16px', color: 'var(--primary)' }}>{syncAccuracy !== null ? `${syncAccuracy}%` : 'Not Measured'}</strong>
                  </div>
                  <div>
                    <span style={{ fontSize: '11px', color: 'var(--text-muted)', display: 'block' }}>Total Steps</span>
                    <strong style={{ fontSize: '16px', color: 'var(--text-main)' }}>{totalSteps}</strong>
                  </div>
                </div>

                {/* Patient Summary */}
                <div style={{ marginBottom: '16px' }}>
                  <h4 style={{ fontSize: '14px', fontWeight: 700, color: 'var(--text-main)', marginBottom: '6px' }}>
                    Patient Overview:
                  </h4>
                  <p style={{ fontSize: '14px', color: 'var(--text-muted)', lineHeight: 1.6, margin: 0 }}>
                    {aiSummary.summary}
                  </p>
                </div>

                {/* Clinician Notes */}
                <div style={{
                  backgroundColor: 'var(--bg-accent-soft)',
                  borderRadius: 'var(--radius-sm)',
                  padding: '14px',
                  marginBottom: '20px',
                  border: '1px solid rgba(74, 140, 140, 0.2)'
                }}>
                  <span style={{ fontSize: '11px', fontWeight: 800, textTransform: 'uppercase', color: 'var(--primary)', letterSpacing: '0.04em' }}>
                    Physician Tele-Rehab Note
                  </span>
                  <p style={{ fontSize: '13px', color: 'var(--text-main)', marginTop: '4px', margin: 0 }}>
                    {aiSummary.clinician_notes}
                  </p>
                </div>

                <button
                  onClick={() => setShowSummaryModal(false)}
                  style={{
                    backgroundColor: 'var(--primary)',
                    color: '#FFFFFF',
                    width: '100%',
                    height: '48px',
                    borderRadius: 'var(--radius-full)',
                    fontWeight: 700,
                    fontSize: '15px'
                  }}
                >
                  Done
                </button>
              </div>
            ) : null}
          </div>
        </div>
      )}

    </div>
  );
}
