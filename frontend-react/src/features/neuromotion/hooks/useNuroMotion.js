/**
 * React Hook for NuroMotion Sensing Pipeline
 * Provides throttled UI state, high-frequency canvas drawing, and lifecycle management.
 */

import { useState, useEffect, useRef, useCallback } from 'react';
import { NuroMotion } from '../core/NuroMotion';

export function useNuroMotion(options = {}) {
  const [isRunning, setIsRunning] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const [isDemoMode, setIsDemoMode] = useState(false);
  const [trackingState, setTrackingState] = useState('LOST');
  const [confidence, setConfidence] = useState(0);
  const [framing, setFraming] = useState(null);


  // Cadence & Movement Metrics (throttled for UI)
  const [metrics, setMetrics] = useState({
    cadenceSpm: '--',
    leftSteps: 0,
    rightSteps: 0,
    totalSteps: 0,
    balanceScore: 100,
    qualityScore: 85,
    syncScore: null,
    timingErrorMs: null,
    isSynchronized: false,
    syncStatus: 'NO_DATA',
  });

  // Audio metrics
  const [audioState, setAudioState] = useState({
    level: 0,
    activity: false,
  });

  // Performance diagnostics
  const [diagnostics, setDiagnostics] = useState({
    modelName: 'MediaPipe Pose Landmarker Full',
    cameraFps: 0,
    poseFps: 0,
    latencyMs: 0,
    droppedFrames: 0,
  });

  const [lastStep, setLastStep] = useState(null);
  const [errorMessage, setErrorMessage] = useState(null);

  // References
  const nuroMotionRef = useRef(null);
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const latestLandmarksRef = useRef(null);

  // Initialize engine once
  if (!nuroMotionRef.current) {
    nuroMotionRef.current = new NuroMotion(options);
  }

  const optionsRef = useRef(options);
  useEffect(() => {
    optionsRef.current = options;
  }, [options]);

  useEffect(() => {
    const engine = nuroMotionRef.current;

    // Movement event listener
    const unsubStep = engine.onMovementEvent((ev) => {
      setLastStep(ev);
      if (ev.syncScore !== undefined) {
        setMetrics((prev) => ({
          ...prev,
          syncScore: ev.syncScore,
          timingErrorMs: ev.timingErrorMs ?? prev.timingErrorMs,
          isSynchronized: Boolean(ev.timingErrorMs !== undefined ? ev.timingErrorMs <= 200 : prev.isSynchronized),
        }));
      }
      if (optionsRef.current?.onStep) {
        optionsRef.current.onStep(ev);
      }
    });

    // Throttled high-level metrics listener
    const unsubMetrics = engine.onMetrics((data) => {
      setMetrics((prev) => ({
        ...prev,
        cadenceSpm: data.cadence.displaySpm,
        leftSteps: data.cadence.leftSteps,
        rightSteps: data.cadence.rightSteps,
        totalSteps: data.cadence.totalSteps,
        balanceScore: data.balance.balanceScore,
        qualityScore: data.quality.qualityScore,
        syncScore: data.sync ? data.sync.syncScore : prev.syncScore,
        timingErrorMs: data.sync ? data.sync.timingErrorMs : prev.timingErrorMs,
      }));

      setAudioState({
        level: data.audio?.level ?? 0,
        activity: data.audio?.activity ?? false,
      });

      setDiagnostics(data.diagnostics);
      setTrackingState(data.diagnostics.trackingState);
    });

    // Pose update listener for direct canvas rendering
    const unsubPose = engine.onPoseUpdate((poseData) => {
      latestLandmarksRef.current = poseData;
      setConfidence(poseData.quality.confidence);
      setTrackingState(poseData.quality.state);
      if (poseData.quality?.framing) {
        setFraming(poseData.quality.framing);
      }

      if (optionsRef.current?.onPoseUpdate) {
        optionsRef.current.onPoseUpdate(poseData);
      }
    });


    // Telemetry stream listener
    const unsubTelemetry = engine.onTelemetry((packet) => {
      if (optionsRef.current?.onTelemetry) {
        optionsRef.current.onTelemetry(packet);
      }
    });

    // Error listener
    const unsubError = engine.onError((err) => {
      setErrorMessage(err?.message || 'Camera or model error');
    });

    return () => {
      unsubStep();
      unsubMetrics();
      unsubPose();
      unsubTelemetry();
      unsubError();
      engine.stop();
    };
  }, []);

  const startCamera = useCallback(async (targetBpm = 60, sessionId = null) => {
    setErrorMessage(null);
    if (!videoRef.current) return false;

    const success = await nuroMotionRef.current.start({
      videoElement: videoRef.current,
      targetBpm,
      sessionId,
      useMic: true,
    });

    setIsRunning(true);
    setIsPaused(false);
    setIsDemoMode(!success);
    return success;
  }, []);

  const startDemo = useCallback((targetBpm = 60, sessionId = null) => {
    setErrorMessage(null);
    nuroMotionRef.current.startDemo(targetBpm, sessionId);
    setIsRunning(true);
    setIsPaused(false);
    setIsDemoMode(true);
    setTrackingState('TRACKING');
    setConfidence(0.95);
  }, []);

  const pause = useCallback(() => {
    nuroMotionRef.current.pause();
    setIsPaused(true);
  }, []);

  const resume = useCallback(() => {
    nuroMotionRef.current.resume();
    setIsPaused(false);
  }, []);

  const stop = useCallback(() => {
    nuroMotionRef.current.stop();
    setIsRunning(false);
    setIsPaused(false);
    setTrackingState('LOST');
  }, []);

  const reset = useCallback(() => {
    nuroMotionRef.current.reset();
    setMetrics({
      cadenceSpm: '--',
      leftSteps: 0,
      rightSteps: 0,
      totalSteps: 0,
      balanceScore: 100,
      qualityScore: 85,
      syncScore: null,
      timingErrorMs: null,
      isSynchronized: false,
      syncStatus: 'NO_DATA',
    });
    setLastStep(null);
  }, []);

  const registerBeat = useCallback((timestamp) => {
    if (nuroMotionRef.current) {
      nuroMotionRef.current.registerBeat(timestamp);
    }
  }, []);

  return {
    videoRef,
    canvasRef,
    latestLandmarksRef,
    isRunning,
    isPaused,
    isDemoMode,
    trackingState,
    confidence,
    framing,
    metrics,
    audioState,
    diagnostics,

    lastStep,
    errorMessage,
    startCamera,
    startDemo,
    pause,
    resume,
    stop,
    reset,
    registerBeat,
  };
}
