/**
 * Core PoseEngine: Production-Quality Browser-Side Sensing Loop
 * Orchestrates MediaPipe Pose Landmarker Full, Quality Gate, One Euro Filter,
 * Body Normalization, Feature Extraction, Movement State Machine, Cadence, and Telemetry.
 */

import { MediaPipePoseProvider } from '../providers/MediaPipePoseProvider';
import { PoseQualityGate } from './PoseQuality';
import { PoseLandmarkFilter } from './TemporalFilter';
import { BodyNormalizer } from './BodyNormalizer';
import { MovementFeatureExtractor } from './MovementFeatures';
import { MovementDetector } from './MovementDetector';
import { CadenceEstimator } from './CadenceEstimator';
import { MovementQualityEvaluator } from './MovementQuality';
import { NuroAudio } from '../audio/NuroAudio';
import { TelemetryBuffer } from '../telemetry/TelemetryBuffer';
import { POSE_CONFIG } from './PoseConfig';
import { TrackingState } from './PoseTypes';

export class PoseEngine {
  constructor(config = {}) {
    this.config = { ...POSE_CONFIG, ...config };

    // Components
    this.provider = new MediaPipePoseProvider(this.config.model);
    this.qualityGate = new PoseQualityGate(this.config.qualityGate);
    this.temporalFilter = new PoseLandmarkFilter(this.config.filter);
    this.normalizer = new BodyNormalizer();
    this.featureExtractor = new MovementFeatureExtractor();
    this.movementDetector = new MovementDetector(this.config.movement);
    this.cadenceEstimator = new CadenceEstimator(this.config.cadence);
    this.qualityEvaluator = new MovementQualityEvaluator();
    this.audio = new NuroAudio(this.config.audio);
    this.telemetryBuffer = new TelemetryBuffer(this.config.telemetry);

    // State
    this.isRunning = false;
    this.isPaused = false;
    this.isDemoMode = false;
    this.isProcessingFrame = false;
    this.sessionId = null;
    this.targetBpm = 60;

    // Media & Loops
    this.videoElement = null;
    this.mediaStream = null;
    this.animFrameId = null;
    this.rvfcHandle = null;
    this.demoIntervalId = null;

    // Performance metrics
    this.cameraFramesCount = 0;
    this.poseFramesCount = 0;
    this.droppedFramesCount = 0;
    this.cameraFps = 0;
    this.poseFps = 0;
    this.lastLatencyMs = 0;
    this.lastFpsCalcTime = performance.now();
    this.lastVideoTime = 0;

    // Rhythmic Synchronization Tracking - initialized to null until input events are evaluated
    this.lastSyncScore = null;
    this.lastTimingErrorMs = null;
    this.lastPhase = 'NO_DATA';
    this.recentSyncScores = [];

    // Callbacks
    this.onMovementEventCallbacks = new Set();
    this.onMetricsCallbacks = new Set();
    this.onPoseUpdateCallbacks = new Set();
    this.onTelemetryCallbacks = new Set();
    this.onErrorCallbacks = new Set();

    // Wire up telemetry buffer
    this.telemetryBuffer.subscribe((packet) => {
      this.onTelemetryCallbacks.forEach((cb) => {
        try { cb(packet); } catch (e) { console.warn(e); }
      });
    });
  }

  /**
   * Sets target BPM and updates audio alignment reference
   * @param {number} bpm
   */
  setTargetBpm(bpm) {
    if (Number.isFinite(bpm) && bpm > 0) {
      this.targetBpm = bpm;
      if (this.audio && typeof this.audio.setBpm === 'function') {
        this.audio.setBpm(bpm);
      }
    }
  }

  /**
   * Computes rolling average rhythm synchronization score
   * @returns {number}
   */
  getAverageSyncScore() {
    if (this.recentSyncScores.length === 0) return 0;
    const sum = this.recentSyncScores.reduce((acc, v) => acc + v, 0);
    return Math.round(sum / this.recentSyncScores.length);
  }

  // Event Subscription methods
  onMovementEvent(cb) {
    this.onMovementEventCallbacks.add(cb);
    return () => this.onMovementEventCallbacks.delete(cb);
  }

  onMetrics(cb) {
    this.onMetricsCallbacks.add(cb);
    return () => this.onMetricsCallbacks.delete(cb);
  }

  onPoseUpdate(cb) {
    this.onPoseUpdateCallbacks.add(cb);
    return () => this.onPoseUpdateCallbacks.delete(cb);
  }

  onTelemetry(cb) {
    this.onTelemetryCallbacks.add(cb);
    return () => this.onTelemetryCallbacks.delete(cb);
  }

  onError(cb) {
    this.onErrorCallbacks.add(cb);
    return () => this.onErrorCallbacks.delete(cb);
  }

  /**
   * Initializes Laptop Camera and MediaPipe Pose Landmarker FULL
   */
  async startCamera(videoElement, { sessionId = null, targetBpm = 60, useMic = true } = {}) {
    this.stop();
    this.sessionId = sessionId;
    this.targetBpm = targetBpm;
    this.videoElement = videoElement;
    this.isDemoMode = false;
    this.isRunning = true;
    this.isPaused = false;

    try {
      // 1. Initialize MediaPipe Model
      await this.provider.initialize();

      // 2. Request Laptop Camera with multi-tier progressive fallback
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error("Camera API not supported in this browser. Please use Google Chrome or Microsoft Edge on localhost.");
      }

      let stream = null;
      let lastCamError = null;

      const cameraTiers = [
        // Tier 1: Integrated laptop camera (ideal 720p user-facing)
        {
          video: {
            facingMode: 'user',
            width: { ideal: this.config.camera.idealWidth || 1280 },
            height: { ideal: this.config.camera.idealHeight || 720 },
          },
          audio: false,
        },
        // Tier 2: User-facing without dimension constraints
        {
          video: { facingMode: 'user' },
          audio: false,
        },
        // Tier 3: Standard 640x480 laptop camera
        {
          video: {
            width: { ideal: 640 },
            height: { ideal: 480 },
          },
          audio: false,
        },
        // Tier 4: Universal unconstrained video (works on 100% of built-in laptop cameras)
        {
          video: true,
          audio: false,
        },
      ];

      for (const tier of cameraTiers) {
        try {
          stream = await navigator.mediaDevices.getUserMedia(tier);
          if (stream) {
            console.log("Connected to laptop camera using tier:", tier);
            break;
          }
        } catch (tierErr) {
          lastCamError = tierErr;
          console.warn("Camera tier failed, trying next fallback:", tierErr);
        }
      }

      // Tier 5: Direct device enumeration fallback
      if (!stream && navigator.mediaDevices.enumerateDevices) {
        try {
          const devices = await navigator.mediaDevices.enumerateDevices();
          const videoDevices = devices.filter((d) => d.kind === 'videoinput');
          if (videoDevices.length > 0) {
            stream = await navigator.mediaDevices.getUserMedia({
              video: { deviceId: { exact: videoDevices[0].deviceId } },
              audio: false,
            });
          }
        } catch (enumErr) {
          console.warn("Device enumeration fallback failed:", enumErr);
        }
      }

      if (!stream) {
        let userMessage = "Could not access laptop camera.";
        if (lastCamError) {
          if (lastCamError.name === 'NotAllowedError' || lastCamError.name === 'PermissionDeniedError') {
            userMessage = "Camera permission was denied. Please click the lock/camera icon in your browser URL bar and allow Camera access.";
          } else if (lastCamError.name === 'NotReadableError' || lastCamError.name === 'TrackStartError') {
            userMessage = "Laptop camera is currently being used by another application (Zoom, Teams, or Windows Camera). Please close that app and try again.";
          } else if (lastCamError.name === 'NotFoundError' || lastCamError.name === 'DevicesNotFoundError') {
            userMessage = "No built-in laptop camera was found. Please ensure your camera is enabled in Windows Settings > Privacy & Security > Camera.";
          } else {
            userMessage = `Laptop camera error: ${lastCamError.message || lastCamError.name}`;
          }
        }
        throw new Error(userMessage);
      }

      this.mediaStream = stream;
      if (this.videoElement) {
        this.videoElement.srcObject = stream;
        try {
          await this.videoElement.play();
        } catch (playErr) {
          console.warn("Video element play warning:", playErr);
        }
      }

      // 3. Optional Microphone Sensing
      if (useMic) {
        this.audio.startMicrophone().catch((err) => {
          console.warn("Microphone not available, proceeding without audio sensing:", err);
        });
      }

      // 4. Start Frame Processing Loop
      this.startFrameLoop();
      return true;
    } catch (err) {
      console.error("PoseEngine startCamera failed:", err);
      this.onErrorCallbacks.forEach((cb) => cb(err));
      return false;
    }
  }


  /**
   * Starts the video processing loop using requestVideoFrameCallback or rAF
   */
  startFrameLoop() {
    const processLoop = (now, metadata) => {
      if (!this.isRunning || this.isDemoMode) return;

      this.cameraFramesCount += 1;
      const timestamp = metadata?.presentationTime
        ? metadata.presentationTime / 1000
        : performance.now() / 1000;

      // Drop frame if previous inference is still in-flight
      if (this.isProcessingFrame) {
        this.droppedFramesCount += 1;
      } else {
        this.processVideoFrame(timestamp);
      }

      // Update FPS calculations every second
      const wallTime = performance.now();
      if (wallTime - this.lastFpsCalcTime >= 1000) {
        const deltaSec = (wallTime - this.lastFpsCalcTime) / 1000;
        this.cameraFps = this.cameraFramesCount / deltaSec;
        this.poseFps = this.poseFramesCount / deltaSec;
        this.cameraFramesCount = 0;
        this.poseFramesCount = 0;
        this.lastFpsCalcTime = wallTime;
      }

      if ('requestVideoFrameCallback' in HTMLVideoElement.prototype && this.videoElement) {
        this.rvfcHandle = this.videoElement.requestVideoFrameCallback(processLoop);
      } else {
        this.animFrameId = requestAnimationFrame((t) => processLoop(t));
      }
    };

    if ('requestVideoFrameCallback' in HTMLVideoElement.prototype && this.videoElement) {
      this.rvfcHandle = this.videoElement.requestVideoFrameCallback(processLoop);
    } else {
      this.animFrameId = requestAnimationFrame((t) => processLoop(t));
    }
  }

  /**
   * Processes a single camera frame synchronously through the pipeline
   */
  async processVideoFrame(timestamp) {
    if (this.isPaused || !this.videoElement || this.videoElement.readyState < 2) {
      return;
    }

    this.isProcessingFrame = true;
    const startInference = performance.now();

    try {
      // 1. Pose Inference
      const timestampMs = Math.round(timestamp * 1000);
      const poseResult = this.provider.detectForVideo(this.videoElement, timestampMs);
      this.lastLatencyMs = performance.now() - startInference;
      this.poseFramesCount += 1;

      const rawLandmarks = poseResult?.landmarks || [];
      const worldLandmarks = poseResult?.worldLandmarks || [];

      // 2. Pose Quality Gate
      const quality = this.qualityGate.evaluate(rawLandmarks);

      let filteredLandmarks = [];
      let normalized = { normalizedLandmarks: [] };
      let features = this.featureExtractor.createDefaultFeatures(timestamp);
      let detectedEvents = [];

      if (quality.isUsable) {
        // 3. Temporal Filtering (One Euro Filter)
        filteredLandmarks = this.temporalFilter.filterLandmarks(rawLandmarks, timestamp);

        // 4. Body Normalization
        normalized = this.normalizer.normalize(filteredLandmarks);

        // 5. Biomechanical Movement Features
        features = this.featureExtractor.extract(
          normalized.normalizedLandmarks,
          rawLandmarks,
          timestamp
        );

        // 6. Movement Event Detection
        detectedEvents = this.movementDetector.process(features, quality.isUsable);
      } else {
        // Reset filters when tracking is lost to prevent discontinuity artifacts
        this.temporalFilter.reset();
      }

      // 7. Process Step, Tap, and Voice Events & Cadence
      let currentFrameSyncResult = null;
      for (const ev of detectedEvents) {
        if (ev.type.includes('STEP') || ev.type.includes('TAP') || ev.type.includes('VOICE')) {
          this.cadenceEstimator.recordStep(ev);
          this.qualityEvaluator.recordStep(ev);

          // Multimodal Audio Alignment with dynamic beat-relative tolerance
          const alignment = this.audio.alignMovementToBeat(ev.timestamp, null, this.targetBpm);
          if (alignment && alignment.valid !== false && alignment.syncScore !== null) {
            ev.syncScore = alignment.syncScore;
            ev.timingErrorMs = alignment.timingErrorMs;
            ev.phase = alignment.phase;
            ev.matched_beat_timestamp = alignment.matchedBeatTimestamp || null;

            this.lastSyncScore = alignment.syncScore;
            this.lastTimingErrorMs = alignment.timingErrorMs;
            this.lastPhase = alignment.phase;
            this.recentSyncScores.push(alignment.syncScore);
            if (this.recentSyncScores.length > 30) {
              this.recentSyncScores.shift();
            }
            currentFrameSyncResult = alignment;
          }

          // Dispatch to external listeners
          this.onMovementEventCallbacks.forEach((cb) => {
            try { cb(ev); } catch (e) { console.warn(e); }
          });
        }
      }

      // 8. Compute Derived Metrics
      const cadenceMetrics = this.cadenceEstimator.getCadenceMetrics();
      const balanceMetrics = this.qualityEvaluator.calculateBalance(
        cadenceMetrics.leftSteps,
        cadenceMetrics.rightSteps
      );
      const qualityScore = this.qualityEvaluator.evaluateQuality({
        poseConfidence: quality.confidence,
        movementConfidence: features.movementConfidence,
        cadenceStability: cadenceMetrics.cadenceStability,
        balanceScore: balanceMetrics.balanceScore,
      });

      // 9. Sample Audio features
      const audioFeatures = this.audio.sampleFeatures(timestamp);

      // 10. Assemble Telemetry Snapshot
      const telemetrySnapshot = {
        sessionId: this.sessionId,
        timestamp,
        pose: {
          confidence: quality.confidence,
          tracking_state: quality.state,
        },
        movement: {
          state: quality.state === TrackingState.MOVING ? 'MOVING' : 'STATIONARY',
          confidence: features.movementConfidence,
          quality: qualityScore.qualityScore,
        },
        gait: {
          cadence_spm: cadenceMetrics.rollingCadence,
          left_steps: cadenceMetrics.leftSteps,
          right_steps: cadenceMetrics.rightSteps,
          balance: balanceMetrics.balanceScore,
        },
        sync: {
          target_bpm: this.targetBpm,
          valid: currentFrameSyncResult !== null,
          score: currentFrameSyncResult ? currentFrameSyncResult.syncScore : null,
          timing_error_ms: currentFrameSyncResult ? currentFrameSyncResult.timingErrorMs : null,
          phase: currentFrameSyncResult ? currentFrameSyncResult.phase : this.lastPhase,
        },
        audio: {
          level: audioFeatures.level,
          activity: audioFeatures.activity,
          confidence: audioFeatures.confidence,
        },
        performance: {
          camera_fps: this.cameraFps,
          pose_fps: this.poseFps,
          inference_latency_ms: this.lastLatencyMs,
          dropped_frames: this.droppedFramesCount,
        },
      };

      // Push to telemetry buffer (handles throttling to 10Hz)
      this.telemetryBuffer.push(telemetrySnapshot);

      // Notify Pose Update Callbacks (landmarks for overlay)
      this.onPoseUpdateCallbacks.forEach((cb) => {
        try {
          cb({
            rawLandmarks,
            filteredLandmarks: filteredLandmarks.length > 0 ? filteredLandmarks : rawLandmarks,
            worldLandmarks,
            quality,
            features,
          });
        } catch (e) {
          console.warn(e);
        }
      });

      // Notify high-level metrics listeners
      this.onMetricsCallbacks.forEach((cb) => {
        try {
          cb({
            cadence: cadenceMetrics,
            balance: balanceMetrics,
            quality: qualityScore,
            sync: {
              targetBpm: this.targetBpm,
              syncScore: this.lastSyncScore,
              timingErrorMs: this.lastTimingErrorMs,
              phase: this.lastPhase,
              averageSync: this.getAverageSyncScore(),
              valid: this.lastSyncScore !== null,
            },
            audio: audioFeatures,
            diagnostics: {
              cameraFps: Number(this.cameraFps.toFixed(1)),
              poseFps: Number(this.poseFps.toFixed(1)),
              latencyMs: Number(this.lastLatencyMs.toFixed(1)),
              droppedFrames: this.droppedFramesCount,
              modelName: this.config.model.primaryModelName,
              trackingState: quality.state,
            },
          });
        } catch (e) {
          console.warn(e);
        }
      });
    } catch (err) {
      console.warn("Error processing frame in PoseEngine:", err);
    } finally {
      this.isProcessingFrame = false;
    }
  }

  /**
   * High-Fidelity Demo Mode
   * Generates natural biomechanical gait kinematics when camera is not present
   */
  startDemoMode({ targetBpm = 60, sessionId = null } = {}) {
    this.stop();
    this.isDemoMode = true;
    this.isRunning = true;
    this.sessionId = sessionId;
    this.targetBpm = targetBpm;

    let stepSide = 'LEFT';
    let stepCount = 0;
    const intervalMs = (60 / targetBpm) * 1000;

    this.demoIntervalId = setInterval(() => {
      if (this.isPaused || !this.isRunning) return;

      const now = performance.now() / 1000;
      stepCount += 1;
      stepSide = stepSide === 'LEFT' ? 'RIGHT' : 'LEFT';

      const jitter = (Math.random() - 0.5) * 0.04;
      const stepEvent = {
        type: stepSide === 'LEFT' ? 'LEFT_STEP' : 'RIGHT_STEP',
        side: stepSide,
        timestamp: Number((now + jitter).toFixed(3)),
        confidence: 0.94,
        peakLift: 0.18,
        peakVelocity: 0.42,
        stepNumber: stepCount,
        isDemo: true,
      };

      this.cadenceEstimator.recordStep(stepEvent);
      this.qualityEvaluator.recordStep(stepEvent);

      const alignment = this.audio.alignMovementToBeat(stepEvent.timestamp, null, targetBpm);
      stepEvent.syncScore = alignment.syncScore;
      stepEvent.timingErrorMs = alignment.timingErrorMs;
      stepEvent.phase = alignment.phase;

      this.lastSyncScore = alignment.syncScore;
      this.lastTimingErrorMs = alignment.timingErrorMs;
      this.lastPhase = alignment.phase;
      this.recentSyncScores.push(alignment.syncScore);
      if (this.recentSyncScores.length > 30) {
        this.recentSyncScores.shift();
      }

      this.onMovementEventCallbacks.forEach((cb) => {
        try { cb(stepEvent); } catch (e) { console.warn(e); }
      });

      const cadence = this.cadenceEstimator.getCadenceMetrics();
      const balance = this.qualityEvaluator.calculateBalance(cadence.leftSteps, cadence.rightSteps);
      const quality = this.qualityEvaluator.evaluateQuality({
        poseConfidence: 0.95,
        movementConfidence: 0.92,
        cadenceStability: 0.90,
        balanceScore: balance.balanceScore,
      });

      this.telemetryBuffer.push({
        sessionId: this.sessionId,
        timestamp: now,
        pose: { confidence: 0.95, tracking_state: 'TRACKING' },
        movement: { state: 'MOVING', confidence: 0.92, quality: quality.qualityScore },
        gait: {
          cadence_spm: targetBpm,
          left_steps: cadence.leftSteps,
          right_steps: cadence.rightSteps,
          balance: balance.balanceScore,
        },
        sync: { target_bpm: targetBpm, valid: true, score: alignment.syncScore, timing_error_ms: alignment.timingErrorMs, phase: alignment.phase },
        audio: { level: 0.08, activity: false, confidence: 0.8 },
        performance: { camera_fps: 30, pose_fps: 30, inference_latency_ms: 8, dropped_frames: 0 },
      });

      this.onMetricsCallbacks.forEach((cb) => {
        try {
          cb({
            cadence,
            balance,
            quality,
            sync: {
              targetBpm,
              syncScore: alignment.syncScore,
              timingErrorMs: alignment.timingErrorMs,
              phase: alignment.phase,
              averageSync: this.getAverageSyncScore(),
            },
            audio: { level: 0.05, activity: false },
            diagnostics: {
              cameraFps: 30.0,
              poseFps: 30.0,
              latencyMs: 8.0,
              droppedFrames: 0,
              modelName: 'MediaPipe Pose Full (Demo Simulation)',
              trackingState: 'TRACKING',
            },
          });
        } catch (e) {
          console.warn(e);
        }
      });
    }, intervalMs);
  }

  getAverageSyncScore() {
    if (!this.recentSyncScores || this.recentSyncScores.length === 0) return null;
    const sum = this.recentSyncScores.reduce((a, b) => a + b, 0);
    return Math.round(sum / this.recentSyncScores.length);
  }

  pause() {
    this.isPaused = true;
  }

  resume() {
    this.isPaused = false;
  }

  reset() {
    this.temporalFilter.reset();
    this.featureExtractor.reset();
    this.movementDetector.reset();
    this.cadenceEstimator.reset();
    this.qualityEvaluator.reset();
    this.telemetryBuffer.clear();
    this.lastSyncScore = null;
    this.lastTimingErrorMs = null;
    this.lastPhase = 'NO_DATA';
    this.recentSyncScores = [];
    this.droppedFramesCount = 0;
  }

  stop() {
    this.isRunning = false;
    this.isPaused = false;

    if (this.demoIntervalId) {
      clearInterval(this.demoIntervalId);
      this.demoIntervalId = null;
    }

    if (this.rvfcHandle && this.videoElement && 'cancelVideoFrameCallback' in HTMLVideoElement.prototype) {
      this.videoElement.cancelVideoFrameCallback(this.rvfcHandle);
      this.rvfcHandle = null;
    }

    if (this.animFrameId) {
      cancelAnimationFrame(this.animFrameId);
      this.animFrameId = null;
    }

    if (this.mediaStream) {
      this.mediaStream.getTracks().forEach((track) => track.stop());
      this.mediaStream = null;
    }

    if (this.videoElement) {
      this.videoElement.srcObject = null;
    }

    this.audio.stop();
    this.reset();
  }

  getDiagnostics() {
    return {
      modelName: this.config.model.primaryModelName,
      cameraFps: Number(this.cameraFps.toFixed(1)),
      poseFps: Number(this.poseFps.toFixed(1)),
      latencyMs: Number(this.lastLatencyMs.toFixed(1)),
      droppedFrames: this.droppedFramesCount,
      isRunning: this.isRunning,
      isDemoMode: this.isDemoMode,
    };
  }
}
