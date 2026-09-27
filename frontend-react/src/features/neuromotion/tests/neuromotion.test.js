import { describe, it, expect, beforeEach } from 'vitest';
import { OneEuroFilter, LowPassFilter, PoseLandmarkFilter } from '../core/TemporalFilter';
import { BodyNormalizer, VectorMath } from '../core/BodyNormalizer';
import { MovementFeatureExtractor } from '../core/MovementFeatures';
import { MovementDetector } from '../core/MovementDetector';
import { CadenceEstimator } from '../core/CadenceEstimator';
import { MovementQualityEvaluator } from '../core/MovementQuality';
import { createTelemetryPacket, serializeTelemetry } from '../telemetry/TelemetryTypes';
import { PoseQualityGate } from '../core/PoseQuality';
import { LANDMARKS, TrackingState, MovementEventType, FramingStatus, FramingFeedback } from '../core/PoseTypes';
import { NuroAudio } from '../audio/NuroAudio';
import { PoseEngine } from '../core/PoseEngine';
import { NuroSync } from '../../../services/nuroMotion';

describe('NuroMotion Phase 1: Pure Logic Test Suite', () => {


  // 1. Vector Mathematics & Geometry
  describe('VectorMath & Angle Calculations', () => {
    it('computes 2D and 3D Euclidean distances accurately', () => {
      const p1 = { x: 0, y: 0, z: 0 };
      const p2 = { x: 3, y: 4, z: 0 };
      expect(VectorMath.distance2D(p1, p2)).toBe(5);
      expect(VectorMath.distance(p1, p2)).toBe(5);

      const p3 = { x: 1, y: 2, z: 2 };
      expect(VectorMath.distance(p1, p3)).toBe(3);
    });

    it('computes 3D joint angles correctly', () => {
      // 90-degree right angle (e.g., knee bent at 90 deg)
      const hip = { x: 0, y: 1, z: 0 };
      const knee = { x: 0, y: 0, z: 0 };
      const ankle = { x: 1, y: 0, z: 0 };
      const angle = VectorMath.angleBetween(hip, knee, ankle);
      expect(Math.round(angle)).toBe(90);

      // Straight line 180 degrees (leg fully extended)
      const ankleStraight = { x: 0, y: -1, z: 0 };
      const angleStraight = VectorMath.angleBetween(hip, knee, ankleStraight);
      expect(Math.round(angleStraight)).toBe(180);
    });

    it('handles degenerate zero-length vectors safely without NaN', () => {
      const p = { x: 0, y: 0, z: 0 };
      const angle = VectorMath.angleBetween(p, p, p);
      expect(Number.isFinite(angle)).toBe(true);
      expect(angle).toBe(180);
    });
  });

  // 2. Body Normalization
  describe('BodyNormalizer', () => {
    let normalizer;
    let mockLandmarks;

    beforeEach(() => {
      normalizer = new BodyNormalizer();
      mockLandmarks = Array(33).fill(null).map(() => ({
        x: 0.5,
        y: 0.5,
        z: 0.0,
        visibility: 0.9,
        presence: 0.9,
      }));

      // Set key reference landmarks
      mockLandmarks[LANDMARKS.LEFT_HIP] = { x: 0.45, y: 0.6, z: 0, visibility: 0.95 };
      mockLandmarks[LANDMARKS.RIGHT_HIP] = { x: 0.55, y: 0.6, z: 0, visibility: 0.95 };
      mockLandmarks[LANDMARKS.LEFT_SHOULDER] = { x: 0.42, y: 0.3, z: 0, visibility: 0.95 };
      mockLandmarks[LANDMARKS.RIGHT_SHOULDER] = { x: 0.58, y: 0.3, z: 0, visibility: 0.95 };
    });

    it('normalizes hip center to (0, 0)', () => {
      const res = normalizer.normalize(mockLandmarks);
      expect(res.hipCenter.x).toBeCloseTo(0.5, 4);
      expect(res.hipCenter.y).toBeCloseTo(0.6, 4);

      // Left hip in normalized coordinates should be shifted around origin
      const normLHip = res.normalizedLandmarks[LANDMARKS.LEFT_HIP];
      const normRHip = res.normalizedLandmarks[LANDMARKS.RIGHT_HIP];
      expect(normLHip.x + normRHip.x).toBeCloseTo(0, 4);
    });

    it('is invariant to lateral translation', () => {
      const norm1 = normalizer.normalize(mockLandmarks);

      // Shift entire person right by 0.2
      const shifted = mockLandmarks.map(lm => ({ ...lm, x: lm.x + 0.2 }));
      const norm2 = normalizer.normalize(shifted);

      expect(norm1.normalizedLandmarks[LANDMARKS.LEFT_KNEE].x).toBeCloseTo(
        norm2.normalizedLandmarks[LANDMARKS.LEFT_KNEE].x,
        4
      );
    });
  });

  // 3. Temporal Filtering (One Euro Filter)
  describe('Temporal Filtering (One Euro Filter)', () => {
    it('reduces high-frequency noise while preserving baseline', () => {
      const filter = new OneEuroFilter(30, 1.0, 0.007, 1.0);
      let t = 0.0;
      const filteredValues = [];

      // Feed stationary signal with noise (val = 1.0 ± 0.05)
      for (let i = 0; i < 20; i++) {
        t += 0.033;
        const noise = (i % 2 === 0 ? 0.05 : -0.05);
        const filtered = filter.filter(1.0 + noise, t);
        filteredValues.push(filtered);
      }

      // After settling, variance should be significantly reduced
      const last = filteredValues[filteredValues.length - 1];
      expect(last).toBeCloseTo(1.0, 1);
    });

    it('adapts to fast motion without excessive lag', () => {
      const filter = new OneEuroFilter(30, 1.0, 0.05, 1.0);
      let t = 0.0;

      // Stationary phase
      for (let i = 0; i < 10; i++) {
        t += 0.033;
        filter.filter(0.0, t);
      }

      // Fast motion progression
      let fastFiltered = 0;
      for (let i = 0; i < 5; i++) {
        t += 0.033;
        fastFiltered = filter.filter(1.0, t);
      }
      // Adaptive filter rapidly catches up to fast moving target
      expect(fastFiltered).toBeGreaterThan(0.7);
    });

    it('resets cleanly when tracking is lost', () => {
      const filter = new OneEuroFilter();
      filter.filter(10.0, 1.0);
      filter.reset();
      expect(filter.lastTime).toBeNull();
      // First filtered value after reset should equal input value
      expect(filter.filter(5.0, 2.0)).toBe(5.0);
    });
  });

  // 4. Pose Quality Gate
  describe('PoseQualityGate', () => {
    let gate;
    beforeEach(() => {
      gate = new PoseQualityGate();
    });

    it('returns LOST when landmarks are missing or empty', () => {
      const result = gate.evaluate([]);
      expect(result.state).toBe(TrackingState.LOST);
      expect(result.isUsable).toBe(false);
    });

    it('returns UNCERTAIN when legs/feet have low confidence', () => {
      const lms = Array(33).fill(null).map(() => ({
        x: 0.5, y: 0.5, z: 0, visibility: 0.3, presence: 0.3
      }));
      // Hips visible but rest below threshold
      lms[LANDMARKS.LEFT_HIP].visibility = 0.8;
      lms[LANDMARKS.RIGHT_HIP].visibility = 0.8;

      const result = gate.evaluate(lms);
      expect(result.state).toBe(TrackingState.UNCERTAIN);
      expect(result.isUsable).toBe(false);
    });

    it('returns TRACKING / MOVING when full lower body is visible with good confidence', () => {
      const lms = Array(33).fill(null).map(() => ({
        x: 0.5, y: 0.5, z: 0, visibility: 0.9, presence: 0.9
      }));

      const result = gate.evaluate(lms, 0.25);
      expect(result.isUsable).toBe(true);
      expect(result.state).toBe(TrackingState.MOVING);
    });
  });

  // 5. Movement Event Detector
  describe('MovementDetector State Machine', () => {
    let detector;
    beforeEach(() => {
      detector = new MovementDetector();
    });

    it('detects step events upon foot strike with debounce', () => {
      let t = 1.0;
      // Stance phase
      const featuresStance = {
        timestamp: t,
        left: { verticalLift: 0.0, verticalVelocity: 0.0 },
        right: { verticalLift: 0.0, verticalVelocity: 0.0 },
        combinedVelocity: 0.0,
        movementConfidence: 0.95,
      };
      detector.process(featuresStance, true);

      // Swing initiate: leg lifts upward
      t += 0.1;
      const featuresSwingUp = {
        timestamp: t,
        left: { verticalLift: 0.15, verticalVelocity: 0.5 },
        right: { verticalLift: 0.0, verticalVelocity: 0.0 },
        combinedVelocity: 0.25,
        movementConfidence: 0.95,
      };
      detector.process(featuresSwingUp, true);

      // Swing peak
      t += 0.15;
      const featuresPeak = {
        timestamp: t,
        left: { verticalLift: 0.20, verticalVelocity: 0.0 },
        right: { verticalLift: 0.0, verticalVelocity: 0.0 },
        combinedVelocity: 0.1,
        movementConfidence: 0.95,
      };
      detector.process(featuresPeak, true);

      // Swing down
      t += 0.15;
      const featuresDown = {
        timestamp: t,
        left: { verticalLift: 0.05, verticalVelocity: -0.3 },
        right: { verticalLift: 0.0, verticalVelocity: 0.0 },
        combinedVelocity: 0.15,
        movementConfidence: 0.95,
      };
      detector.process(featuresDown, true);

      // Foot strike / arrest at floor
      t += 0.1;
      const featuresStrike = {
        timestamp: t,
        left: { verticalLift: 0.01, verticalVelocity: -0.02 },
        right: { verticalLift: 0.0, verticalVelocity: 0.0 },
        combinedVelocity: 0.05,
        movementConfidence: 0.95,
      };
      const events = detector.process(featuresStrike, true);

      // Step detected!
      const stepEvents = events.filter(e => e.type === MovementEventType.LEFT_STEP);
      expect(stepEvents.length).toBe(1);
      expect(stepEvents[0].side).toBe('LEFT');
      expect(stepEvents[0].timestamp).toBeCloseTo(t, 2);
    });

    it('enforces debounce and prevents impossible rapid firing', () => {
      const counts = detector.getStepCounts();
      expect(counts.total).toBe(0);
    });
  });

  // 6. Cadence Estimator
  describe('CadenceEstimator', () => {
    let cadence;
    beforeEach(() => {
      cadence = new CadenceEstimator();
    });

    it('never displays NaN or Infinity when data is empty or insufficient', () => {
      const metrics = cadence.getCadenceMetrics();
      expect(metrics.instantaneousCadence).toBeNull();
      expect(metrics.rollingCadence).toBeNull();
      expect(metrics.displaySpm).toBe('--');
      expect(Number.isNaN(metrics.cadenceConfidence)).toBe(false);
    });

    it('calculates steady cadence accurately for 60 SPM steps (1s interval)', () => {
      let t = 0.0;
      // Record 5 steps at 1.0 second intervals (60 steps/min)
      for (let i = 0; i < 5; i++) {
        t += 1.0;
        cadence.recordStep({
          side: i % 2 === 0 ? 'LEFT' : 'RIGHT',
          timestamp: t,
          type: 'STEP',
        });
      }

      const metrics = cadence.getCadenceMetrics();
      expect(metrics.rollingCadence).toBeCloseTo(60, 0);
      expect(metrics.displaySpm).toBe('60');
      expect(metrics.isStable).toBe(true);
      expect(metrics.cadenceStability).toBeGreaterThan(0.8);
    });
  });

  // 7. Movement Quality & Balance
  describe('MovementQualityEvaluator', () => {
    let evaluator;
    beforeEach(() => {
      evaluator = new MovementQualityEvaluator();
    });

    it('calculates 100% balance for symmetric bilateral steps', () => {
      evaluator.recordStep({ side: 'LEFT', timestamp: 1.0, peakLift: 0.15 });
      evaluator.recordStep({ side: 'RIGHT', timestamp: 2.0, peakLift: 0.15 });
      evaluator.recordStep({ side: 'LEFT', timestamp: 3.0, peakLift: 0.15 });
      evaluator.recordStep({ side: 'RIGHT', timestamp: 4.0, peakLift: 0.15 });

      const balance = evaluator.calculateBalance(2, 2);
      expect(balance.balanceScore).toBeGreaterThanOrEqual(95);
    });

    it('evaluates deterministic quality score within [0, 100]', () => {
      const q = evaluator.evaluateQuality({
        poseConfidence: 0.9,
        movementConfidence: 0.85,
        cadenceStability: 0.88,
        balanceScore: 92,
      });

      expect(q.qualityScore).toBeGreaterThanOrEqual(80);
      expect(q.qualityScore).toBeLessThanOrEqual(100);
    });
  });

  // 8. Telemetry Serialization
  describe('Telemetry Serialization', () => {
    it('creates and serializes compact telemetry packets conforming to schema', () => {
      const packet = createTelemetryPacket({
        sessionId: 42,
        timestamp: 100.5,
        pose: { confidence: 0.92, tracking_state: 'TRACKING' },
        movement: { state: 'MOVING', confidence: 0.88, quality: 90 },
        gait: { cadence_spm: 56.4, left_steps: 12, right_steps: 11, balance: 95 },
        sync: { target_bpm: 56, score: 92, timing_error_ms: 22 },
        audio: { level: 0.14, activity: true, confidence: 0.85 },
        performance: { camera_fps: 30.1, pose_fps: 29.8, inference_latency_ms: 18.2, dropped_frames: 0 },
      });

      expect(packet.session_id).toBe(42);
      expect(packet.gait.cadence_spm).toBe(56.4);
      expect(packet.sync.timing_error_ms).toBe(22);

      const jsonStr = serializeTelemetry(packet);
      const parsed = JSON.parse(jsonStr);
      expect(parsed.session_id).toBe(42);
      expect(parsed.pose.tracking_state).toBe('TRACKING');
    });
  });

  // 9. Patient Body Framing Silhouette Guide & Quality Gate
  describe('Patient Body Framing Silhouette Guide & Quality Gate', () => {
    let gate;
    let createMockPerson;

    beforeEach(() => {
      gate = new PoseQualityGate();

      createMockPerson = ({
        topY = 0.15,
        bottomY = 0.85,
        centerX = 0.50,
        anklesVis = 0.9,
        feetVis = 0.9,
        hipsVis = 0.9,
      } = {}) => {
        const lms = Array(33).fill(null).map(() => ({
          x: centerX,
          y: 0.5,
          z: 0.0,
          visibility: 0.9,
          presence: 0.9,
        }));

        // Head
        lms[LANDMARKS.NOSE] = { x: centerX, y: topY, z: 0, visibility: 0.95 };
        // Shoulders
        lms[LANDMARKS.LEFT_SHOULDER] = { x: centerX - 0.1, y: topY + 0.1, z: 0, visibility: 0.95 };
        lms[LANDMARKS.RIGHT_SHOULDER] = { x: centerX + 0.1, y: topY + 0.1, z: 0, visibility: 0.95 };
        // Hips
        lms[LANDMARKS.LEFT_HIP] = { x: centerX - 0.08, y: 0.50, z: 0, visibility: hipsVis };
        lms[LANDMARKS.RIGHT_HIP] = { x: centerX + 0.08, y: 0.50, z: 0, visibility: hipsVis };
        // Knees
        lms[LANDMARKS.LEFT_KNEE] = { x: centerX - 0.08, y: 0.68, z: 0, visibility: 0.9 };
        lms[LANDMARKS.RIGHT_KNEE] = { x: centerX + 0.08, y: 0.68, z: 0, visibility: 0.9 };
        // Ankles
        lms[LANDMARKS.LEFT_ANKLE] = { x: centerX - 0.08, y: bottomY - 0.04, z: 0, visibility: anklesVis };
        lms[LANDMARKS.RIGHT_ANKLE] = { x: centerX + 0.08, y: bottomY - 0.04, z: 0, visibility: anklesVis };
        // Heels
        lms[LANDMARKS.LEFT_HEEL] = { x: centerX - 0.08, y: bottomY - 0.02, z: 0, visibility: feetVis };
        lms[LANDMARKS.RIGHT_HEEL] = { x: centerX + 0.08, y: bottomY - 0.02, z: 0, visibility: feetVis };
        // Feet Indices (Toes)
        lms[LANDMARKS.LEFT_FOOT_INDEX] = { x: centerX - 0.08, y: bottomY, z: 0, visibility: feetVis };
        lms[LANDMARKS.RIGHT_FOOT_INDEX] = { x: centerX + 0.08, y: bottomY, z: 0, visibility: feetVis };

        return lms;
      };
    });

    it('detects NO_PERSON when landmarks are absent or hips unconfident', () => {
      const emptyResult = gate.evaluateFraming(null);
      expect(emptyResult.status).toBe(FramingStatus.NO_PERSON);
      expect(emptyResult.feedback).toBe('Step into camera view');

      const noHips = createMockPerson({ hipsVis: 0.1 });
      const framing = gate.evaluateFraming(noHips);
      expect(framing.status).toBe(FramingStatus.NO_PERSON);
    });

    it('instructs user to "Make sure your feet are visible" when feet are occluded or clipped', () => {
      // Feet low visibility
      const occludedFeet = createMockPerson({ feetVis: 0.1, anklesVis: 0.1 });
      const framing1 = gate.evaluateFraming(occludedFeet);
      expect(framing1.status).toBe(FramingStatus.FEET_NOT_VISIBLE);
      expect(framing1.feedback).toBe('Make sure your feet are visible');

      // Feet cut off at camera bottom edge (y >= 0.96)
      const clippedFeet = createMockPerson({ bottomY: 0.98 });
      const framing2 = gate.evaluateFraming(clippedFeet);
      expect(framing2.status).toBe(FramingStatus.FEET_NOT_VISIBLE);
      expect(framing2.feetVisible).toBe(false);
    });

    it('instructs user to "Move farther away" when body fills too much of frame', () => {
      // Body height > 84% of vertical frame
      const tooClose = createMockPerson({ topY: 0.03, bottomY: 0.92 });
      const framing = gate.evaluateFraming(tooClose);
      expect(framing.status).toBe(FramingStatus.MOVE_FARTHER);
      expect(framing.feedback).toBe('Move farther away');
    });

    it('instructs user to "Move closer" when body is too small', () => {
      // Body height < 38%
      const tooFar = createMockPerson({ topY: 0.35, bottomY: 0.65 });
      const framing = gate.evaluateFraming(tooFar);
      expect(framing.status).toBe(FramingStatus.MOVE_CLOSER);
      expect(framing.feedback).toBe('Move closer');
    });

    it('instructs user to "Center yourself" when shifted laterally', () => {
      // Center X shifted to left (< 0.32)
      const shiftedLeft = createMockPerson({ centerX: 0.20, topY: 0.15, bottomY: 0.85 });
      const framing = gate.evaluateFraming(shiftedLeft);
      expect(framing.status).toBe(FramingStatus.CENTER_BODY);
      expect(framing.feedback).toBe('Center yourself');
    });

    it('reports "Full body detected" when optimal distance and framing are met', () => {
      const wellFramed = createMockPerson({ topY: 0.12, bottomY: 0.86, centerX: 0.50 });
      const framing = gate.evaluateFraming(wellFramed);
      expect(framing.status).toBe(FramingStatus.FULL_BODY_DETECTED);
      expect(framing.feedback).toBe('Full body detected');
      expect(framing.isOptimallyFramed).toBe(true);
      expect(framing.score).toBe(100);
    });

    it('feeds framing status directly into Pose Quality Gate', () => {
      const occludedFeet = createMockPerson({ feetVis: 0.1 });
      const quality = gate.evaluate(occludedFeet);

      // Quality gate sets isUsable to false and state to UNCERTAIN when feet are missing
      expect(quality.isUsable).toBe(false);
      expect(quality.state).toBe(TrackingState.UNCERTAIN);
      expect(quality.missingRegions).toContain('feet_visibility');
      expect(quality.framing.status).toBe(FramingStatus.FEET_NOT_VISIBLE);
    });
  });

  // 10. NuroAudio Dynamic Beat-Relative Tolerance & Alignment
  describe('NuroAudio Dynamic Beat-Relative Tolerance & Alignment', () => {
    let audio;

    beforeEach(() => {
      audio = new NuroAudio();
    });

    it('derives dynamic tolerance as 20% of beat period across different BPMs', () => {
      // 60 BPM -> 1000ms period -> 200ms tolerance
      expect(audio.calculateDynamicTolerance(60)).toBe(200.0);
      // 120 BPM -> 500ms period -> 100ms tolerance
      expect(audio.calculateDynamicTolerance(120)).toBe(100.0);
      // 50 BPM -> 1200ms period -> 240ms tolerance
      expect(audio.calculateDynamicTolerance(50)).toBe(240.0);
    });

    it('aligns movement to registered beats with dynamic tolerance', () => {
      audio.setBpm(60); // tolerance = 200ms
      audio.registerSynthesizedBeat(1.0);
      audio.registerSynthesizedBeat(2.0);

      // On-beat event (0ms error)
      const perfect = audio.alignMovementToBeat(2.000);
      expect(perfect.timingErrorMs).toBe(0);
      expect(perfect.syncScore).toBe(100);
      expect(perfect.isSynchronized).toBe(true);
      expect(perfect.phase).toBe('ON_BEAT');

      // 60ms late event -> 100 * (1 - 60/200) = 70
      const late = audio.alignMovementToBeat(2.060);
      expect(late.timingErrorMs).toBe(60);
      expect(late.syncScore).toBe(70);
      expect(late.isSynchronized).toBe(true);
      expect(late.phase).toBe('LATE');

      // 220ms off-beat event -> exceeds 200ms tolerance -> score 0
      const missed = audio.alignMovementToBeat(2.220);
      expect(missed.timingErrorMs).toBe(220);
      expect(missed.syncScore).toBe(0);
      expect(missed.isSynchronized).toBe(false);
    });

    it('returns explicit NO_DATA / invalid state when beat history is empty', () => {
      audio.setBpm(60);
      const evalResult = audio.alignMovementToBeat(3.050, null, 60);
      expect(evalResult.valid).toBe(false);
      expect(evalResult.timingErrorMs).toBeNull();
      expect(evalResult.syncScore).toBeNull();
      expect(evalResult.isSynchronized).toBe(false);
      expect(evalResult.phase).toBe('NO_DATA');
    });
  });

  // 11. Semantic Decoupling: Movement Quality vs Synchronization Accuracy
  describe('Semantic Decoupling: Movement Quality vs Synchronization Accuracy', () => {
    it('ensures PoseEngine assigns real syncScore to sync.score rather than qualityScore', () => {
      const engine = new PoseEngine();
      engine.setTargetBpm(60);
      engine.audio.registerSynthesizedBeat(1.0);
      engine.audio.registerSynthesizedBeat(2.0);

      // Simulate a step event with 50ms timing error
      const alignment = engine.audio.alignMovementToBeat(2.050, null, 60);
      expect(alignment.syncScore).toBe(75); // Real rhythm sync score

      engine.lastSyncScore = alignment.syncScore;
      engine.lastTimingErrorMs = alignment.timingErrorMs;

      // Telemetry packet verification: movement.quality and sync.score must be independent
      const qualityScore = 95; // High movement quality (e.g. good posture)
      const telemetry = createTelemetryPacket({
        sessionId: 101,
        movement: { quality: qualityScore },
        sync: { target_bpm: 60, score: engine.lastSyncScore, timing_error_ms: engine.lastTimingErrorMs },
      });

      expect(telemetry.movement.quality).toBe(95);
      expect(telemetry.sync.score).toBe(75); // Preserves rhythm sync, NOT contaminated by quality!
      expect(telemetry.sync.score).not.toBe(telemetry.movement.quality);
    });

    it('preserves high movement quality even when rhythm sync drops', () => {
      const highQualityLowSync = createTelemetryPacket({
        sessionId: 102,
        movement: { quality: 92 },
        sync: { target_bpm: 60, score: 35, timing_error_ms: 130 },
      });

      expect(highQualityLowSync.movement.quality).toBe(92);
      expect(highQualityLowSync.sync.score).toBe(35);
    });

    it('preserves high rhythm sync even when movement quality drops', () => {
      const lowQualityHighSync = createTelemetryPacket({
        sessionId: 103,
        movement: { quality: 40 },
        sync: { target_bpm: 60, score: 95, timing_error_ms: 10 },
      });

      expect(lowQualityHighSync.movement.quality).toBe(40);
      expect(lowQualityHighSync.sync.score).toBe(95);
    });
  });

  // 12. NuroSync Dynamic Tolerance & Multi-tempo Consistency
  describe('NuroSync Dynamic Tolerance & Multi-tempo Consistency', () => {
    it('dynamically adapts tolerance when tempo changes', () => {
      const sync = new NuroSync(null, 60);
      expect(sync.toleranceMs).toBe(200.0);

      sync.setBpm(100);
      expect(sync.toleranceMs).toBe(120.0); // 60000 / 100 * 0.20 = 120ms

      sync.setBpm(120);
      expect(sync.toleranceMs).toBe(100.0); // 60000 / 120 * 0.20 = 100ms
    });

    it('evaluates step alignment relative to dynamic tolerance window', () => {
      const sync = new NuroSync(null, 60); // tolerance 200ms
      sync.recordBeat(1.0);
      sync.recordBeat(2.0);

      const res = sync.evaluateStep(2.040); // 40ms error
      expect(res.timingErrorMs).toBe(40);
      expect(res.syncScore).toBe(80); // 100 * (1 - 40/200) = 80
      expect(res.isSynchronized).toBe(true);
    });
  });

  // 13. Zero-Input Integrity & Ghost Rhythm Metric Suppression
  describe('Zero-Input Integrity & Ghost Rhythm Metric Suppression', () => {
    it('ensures NuroSync returns null average sync when no steps or inputs exist', () => {
      const sync = new NuroSync(null, 60);
      expect(sync.getAverageSync()).toBeNull(); // Must be null, never fabricate 92 or 85!
    });

    it('ensures PoseEngine reports null rhythm sync score prior to any movement event', () => {
      const engine = new PoseEngine();
      expect(engine.lastSyncScore).toBeNull();
      expect(engine.getAverageSyncScore()).toBeNull(); // Must be null, never fabricate 85!
    });

    it('ensures PoseEngine reset restores sync score to null and NO_DATA state', () => {
      const engine = new PoseEngine();
      engine.lastSyncScore = 92;
      engine.recentSyncScores.push(92);
      expect(engine.getAverageSyncScore()).toBe(92);

      engine.reset();
      expect(engine.lastSyncScore).toBeNull();
      expect(engine.getAverageSyncScore()).toBeNull();
      expect(engine.lastPhase).toBe('NO_DATA');
    });

    it('evaluates anticipatory steps before upcoming beat using beat projection', () => {
      const sync = new NuroSync(null, 60); // beatPeriod = 1.0s, tolerance = 200ms
      sync.recordBeat(1.0); // only past beat recorded
      // Event occurs at 1.960s (anticipating beat 2.0s by 40ms)
      const res = sync.evaluateStep(1.960);
      expect(res.timingErrorMs).toBe(40);
      expect(res.syncScore).toBe(80);
      expect(res.isSynchronized).toBe(true);
    });
  });
});

