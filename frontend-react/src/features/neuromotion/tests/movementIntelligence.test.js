import { describe, it, expect, beforeEach } from 'vitest';
import { MovementIntelligence, MOVEMENT_INTELLIGENCE_CONFIG } from '../core/MovementIntelligence';
import { LANDMARKS } from '../core/PoseTypes';

describe('Movement Intelligence & Rhythm Intelligence Test Suite (P1)', () => {
  let mi;

  // Helper: Generates a valid 33-landmark pose structure
  const createBaseLandmarks = (kneeBendOffset = 0, yOffset = 0, leftScale = 1.0, rightScale = 1.0) => {
    const lms = Array(33).fill(null).map(() => ({
      x: 0.5,
      y: 0.5 + yOffset,
      z: 0.0,
      visibility: 0.95,
      presence: 0.95
    }));

    // Torso anchor
    lms[LANDMARKS.LEFT_SHOULDER] = { x: 0.45, y: 0.3 + yOffset, z: 0.0, visibility: 0.95 };
    lms[LANDMARKS.RIGHT_SHOULDER] = { x: 0.55, y: 0.3 + yOffset, z: 0.0, visibility: 0.95 };
    lms[LANDMARKS.LEFT_HIP] = { x: 0.46, y: 0.55 + yOffset, z: 0.0, visibility: 0.95 };
    lms[LANDMARKS.RIGHT_HIP] = { x: 0.54, y: 0.55 + yOffset, z: 0.0, visibility: 0.95 };

    // Legs
    // Left leg
    lms[LANDMARKS.LEFT_KNEE] = {
      x: 0.46 + (kneeBendOffset * 0.1 * leftScale),
      y: 0.72 + yOffset,
      z: (kneeBendOffset * 0.1 * leftScale),
      visibility: 0.95
    };
    lms[LANDMARKS.LEFT_ANKLE] = {
      x: 0.46,
      y: 0.88 + yOffset - (kneeBendOffset * 0.15 * leftScale),
      z: 0.0,
      visibility: 0.95
    };
    lms[LANDMARKS.LEFT_HEEL] = { x: 0.45, y: 0.90 + yOffset, z: 0.0, visibility: 0.95 };
    lms[LANDMARKS.LEFT_FOOT_INDEX] = { x: 0.46, y: 0.92 + yOffset, z: 0.0, visibility: 0.95 };

    // Right leg
    lms[LANDMARKS.RIGHT_KNEE] = {
      x: 0.54 - (kneeBendOffset * 0.1 * rightScale),
      y: 0.72 + yOffset,
      z: (kneeBendOffset * 0.1 * rightScale),
      visibility: 0.95
    };
    lms[LANDMARKS.RIGHT_ANKLE] = {
      x: 0.54,
      y: 0.88 + yOffset - (kneeBendOffset * 0.15 * rightScale),
      z: 0.0,
      visibility: 0.95
    };
    lms[LANDMARKS.RIGHT_HEEL] = { x: 0.55, y: 0.90 + yOffset, z: 0.0, visibility: 0.95 };
    lms[LANDMARKS.RIGHT_FOOT_INDEX] = { x: 0.54, y: 0.92 + yOffset, z: 0.0, visibility: 0.95 };

    return lms;
  };

  beforeEach(() => {
    mi = new MovementIntelligence();
  });

  // TEST 1: User stationary -> movementState = IDLE
  it('TEST 1: correctly classifies stationary user with diffScore < MOTION_THRESHOLD as IDLE', () => {
    let state;
    for (let i = 0; i < 20; i++) {
      const lms = createBaseLandmarks();
      state = mi.processFrame(lms, 5, i * 0.033); // diffScore = 5 < 12
    }
    expect(state.state).toBe('IDLE');
    expect(state.movement.velocity).toBeCloseTo(0.0, 1);
  });

  // TEST 2: Slow movement -> ACTIVE with lower velocity
  it('TEST 2: detects ACTIVE state during slow motion and reports lower velocity than rapid motion', () => {
    let slowState;
    for (let i = 0; i < 20; i++) {
      const lms = createBaseLandmarks(i * 0.01, i * 0.005);
      slowState = mi.processFrame(lms, 25, i * 0.033);
    }
    expect(slowState.state).toBe('ACTIVE');

    const fastMi = new MovementIntelligence();
    let fastState;
    for (let i = 0; i < 20; i++) {
      const lms = createBaseLandmarks(i * 0.08, i * 0.04);
      fastState = fastMi.processFrame(lms, 65, i * 0.033);
    }

    expect(fastState.state).toBe('ACTIVE');
    expect(slowState.movement.velocity).toBeLessThan(fastState.movement.velocity);
  });

  // TEST 3: Smooth repeated movement -> higher smoothness and consistency
  it('TEST 3: yields high smoothness and consistency on smooth sinusoidal repeated cycles', () => {
    let state;
    // Simulate 60 frames (~2s) of pure sinusoidal smooth motion
    for (let i = 0; i < 60; i++) {
      const t = i * 0.033;
      const wave = Math.sin(t * 2 * Math.PI * 1.0); // 1 Hz smooth cycle
      const lms = createBaseLandmarks(wave * 0.5, wave * 0.05);
      state = mi.processFrame(lms, 30, t);
    }
    expect(state.state).toBe('ACTIVE');
    expect(state.movement.smoothness).toBeGreaterThan(0.70);
    expect(state.movement.quality).toBeGreaterThan(0.60);
  });

  // TEST 4: Irregular movement -> lower smoothness
  it('TEST 4: exhibits lower smoothness and quality on irregular/jerky movement', () => {
    let smoothState;
    for (let i = 0; i < 40; i++) {
      const t = i * 0.033;
      const smoothWave = Math.sin(t * Math.PI);
      smoothState = mi.processFrame(createBaseLandmarks(smoothWave * 0.3), 30, t);
    }

    const jerkyMi = new MovementIntelligence();
    let jerkyState;
    for (let i = 0; i < 40; i++) {
      const t = i * 0.033;
      // High-frequency jittery jerks
      const jerk = (i % 2 === 0 ? 0.8 : -0.8);
      jerkyState = jerkyMi.processFrame(createBaseLandmarks(jerk), 45, t);
    }

    expect(jerkyState.movement.smoothness).toBeLessThan(smoothState.movement.smoothness);
  });

  // TEST 5: Left/right movement -> symmetry responds to differences
  it('TEST 5: symmetry accurately reflects bilateral movement disparity', () => {
    // Bilateral symmetrical movement
    const symMi = new MovementIntelligence();
    let symState;
    for (let i = 0; i < 30; i++) {
      const t = i * 0.033;
      const offset = Math.sin(t * 4);
      symState = symMi.processFrame(createBaseLandmarks(offset, 0, 1.0, 1.0), 30, t);
    }
    expect(symState.movement.symmetry).toBeGreaterThan(0.85);

    // Asymmetric movement (left moving vigorously, right almost immobile)
    const asymMi = new MovementIntelligence();
    let asymState;
    for (let i = 0; i < 30; i++) {
      const t = i * 0.033;
      const offset = Math.sin(t * 4);
      asymState = asymMi.processFrame(createBaseLandmarks(offset, 0, 1.0, 0.15), 30, t);
    }
    expect(asymState.movement.symmetry).toBeLessThan(symState.movement.symmetry);
  });

  // TEST 6: Movement aligned with beat -> higher beatSyncScore
  it('TEST 6: yields high beatSyncScore when movement events align with expected beats', () => {
    const mockSync = {
      currentBpm: 60,
      beatTimestamps: [1.0, 2.0, 3.0, 4.0, 5.0]
    };

    // Movement event occurring exactly on beat 2.0s (0ms timing error)
    const state = mi.processFrame(createBaseLandmarks(0.5), 30, 2.000, mockSync);
    expect(state.rhythm.timingErrorMs).toBe(0);
    expect(state.rhythm.sync).toBe(1.0);
  });

  // TEST 7: Movement deliberately shifted away from beat -> lower beatSyncScore
  it('TEST 7: produces lower beatSyncScore when movement event is shifted away from beat', () => {
    const mockSync = {
      currentBpm: 60,
      beatTimestamps: [1.0, 2.0, 3.0]
    };

    // On-beat frame at 2.000s
    const onBeatState = mi.processFrame(createBaseLandmarks(0.5), 30, 2.000, mockSync);

    const offBeatMi = new MovementIntelligence();
    // Off-beat frame at 2.140s (140ms error; tolerance at 60 BPM is 200ms)
    const offBeatState = offBeatMi.processFrame(createBaseLandmarks(0.5), 30, 2.140, mockSync);

    expect(offBeatState.rhythm.timingErrorMs).toBe(140);
    expect(offBeatState.rhythm.sync).toBeLessThan(onBeatState.rhythm.sync);
    expect(offBeatState.rhythm.sync).toBeCloseTo(1.0 - (140 / 200), 2);
  });

  // TEST 8: Temporary landmark loss -> confidence decreases without application crash
  it('TEST 8: gracefully handles temporary landmark loss without crashing, lowering confidence', () => {
    // Warm up with valid landmarks
    for (let i = 0; i < 15; i++) {
      mi.processFrame(createBaseLandmarks(), 25, i * 0.033);
    }
    const goodConf = mi.getState().confidence;
    expect(goodConf).toBeGreaterThan(0.60);

    // Sudden landmark loss (null landmarks)
    expect(() => {
      mi.processFrame(null, 25, 0.60);
    }).not.toThrow();

    const lostState = mi.getState();
    expect(lostState.state).toBe('LOW_CONFIDENCE');
    expect(lostState.confidence).toBeLessThan(goodConf);
  });

  // TEST 9: BPM changes -> rhythm timing uses new BPM and dynamic tolerance
  it('TEST 9: adjusts dynamic tolerance window when BPM changes from 60 to 120', () => {
    const mockSync60 = { currentBpm: 60, beatTimestamps: [1.0, 2.0] };
    const mockSync120 = { currentBpm: 120, beatTimestamps: [1.0, 1.5, 2.0] };

    // At 60 BPM, beat interval is 1000ms, tolerance is 200ms. An 80ms error has sync = 1 - (80/200) = 0.60
    const state60 = mi.processFrame(createBaseLandmarks(), 30, 1.080, mockSync60);
    expect(state60.rhythm.sync).toBeCloseTo(0.60, 2);

    const mi2 = new MovementIntelligence();
    // At 120 BPM, beat interval is 500ms, tolerance is 100ms. An 80ms error has sync = 1 - (80/100) = 0.20
    const state120 = mi2.processFrame(createBaseLandmarks(), 30, 1.080, mockSync120);
    expect(state120.rhythm.sync).toBeCloseTo(0.20, 2);
  });

  // TEST 10: Long session -> temporal buffer remains bounded and memory does not grow
  it('TEST 10: maintains strictly bounded buffer size over long extended sessions (1000 frames)', () => {
    for (let i = 0; i < 1000; i++) {
      const t = i * 0.033; // 30fps simulation
      mi.processFrame(createBaseLandmarks(Math.sin(t)), 25, t);
    }
    // Buffer should strictly respect maxFrames (90 frames)
    expect(mi.buffer.length).toBeLessThanOrEqual(MOVEMENT_INTELLIGENCE_CONFIG.maxFrames);
    // Buffer time span should be bounded by bufferDurationSec (~2.5s)
    const oldestTimestamp = mi.buffer[0].timestamp;
    const latestTimestamp = mi.buffer[mi.buffer.length - 1].timestamp;
    expect(latestTimestamp - oldestTimestamp).toBeLessThanOrEqual(MOVEMENT_INTELLIGENCE_CONFIG.bufferDurationSec + 0.05);
  });

  // Integration Check: Single Unified Movement State Object Shape
  it('outputs the exact required structured Movement State Object', () => {
    for (let i = 0; i < 15; i++) {
      mi.processFrame(createBaseLandmarks(), 30, i * 0.033);
    }
    const state = mi.getState();

    expect(state).toHaveProperty('timestamp');
    expect(typeof state.timestamp).toBe('number');
    expect(['IDLE', 'ACTIVE', 'LOW_CONFIDENCE', 'WARMING_UP']).toContain(state.state);

    expect(state).toHaveProperty('movement');
    expect(state.movement).toHaveProperty('rom');
    expect(state.movement).toHaveProperty('velocity');
    expect(state.movement).toHaveProperty('smoothness');
    expect(state.movement).toHaveProperty('symmetry');
    expect(state.movement).toHaveProperty('consistency');
    expect(state.movement).toHaveProperty('quality');

    expect(state).toHaveProperty('rhythm');
    expect(state.rhythm).toHaveProperty('sync');
    expect(state.rhythm).toHaveProperty('timingErrorMs');
    expect(state.rhythm).toHaveProperty('missedBeats');

    expect(state).toHaveProperty('confidence');
    expect(typeof state.confidence).toBe('number');
  });

  // ADVERSARIAL TEST 1: NaN and Infinity injection
  it('ADVERSARIAL 1: gracefully handles NaN and Infinity landmark coordinates without producing NaN in state', () => {
    const corruptedLms = createBaseLandmarks();
    corruptedLms[23].x = NaN;
    corruptedLms[24].y = Infinity;
    corruptedLms[25].z = -Infinity;
    corruptedLms[26].visibility = NaN;

    let state;
    for (let i = 0; i < 20; i++) {
      state = mi.processFrame(corruptedLms, 30, i * 0.033);
    }

    expect(Number.isFinite(state.confidence)).toBe(true);
    expect(Number.isFinite(state.movement.rom)).toBe(true);
    expect(Number.isFinite(state.movement.velocity)).toBe(true);
    expect(Number.isFinite(state.movement.quality)).toBe(true);
    expect(state.rhythm.sync === null || Number.isFinite(state.rhythm.sync)).toBe(true);
    expect(Number.isNaN(state.rhythm.sync)).toBe(false);
  });

  // ADVERSARIAL TEST 2: Confidence boundary sweep
  it('ADVERSARIAL 2: transitions states accurately across confidence boundaries [0.0, 0.44, 0.45, 0.49, 0.50, 1.0]', () => {
    const boundaries = [0.0, 0.44, 0.45, 0.49, 0.50, 1.0];
    for (const conf of boundaries) {
      const lms = createBaseLandmarks();
      // Set key joints visibility
      [11, 12, 23, 24, 25, 26, 27, 28].forEach(idx => {
        lms[idx].visibility = conf;
      });

      mi.reset();
      let state;
      for (let i = 0; i < 20; i++) {
        state = mi.processFrame(lms, 30, i * 0.033);
      }
      expect(Number.isFinite(state.confidence)).toBe(true);
      expect(state.confidence).toBeGreaterThanOrEqual(0.0);
      expect(state.confidence).toBeLessThanOrEqual(1.0);
      if (state.confidence < 0.40) {
        expect(state.state).toBe('LOW_CONFIDENCE');
      }
    }
  });

  // ADVERSARIAL TEST 3: Null, undefined, and truncated landmark arrays
  it('ADVERSARIAL 3: handles null, undefined, or truncated landmark arrays gracefully', () => {
    expect(() => mi.processFrame(null, 0, 0.1)).not.toThrow();
    expect(() => mi.processFrame(undefined, 0, 0.2)).not.toThrow();
    expect(() => mi.processFrame([], 0, 0.3)).not.toThrow();
    expect(() => mi.processFrame([{ x: 0.5, y: 0.5 }], 0, 0.4)).not.toThrow();
    const state = mi.getState();
    expect(state.confidence).toBeLessThanOrEqual(0.40);
  });

  // ADVERSARIAL TEST 4: Alternating high-frequency noise & velocity bounding
  it('ADVERSARIAL 4: bounds velocity and prevents explosion under high-frequency alternating noise', () => {
    let state;
    for (let i = 0; i < 30; i++) {
      const toggle = (i % 2 === 0) ? 1.0 : -1.0;
      const lms = createBaseLandmarks(toggle * 5.0, toggle * 2.0);
      state = mi.processFrame(lms, 50, i * 0.033);
    }
    // Max physical velocity is clamped in calculateVelocityAndAcceleration
    expect(state.movement.velocity).toBeLessThanOrEqual(10.0);
    expect(Number.isFinite(state.movement.velocity)).toBe(true);
  });
});
