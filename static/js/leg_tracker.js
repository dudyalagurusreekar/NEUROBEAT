/**
 * Nuro-Beats Movement Kinematics & Measurement Engine (v2.0)
 * 
 * Pipeline:
 * RAW POSE -> POSE QUALITY GATE -> OUTLIER REJECTION -> ONE EURO FILTER ->
 * BODY-NORMALIZED KINEMATICS -> MULTI-SIGNAL EVENT DETECTOR -> VALIDATED MEASUREMENTS ->
 * UNCERTAINTY / QUALITY -> VERSIONED TELEMETRY -> SESSION METRICS
 * 
 * Compliant with MEASUREMENT_SPEC.md (schema_version: "2.0").
 */

const MOTION_THRESHOLD = 12;

/**
 * 1. Single Authoritative Landmark-to-Canvas Projection Function.
 * Translates normalized [0..1] MediaPipe landmark coordinates into pixel coordinates
 * with letterbox/pillarbox/mirror awareness.
 */
function transformLandmarkToCanvas(
    landmark,
    videoWidth,
    videoHeight,
    canvasWidth,
    canvasHeight,
    renderMode = 'fill',
    mirrored = false
) {
    if (!landmark) return null;

    const vW = videoWidth > 0 ? videoWidth : 640;
    const vH = videoHeight > 0 ? videoHeight : 480;
    const cW = canvasWidth > 0 ? canvasWidth : vW;
    const cH = canvasHeight > 0 ? canvasHeight : vH;

    let renderW = cW;
    let renderH = cH;
    let offsetX = 0;
    let offsetY = 0;

    if (renderMode === 'contain') {
        const videoRatio = vW / vH;
        const canvasRatio = cW / cH;
        if (canvasRatio > videoRatio) {
            renderH = cH;
            renderW = cH * videoRatio;
            offsetX = (cW - renderW) / 2;
            offsetY = 0;
        } else {
            renderW = cW;
            renderH = cW / videoRatio;
            offsetX = 0;
            offsetY = (cH - renderH) / 2;
        }
    } else if (renderMode === 'cover') {
        const videoRatio = vW / vH;
        const canvasRatio = cW / cH;
        if (canvasRatio > videoRatio) {
            renderW = cW;
            renderH = cW / videoRatio;
            offsetX = 0;
            offsetY = (cH - renderH) / 2;
        } else {
            renderH = cH;
            renderW = cH * videoRatio;
            offsetX = (cW - renderW) / 2;
            offsetY = 0;
        }
    }

    const normX = mirrored ? (1 - landmark.x) : landmark.x;
    const normY = landmark.y;

    return {
        x: offsetX + normX * renderW,
        y: offsetY + normY * renderH,
        z: landmark.z || 0,
        visibility: typeof landmark.visibility === 'number' ? landmark.visibility : 1.0,
        renderRect: {
            width: renderW,
            height: renderH,
            offsetX: offsetX,
            offsetY: offsetY
        }
    };
}

/**
 * 2. Adaptive One Euro Filter (Casiez et al., 2012)
 * Smooths jitter during stationary periods while preserving rapid velocity during foot strikes.
 */
class LowPassFilter {
    constructor(alpha = 1.0) {
        this.setAlpha(alpha);
        this.s = null;
    }

    setAlpha(alpha) {
        this.alpha = Math.max(0.0, Math.min(1.0, alpha));
    }

    filter(value) {
        if (this.s === null) {
            this.s = value;
        } else {
            this.s = this.alpha * value + (1.0 - this.alpha) * this.s;
        }
        return this.s;
    }

    hasLastRawValue() {
        return this.s !== null;
    }

    lastRawValue() {
        return this.s;
    }

    reset() {
        this.s = null;
    }
}

class OneEuroFilter {
    constructor(minCutoff = 1.2, beta = 0.008, dCutoff = 1.0) {
        this.minCutoff = minCutoff;
        this.beta = beta;
        this.dCutoff = dCutoff;
        this.xFilter = new LowPassFilter();
        this.dxFilter = new LowPassFilter();
        this.lastTime = null;
    }

    alpha(rate, cutoff) {
        const tau = 1.0 / (2.0 * Math.PI * cutoff);
        const te = 1.0 / rate;
        return 1.0 / (1.0 + tau / te);
    }

    filter(value, timestamp) {
        if (this.lastTime === null || timestamp === undefined) {
            this.lastTime = timestamp;
            return this.xFilter.filter(value);
        }

        const dt = Math.max(0.005, (timestamp - this.lastTime) / 1000.0);
        this.lastTime = timestamp;

        // Reset if frame gap is too large
        if (dt > 0.25) {
            this.reset();
            return this.xFilter.filter(value);
        }

        const rate = 1.0 / dt;
        const prevX = this.xFilter.hasLastRawValue() ? this.xFilter.lastRawValue() : value;
        const dx = (value - prevX) * rate;

        const edx = this.dxFilter.filter(dx);
        this.dxFilter.setAlpha(this.alpha(rate, this.dCutoff));

        const cutoff = this.minCutoff + this.beta * Math.abs(edx);
        this.xFilter.setAlpha(this.alpha(rate, cutoff));

        return this.xFilter.filter(value);
    }

    reset() {
        this.xFilter.reset();
        this.dxFilter.reset();
        this.lastTime = null;
    }
}

/**
 * 2B. Velocity-Aware Adaptive Landmark Filter & Outlier Rejector
 * Provides separate clean streams for display and measurement.
 * - Outlier Rejection: detects sudden single-frame spatial jumps (> 0.18 normalized dist) and clamps them.
 * - Velocity-Aware Filtering: dynamically adjusts smoothing factor alpha:
 *     alpha = 0.68 for slow/stationary postures (eradicates sub-pixel jitter)
 *     alpha = 0.90 for rapid intentional limb movement (prevents lag/phase delay)
 * Preserves raw landmarks for debugging/diagnostics.
 */
class AdaptiveLandmarkFilter {
    constructor(alphaMin = 0.68, alphaMax = 0.90) {
        this.alphaMin = alphaMin;
        this.alphaMax = alphaMax;
        this.prevFiltered = null;
        this.prevTime = null;
        this.outlierCounts = new Array(33).fill(0);
        this.latestRaw = null;
        this.latestDisplay = null;
        this.latestMeasurement = null;
        this.latestJitter = 0.0;
    }

    reset() {
        this.prevFiltered = null;
        this.prevTime = null;
        this.outlierCounts.fill(0);
        this.latestRaw = null;
        this.latestDisplay = null;
        this.latestMeasurement = null;
        this.latestJitter = 0.0;
    }

    filter(rawLandmarks, timestampMs) {
        if (!rawLandmarks || rawLandmarks.length < 33) {
            this.reset();
            return { rawLandmarks, displayLandmarks: rawLandmarks, measurementLandmarks: rawLandmarks, jitter: 0 };
        }

        this.latestRaw = rawLandmarks;
        const now = typeof timestampMs === 'number' ? timestampMs : performance.now();

        if (!this.prevFiltered || this.prevTime === null) {
            this.prevTime = now;
            this.prevFiltered = rawLandmarks.map(lm => lm ? { x: lm.x, y: lm.y, z: lm.z || 0, visibility: lm.visibility ?? 1 } : null);
            this.latestDisplay = this.prevFiltered.map(lm => lm ? { ...lm } : null);
            this.latestMeasurement = this.prevFiltered.map(lm => lm ? { ...lm } : null);
            return {
                rawLandmarks,
                displayLandmarks: this.latestDisplay,
                measurementLandmarks: this.latestMeasurement,
                jitter: 0
            };
        }

        const dt = Math.max(0.008, Math.min(0.25, (now - this.prevTime) / 1000.0));
        this.prevTime = now;

        const display = new Array(rawLandmarks.length);
        const measurement = new Array(rawLandmarks.length);
        let sumDisplacement = 0;
        let countDisplacement = 0;

        for (let i = 0; i < rawLandmarks.length; i++) {
            const raw = rawLandmarks[i];
            const prev = this.prevFiltered[i];

            if (!raw || !prev) {
                display[i] = raw ? { ...raw } : null;
                measurement[i] = raw ? { ...raw } : null;
                continue;
            }

            const rawVis = typeof raw.visibility === 'number' ? raw.visibility : 1.0;
            const dx = raw.x - prev.x;
            const dy = raw.y - prev.y;
            const dz = (raw.z || 0) - (prev.z || 0);
            const dist = Math.hypot(dx, dy);

            // Track landmark jitter on core body landmarks (hips 23,24, knees 25,26, ankles 27,28)
            if (i >= 23 && i <= 28) {
                sumDisplacement += dist;
                countDisplacement++;
            }

            // 1. Outlier Rejection Gating (Section 13)
            // Single-frame teleportation gate: > 0.18 normalized body space in dt
            let targetX = raw.x;
            let targetY = raw.y;
            let targetZ = raw.z || 0;

            const maxAllowedDist = 0.18;
            if (dist > maxAllowedDist && this.outlierCounts[i] < 2) {
                // Outlier detected: clamp displacement vector
                const scale = maxAllowedDist / dist;
                targetX = prev.x + dx * scale;
                targetY = prev.y + dy * scale;
                targetZ = prev.z + dz * scale;
                this.outlierCounts[i]++;
            } else {
                this.outlierCounts[i] = 0;
            }

            // 2. Velocity-Aware Adaptive Smoothing (Section 11, 12)
            // Normalized velocity in screens/sec
            const vel = dist / dt;
            let alpha = this.alphaMin;
            if (vel > 0.25) {
                // Adaptive ramp: fast movement gets lighter smoothing (higher alpha)
                const ramp = Math.min(1.0, (vel - 0.25) / 1.55);
                alpha = this.alphaMin + (this.alphaMax - this.alphaMin) * ramp;
            }

            // Exponential low-pass filter: filtered_t = alpha * raw_t + (1 - alpha) * filtered_(t-1)
            const filtX = alpha * targetX + (1.0 - alpha) * prev.x;
            const filtY = alpha * targetY + (1.0 - alpha) * prev.y;
            const filtZ = alpha * targetZ + (1.0 - alpha) * prev.z;

            const filteredObj = {
                x: filtX,
                y: filtY,
                z: filtZ,
                visibility: rawVis
            };

            this.prevFiltered[i] = filteredObj;
            display[i] = { ...filteredObj };
            measurement[i] = { ...filteredObj };
        }

        this.latestDisplay = display;
        this.latestMeasurement = measurement;
        this.latestJitter = countDisplacement > 0 ? (sumDisplacement / countDisplacement) : 0;

        return {
            rawLandmarks,
            displayLandmarks: display,
            measurementLandmarks: measurement,
            jitter: this.latestJitter
        };
    }

    getDisplayLandmarks() {
        return this.latestDisplay;
    }

    getMeasurementLandmarks() {
        return this.latestMeasurement;
    }
}

/**
 * 2C. FrameDiagnosticsCollector (Sections 4, 5, 29, 30)
 * Lightweight developer-only frame diagnostics object maintaining a bounded rolling 120-frame buffer.
 * Calculates authoritative frame timing statistics, jitter, dropped/out-of-order frames, and temporal stability.
 */
class FrameDiagnosticsCollector {
    constructor(capacity = 120) {
        this.capacity = capacity;
        this.frames = []; // Rolling buffer of frame telemetry records
        this.droppedFramesCount = 0;
        this.duplicateFramesCount = 0;
        this.outOfOrderCount = 0;
        this.frameGapsCount = 0;
        this.lastCaptureTimestamp = null;
        this.lastSummaryTime = 0;
        this.cachedSummary = null;

        // Cumulative totals
        this.totalCaptured = 0;
        this.totalProcessed = 0;
    }

    reset() {
        this.frames = [];
        this.droppedFramesCount = 0;
        this.duplicateFramesCount = 0;
        this.outOfOrderCount = 0;
        this.frameGapsCount = 0;
        this.lastCaptureTimestamp = null;
        this.lastSummaryTime = 0;
        this.cachedSummary = null;
    }

    recordCapture(frameId, captureTimestamp, captureDeltaMs, isFrameGap = false) {
        this.totalCaptured++;
        if (isFrameGap) {
            this.frameGapsCount++;
        }
        this.lastCaptureTimestamp = captureTimestamp;
    }

    recordDropped(frameId, timestamp, reason = 'MUTEX_BUSY') {
        this.droppedFramesCount++;
    }

    recordOutOfOrder(frameId, expectedFrameId) {
        this.outOfOrderCount++;
    }

    recordDuplicate(frameId) {
        this.duplicateFramesCount++;
    }

    recordFrameProcessed(record) {
        this.totalProcessed++;
        const entry = {
            frameId: record.frameId,
            captureTimestamp: record.captureTimestamp,
            processingStart: record.processingStart,
            processingEnd: record.processingEnd,
            captureDeltaMs: record.captureDeltaMs || (record.captureTimestamp - (this.lastCaptureTimestamp || record.captureTimestamp)),
            processingDurationMs: record.processingEnd - record.processingStart,
            inferenceDurationMs: record.inferenceDurationMs || 0,
            renderTimestamp: performance.now(),
            renderDeltaMs: 0,
            droppedFrames: this.droppedFramesCount,
            duplicateFrames: this.duplicateFramesCount,
            queueDepth: 0,
            landmarkTimestamp: record.captureTimestamp,
            resultAgeMs: record.resultAgeMs || (record.processingEnd - record.captureTimestamp),
            landmarkJitter: record.landmarkJitter || 0
        };

        this.frames.push(entry);
        if (this.frames.length > this.capacity) {
            this.frames.shift();
        }
    }

    getSummary() {
        const now = performance.now();
        if (now - this.lastSummaryTime < 80 && this.cachedSummary) {
            return this.cachedSummary;
        }
        this.lastSummaryTime = now;

        if (this.frames.length === 0) {
            return {
                effectiveFPS: 0,
                meanFrameIntervalMs: 0,
                medianFrameIntervalMs: 0,
                p95FrameIntervalMs: 0,
                maxFrameIntervalMs: 0,
                frameTimingJitterMs: 0,
                inferenceDurationMs: 0,
                resultAgeMs: 0,
                landmarkJitter: 0,
                droppedFrames: this.droppedFramesCount,
                duplicateFrames: this.duplicateFramesCount,
                outOfOrderCount: this.outOfOrderCount,
                temporalStability: 1.0
            };
        }

        const intervals = [];
        const inferences = [];
        const resultAges = [];
        const jitters = [];

        for (let i = 1; i < this.frames.length; i++) {
            const dt = this.frames[i].captureTimestamp - this.frames[i - 1].captureTimestamp;
            if (dt > 0 && dt < 500) {
                intervals.push(dt);
            }
        }
        for (const f of this.frames) {
            if (f.inferenceDurationMs > 0) inferences.push(f.inferenceDurationMs);
            if (f.resultAgeMs > 0) resultAges.push(f.resultAgeMs);
            if (f.landmarkJitter > 0) jitters.push(f.landmarkJitter);
        }

        const mean = (arr) => arr.length > 0 ? (arr.reduce((a, b) => a + b, 0) / arr.length) : 0;
        const median = (arr) => {
            if (arr.length === 0) return 0;
            const s = [...arr].sort((a, b) => a - b);
            const m = Math.floor(s.length / 2);
            return s.length % 2 === 0 ? (s[m - 1] + s[m]) / 2 : s[m];
        };
        const p95 = (arr) => {
            if (arr.length === 0) return 0;
            const s = [...arr].sort((a, b) => a - b);
            const idx = Math.min(s.length - 1, Math.floor(s.length * 0.95));
            return s[idx];
        };
        const stdDev = (arr, m) => {
            if (arr.length < 2) return 0;
            const v = arr.reduce((acc, val) => acc + Math.pow(val - m, 2), 0) / (arr.length - 1);
            return Math.sqrt(v);
        };

        const meanInterval = mean(intervals);
        const medInterval = median(intervals);
        const p95Interval = p95(intervals);
        const maxInterval = intervals.length > 0 ? Math.max(...intervals) : 0;
        const timingJitter = stdDev(intervals, meanInterval);

        const meanInference = mean(inferences);
        const meanAge = mean(resultAges);
        const ageVariance = stdDev(resultAges, meanAge);
        const meanLandmarkJitter = mean(jitters);

        const effectiveFPS = meanInterval > 0 ? (1000.0 / meanInterval) : 0;

        // Developer-only Temporal Stability Score (Section 30)
        // temporalStability = 1 - normalized(landmarkJitter + frameTimingJitter + resultAgeVariance)
        const normJitter = Math.min(1.0, meanLandmarkJitter / 0.04);
        const normTimingJitter = Math.min(1.0, timingJitter / 25.0);
        const normAgeVar = Math.min(1.0, ageVariance / 40.0);
        const penalty = (0.35 * normJitter) + (0.40 * normTimingJitter) + (0.25 * normAgeVar);
        const temporalStability = Number(Math.max(0.0, Math.min(1.0, 1.0 - penalty)).toFixed(2));

        this.cachedSummary = {
            effectiveFPS: Number(effectiveFPS.toFixed(1)),
            meanFrameIntervalMs: Number(meanInterval.toFixed(1)),
            medianFrameIntervalMs: Number(medInterval.toFixed(1)),
            p95FrameIntervalMs: Number(p95Interval.toFixed(1)),
            maxFrameIntervalMs: Number(maxInterval.toFixed(1)),
            frameTimingJitterMs: Number(timingJitter.toFixed(1)),
            inferenceDurationMs: Number(meanInference.toFixed(1)),
            resultAgeMs: Number(meanAge.toFixed(1)),
            landmarkJitter: Number(meanLandmarkJitter.toFixed(4)),
            droppedFrames: this.droppedFramesCount,
            duplicateFrames: this.duplicateFramesCount,
            outOfOrderCount: this.outOfOrderCount,
            temporalStability
        };

        return this.cachedSummary;
    }
}

/**
 * 3. Pose Quality Gate
 * Evaluates 10 critical gait landmarks, lower-body completeness, framing, and feet truncation.
 */
class PoseQualityGate {
    constructor() {
        this.recentFrames = []; // Ring buffer for tracking coverage (size 30)
        this.criticalFrames = [];
    }

    evaluate(landmarks, timestamp) {
        if (!landmarks || landmarks.length < 33) {
            this.recordFrame(false, false);
            return {
                state: 'LOST',
                passed: false,
                confidence: 0,
                trackingCoverage: this.getCoverage(),
                criticalCoverage: this.getCriticalCoverage(),
                feetInFrame: false,
                reason: "Position your full body in the camera frame."
            };
        }

        // 10 Critical Gait Landmarks:
        // Hips (23, 24), Knees (25, 26), Ankles (27, 28), Heels (29, 30), Foot Indices (31, 32)
        const leftHip = landmarks[23];
        const rightHip = landmarks[24];
        const leftKnee = landmarks[25];
        const rightKnee = landmarks[26];
        const leftAnkle = landmarks[27];
        const rightAnkle = landmarks[28];
        const leftHeel = landmarks[29];
        const rightHeel = landmarks[30];
        const leftToe = landmarks[31];
        const rightToe = landmarks[32];

        const lowerLimbLandmarks = [
            leftHip, rightHip, leftKnee, rightKnee,
            leftAnkle, rightAnkle, leftHeel, rightHeel, leftToe, rightToe
        ];

        let sumVis = 0;
        let visibleCount = 0;
        let feetAtBottom = false;

        lowerLimbLandmarks.forEach(lm => {
            const v = lm?.visibility || 0;
            sumVis += v;
            if (v >= 0.35) visibleCount++;
            if (lm && lm.y >= 0.96) feetAtBottom = true; // Truncation within 4% of frame bottom
        });

        const meanVis = sumVis / 10.0;
        const lowerBodyComplete = visibleCount >= 8;
        const feetVisible = (leftAnkle?.visibility >= 0.35 || leftToe?.visibility >= 0.35) &&
                            (rightAnkle?.visibility >= 0.35 || rightToe?.visibility >= 0.35) &&
                            !feetAtBottom;

        this.recordFrame(true, lowerBodyComplete && feetVisible);

        const coverage = this.getCoverage();
        const criticalCoverage = this.getCriticalCoverage();

        // Quality State Evaluation
        let state = 'GOOD';
        let reason = null;
        let passed = true;

        if (feetAtBottom) {
            state = 'POOR';
            passed = false;
            reason = "Feet truncated at bottom edge — step back slightly so your feet are fully visible.";
        } else if (!feetVisible) {
            state = 'POOR';
            passed = false;
            reason = "Feet not clearly visible — adjust camera angle or lighting.";
        } else if (meanVis < 0.40 || criticalCoverage < 0.60) {
            state = 'FAIR';
            passed = false;
            reason = "Tracking quality low — check room lighting and camera framing.";
        } else if (meanVis >= 0.80 && criticalCoverage >= 0.90) {
            state = 'EXCELLENT';
            passed = true;
        } else {
            state = 'GOOD';
            passed = true;
        }

        return {
            state,
            passed,
            confidence: Number(meanVis.toFixed(2)),
            trackingCoverage: coverage,
            criticalCoverage: criticalCoverage,
            feetInFrame: feetVisible,
            reason
        };
    }

    recordFrame(valid, critical) {
        this.recentFrames.push(valid ? 1 : 0);
        this.criticalFrames.push(critical ? 1 : 0);
        if (this.recentFrames.length > 30) this.recentFrames.shift();
        if (this.criticalFrames.length > 30) this.criticalFrames.shift();
    }

    getCoverage() {
        if (this.recentFrames.length === 0) return 1.0;
        const sum = this.recentFrames.reduce((a, b) => a + b, 0);
        return Number((sum / this.recentFrames.length).toFixed(2));
    }

    getCriticalCoverage() {
        if (this.criticalFrames.length === 0) return 1.0;
        const sum = this.criticalFrames.reduce((a, b) => a + b, 0);
        return Number((sum / this.criticalFrames.length).toFixed(2));
    }

    reset() {
        this.recentFrames = [];
        this.criticalFrames = [];
    }
}

/**
 * 4. Multi-Signal Gait Event & Heel-Strike Detector
 * Evaluates vertical trajectory, velocity, heel deceleration, knee flexion, and bilateral alternation.
 */
class MultiSignalStepDetector {
    constructor() {
        this.lastLeftStepTime = 0;
        this.lastRightStepTime = 0;
        this.leftDuration = 0;
        this.rightDuration = 0;
        this.leftStepsCount = 0;
        this.rightStepsCount = 0;
        this.validSteps = 0;
        this.rejectedCandidates = 0;
        this.minStepCooldown = 350; // ms

        // Trajectory tracking
        this.prevLeftAnkleY = null;
        this.prevRightAnkleY = null;
        this.prevLeftHeelY = null;
        this.prevRightHeelY = null;
        this.prevTime = null;

        // Velocities
        this.leftVel = 0;
        this.rightVel = 0;
        this.prevLeftVel = 0;
        this.prevRightVel = 0;

        // Step timestamps
        this.stepTimestamps = [];
        this.leftStepTimestamps = [];
        this.rightStepTimestamps = [];
        this.stepIntervals = [];

        // Last detected event type per side
        this.lastLeftEventType = 'IDLE';
        this.lastRightEventType = 'IDLE';
    }

    evaluate(normCoords, now, qualityConfidence) {
        if (!normCoords || this.prevTime === null) {
            this.prevTime = now;
            this.updatePreviousCoords(normCoords);
            return { stepDetected: false, stepLeg: 'none', eventType: 'IDLE' };
        }

        const dt = Math.max(0.01, (now - this.prevTime) / 1000.0);
        this.prevTime = now;

        const leftAnkleY = normCoords.leftAnkle.y;
        const rightAnkleY = normCoords.rightAnkle.y;
        const leftHeelY = normCoords.leftHeel.y;
        const rightHeelY = normCoords.rightHeel.y;

        // Biomechanical sign convention: Y is positive upwards.
        // Downward motion (toward floor impact) has negative velocity.
        this.leftVel = (leftAnkleY - (this.prevLeftAnkleY !== null ? this.prevLeftAnkleY : leftAnkleY)) / dt;
        this.rightVel = (rightAnkleY - (this.prevRightAnkleY !== null ? this.prevRightAnkleY : rightAnkleY)) / dt;

        let leftCandidate = false;
        let rightCandidate = false;
        let leftScore = 0;
        let rightScore = 0;

        // Multi-signal evidence accumulation for Left Foot
        if (this.prevLeftVel < -0.35 && this.leftVel >= -0.35) {
            // Deceleration towards foot contact
            const trajEvidence = Math.min(1.0, Math.abs(leftAnkleY - normCoords.baselineFootY) / 0.15);
            const velEvidence = Math.min(1.0, Math.abs(this.prevLeftVel) / 1.5);
            const heelEvidence = (this.prevLeftHeelY !== null && leftHeelY <= this.prevLeftHeelY) ? 0.9 : 0.4;
            const kneeEvidence = normCoords.leftKneeAngle < 165 ? 0.9 : 0.5; // flexed knee context
            const tempoEvidence = (now - this.lastLeftStepTime) >= this.minStepCooldown ? 1.0 : 0.0;

            leftScore = (
                0.25 * trajEvidence +
                0.20 * velEvidence +
                0.15 * heelEvidence +
                0.15 * kneeEvidence +
                0.15 * tempoEvidence +
                0.10 * qualityConfidence
            );

            if (leftScore >= 0.55 && (now - this.lastLeftStepTime) >= this.minStepCooldown) {
                leftCandidate = true;
            } else if (leftScore > 0.4) {
                this.rejectedCandidates++;
            }
        }

        // Multi-signal evidence accumulation for Right Foot
        if (this.prevRightVel < -0.35 && this.rightVel >= -0.35) {
            const trajEvidence = Math.min(1.0, Math.abs(rightAnkleY - normCoords.baselineFootY) / 0.15);
            const velEvidence = Math.min(1.0, Math.abs(this.prevRightVel) / 1.5);
            const heelEvidence = (this.prevRightHeelY !== null && rightHeelY <= this.prevRightHeelY) ? 0.9 : 0.4;
            const kneeEvidence = normCoords.rightKneeAngle < 165 ? 0.9 : 0.5;
            const tempoEvidence = (now - this.lastRightStepTime) >= this.minStepCooldown ? 1.0 : 0.0;

            rightScore = (
                0.25 * trajEvidence +
                0.20 * velEvidence +
                0.15 * heelEvidence +
                0.15 * kneeEvidence +
                0.15 * tempoEvidence +
                0.10 * qualityConfidence
            );

            if (rightScore >= 0.55 && (now - this.lastRightStepTime) >= this.minStepCooldown) {
                rightCandidate = true;
            } else if (rightScore > 0.4) {
                this.rejectedCandidates++;
            }
        }

        this.prevLeftVel = this.leftVel;
        this.prevRightVel = this.rightVel;
        this.updatePreviousCoords(normCoords);

        let stepDetected = false;
        let stepLeg = 'none';
        let eventType = 'STEP_EVENT';

        if (leftCandidate && rightCandidate) {
            stepDetected = true;
            stepLeg = leftScore >= rightScore ? 'left' : 'right';
        } else if (leftCandidate) {
            stepDetected = true;
            stepLeg = 'left';
        } else if (rightCandidate) {
            stepDetected = true;
            stepLeg = 'right';
        }

        if (stepDetected) {
            this.validSteps++;
            const tSec = now / 1000.0;

            if (stepLeg === 'left') {
                if (this.lastLeftStepTime > 0) {
                    this.leftDuration = now - this.lastLeftStepTime;
                }
                this.lastLeftStepTime = now;
                this.leftStepsCount++;
                this.leftStepTimestamps.push(tSec);
                if (this.leftStepTimestamps.length > 50) this.leftStepTimestamps.shift();
            } else {
                if (this.lastRightStepTime > 0) {
                    this.rightDuration = now - this.lastRightStepTime;
                }
                this.lastRightStepTime = now;
                this.rightStepsCount++;
                this.rightStepTimestamps.push(tSec);
                if (this.rightStepTimestamps.length > 50) this.rightStepTimestamps.shift();
            }

            if (this.stepTimestamps.length > 0) {
                const prevStepT = this.stepTimestamps[this.stepTimestamps.length - 1];
                const dtStep = tSec - prevStepT;
                if (dtStep >= 0.25 && dtStep <= 3.5) {
                    this.stepIntervals.push(dtStep);
                    if (this.stepIntervals.length > 40) this.stepIntervals.shift();
                }
            }
            this.stepTimestamps.push(tSec);
            if (this.stepTimestamps.length > 60) this.stepTimestamps.shift();

            // Distinguish HEEL_STRIKE_CANDIDATE if heel leads impact
            if ((stepLeg === 'left' && leftScore > 0.75) || (stepLeg === 'right' && rightScore > 0.75)) {
                eventType = 'HEEL_STRIKE_CANDIDATE';
            }
        }

        return {
            stepDetected,
            stepLeg,
            eventType,
            leftScore,
            rightScore
        };
    }

    updatePreviousCoords(normCoords) {
        if (!normCoords) return;
        this.prevLeftAnkleY = normCoords.leftAnkle?.y ?? null;
        this.prevRightAnkleY = normCoords.rightAnkle?.y ?? null;
        this.prevLeftHeelY = normCoords.leftHeel?.y ?? null;
        this.prevRightHeelY = normCoords.rightHeel?.y ?? null;
    }

    reset() {
        this.lastLeftStepTime = 0;
        this.lastRightStepTime = 0;
        this.leftDuration = 0;
        this.rightDuration = 0;
        this.leftStepsCount = 0;
        this.rightStepsCount = 0;
        this.validSteps = 0;
        this.rejectedCandidates = 0;
        this.prevLeftAnkleY = null;
        this.prevRightAnkleY = null;
        this.prevLeftHeelY = null;
        this.prevRightHeelY = null;
        this.prevTime = null;
        this.stepTimestamps = [];
        this.leftStepTimestamps = [];
        this.rightStepTimestamps = [];
        this.stepIntervals = [];
    }
}

/**
 * 5. Robust Cadence & Interval Estimator
 */
class CadenceEstimator {
    constructor() {
        this.stepIntervals = [];
    }

    update(intervals) {
        this.stepIntervals = intervals || [];
    }

    getMetrics() {
        if (this.stepIntervals.length < 2) {
            return {
                valid: false,
                cadence_spm: 0,
                cadence_median_spm: 0,
                step_interval_mean_s: 0,
                step_interval_median_s: 0,
                step_interval_sd_s: 0,
                cadence_cv: 0,
                cadence_mad: 0,
                confidence: 0
            };
        }

        const sorted = [...this.stepIntervals].sort((a, b) => a - b);
        const mid = Math.floor(sorted.length / 2);
        const medianInterval = sorted.length % 2 === 0 ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid];

        const sum = this.stepIntervals.reduce((a, b) => a + b, 0);
        const meanInterval = sum / this.stepIntervals.length;

        const variance = this.stepIntervals.reduce((a, b) => a + Math.pow(b - meanInterval, 2), 0) / (this.stepIntervals.length - 1);
        const sdInterval = Math.sqrt(variance);

        const cv = meanInterval > 0 ? (sdInterval / meanInterval) : 0;

        // Median Absolute Deviation (MAD)
        const deviations = sorted.map(v => Math.abs(v - medianInterval)).sort((a, b) => a - b);
        const mad = deviations.length % 2 === 0 ? (deviations[mid - 1] + deviations[mid]) / 2 : deviations[mid];

        const cadenceSpm = medianInterval > 0 ? Number((60.0 / medianInterval).toFixed(1)) : 0;
        const meanCadenceSpm = meanInterval > 0 ? Number((60.0 / meanInterval).toFixed(1)) : 0;

        // Confidence: higher when intervals are regular (low CV)
        const confidence = Number(Math.max(0.2, Math.min(1.0, 1.0 - cv * 2.0)).toFixed(2));

        return {
            valid: true,
            cadence_spm: meanCadenceSpm,
            cadence_median_spm: cadenceSpm,
            step_interval_mean_s: Number(meanInterval.toFixed(3)),
            step_interval_median_s: Number(medianInterval.toFixed(3)),
            step_interval_sd_s: Number(sdInterval.toFixed(3)),
            cadence_cv: Number(cv.toFixed(3)),
            cadence_mad: Number(mad.toFixed(3)),
            confidence
        };
    }
}

/**
 * 6. Multi-Dimensional Gait Symmetry Calculator
 */
class GaitSymmetryCalculator {
    constructor() {
        this.symmetryHistory = [];
    }

    /**
     * Standard temporal symmetry formulation:
     * SI = ((T_right - T_left) / (0.5 * (|T_right| + |T_left|))) * 100
     * temporal_asymmetry_pct = abs(SI)
     */
    calculate(leftDuration, rightDuration) {
        if (!leftDuration || !rightDuration || leftDuration <= 0 || rightDuration <= 0) {
            return {
                valid: false,
                temporal_asymmetry_pct: 0,
                symmetry_index: 0,
                display_symmetry_pct: 100
            };
        }

        const denom = 0.5 * (Math.abs(rightDuration) + Math.abs(leftDuration));
        if (denom === 0) return { valid: false, temporal_asymmetry_pct: 0, symmetry_index: 0, display_symmetry_pct: 100 };

        const si = ((rightDuration - leftDuration) / denom) * 100;
        const temporalAsymmetry = Number(Math.abs(si).toFixed(1));

        // Rolling display percentage for UI: 100% minus asymmetry
        const displayScore = Math.max(40, Math.min(100, Math.round(100 - temporalAsymmetry)));

        this.symmetryHistory.push(displayScore);
        if (this.symmetryHistory.length > 20) this.symmetryHistory.shift();

        const avgScore = Math.round(this.symmetryHistory.reduce((a, b) => a + b, 0) / this.symmetryHistory.length);

        return {
            valid: true,
            temporal_asymmetry_pct: temporalAsymmetry,
            symmetry_index: Number(si.toFixed(1)),
            display_symmetry_pct: avgScore
        };
    }

    reset() {
        this.symmetryHistory = [];
    }
}

/**
 * 7. Postural Balance & Stability Evaluator
 */
class BalanceKinematicsEvaluator {
    constructor() {
        this.swayHistory = []; // lateral displacement values
        this.pathLength = 0;
        this.lastSway = null;
        this.lastTime = null;
        this.stabilityIndex = 100;
        this.lastWeightDistribution = 'Centered';
    }

    evaluate(landmarks, now) {
        if (!landmarks || landmarks.length < 33) {
            return { valid: false, stabilityIndex: 100, weightDistribution: 'Centered' };
        }

        const leftShoulder = landmarks[11];
        const rightShoulder = landmarks[12];
        const leftHip = landmarks[23];
        const rightHip = landmarks[24];
        const leftAnkle = landmarks[27];
        const rightAnkle = landmarks[28];

        if (!leftShoulder || !rightShoulder || !leftHip || !rightHip) {
            return { valid: false, stabilityIndex: 100, weightDistribution: 'Centered' };
        }

        const midShouldersX = (leftShoulder.x + rightShoulder.x) / 2;
        const midHipsX = (leftHip.x + rightHip.x) / 2;
        const trunkCenterX = (midShouldersX + midHipsX) / 2;
        const baseCenterX = ((leftAnkle?.x || midHipsX) + (rightAnkle?.x || midHipsX)) / 2;

        // Lateral displacement proxy: trunk center minus base of support center
        const lateralSway = Number((trunkCenterX - baseCenterX).toFixed(4));

        this.swayHistory.push(lateralSway);
        if (this.swayHistory.length > 40) this.swayHistory.shift();

        if (this.lastSway !== null) {
            this.pathLength += Math.abs(lateralSway - this.lastSway);
        }
        this.lastSway = lateralSway;

        // Window RMS Sway
        const meanSway = this.swayHistory.reduce((a, b) => a + b, 0) / this.swayHistory.length;
        const sumSq = this.swayHistory.reduce((a, b) => a + Math.pow(b - meanSway, 2), 0);
        const rmsSway = Math.sqrt(sumSq / this.swayHistory.length);

        // Postural Stability Index: 100 - (2.5 * RMS sway * 100)
        this.stabilityIndex = Math.max(40, Math.min(100, Math.round(100 - (rmsSway * 250))));

        let weightDist = 'Centered';
        if (lateralSway < -0.035) {
            weightDist = 'Shifted Left';
        } else if (lateralSway > 0.035) {
            weightDist = 'Shifted Right';
        }
        this.lastWeightDistribution = weightDist;

        return {
            valid: true,
            lateral_sway: lateralSway,
            sway_rms: Number(rmsSway.toFixed(4)),
            path_length: Number(this.pathLength.toFixed(3)),
            stability_index: this.stabilityIndex,
            weight_distribution: weightDist
        };
    }

    reset() {
        this.swayHistory = [];
        this.pathLength = 0;
        this.lastSway = null;
        this.lastTime = null;
        this.stabilityIndex = 100;
        this.lastWeightDistribution = 'Centered';
    }
}

/**
 * 8. Primary Kinematics Tracker Orchestrator
 */
class LegKinematicsTracker {
    constructor() {
        this.videoElement = null;
        this.canvasElement = null;
        this.overlayCanvas = null;
        this.overlayCtx = null;
        this.cameraStage = null;
        this.poseModel = null;
        this.isModelLoaded = false;
        this.isProcessing = false;
        this.resizeObserver = null;

        // Protocol state: READY -> CALIBRATING -> TRACKING -> MEASUREMENT -> COMPLETE
        this.protocolState = 'READY';
        this.trackingMode = 'gait';

        // Authoritative Subcomponents
        this.qualityGate = new PoseQualityGate();
        this.stepDetector = new MultiSignalStepDetector();
        this.cadenceEstimator = new CadenceEstimator();
        this.symmetryCalculator = new GaitSymmetryCalculator();
        this.balanceEvaluator = new BalanceKinematicsEvaluator();
        this.adaptiveFilter = new AdaptiveLandmarkFilter();
        this.diagnostics = new FrameDiagnosticsCollector();
        window.frameDiagnostics = this.diagnostics;

        // Frame timing & synchronization contract (Sections 7, 8, 9, 10)
        this.latestAcceptedFrameId = -1;
        this.latestPoseTimestamp = 0;
        this.inFlightFrameId = null;
        this.inFlightTimestamp = null;
        this.inFlightStart = 0;
        this.internalFrameCounter = 0;
        this.latestQuality = null;
        this.lastUiUpdateTime = 0;

        // 1 Euro Filters for key joints (X, Y)
        this.filters = {
            leftAnkleX: new OneEuroFilter(1.2, 0.008),
            leftAnkleY: new OneEuroFilter(1.2, 0.008),
            rightAnkleX: new OneEuroFilter(1.2, 0.008),
            rightAnkleY: new OneEuroFilter(1.2, 0.008),
            leftHeelY: new OneEuroFilter(1.2, 0.008),
            rightHeelY: new OneEuroFilter(1.2, 0.008),
            pelvisX: new OneEuroFilter(0.8, 0.003),
            pelvisY: new OneEuroFilter(0.8, 0.003)
        };

        // Smoothed body scale
        this.smoothedScale = 0.25;
        this.baselineFootY = -1.8;

        // Diagnostics
        this.fps = 0;
        this.latencyMs = 0;
        this.lastFrameTime = performance.now();
        this.frameCount = 0;

        // Authoritative Render Rect
        this.renderRect = {
            width: 640,
            height: 480,
            videoWidth: 640,
            videoHeight: 480,
            scale: 1,
            offsetX: 0,
            offsetY: 0,
            mirrored: false
        };

        this.latestResults = null;
        this.averageSymmetry = 100;
        this.balanceStabilityScore = 100;

        // Bindings
        this.handlePoseResults = this.handlePoseResults.bind(this);
        this.updateDimensions = this.updateDimensions.bind(this);
    }

    setTrackingMode(mode) {
        this.trackingMode = mode === 'balance' ? 'balance' : 'gait';
        console.log(`[TRACKING_MODE] Active sensing mode set to: ${this.trackingMode}`);
    }

    async initialize(videoEl, canvasEl, overlayCanvasEl) {
        this.videoElement = videoEl || document.getElementById('cameraFeed');
        this.canvasElement = canvasEl || document.getElementById('cameraCanvas');
        this.overlayCanvas = overlayCanvasEl || document.getElementById('cameraOverlayCanvas');
        this.cameraStage = document.getElementById('cameraStage') || document.getElementById('cameraBox');

        if (this.overlayCanvas) {
            this.overlayCtx = this.overlayCanvas.getContext('2d');
        }

        if (this.videoElement) {
            this.videoElement.addEventListener('loadedmetadata', this.updateDimensions);
            this.videoElement.addEventListener('resize', this.updateDimensions);
        }

        if (this.cameraStage && typeof ResizeObserver !== 'undefined') {
            this.resizeObserver = new ResizeObserver(() => {
                this.updateDimensions();
            });
            this.resizeObserver.observe(this.cameraStage);
        }

        this.updateDimensions();

        if (typeof Pose !== 'undefined') {
            try {
                this.poseModel = new Pose({
                    locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}`
                });
                this.poseModel.setOptions({
                    modelComplexity: 1,
                    smoothLandmarks: false, // We use our superior configurable OneEuroFilter
                    minDetectionConfidence: 0.5,
                    minTrackingConfidence: 0.5
                });
                this.poseModel.onResults(this.handlePoseResults);
                this.isModelLoaded = true;
                console.log(`[POSE_INIT] Model initialized for mode=${this.trackingMode}`);
            } catch (err) {
                console.warn('[POSE_INIT] MediaPipe initialization warning:', err);
            }
        }
    }

    updateDimensions() {
        if (!this.videoElement) return;

        const vW = this.videoElement.videoWidth;
        const vH = this.videoElement.videoHeight;

        if (vW > 0 && vH > 0) {
            if (this.overlayCanvas) {
                if (this.overlayCanvas.width !== vW || this.overlayCanvas.height !== vH) {
                    this.overlayCanvas.width = vW;
                    this.overlayCanvas.height = vH;
                }
            }

            if (this.renderRect.videoWidth !== vW || this.renderRect.videoHeight !== vH) {
                if (this.cameraStage) {
                    this.cameraStage.style.setProperty('--camera-aspect-ratio', `${vW} / ${vH}`);
                }
                this.renderRect.videoWidth = vW;
                this.renderRect.videoHeight = vH;
                this.renderRect.width = vW;
                this.renderRect.height = vH;
            }
        }
    }

    handlePoseResults(results) {
        const arrivalTime = performance.now();
        const fId = this.inFlightFrameId;
        const tMs = this.inFlightTimestamp;

        this.isProcessing = false;

        // Prevent Out-Of-Order Results (Section 8)
        if (typeof fId === 'number' && fId < this.latestAcceptedFrameId) {
            if (this.diagnostics) {
                this.diagnostics.recordOutOfOrder(fId, this.latestAcceptedFrameId);
            }
            return;
        }

        this.latestAcceptedFrameId = fId;
        this.latestPoseTimestamp = tMs || arrivalTime;
        this.latestResults = results;

        this.latencyMs = Number((arrivalTime - (this.inFlightStart || arrivalTime)).toFixed(1));
        this.frameCount++;
        if (this.frameCount % 10 === 0) {
            this.fps = Number((1000.0 / Math.max(1, this.latencyMs)).toFixed(1));
        }
        this.lastFrameTime = arrivalTime;
    }

    /**
     * Backward-compatible quality gate wrapper
     */
    checkQualityGate(landmarks) {
        return this.qualityGate.evaluate(landmarks, performance.now());
    }

    /**
     * Estimates anatomical movement & landmarks based on active mode
     * Complies with Section 8 (no out-of-order), Section 9 (single in-flight), Section 10 (no stale frames),
     * Section 11/12 (velocity-aware adaptive filtering), Section 13 (outlier rejection), Section 15 (display vs measurement).
     */
    async estimatePose(videoEl, frameId = null, timestampMs = null) {
        if (!videoEl || videoEl.readyState < 2) return null;

        const processingStart = performance.now();
        const fId = typeof frameId === 'number' ? frameId : ++this.internalFrameCounter;
        const tMs = typeof timestampMs === 'number' ? timestampMs : performance.now();

        // 1. Single In-Flight Inference check (Section 9)
        if (this.isProcessing) {
            if (this.diagnostics) {
                this.diagnostics.recordDropped(fId, tMs, 'INFERENCE_BUSY');
            }
            return {
                isNewResult: false,
                qualityPassed: this.latestQuality ? this.latestQuality.passed : false,
                confidence: this.latestQuality ? this.latestQuality.confidence : 0,
                resultAgeMs: Number((tMs - this.latestPoseTimestamp).toFixed(1))
            };
        }

        if (this.isModelLoaded && this.poseModel) {
            this.isProcessing = true;
            this.inFlightFrameId = fId;
            this.inFlightTimestamp = tMs;
            this.inFlightStart = processingStart;

            try {
                await this.poseModel.send({ image: videoEl });
            } catch (e) {
                console.warn('[POSE_INFERENCE] Error sending frame to model:', e);
                this.isProcessing = false;
                return null;
            }
        }

        const results = this.latestResults;
        if (!results || !results.poseLandmarks || results.poseLandmarks.length < 33) {
            return null;
        }

        // Verify this is a new result matching this frame
        if (this.latestAcceptedFrameId !== fId) {
            return {
                isNewResult: false,
                qualityPassed: false,
                confidence: 0,
                resultAgeMs: Number((tMs - this.latestPoseTimestamp).toFixed(1))
            };
        }

        const rawLm = results.poseLandmarks;
        const now = tMs;

        // 1. Run Pose Quality Gate
        const quality = this.qualityGate.evaluate(rawLm, now);
        this.latestQuality = quality;
        this.updateQualityBanner(quality);

        // 2. Adaptive Landmark Filtering (Velocity-Aware + Outlier Rejection)
        const filterResult = this.adaptiveFilter.filter(rawLm, now);
        const displayLm = filterResult.displayLandmarks;
        const measurementLm = filterResult.measurementLandmarks;

        // 3. Draw Skeleton Overlay using display landmarks
        this.drawSkeletonOverlay(displayLm);

        const processingEnd = performance.now();
        const inferenceDurationMs = processingEnd - processingStart;

        if (this.diagnostics) {
            this.diagnostics.recordFrameProcessed({
                frameId: fId,
                captureTimestamp: tMs,
                processingStart,
                processingEnd,
                inferenceDurationMs,
                resultAgeMs: processingEnd - tMs,
                landmarkJitter: filterResult.jitter
            });
        }

        if (!quality.passed) {
            return {
                isNewResult: true,
                qualityPassed: false,
                confidence: 0,
                rawLandmarks: rawLm,
                measurementLandmarks: measurementLm,
                displayLandmarks: displayLm,
                resultAgeMs: Number((processingEnd - tMs).toFixed(1))
            };
        }

        // 4. Body Normalization & Kinematics using measurement landmarks
        const normCoords = this.normalizeAndFilter(measurementLm, now);

        let kinematicsResult = null;
        if (this.trackingMode === 'balance') {
            kinematicsResult = this.processBalanceKinematics(measurementLm, now);
        } else {
            kinematicsResult = this.processGaitKinematics(normCoords, measurementLm, now, quality.confidence);
        }

        return {
            ...kinematicsResult,
            isNewResult: true,
            qualityPassed: true,
            rawLandmarks: rawLm,
            measurementLandmarks: measurementLm,
            displayLandmarks: displayLm,
            resultAgeMs: Number((processingEnd - tMs).toFixed(1))
        };
    }

    /**
     * Normalizes landmarks to pelvis center and applies OneEuroFilter
     */
    normalizeAndFilter(lm, now) {
        const leftHip = lm[23];
        const rightHip = lm[24];
        const leftShoulder = lm[11];
        const rightShoulder = lm[12];
        const leftAnkle = lm[27];
        const rightAnkle = lm[28];
        const leftHeel = lm[29];
        const rightHeel = lm[30];

        const pelvisX = (leftHip.x + rightHip.x) / 2.0;
        const pelvisY = (leftHip.y + rightHip.y) / 2.0;

        // Torso length as primary scale
        let torsoDist = 0.25;
        if (leftShoulder && rightShoulder) {
            const shoulderX = (leftShoulder.x + rightShoulder.x) / 2.0;
            const shoulderY = (leftShoulder.y + rightShoulder.y) / 2.0;
            torsoDist = Math.hypot(shoulderX - pelvisX, shoulderY - pelvisY) || 0.25;
        } else {
            torsoDist = Math.hypot(leftHip.x - rightHip.x, leftHip.y - rightHip.y) * 1.5 || 0.25;
        }

        this.smoothedScale = 0.05 * torsoDist + 0.95 * this.smoothedScale;
        const S = this.smoothedScale;

        // Biomechanical sign convention: positive upwards
        const normLeftAnkleY = -(leftAnkle.y - pelvisY) / S;
        const normRightAnkleY = -(rightAnkle.y - pelvisY) / S;
        const normLeftHeelY = -(leftHeel.y - pelvisY) / S;
        const normRightHeelY = -(rightHeel.y - pelvisY) / S;

        // Filter coordinates
        const filtLeftAnkleY = this.filters.leftAnkleY.filter(normLeftAnkleY, now);
        const filtRightAnkleY = this.filters.rightAnkleY.filter(normRightAnkleY, now);
        const filtLeftHeelY = this.filters.leftHeelY.filter(normLeftHeelY, now);
        const filtRightHeelY = this.filters.rightHeelY.filter(normRightHeelY, now);

        const leftKneeAngle = this.calculateAngle(leftHip, lm[25], leftAnkle) || 170;
        const rightKneeAngle = this.calculateAngle(rightHip, lm[26], rightAnkle) || 170;

        return {
            leftAnkle: { x: (leftAnkle.x - pelvisX) / S, y: filtLeftAnkleY },
            rightAnkle: { x: (rightAnkle.x - pelvisX) / S, y: filtRightAnkleY },
            leftHeel: { x: (leftHeel.x - pelvisX) / S, y: filtLeftHeelY },
            rightHeel: { x: (rightHeel.x - pelvisX) / S, y: filtRightHeelY },
            leftKneeAngle,
            rightKneeAngle,
            baselineFootY: this.baselineFootY
        };
    }

    /**
     * Gait Mode Kinematics (Multi-signal detection, Cadence, SI Temporal Symmetry)
     */
    processGaitKinematics(normCoords, rawLm, now, qualityConfidence) {
        const detection = this.stepDetector.evaluate(normCoords, now, qualityConfidence);

        // Update cadence estimator with intervals
        this.cadenceEstimator.update(this.stepDetector.stepIntervals);
        const cadenceMetrics = this.cadenceEstimator.getMetrics();

        // Calculate standard temporal symmetry index
        const symmetryMetrics = this.symmetryCalculator.calculate(
            this.stepDetector.leftDuration,
            this.stepDetector.rightDuration
        );

        this.averageSymmetry = symmetryMetrics.display_symmetry_pct;

        // NuroSync beat association
        if (detection.stepDetected && window.nuroSync) {
            window.nuroSync.evaluateStep({
                timestamp: now / 1000.0,
                side: detection.stepLeg.toUpperCase(),
                type: detection.eventType,
                confidence: qualityConfidence
            });
        }

        const leftConf = ((rawLm[23]?.visibility || 0.8) + (rawLm[25]?.visibility || 0.8) + (rawLm[27]?.visibility || 0.8)) / 3;
        const rightConf = ((rawLm[24]?.visibility || 0.8) + (rawLm[26]?.visibility || 0.8) + (rawLm[28]?.visibility || 0.8)) / 3;

        // Update global session data
        if (typeof sessionData !== 'undefined') {
            sessionData.leftStepsCount = this.stepDetector.leftStepsCount;
            sessionData.rightStepsCount = this.stepDetector.rightStepsCount;
            sessionData.averageSymmetry = symmetryMetrics.display_symmetry_pct;
            if (window.nuroSync) {
                sessionData.accuracyScore = window.nuroSync.getCurrentAccuracy();
                if (typeof updateAccuracyDisplay === 'function') {
                    updateAccuracyDisplay(sessionData.accuracyScore);
                }
            }
        }

        this.updateLegUI({
            status: detection.stepLeg !== 'none' ? `${detection.stepLeg.toUpperCase()} Step` : 'Tracking Active',
            activeLeg: detection.stepLeg.toUpperCase(),
            leftConfidence: leftConf,
            rightConfidence: rightConf,
            isLeftStepping: detection.stepLeg === 'left',
            isRightStepping: detection.stepLeg === 'right',
            cadenceSpm: cadenceMetrics.cadence_median_spm,
            symmetryPct: symmetryMetrics.display_symmetry_pct
        });

        return {
            mode: 'gait',
            qualityPassed: true,
            stepDetected: detection.stepDetected,
            stepLeg: detection.stepLeg,
            eventType: detection.eventType,
            cadence: cadenceMetrics,
            symmetry: symmetryMetrics,
            leftSteps: this.stepDetector.leftStepsCount,
            rightSteps: this.stepDetector.rightStepsCount,
            confidence: qualityConfidence
        };
    }

    /**
     * Balance Mode Kinematics (Postural Sway, Base Geometry, Stability Index)
     */
    processBalanceKinematics(rawLm, now) {
        const balanceMetrics = this.balanceEvaluator.evaluate(rawLm, now);
        this.balanceStabilityScore = balanceMetrics.stability_index;

        if (typeof sessionData !== 'undefined') {
            sessionData.accuracyScore = balanceMetrics.stability_index;
            sessionData.averageSymmetry = balanceMetrics.stability_index;
            if (typeof updateAccuracyDisplay === 'function') {
                updateAccuracyDisplay(balanceMetrics.stability_index);
            }
        }

        this.updateBalanceUI({
            weightDistribution: balanceMetrics.weight_distribution,
            stability: balanceMetrics.stability_index,
            swayMagnitude: balanceMetrics.sway_rms
        });

        return {
            mode: 'balance',
            qualityPassed: balanceMetrics.valid,
            stability: balanceMetrics.stability_index,
            weightDistribution: balanceMetrics.weight_distribution,
            confidence: 0.9
        };
    }

    calculateAngle(p1, p2, p3) {
        if (!p1 || !p2 || !p3) return null;
        const v1x = p1.x - p2.x;
        const v1y = p1.y - p2.y;
        const v2x = p3.x - p2.x;
        const v2y = p3.y - p2.y;
        const dot = v1x * v2x + v1y * v2y;
        const mag1 = Math.hypot(v1x, v1y);
        const mag2 = Math.hypot(v2x, v2y);
        if (mag1 === 0 || mag2 === 0) return null;
        const cosAngle = Math.max(-1, Math.min(1, dot / (mag1 * mag2)));
        return Math.round(Math.acos(cosAngle) * (180 / Math.PI));
    }

    drawSkeletonOverlay(landmarks) {
        if (!this.overlayCanvas || !this.overlayCtx) return;

        this.updateDimensions();

        const w = this.overlayCanvas.width;
        const h = this.overlayCanvas.height;
        const ctx = this.overlayCtx;

        ctx.clearRect(0, 0, w, h);

        if (!landmarks || landmarks.length < 33) return;

        const vW = this.videoElement?.videoWidth || w;
        const vH = this.videoElement?.videoHeight || h;

        const project = (idx) => {
            const rawLm = landmarks[idx];
            if (!rawLm) return null;
            return transformLandmarkToCanvas(rawLm, vW, vH, w, h, 'fill', false);
        };

        const coreTorso = [[11, 12], [11, 23], [12, 24], [23, 24]];
        const headFace = [[0, 1], [1, 2], [2, 3], [3, 7], [0, 4], [4, 5], [5, 6], [6, 8], [9, 10]];
        const leftArm = [[11, 13], [13, 15], [15, 17], [15, 19], [15, 21]];
        const rightArm = [[12, 14], [14, 16], [16, 18], [16, 20], [16, 22]];
        const leftLeg = [[23, 25], [25, 27], [27, 29], [29, 31], [27, 31]];
        const rightLeg = [[24, 26], [26, 28], [28, 30], [30, 32], [28, 32]];

        const drawSegment = (connections, strokeStyle, lineWidth) => {
            ctx.strokeStyle = strokeStyle;
            ctx.lineWidth = lineWidth;
            ctx.lineCap = 'round';
            ctx.lineJoin = 'round';

            connections.forEach(([i, j]) => {
                const p1 = project(i);
                const p2 = project(j);
                if (p1 && p2 && p1.visibility > 0.30 && p2.visibility > 0.30) {
                    ctx.beginPath();
                    ctx.moveTo(p1.x, p1.y);
                    ctx.lineTo(p2.x, p2.y);
                    ctx.stroke();
                }
            });
        };

        drawSegment(coreTorso, '#01aac5', 4.5);
        drawSegment(headFace, 'rgba(148, 163, 184, 0.65)', 2);
        drawSegment(leftArm, '#00e5ff', 3.5);
        drawSegment(rightArm, '#10b981', 3.5);
        drawSegment(leftLeg, '#00e5ff', 3.5);
        drawSegment(rightLeg, '#10b981', 3.5);

        for (let i = 0; i < landmarks.length; i++) {
            const pt = project(i);
            if (!pt || pt.visibility <= 0.30) continue;

            let fillColor = '#01aac5';
            let radius = 4;

            if (i >= 11 && i % 2 === 1) {
                fillColor = '#00e5ff';
                radius = (i === 11 || i === 23 || i === 25) ? 6 : 4.5;
            } else if (i >= 12 && i % 2 === 0) {
                fillColor = '#10b981';
                radius = (i === 12 || i === 24 || i === 26) ? 6 : 4.5;
            } else if (i === 0) {
                fillColor = '#38bdf8';
                radius = 4;
            } else if (i <= 10) {
                fillColor = '#94a3b8';
                radius = 2.5;
            }

            ctx.beginPath();
            ctx.arc(pt.x, pt.y, radius + 1.5, 0, 2 * Math.PI);
            ctx.fillStyle = '#ffffff';
            ctx.fill();

            ctx.beginPath();
            ctx.arc(pt.x, pt.y, radius, 0, 2 * Math.PI);
            ctx.fillStyle = fillColor;
            ctx.fill();
        }
    }

    updateQualityBanner(quality) {
        let banner = document.getElementById('cameraQualityBanner');
        if (!quality.passed) {
            if (!banner) {
                const stage = this.cameraStage || document.getElementById('cameraBox');
                if (stage) {
                    banner = document.createElement('div');
                    banner.id = 'cameraQualityBanner';
                    banner.className = 'camera-quality-banner';
                    stage.appendChild(banner);
                }
            }
            if (banner) {
                banner.textContent = quality.reason;
                banner.style.display = 'block';
            }
        } else {
            if (banner) {
                banner.style.display = 'none';
            }
        }
    }

    updateLegUI(data) {
        if (this.trackingMode === 'balance') return;

        // Throttle DOM mutations to ~10 FPS (100ms) unless stepping state changes
        const now = performance.now();
        const stepStateChanged = (data.isLeftStepping !== this._lastLeftStepping) || (data.isRightStepping !== this._lastRightStepping);
        if (!stepStateChanged && (now - this.lastUiUpdateTime < 100)) {
            return;
        }
        this.lastUiUpdateTime = now;
        this._lastLeftStepping = data.isLeftStepping;
        this._lastRightStepping = data.isRightStepping;

        const leftBadge = document.getElementById('leftLegBadge');
        const leftState = document.getElementById('leftLegState');
        const rightBadge = document.getElementById('rightLegBadge');
        const rightState = document.getElementById('rightLegState');
        const symmetryVal = document.getElementById('gaitSymmetryValue');

        if (leftBadge && leftState) {
            if (data.isLeftStepping) {
                leftBadge.style.background = '#01aac5';
                leftBadge.style.color = '#ffffff';
                leftState.textContent = 'Stepping';
            } else {
                leftBadge.style.background = '#64748b';
                leftBadge.style.color = '#ffffff';
                leftState.textContent = 'Idle';
            }
        }

        if (rightBadge && rightState) {
            if (data.isRightStepping) {
                rightBadge.style.background = '#10b981';
                rightBadge.style.color = '#ffffff';
                rightState.textContent = 'Stepping';
            } else {
                rightBadge.style.background = '#64748b';
                rightBadge.style.color = '#ffffff';
                rightState.textContent = 'Idle';
            }
        }

        if (symmetryVal) {
            if (this.averageSymmetry > 0) {
                symmetryVal.textContent = `${this.averageSymmetry}%`;
                if (this.averageSymmetry >= 80) {
                    symmetryVal.className = 'fw-bold text-success';
                } else if (this.averageSymmetry >= 70) {
                    symmetryVal.className = 'fw-bold text-info';
                } else {
                    symmetryVal.className = 'fw-bold text-warning';
                }
            } else {
                symmetryVal.textContent = '--%';
            }
        }
    }

    updateBalanceUI(data) {
        const now = performance.now();
        if (now - this.lastUiUpdateTime < 100) return;
        this.lastUiUpdateTime = now;

        const leftBadge = document.getElementById('leftLegBadge');
        const rightBadge = document.getElementById('rightLegBadge');
        const stabilityVal = document.getElementById('gaitSymmetryValue');

        if (leftBadge && rightBadge) {
            if (data.weightDistribution === 'Shifted Left') {
                leftBadge.style.background = '#01aac5';
                leftBadge.style.color = '#ffffff';
                leftBadge.textContent = 'Left Weight: Loaded';
                rightBadge.style.background = '#64748b';
                rightBadge.style.color = '#ffffff';
                rightBadge.textContent = 'Right Weight: Light';
            } else if (data.weightDistribution === 'Shifted Right') {
                rightBadge.style.background = '#10b981';
                rightBadge.style.color = '#ffffff';
                rightBadge.textContent = 'Right Weight: Loaded';
                leftBadge.style.background = '#64748b';
                leftBadge.style.color = '#ffffff';
                leftBadge.textContent = 'Left Weight: Light';
            } else {
                leftBadge.style.background = '#01aac5';
                leftBadge.style.color = '#ffffff';
                leftBadge.textContent = 'Left Weight: Centered';
                rightBadge.style.background = '#10b981';
                rightBadge.style.color = '#ffffff';
                rightBadge.textContent = 'Right Weight: Centered';
            }
        }

        if (stabilityVal) {
            stabilityVal.textContent = `${data.stability}%`;
            stabilityVal.className = data.stability >= 80 ? 'fw-bold text-success' : 'fw-bold text-warning';
        }
    }

    /**
     * Constructs unified telemetry packet adhering to schema_version: "2.0"
     */
    getTelemetryPayload(sessionId) {
        const qualityCoverage = this.qualityGate.getCoverage();
        const criticalCoverage = this.qualityGate.getCriticalCoverage();
        const cadence = this.cadenceEstimator.getMetrics();
        const symmetry = this.symmetryCalculator.calculate(
            this.stepDetector.leftDuration,
            this.stepDetector.rightDuration
        );
        const syncMetrics = window.nuroSync ? window.nuroSync.getMetricsSummary() : null;
        const diagSummary = this.diagnostics ? this.diagnostics.getSummary() : null;

        return {
            schema_version: "2.0",
            session_id: sessionId,
            timestamp: Number((performance.now() / 1000.0).toFixed(3)),
            session_type: this.trackingMode === 'balance' ? 'balance_training' : 'gait_trainer',
            measurement_quality: {
                state: this.qualityGate.recentFrames.length > 0 ? 'GOOD' : 'INITIALIZING',
                confidence: qualityCoverage,
                tracking_coverage: qualityCoverage,
                critical_landmark_coverage: criticalCoverage,
                fps: diagSummary ? diagSummary.effectiveFPS : this.fps,
                latency_ms: diagSummary ? diagSummary.inferenceDurationMs : this.latencyMs,
                outliers_rejected: this.stepDetector.rejectedCandidates,
                dropped_frames: diagSummary ? diagSummary.droppedFrames : 0,
                temporal_stability: diagSummary ? diagSummary.temporalStability : 1.0
            },
            gait: {
                valid: cadence.valid,
                cadence_spm: cadence.cadence_spm,
                cadence_median_spm: cadence.cadence_median_spm,
                cadence_cv: cadence.cadence_cv,
                step_interval_mean_s: cadence.step_interval_mean_s,
                step_interval_sd_s: cadence.step_interval_sd_s,
                total_steps: this.stepDetector.leftStepsCount + this.stepDetector.rightStepsCount,
                left_steps: this.stepDetector.leftStepsCount,
                right_steps: this.stepDetector.rightStepsCount,
                temporal_asymmetry_pct: symmetry.temporal_asymmetry_pct,
                symmetry_index: symmetry.symmetry_index
            },
            balance: {
                valid: this.trackingMode === 'balance',
                stability_index: this.balanceEvaluator.stabilityIndex,
                weight_distribution: this.balanceEvaluator.lastWeightDistribution,
                path_length: Number(this.balanceEvaluator.pathLength.toFixed(3))
            },
            sync: syncMetrics || {
                valid: (window.latestMovementState && window.latestMovementState.rhythm && typeof window.latestMovementState.rhythm.sync === 'number'),
                rhythm_alignment_score: (window.latestMovementState && window.latestMovementState.rhythm && typeof window.latestMovementState.rhythm.sync === 'number')
                    ? Math.round(window.latestMovementState.rhythm.sync * 100)
                    : 0
            },
            movement_intelligence: window.latestMovementState || (window.movementIntelligence && typeof window.movementIntelligence.getState === 'function' ? window.movementIntelligence.getState() : null),
            adaptive_state: window.latestAdaptiveState || (window.adaptiveEngine && typeof window.adaptiveEngine.getCurrentState === 'function' ? window.adaptiveEngine.getCurrentState() : null),
            adaptation_decision: window.adaptiveEngine && typeof window.adaptiveEngine.getLastDecision === 'function' ? window.adaptiveEngine.getLastDecision() : null,
            agent_state: window.latestAgentState || (window.nuroAgent && typeof window.nuroAgent.getCurrentState === 'function' ? window.nuroAgent.getCurrentState() : null),
            agent_decision: window.nuroAgent && typeof window.nuroAgent.getCurrentReasoning === 'function' ? window.nuroAgent.getCurrentReasoning() : null,
            agent_summary: window.nuroAgent && typeof window.nuroAgent.generateSessionSummary === 'function' ? window.nuroAgent.generateSessionSummary() : null
        };
    }

    useLightweightFallback(diffScore) {
        if (this.trackingMode === 'balance') {
            this.updateBalanceUI({
                weightDistribution: 'Centered',
                stability: 95,
                swayMagnitude: 0.01
            });
            return;
        }

        this.updateLegUI({
            status: diffScore >= MOTION_THRESHOLD ? 'Active Movement' : 'Idle / Standing Still',
            activeLeg: diffScore >= MOTION_THRESHOLD ? 'Both' : 'None',
            leftConfidence: diffScore >= MOTION_THRESHOLD ? 0.5 : 0,
            rightConfidence: diffScore >= MOTION_THRESHOLD ? 0.5 : 0
        });
    }

    getDiagnostics() {
        const qualityCoverage = this.qualityGate ? this.qualityGate.getCoverage() : 1.0;
        const criticalCoverage = this.qualityGate ? this.qualityGate.getCriticalCoverage() : 1.0;
        let qualityState = 'EXCELLENT';
        if (qualityCoverage < 0.6) qualityState = 'FAIR';
        if (qualityCoverage < 0.3) qualityState = 'POOR';

        const summary = this.diagnostics ? this.diagnostics.getSummary() : null;

        return {
            fps: summary ? summary.effectiveFPS : (this.fps || 30),
            latency_ms: summary ? summary.inferenceDurationMs : (this.latencyMs || 15),
            tracking_coverage: qualityCoverage,
            critical_landmark_coverage: criticalCoverage,
            valid_steps: this.stepDetector ? (this.stepDetector.leftStepsCount + this.stepDetector.rightStepsCount) : 0,
            rejected_step_candidates: this.stepDetector ? this.stepDetector.rejectedCandidates : 0,
            quality_state: qualityState,
            // Telemetry & frame diagnostics (Sections 4, 5, 29, 30)
            effective_fps: summary ? summary.effectiveFPS : (this.fps || 30),
            frame_interval_ms: summary ? summary.meanFrameIntervalMs : 33.3,
            frame_timing_jitter_ms: summary ? summary.frameTimingJitterMs : 0,
            inference_duration_ms: summary ? summary.inferenceDurationMs : (this.latencyMs || 15),
            result_age_ms: summary ? summary.resultAgeMs : 0,
            landmark_jitter: summary ? summary.landmarkJitter : 0,
            dropped_frames: summary ? summary.droppedFrames : 0,
            duplicate_frames: summary ? summary.duplicateFrames : 0,
            out_of_order_count: summary ? summary.outOfOrderCount : 0,
            temporal_stability: summary ? summary.temporalStability : 1.0
        };
    }

    reset() {
        if (this.diagnostics) this.diagnostics.reset();
        if (this.adaptiveFilter) this.adaptiveFilter.reset();
        this.latestAcceptedFrameId = -1;
        this.latestPoseTimestamp = 0;
        this.inFlightFrameId = null;
        this.inFlightTimestamp = null;
        this.inFlightStart = 0;
        this.isInferenceInFlight = false;
        this.internalFrameCounter = 0;
        this.latestQuality = null;
        this.lastUiUpdateTime = 0;

        this.stepDetector.reset();
        this.symmetryCalculator.reset();
        this.balanceEvaluator.reset();
        this.qualityGate.reset();
        Object.values(this.filters).forEach(f => f.reset());
        this.averageSymmetry = 100;
        this.balanceStabilityScore = 100;

        if (window.movementIntelligence && typeof window.movementIntelligence.reset === 'function') {
            window.movementIntelligence.reset();
        }
        if (window.p4PhaseIntelligence && typeof window.p4PhaseIntelligence.reset === 'function') {
            window.p4PhaseIntelligence.reset();
        }
        if (window.adaptiveEngine && typeof window.adaptiveEngine.reset === 'function') {
            window.adaptiveEngine.reset();
        }
        if (window.nuroAgent && typeof window.nuroAgent.reset === 'function') {
            window.nuroAgent.reset();
        }

        if (this.overlayCanvas && this.overlayCtx) {
            this.overlayCtx.clearRect(0, 0, this.overlayCanvas.width, this.overlayCanvas.height);
        }

        const banner = document.getElementById('cameraQualityBanner');
        if (banner) banner.style.display = 'none';

        if (this.trackingMode === 'balance') {
            this.updateBalanceUI({ weightDistribution: 'Centered', stability: 100 });
        } else {
            this.updateLegUI({
                status: 'Idle / Standing Still',
                activeLeg: 'None',
                leftConfidence: 0,
                rightConfidence: 0
            });
        }
    }

    teardown() {
        this.reset();
        if (this.resizeObserver) {
            this.resizeObserver.disconnect();
            this.resizeObserver = null;
        }
        if (this.videoElement) {
            this.videoElement.removeEventListener('loadedmetadata', this.updateDimensions);
            this.videoElement.removeEventListener('resize', this.updateDimensions);
        }
    }
}

// Global instance
const legTrackerInstance = new LegKinematicsTracker();
window.legTracker = legTrackerInstance;
window.transformLandmarkToCanvas = transformLandmarkToCanvas;

// Track last evaluation timestamps for P2/P3 to enforce window-level execution (Section 26)
let lastP2EvaluationSec = 0;
let lastP3ObservationSec = 0;

async function runFusedLegTracking(diffScore, frameId = null, captureTimestampMs = null) {
    const videoElement = document.getElementById('cameraFeed');
    if (!videoElement || videoElement.readyState < 2) return;

    const fId = typeof frameId === 'number' ? frameId : ++legTrackerInstance.internalFrameCounter;
    const captureMs = typeof captureTimestampMs === 'number' ? captureTimestampMs : performance.now();
    const captureSec = captureMs / 1000.0;

    try {
        const poseResults = await legTrackerInstance.estimatePose(videoElement, fId, captureMs);

        // Movement Intelligence P1: Process ONLY when new valid pose result is returned (Section 9 & 10)
        if (poseResults && poseResults.isNewResult && window.movementIntelligence) {
            const measurementLandmarks = poseResults.measurementLandmarks;
            const miState = window.movementIntelligence.processFrame(
                measurementLandmarks,
                diffScore,
                captureSec,
                window.nuroSync,
                fId
            );
            window.latestMovementState = miState;
            if (typeof sessionData !== 'undefined' && miState.rhythm && typeof miState.rhythm.sync === 'number') {
                sessionData.cameraSyncAccuracy = Math.round(miState.rhythm.sync * 100);
            }

            // Learned Temporal Motion Intelligence P4: Continuous Phase & Cycle Tracking
            if (window.p4PhaseIntelligence) {
                const currentBpm = (typeof sessionData !== 'undefined' && sessionData.currentBPM)
                    ? sessionData.currentBPM
                    : (window.nuroSync ? window.nuroSync.currentBpm : 60);

                const prevCapture = legTrackerInstance.lastCaptureMs || (captureMs - 33.33);
                const deltaMs = Math.max(1.0, captureMs - prevCapture);
                legTrackerInstance.lastCaptureMs = captureMs;

                const p4Observation = {
                    frameId: fId,
                    captureTimestampMs: captureMs,
                    deltaTimeMs: deltaMs,
                    landmarks: measurementLandmarks,
                    landmarkConfidence: miState.confidence || 0.9,
                    bodyScale: legTrackerInstance.smoothedScale || 1.0,
                    movementState: miState.state || 'ACTIVE',
                    bpm: currentBpm,
                    isNewPoseResult: true
                };

                const p4State = window.p4PhaseIntelligence.processObservation(p4Observation);
                if (p4State) {
                    window.latestP4State = p4State;
                    miState.temporal = p4State;
                }
            }

            // Adaptive Intelligence P2: Enforce Window-Level Evaluation (Section 26 - at least 1.0s interval)
            if (window.adaptiveEngine && (captureSec - lastP2EvaluationSec >= 1.0)) {
                lastP2EvaluationSec = captureSec;
                const currentBpm = (typeof sessionData !== 'undefined' && sessionData.currentBPM)
                    ? sessionData.currentBPM
                    : (window.nuroSync ? window.nuroSync.currentBpm : 60);
                const adaptiveState = window.adaptiveEngine.evaluate(miState, currentBpm, captureSec);
                window.latestAdaptiveState = adaptiveState;
            }

            // Nuro Agent P3: Enforce Window-Level Observation (Section 26 - at least 2.0s interval)
            if (window.nuroAgent && window.latestAdaptiveState && (captureSec - lastP3ObservationSec >= 2.0)) {
                lastP3ObservationSec = captureSec;
                const agentState = window.nuroAgent.observe(miState, window.latestAdaptiveState, captureSec);
                window.latestAgentState = agentState;
            }
        }

        if (diffScore < MOTION_THRESHOLD) {
            if (legTrackerInstance.trackingMode === 'gait') {
                legTrackerInstance.updateLegUI({
                    status: 'Idle / Standing Still',
                    activeLeg: 'None',
                    leftConfidence: 0,
                    rightConfidence: 0
                });
            }
            return;
        }

        if (!poseResults || !poseResults.qualityPassed) {
            legTrackerInstance.useLightweightFallback(diffScore);
        }
    } catch (err) {
        console.warn('Pose tracking error, falling back to pixel motion:', err);
        legTrackerInstance.useLightweightFallback(diffScore);
    }
}

window.runFusedLegTracking = runFusedLegTracking;
