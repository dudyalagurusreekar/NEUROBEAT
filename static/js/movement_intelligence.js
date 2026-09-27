/**
 * MovementIntelligence - Centralized Movement & Rhythm Intelligence Layer (P1)
 * 
 * Transforms raw pose landmarks into reliable, explainable, temporal movement metrics
 * and integrates with NuroSync for rhythm alignment.
 * 
 * Pipeline:
 * RAW POSE -> POSE NORMALIZATION -> TEMPORAL BUFFER (1-3s) ->
 * FEATURE EXTRACTION (ROM, Velocity, Acceleration, Smoothness, Symmetry, Consistency) ->
 * MOVEMENT QUALITY & CONFIDENCE -> RHYTHM INTELLIGENCE ->
 * CANONICAL MOVEMENT STATE OBJECT
 * 
 * Engineering and biomechanical movement metrics only. Zero medical/diagnostic claims.
 */

(function (root, factory) {
    if (typeof define === 'function' && define.amd) {
        define([], factory);
    } else if (typeof module === 'object' && module.exports) {
        module.exports = factory();
    } else {
        const exports = factory();
        root.MovementIntelligence = exports.MovementIntelligence;
        root.MOVEMENT_INTELLIGENCE_CONFIG = exports.MOVEMENT_INTELLIGENCE_CONFIG;
        root.movementIntelligence = new exports.MovementIntelligence();
    }
}(typeof self !== 'undefined' ? self : this, function () {

    const MOVEMENT_INTELLIGENCE_CONFIG = {
        bufferDurationSec: 2.5,        // 1 to 3 seconds rolling temporal window
        maxFrames: 90,                 // Bounded frame capacity (prevents memory leak)
        minFramesForMetrics: 12,       // Warming up threshold
        motionThreshold: 12,           // Motion threshold matching MOTION_THRESHOLD
        weights: {
            rom: 0.25,
            smoothness: 0.30,
            symmetry: 0.25,
            consistency: 0.20
        },
        expectedKneeRomDeg: 75.0,      // Reference knee flexion ROM during gait/stepping
        expectedHipRomDeg: 60.0,       // Reference hip flexion ROM
        expectedArmRomDeg: 80.0,       // Reference elbow/shoulder ROM
        referenceAccVar: 45.0,         // Reference acceleration variance for smoothness scale
        minConfidenceThreshold: 0.40   // Cutoff for LOW_CONFIDENCE state
    };

    class MovementIntelligence {
        constructor(config = {}) {
            this.config = {
                ...MOVEMENT_INTELLIGENCE_CONFIG,
                ...config,
                weights: { ...MOVEMENT_INTELLIGENCE_CONFIG.weights, ...(config.weights || {}) }
            };

            // Rolling temporal buffer: [{ timestamp, landmarks, normalizedLandmarks, confidence }]
            this.buffer = [];

            // Landmark tracking history
            this.prevDistalPositions = null;
            this.prevVelocities = null;
            this.smoothedScale = 0.25;

            // Repetition tracking
            this.repetitions = [];       // [{ timestamp, duration, amplitude }]
            this.lastRepTimestamp = 0;
            this.repPeakCandidate = null;
            this.repDirection = 0;       // +1 = increasing, -1 = decreasing

            // Rhythm intelligence tracking
            this.rhythmHistory = [];     // [{ timestamp, timingErrorMs, isWithinTolerance }]
            this.missedBeatsCount = 0;
            this.successfulBeatsCount = 0;
            this.lastEvaluatedBeatTime = null;

            // Active mode
            this.trackingMode = 'gait';   // 'gait' | 'balance' | 'upper_limb'

            // Cached latest movement state
            this.latestState = this.createDefaultState('WARMING_UP', 0);
        }

        setTrackingMode(mode) {
            this.trackingMode = mode || 'gait';
        }

        setWeights(newWeights) {
            if (typeof newWeights === 'object' && newWeights !== null) {
                this.config.weights = { ...this.config.weights, ...newWeights };
            }
        }

        reset() {
            this.buffer = [];
            this.prevDistalPositions = null;
            this.prevVelocities = null;
            this.smoothedScale = 0.25;
            this.repetitions = [];
            this.lastRepTimestamp = 0;
            this.repPeakCandidate = null;
            this.repDirection = 0;
            this.rhythmHistory = [];
            this.missedBeatsCount = 0;
            this.successfulBeatsCount = 0;
            this.lastEvaluatedBeatTime = null;
            this.latestState = this.createDefaultState('WARMING_UP', performance.now() / 1000);
        }

        createDefaultState(state = 'WARMING_UP', timestampSec = 0) {
            return {
                timestamp: Math.round(timestampSec * 1000),
                state: state,
                movement: {
                    rom: 0.0,
                    velocity: 0.0,
                    smoothness: 0.85,
                    symmetry: 1.0,
                    consistency: 0.85,
                    quality: 0.85
                },
                rhythm: {
                    sync: null,
                    timingErrorMs: null,
                    missedBeats: 0,
                    averageErrorMs: 0,
                    timingVariance: 0,
                    successfulBeats: 0
                },
                confidence: 0.0
            };
        }

        /**
         * 3. Body-Centered Landmark Normalization
         * Normalizes landmark coordinates relative to hip center and torso scale.
         * Inverted Y coordinate so positive Y represents upward biomechanical lift.
         */
        normalizeLandmarks(rawLandmarks) {
            if (!rawLandmarks || rawLandmarks.length < 33) {
                return null;
            }

            const leftHip = rawLandmarks[23];
            const rightHip = rawLandmarks[24];
            const leftShoulder = rawLandmarks[11];
            const rightShoulder = rawLandmarks[12];

            // Pelvis Center
            const hipCenterX = ((leftHip?.x ?? 0.5) + (rightHip?.x ?? 0.5)) / 2.0;
            const hipCenterY = ((leftHip?.y ?? 0.6) + (rightHip?.y ?? 0.6)) / 2.0;

            // Torso Length as Body Scale Reference
            let torsoLength = 0.25;
            if (leftShoulder && rightShoulder) {
                const shoulderCenterX = (leftShoulder.x + rightShoulder.x) / 2.0;
                const shoulderCenterY = (leftShoulder.y + rightShoulder.y) / 2.0;
                torsoLength = Math.hypot(shoulderCenterX - hipCenterX, shoulderCenterY - hipCenterY);
            } else if (leftHip && rightHip) {
                torsoLength = Math.hypot(leftHip.x - rightHip.x, leftHip.y - rightHip.y) * 1.5;
            }

            const clampedTorso = Math.max(0.10, Math.min(0.85, torsoLength || 0.25));
            this.smoothedScale = 0.10 * clampedTorso + 0.90 * this.smoothedScale;
            const S = this.smoothedScale;

            const normalizedLandmarks = new Array(rawLandmarks.length);
            for (let i = 0; i < rawLandmarks.length; i++) {
                const lm = rawLandmarks[i];
                if (!lm) {
                    normalizedLandmarks[i] = { x: 0, y: 0, z: 0, visibility: 0 };
                    continue;
                }
                normalizedLandmarks[i] = {
                    x: (lm.x - hipCenterX) / S,
                    y: -(lm.y - hipCenterY) / S, // Positive Y = upward
                    z: (lm.z || 0) / S,
                    visibility: typeof lm.visibility === 'number' ? lm.visibility : (lm.presence ?? 1.0)
                };
            }

            return {
                normalizedLandmarks,
                center: { x: hipCenterX, y: hipCenterY },
                scale: S
            };
        }

        /**
         * Computes 3D Euclidean angle in degrees at vertex B: angle(A, B, C)
         */
        computeAngle(a, b, c) {
            if (!a || !b || !c) return 180.0;
            const v1x = a.x - b.x;
            const v1y = a.y - b.y;
            const v1z = (a.z || 0) - (b.z || 0);

            const v2x = c.x - b.x;
            const v2y = c.y - b.y;
            const v2z = (c.z || 0) - (b.z || 0);

            const mag1 = Math.sqrt(v1x * v1x + v1y * v1y + v1z * v1z);
            const mag2 = Math.sqrt(v2x * v2x + v2y * v2y + v2z * v2z);

            if (mag1 < 1e-6 || mag2 < 1e-6) return 180.0;

            const dot = v1x * v2x + v1y * v2y + v1z * v2z;
            const cosTheta = Math.max(-1.0, Math.min(1.0, dot / (mag1 * mag2)));
            return (Math.acos(cosTheta) * 180.0) / Math.PI;
        }

        /**
         * 4A. Range of Motion (ROM)
         * Computes dynamic range of motion across active joints within the temporal buffer.
         */
        calculateRangeOfMotion() {
            if (this.buffer.length < 2) {
                return { rom: 0.0, leftRom: 0.0, rightRom: 0.0 };
            }

            let minLeftKnee = Infinity, maxLeftKnee = -Infinity;
            let minRightKnee = Infinity, maxRightKnee = -Infinity;
            let minLeftArm = Infinity, maxLeftArm = -Infinity;
            let minRightArm = Infinity, maxRightArm = -Infinity;

            for (let i = 0; i < this.buffer.length; i++) {
                const normLm = this.buffer[i].normalizedLandmarks;
                if (!normLm) continue;

                // Knee angles (Hips: 23, 24 | Knees: 25, 26 | Ankles: 27, 28)
                const lKnee = this.computeAngle(normLm[23], normLm[25], normLm[27]);
                const rKnee = this.computeAngle(normLm[24], normLm[26], normLm[28]);

                if (lKnee < minLeftKnee) minLeftKnee = lKnee;
                if (lKnee > maxLeftKnee) maxLeftKnee = lKnee;
                if (rKnee < minRightKnee) minRightKnee = rKnee;
                if (rKnee > maxRightKnee) maxRightKnee = rKnee;

                // Arm angles (Shoulders: 11, 12 | Elbows: 13, 14 | Wrists: 15, 16)
                const lArm = this.computeAngle(normLm[11], normLm[13], normLm[15]);
                const rArm = this.computeAngle(normLm[12], normLm[14], normLm[16]);

                if (lArm < minLeftArm) minLeftArm = lArm;
                if (lArm > maxLeftArm) maxLeftArm = lArm;
                if (rArm < minRightArm) minRightArm = rArm;
                if (rArm > maxRightArm) maxRightArm = rArm;
            }

            const leftKneeRom = maxLeftKnee > minLeftKnee ? (maxLeftKnee - minLeftKnee) : 0.0;
            const rightKneeRom = maxRightKnee > minRightKnee ? (maxRightKnee - minRightKnee) : 0.0;
            const leftArmRom = maxLeftArm > minLeftArm ? (maxLeftArm - minLeftArm) : 0.0;
            const rightArmRom = maxRightArm > minRightArm ? (maxRightArm - minRightArm) : 0.0;

            let activeLeftRom = leftKneeRom;
            let activeRightRom = rightKneeRom;
            let expectedRom = this.config.expectedKneeRomDeg;

            if (this.trackingMode === 'upper_limb') {
                activeLeftRom = leftArmRom;
                activeRightRom = rightArmRom;
                expectedRom = this.config.expectedArmRomDeg;
            }

            const avgRomDeg = (activeLeftRom + activeRightRom) / 2.0;
            const normalizedRom = Math.max(0.0, Math.min(1.0, avgRomDeg / expectedRom));

            return {
                rom: Number(normalizedRom.toFixed(3)),
                leftRom: Number(activeLeftRom.toFixed(1)),
                rightRom: Number(activeRightRom.toFixed(1)),
                rawRomDeg: Number(avgRomDeg.toFixed(1))
            };
        }

        /**
         * 4B & 4C. Velocity and Acceleration Analysis
         * Computes stable aggregate velocities with occlusion spike protection.
         */
        calculateVelocityAndAcceleration(timestampSec) {
            if (this.buffer.length < 2) {
                return {
                    velocity: 0.0,
                    rawVelocity: 0.0,
                    acceleration: 0.0,
                    leftVel: 0.0,
                    rightVel: 0.0,
                    accelerations: []
                };
            }

            const currFrame = this.buffer[this.buffer.length - 1];
            const prevFrame = this.buffer[this.buffer.length - 2];
            const dt = Math.max(0.008, currFrame.timestamp - prevFrame.timestamp);

            const currNorm = currFrame.normalizedLandmarks;
            const prevNorm = prevFrame.normalizedLandmarks;

            let lCurr = currNorm[27], rCurr = currNorm[28]; // Ankles
            let lPrev = prevNorm[27], rPrev = prevNorm[28];

            if (this.trackingMode === 'upper_limb') {
                lCurr = currNorm[15]; rCurr = currNorm[16]; // Wrists
                lPrev = prevNorm[15]; rPrev = prevNorm[16];
            }

            const calcDist = (p1, p2) => Math.sqrt(
                Math.pow(p1.x - p2.x, 2) + Math.pow(p1.y - p2.y, 2) + Math.pow((p1.z || 0) - (p2.z || 0), 2)
            );

            // Clamp velocity spikes caused by occlusion/teleportation
            const maxPhysVelocity = 8.0; // body lengths per second
            const lDist = calcDist(lCurr, lPrev);
            const rDist = calcDist(rCurr, rPrev);

            const lVel = Math.min(maxPhysVelocity, lDist / dt);
            const rVel = Math.min(maxPhysVelocity, rDist / dt);
            const avgVel = (lVel + rVel) / 2.0;

            // Historical vector acceleration and jerk series over buffer
            const accelVectors = [];
            const accelerations = [];
            const jerks = [];
            for (let i = 2; i < this.buffer.length; i++) {
                const fCurr = this.buffer[i].normalizedLandmarks;
                const fPrev = this.buffer[i - 1].normalizedLandmarks;
                const fPPrev = this.buffer[i - 2].normalizedLandmarks;
                if (!fCurr || !fPrev || !fPPrev) continue;

                const stepDt = Math.max(0.008, this.buffer[i].timestamp - this.buffer[i - 1].timestamp);

                // Frame gap protection (Sections 6 & 21):
                // Do not differentiate across missing frame intervals (> 150ms)
                if (stepDt > 0.15) {
                    accelVectors.length = 0;
                    continue;
                }

                let cL = fCurr[27], cR = fCurr[28];
                let pL = fPrev[27], pR = fPrev[28];
                let ppL = fPPrev[27], ppR = fPPrev[28];

                if (this.trackingMode === 'upper_limb') {
                    cL = fCurr[15]; cR = fCurr[16];
                    pL = fPrev[15]; pR = fPrev[16];
                    ppL = fPPrev[15]; ppR = fPPrev[16];
                }

                // Vector velocity components
                const vx1 = (((cL.x - pL.x) + (cR.x - pR.x)) / 2.0) / stepDt;
                const vy1 = (((cL.y - pL.y) + (cR.y - pR.y)) / 2.0) / stepDt;
                const vz1 = ((((cL.z || 0) - (pL.z || 0)) + (((cR.z || 0) - (pR.z || 0)))) / 2.0) / stepDt;

                const vx0 = (((pL.x - ppL.x) + (pR.x - ppR.x)) / 2.0) / stepDt;
                const vy0 = (((pL.y - ppL.y) + (pR.y - ppR.y)) / 2.0) / stepDt;
                const vz0 = ((((pL.z || 0) - (ppL.z || 0)) + (((pR.z || 0) - (ppR.z || 0)))) / 2.0) / stepDt;

                // Vector acceleration
                const ax = (vx1 - vx0) / stepDt;
                const ay = (vy1 - vy0) / stepDt;
                const az = (vz1 - vz0) / stepDt;
                const aMag = Math.sqrt(ax * ax + ay * ay + az * az);

                accelVectors.push({ ax, ay, az });
                accelerations.push(Math.min(100.0, aMag));

                // Jerk: rate of change of acceleration vector
                if (accelVectors.length >= 2) {
                    const prevA = accelVectors[accelVectors.length - 2];
                    const jx = (ax - prevA.ax) / stepDt;
                    const jy = (ay - prevA.ay) / stepDt;
                    const jz = (az - prevA.az) / stepDt;
                    const jMag = Math.sqrt(jx * jx + jy * jy + jz * jz);
                    jerks.push(jMag);
                }
            }

            const latestAccel = accelerations.length > 0 ? accelerations[accelerations.length - 1] : 0.0;
            const normalizedVel = Math.max(0.0, Math.min(1.0, avgVel / 2.5));

            return {
                velocity: Number(normalizedVel.toFixed(3)),
                rawVelocity: Number(avgVel.toFixed(3)),
                acceleration: Number(latestAccel.toFixed(2)),
                leftVel: Number(lVel.toFixed(3)),
                rightVel: Number(rVel.toFixed(3)),
                accelerations,
                jerks
            };
        }

        /**
         * 4D. Smoothness Metric
         * Quantifies movement smoothness based on normalized jerk (rate of change of acceleration vector).
         * 1.0 = smooth continuous motion, 0.0 = highly irregular/jerky motion.
         */
        calculateSmoothness(jerks) {
            if (!jerks || jerks.length < 2) {
                return 0.85; // Initial neutral smoothness
            }

            const meanJerk = jerks.reduce((sum, j) => sum + j, 0) / jerks.length;
            const refJerk = this.config.referenceJerk || 350.0;

            const smoothness = Math.max(0.0, Math.min(1.0, 1.0 - (meanJerk / refJerk)));
            return Number(smoothness.toFixed(3));
        }

        getJerkMetrics(jerks) {
            if (!jerks || jerks.length < 2) {
                return { rawJerk: 0.0, filteredJerk: 0.0 };
            }
            const meanJerk = jerks.reduce((sum, j) => sum + j, 0) / jerks.length;
            const refJerk = this.config.referenceJerk || 350.0;
            return {
                rawJerk: Number(meanJerk.toFixed(1)),
                filteredJerk: Number(Math.min(refJerk * 2.0, meanJerk).toFixed(1))
            };
        }

        /**
         * 4E. Left / Right Symmetry
         * Compares bilateral movement kinematics (ROM ratio and velocity balance).
         */
        calculateSymmetry(leftRom, rightRom, leftVel, rightVel) {
            const maxRom = Math.max(leftRom, rightRom, 1.0);
            const minRom = Math.min(leftRom, rightRom);
            const romRatio = minRom / maxRom;

            const maxVel = Math.max(leftVel, rightVel, 0.05);
            const minVel = Math.min(leftVel, rightVel);
            const velRatio = minVel / maxVel;

            const symmetry = (0.50 * romRatio) + (0.50 * velRatio);
            return Number(Math.max(0.0, Math.min(1.0, symmetry)).toFixed(3));
        }

        /**
         * 4F. Repetition Consistency (Sections 17, 18, 19)
         * Tracks cyclic movement peaks with confirmed local reversal and tempo-derived refractory period.
         */
        updateRepetitionTracker(timestampSec, signalAmplitude, currentBpm = 60) {
            if (this.repPeakCandidate === null) {
                this.repPeakCandidate = signalAmplitude;
                this.lastRepTimestamp = timestampSec;
                return;
            }

            const delta = signalAmplitude - this.repPeakCandidate;
            // Configurable refractory period derived from tempo (Section 18)
            const minPeakSeparation = Math.max(0.35, (60.0 / Math.max(40, currentBpm || 60)) * 0.40);

            if (this.repDirection >= 0 && delta < -0.04) {
                // Crest confirmed: transition to descending
                const duration = timestampSec - this.lastRepTimestamp;
                if (duration > minPeakSeparation && duration < 3.5) {
                    this.repetitions.push({
                        timestamp: timestampSec,
                        duration: Number(duration.toFixed(3)),
                        amplitude: Number(this.repPeakCandidate.toFixed(3))
                    });
                    if (this.repetitions.length > 8) this.repetitions.shift();
                    this.lastRepTimestamp = timestampSec;
                }
                this.repDirection = -1;
                this.repPeakCandidate = signalAmplitude;
            } else if (this.repDirection <= 0 && delta > 0.04) {
                // Trough detected: transition to ascending
                this.repDirection = 1;
                this.repPeakCandidate = signalAmplitude;
            } else {
                if (this.repDirection > 0 && signalAmplitude > this.repPeakCandidate) {
                    this.repPeakCandidate = signalAmplitude;
                } else if (this.repDirection < 0 && signalAmplitude < this.repPeakCandidate) {
                    this.repPeakCandidate = signalAmplitude;
                }
            }
        }

        calculateRepetitionConsistency() {
            if (this.repetitions.length < 2) {
                return 0.85; // Prior expected baseline
            }

            const durations = this.repetitions.map(r => r.duration);
            const amplitudes = this.repetitions.map(r => r.amplitude);

            const calcCv = (arr) => {
                const mean = arr.reduce((a, b) => a + b, 0) / arr.length;
                if (mean <= 0.001) return 0;
                const variance = arr.reduce((sum, v) => sum + Math.pow(v - mean, 2), 0) / arr.length;
                return Math.sqrt(variance) / mean;
            };

            const cvDur = calcCv(durations);
            const cvAmp = calcCv(amplitudes);

            const durConsistency = Math.max(0.0, 1.0 - Math.min(1.0, cvDur));
            const ampConsistency = Math.max(0.0, 1.0 - Math.min(1.0, cvAmp));

            const consistency = (0.50 * durConsistency) + (0.50 * ampConsistency);
            return Number(Math.max(0.0, Math.min(1.0, consistency)).toFixed(3));
        }

        /**
         * 5. Tracking Confidence Score
         * Separate from movement quality: reflects measurement integrity and landmark visibility.
         */
        calculateConfidence(rawLandmarks) {
            if (!rawLandmarks || rawLandmarks.length < 33) return 0.0;

            // Relevant key joint landmarks
            const jointIndices = [11, 12, 23, 24, 25, 26, 27, 28]; // Shoulders, Hips, Knees, Ankles
            let visSum = 0;
            for (let idx of jointIndices) {
                const lm = rawLandmarks[idx];
                visSum += (typeof lm?.visibility === 'number' ? lm.visibility : (lm?.presence ?? 1.0));
            }
            const avgVisibility = visSum / jointIndices.length;

            // Buffer length sufficiency
            const bufferSufficiency = Math.min(1.0, this.buffer.length / this.config.minFramesForMetrics);

            const confidence = (0.60 * avgVisibility) + (0.40 * bufferSufficiency);
            return Number(Math.max(0.0, Math.min(1.0, confidence)).toFixed(3));
        }

        /**
         * 6. Composite Movement Quality Score
         * Weighted combination of ROM, Smoothness, Symmetry, and Repetition Consistency.
         */
        calculateMovementQuality(rom, smoothness, symmetry, consistency) {
            const w = this.config.weights;
            const quality = (w.rom * rom) + (w.smoothness * smoothness) + (w.symmetry * symmetry) + (w.consistency * consistency);
            return Number(Math.max(0.0, Math.min(1.0, quality)).toFixed(3));
        }

        /**
         * 7. Rhythm Intelligence
         * Compares movement events with beats from NuroSync or canonical tempo grid.
         */
        evaluateRhythm(timestampSec, currentBpm = 60, nuroSyncInstance = null) {
            const bpm = Number.isFinite(currentBpm) && currentBpm > 0 ? currentBpm : 60;
            const beatPeriod = 60.0 / bpm;
            const toleranceMs = (60000.0 / bpm) * 0.20;

            let nearestBeat = null;
            if (nuroSyncInstance && Array.isArray(nuroSyncInstance.beatTimestamps) && nuroSyncInstance.beatTimestamps.length > 0) {
                const beats = nuroSyncInstance.beatTimestamps;
                let minDiff = Infinity;
                for (let i = 0; i < beats.length; i++) {
                    const diff = Math.abs(timestampSec - beats[i]);
                    if (diff < minDiff) {
                        minDiff = diff;
                        nearestBeat = beats[i];
                    }
                }
            } else {
                // Cannot calculate synchronization without beat timestamps
                return {
                    sync: null,
                    timingErrorMs: null,
                    missedBeats: this.missedBeatsCount,
                    averageErrorMs: 0,
                    timingVariance: 0,
                    successfulBeats: this.successfulBeatsCount
                };
            }

            const timingErrorSec = nearestBeat !== null ? (timestampSec - nearestBeat) : 0;
            const timingErrorMs = Math.round(timingErrorSec * 1000);
            const absErrorMs = Math.abs(timingErrorMs);

            const isWithinTolerance = absErrorMs <= toleranceMs;
            if (isWithinTolerance) {
                this.successfulBeatsCount += 1;
            }

            this.rhythmHistory.push({
                timestamp: timestampSec,
                timingErrorMs,
                absErrorMs,
                isWithinTolerance
            });
            if (this.rhythmHistory.length > 20) this.rhythmHistory.shift();

            // Missing beats tracking
            if (this.lastEvaluatedBeatTime !== null && (timestampSec - this.lastEvaluatedBeatTime) > (beatPeriod * 1.8)) {
                const missed = Math.floor((timestampSec - this.lastEvaluatedBeatTime) / beatPeriod) - 1;
                if (missed > 0) {
                    this.missedBeatsCount += missed;
                }
            }
            this.lastEvaluatedBeatTime = timestampSec;

            // Rolling statistics
            const totalEvents = this.rhythmHistory.length;
            const avgErrorMs = totalEvents > 0 
                ? (this.rhythmHistory.reduce((s, e) => s + e.absErrorMs, 0) / totalEvents) 
                : 0;

            const signedMean = totalEvents > 0
                ? (this.rhythmHistory.reduce((s, e) => s + e.timingErrorMs, 0) / totalEvents)
                : 0;
            const variance = totalEvents > 1
                ? (this.rhythmHistory.reduce((s, e) => s + Math.pow(e.timingErrorMs - signedMean, 2), 0) / (totalEvents - 1))
                : 0;

            // Dynamic instant sync score
            const syncScore = Math.max(0.0, Math.min(1.0, 1.0 - (absErrorMs / toleranceMs)));

            return {
                sync: Number(syncScore.toFixed(2)),
                timingErrorMs: timingErrorMs,
                missedBeats: this.missedBeatsCount,
                averageErrorMs: Number(avgErrorMs.toFixed(1)),
                timingVariance: Number(variance.toFixed(1)),
                successfulBeats: this.successfulBeatsCount
            };
        }

        /**
         * Main Frame Processing Loop
         * Accepts raw landmarks, runs normalization, buffers frames, extracts metrics,
         * evaluates rhythm, and produces the canonical Movement State Object.
         * 
         * @param {Array<Object>|null} rawLandmarks 
         * @param {number} diffScore 
         * @param {number} timestampSec 
         * @param {Object} nuroSyncInstance 
         * @param {number|null} frameId
         * @returns {Object} Structured Movement State Object
         */
        processFrame(rawLandmarks, diffScore = 0, timestampSec = null, nuroSyncInstance = null, frameId = null) {
            const now = typeof timestampSec === 'number' && Number.isFinite(timestampSec)
                ? timestampSec
                : (performance.now() / 1000.0);

            // Error Handling: Landmark Loss or Camera Occlusion
            if (!rawLandmarks || rawLandmarks.length < 33) {
                const degradedConfidence = Math.max(0.0, this.latestState.confidence * 0.85);
                const state = diffScore < this.config.motionThreshold ? 'IDLE' : 'LOW_CONFIDENCE';
                this.latestState = {
                    ...this.latestState,
                    timestamp: Math.round(now * 1000),
                    state,
                    confidence: Number(degradedConfidence.toFixed(2))
                };
                return this.latestState;
            }

            // Normalization
            const normResult = this.normalizeLandmarks(rawLandmarks);
            const confidence = this.calculateConfidence(rawLandmarks);

            // 2. Bounded Temporal Buffer
            this.buffer.push({
                timestamp: now,
                frameId: frameId,
                landmarks: rawLandmarks,
                normalizedLandmarks: normResult ? normResult.normalizedLandmarks : null,
                confidence
            });

            // Discard frames exceeding time window or max frames limit
            while (
                this.buffer.length > this.config.maxFrames ||
                (this.buffer.length > 2 && (now - this.buffer[0].timestamp) > this.config.bufferDurationSec)
            ) {
                this.buffer.shift();
            }

            // 8. Distinguish State: IDLE vs WARMING_UP vs LOW_CONFIDENCE vs ACTIVE
            let state = 'ACTIVE';
            if (diffScore < this.config.motionThreshold) {
                state = 'IDLE';
            } else if (this.buffer.length < this.config.minFramesForMetrics) {
                state = 'WARMING_UP';
            } else if (confidence < this.config.minConfidenceThreshold) {
                state = 'LOW_CONFIDENCE';
            }

            // 4. Feature Extraction
            const romMetrics = this.calculateRangeOfMotion();
            const velMetrics = this.calculateVelocityAndAcceleration(now);
            const smoothness = this.calculateSmoothness(velMetrics.jerks);
            const jerkMetrics = this.getJerkMetrics(velMetrics.jerks);
            const symmetry = this.calculateSymmetry(
                romMetrics.leftRom,
                romMetrics.rightRom,
                velMetrics.leftVel,
                velMetrics.rightVel
            );

            // 7. Rhythm Intelligence
            const currentBpm = (nuroSyncInstance && typeof nuroSyncInstance.currentBpm === 'number') 
                ? nuroSyncInstance.currentBpm 
                : 60;

            // Update repetition peak tracking with dynamic refractory period
            const liftSignal = normResult?.normalizedLandmarks ? normResult.normalizedLandmarks[27].y : 0;
            this.updateRepetitionTracker(now, liftSignal, currentBpm);
            const consistency = this.calculateRepetitionConsistency();

            // 6. Movement Quality Score
            const movementQuality = this.calculateMovementQuality(
                romMetrics.rom,
                smoothness,
                symmetry,
                consistency
            );

            const rhythm = this.evaluateRhythm(now, currentBpm, nuroSyncInstance);

            // 9. Single Structured Movement State Object
            this.latestState = {
                timestamp: Math.round(now * 1000),
                state,
                movement: {
                    rom: romMetrics.rom,
                    velocity: velMetrics.velocity,
                    smoothness: smoothness,
                    rawJerk: jerkMetrics.rawJerk,
                    filteredJerk: jerkMetrics.filteredJerk,
                    symmetry: symmetry,
                    consistency: consistency,
                    quality: movementQuality
                },
                rhythm: {
                    sync: rhythm.sync,
                    timingErrorMs: rhythm.timingErrorMs,
                    missedBeats: rhythm.missedBeats,
                    averageErrorMs: Math.round(rhythm.averageErrorMs),
                    timingVariance: Math.round(rhythm.timingVariance),
                    successfulBeats: rhythm.successfulBeats
                },
                confidence: Number(confidence.toFixed(2))
            };

            return this.latestState;
        }

        getState() {
            return this.latestState;
        }
    }

    return {
        MovementIntelligence,
        MOVEMENT_INTELLIGENCE_CONFIG
    };
}));
