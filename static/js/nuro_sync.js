/**
 * NuroSync v2.0 - Deterministic Rhythmic Synchronization & Temporal Phase Engine
 * 
 * Compares real movement event timestamps with rhythmic beat timestamps on a canonical monotonic timeline.
 * Calculates:
 * - Signed error (early anticipation vs late lag in ms)
 * - Absolute error (ms)
 * - Mean Absolute Error (MAE), Median Absolute Error, Standard Deviation (SD), Root Mean Square Error (RMSE)
 * - 90th (P90) and 95th (P95) percentile absolute errors
 * - Early, Late, and On-Time distribution counts and On-Time percentage
 * - Cycle-normalized phase error: phase = signed_error / beat_period
 * - Rhythm Alignment Score (0-100 deterministic engineering composite)
 * 
 * ZERO Math.random() - 100% deterministic and reproducible.
 */

class NuroSync {
    constructor(toleranceMs = null) {
        this.currentBpm = 60;
        this.customTolerance = toleranceMs;
        this.toleranceMs = toleranceMs !== null ? toleranceMs : this.calculateDynamicTolerance(this.currentBpm);
        this.beatTimestamps = []; // seconds (monotonic performance.now() / 1000)
        this.stepEvents = [];
        this.signedErrors = [];   // in ms
        this.absoluteErrors = []; // in ms
        this.phaseErrors = [];    // cycle-normalized [-0.5, 0.5]

        this.earlyCount = 0;
        this.lateCount = 0;
        this.onTimeCount = 0;
        this.totalStepsEvaluated = 0;

        // Engineering score: Rhythm Alignment Score (0-100) - initialized to null until input events are evaluated
        this.rhythmAlignmentScore = null;
    }

    calculateDynamicTolerance(bpm) {
        const validBpm = (Number.isFinite(bpm) && bpm > 0) ? bpm : 60;
        const beatPeriodMs = 60000.0 / validBpm;
        return Number((beatPeriodMs * 0.20).toFixed(1));
    }

    setBpm(bpm) {
        if (Number.isFinite(bpm) && bpm > 0) {
            this.currentBpm = bpm;
            if (this.customTolerance === null) {
                this.toleranceMs = this.calculateDynamicTolerance(bpm);
            }
        }
    }

    setTolerance(toleranceMs) {
        if (Number.isFinite(toleranceMs) && toleranceMs > 0) {
            this.customTolerance = toleranceMs;
            this.toleranceMs = toleranceMs;
        } else if (toleranceMs === null) {
            this.customTolerance = null;
            this.toleranceMs = this.calculateDynamicTolerance(this.currentBpm);
        }
    }

    recordBeat(timestampSec) {
        const t = timestampSec !== undefined ? timestampSec : (performance.now() / 1000.0);
        this.beatTimestamps.push(t);
        // Retain rolling window of 60 beats
        if (this.beatTimestamps.length > 60) {
            this.beatTimestamps.shift();
        }
    }

    /**
     * Associates a movement step event with preceding, nearest, and following beats
     * and evaluates timing statistics.
     * @param {Object|number} stepEvent - { timestamp, side, type, confidence } or timestamp in seconds
     */
    evaluateStep(stepEvent) {
        const stepTime = typeof stepEvent === 'object' ? stepEvent.timestamp : stepEvent;
        if (!Number.isFinite(stepTime)) {
            return this.getInstantaneousMetrics(0, 0, 0, false);
        }

        if (this.customTolerance === null) {
            this.toleranceMs = this.calculateDynamicTolerance(this.currentBpm);
        }

        this.stepEvents.push(typeof stepEvent === 'object' ? stepEvent : { timestamp: stepTime });
        if (this.stepEvents.length > 60) this.stepEvents.shift();

        if (this.beatTimestamps.length === 0) {
            return {
                valid: false,
                stepTime,
                timingErrorMs: null,
                timing_error_ms: null,
                signedErrorMs: null,
                signed_error_ms: null,
                absErrorMs: null,
                abs_error_ms: null,
                phaseError: null,
                phase_error: null,
                isWithinTolerance: false,
                is_within_tolerance: false,
                rhythmAlignmentScore: null,
                rhythm_alignment_score: null,
                phase: 'NO_DATA'
            };
        }

        // 1. Identify preceding, nearest, and following beats
        let precedingBeat = null;
        let followingBeat = null;
        let nearestBeat = this.beatTimestamps[0];
        let minDiff = Math.abs(stepTime - nearestBeat);

        for (let i = 0; i < this.beatTimestamps.length; i++) {
            const bt = this.beatTimestamps[i];
            const diff = Math.abs(stepTime - bt);

            if (bt <= stepTime) {
                if (precedingBeat === null || bt > precedingBeat) {
                    precedingBeat = bt;
                }
            } else if (bt > stepTime) {
                if (followingBeat === null || bt < followingBeat) {
                    followingBeat = bt;
                }
            }

            if (diff < minDiff) {
                minDiff = diff;
                nearestBeat = bt;
            }
        }

        // Beat period estimation (from target BPM or local beat delta)
        let beatPeriod = 60.0 / Math.max(30, this.currentBpm);
        if (precedingBeat !== null && followingBeat !== null && (followingBeat - precedingBeat) > 0.2) {
            beatPeriod = followingBeat - precedingBeat;
        }

        // Project upcoming beat if step is anticipating next beat not yet recorded
        if (followingBeat === null && precedingBeat !== null) {
            const projectedNext = precedingBeat + beatPeriod;
            const nextDiff = Math.abs(stepTime - projectedNext);
            if (nextDiff < minDiff) {
                nearestBeat = projectedNext;
                minDiff = nextDiff;
            }
        }

        // Signed error: positive = lag (late), negative = anticipation (early)
        const signedErrorSec = stepTime - nearestBeat;
        const signedErrorMs = Number((signedErrorSec * 1000).toFixed(1));
        const absErrorMs = Number(Math.abs(signedErrorMs).toFixed(1));

        // Cycle-normalized phase error [-0.5, 0.5]
        const phaseError = Number((signedErrorSec / beatPeriod).toFixed(3));

        // Update distribution buffers
        this.signedErrors.push(signedErrorMs);
        this.absoluteErrors.push(absErrorMs);
        this.phaseErrors.push(phaseError);

        if (this.signedErrors.length > 40) this.signedErrors.shift();
        if (this.absoluteErrors.length > 40) this.absoluteErrors.shift();
        if (this.phaseErrors.length > 40) this.phaseErrors.shift();

        this.totalStepsEvaluated += 1;

        const isWithinTolerance = absErrorMs <= this.toleranceMs;
        if (isWithinTolerance) {
            this.onTimeCount += 1;
        }
        if (signedErrorMs < -20) {
            this.earlyCount += 1;
        } else if (signedErrorMs > 20) {
            this.lateCount += 1;
        }

        // Deterministic instant score: 100 at 0 error, 0 at >= toleranceMs
        const instantScore = Math.max(0, Math.min(100, Math.round(100 * (1 - absErrorMs / this.toleranceMs))));
        if (this.totalStepsEvaluated === 1) {
            this.rhythmAlignmentScore = Number(instantScore.toFixed(1));
        } else {
            // Rolling exponential filter (alpha = 0.25)
            this.rhythmAlignmentScore = Number((0.25 * instantScore + 0.75 * this.rhythmAlignmentScore).toFixed(1));
        }

        return this.getInstantaneousMetrics(signedErrorMs, absErrorMs, phaseError, isWithinTolerance);
    }

    getInstantaneousMetrics(signedErrorMs, absErrorMs, phaseError, isWithinTolerance) {
        return {
            signedErrorMs,
            signed_error_ms: signedErrorMs,
            absErrorMs,
            abs_error_ms: absErrorMs,
            phaseError,
            phase_error: phaseError,
            isWithinTolerance,
            is_within_tolerance: isWithinTolerance,
            rhythmAlignmentScore: this.rhythmAlignmentScore,
            rhythm_alignment_score: this.rhythmAlignmentScore,
            medianAbsErrorMs: this.getMedian(this.absoluteErrors),
            rmseMs: this.getRMSE(this.signedErrors)
        };
    }

    getMedian(arr) {
        if (!arr || arr.length === 0) return 0;
        const sorted = [...arr].sort((a, b) => a - b);
        const mid = Math.floor(sorted.length / 2);
        if (sorted.length % 2 === 0) {
            return Number(((sorted[mid - 1] + sorted[mid]) / 2).toFixed(1));
        }
        return Number(sorted[mid].toFixed(1));
    }

    getPercentile(arr, p) {
        if (!arr || arr.length === 0) return 0;
        const sorted = [...arr].sort((a, b) => a - b);
        const idx = Math.min(sorted.length - 1, Math.max(0, Math.floor((p / 100) * sorted.length)));
        return Number(sorted[idx].toFixed(1));
    }

    getRMSE(arr) {
        if (!arr || arr.length === 0) return 0;
        const sumSq = arr.reduce((acc, v) => acc + (v * v), 0);
        return Number(Math.sqrt(sumSq / arr.length).toFixed(1));
    }

    getMean(arr) {
        if (!arr || arr.length === 0) return 0;
        const sum = arr.reduce((acc, v) => acc + v, 0);
        return Number((sum / arr.length).toFixed(1));
    }

    getSD(arr) {
        if (!arr || arr.length < 2) return 0;
        const mean = this.getMean(arr);
        const variance = arr.reduce((acc, v) => acc + Math.pow(v - mean, 2), 0) / (arr.length - 1);
        return Number(Math.sqrt(variance).toFixed(1));
    }

    /**
     * Backward-compatible accuracy getter returning the deterministic Rhythm Alignment Score
     */
    getCurrentAccuracy() {
        if (this.totalStepsEvaluated === 0 || this.absoluteErrors.length === 0) return 0;
        return Math.max(0, Math.min(100, Math.round(this.rhythmAlignmentScore)));
    }

    getRhythmAlignmentScore() {
        if (this.totalStepsEvaluated === 0 || this.absoluteErrors.length === 0) return 0;
        return Math.max(0, Math.min(100, Math.round(this.rhythmAlignmentScore)));
    }

    getMetrics() {
        return this.getMetricsSummary();
    }

    /**
     * Returns full multi-dimensional synchronization metrics packet conforming to schema v2.0
     */
    getMetricsSummary() {
        if (this.absoluteErrors.length === 0) {
            return {
                valid: false,
                signed_error_ms_mean: 0,
                absolute_error_ms_mean: 0,
                median_abs_error_ms: 0,
                sd_ms: 0,
                rmse_ms: 0,
                p90_abs_error_ms: 0,
                p95_abs_error_ms: 0,
                on_time_pct: 0,
                early_events: 0,
                late_events: 0,
                on_time_events: 0,
                total_events: 0,
                rhythm_alignment_score: 0,
                mean_phase_error: 0
            };
        }

        const total = this.absoluteErrors.length;
        const onTime = this.absoluteErrors.filter(e => e <= this.toleranceMs).length;
        const onTimePct = Number(((onTime / total) * 100).toFixed(1));

        return {
            valid: true,
            signed_error_ms_mean: this.getMean(this.signedErrors),
            absolute_error_ms_mean: this.getMean(this.absoluteErrors),
            median_abs_error_ms: this.getMedian(this.absoluteErrors),
            sd_ms: this.getSD(this.signedErrors),
            rmse_ms: this.getRMSE(this.signedErrors),
            p90_abs_error_ms: this.getPercentile(this.absoluteErrors, 90),
            p95_abs_error_ms: this.getPercentile(this.absoluteErrors, 95),
            on_time_pct: onTimePct,
            early_events: this.earlyCount,
            late_events: this.lateCount,
            on_time_events: this.onTimeCount,
            total_events: this.totalStepsEvaluated,
            rhythm_alignment_score: Math.round(this.rhythmAlignmentScore),
            mean_phase_error: this.getMean(this.phaseErrors)
        };
    }

    reset() {
        this.beatTimestamps = [];
        this.stepEvents = [];
        this.signedErrors = [];
        this.absoluteErrors = [];
        this.phaseErrors = [];
        this.earlyCount = 0;
        this.lateCount = 0;
        this.onTimeCount = 0;
        this.totalStepsEvaluated = 0;
        this.rhythmAlignmentScore = null;
    }
}

// Global instance
window.nuroSync = new NuroSync();
