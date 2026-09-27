/**
 * NuroMotion & NuroSync Engines for Nuro-Beats
 * Implements deterministic movement event detection, rhythmic synchronization,
 * and adaptive tempo control as defined in ARCHITECTURE.md and PRD.md.
 */

import { NuroMotion as CoreNuroMotion } from '../features/neuromotion/core/NuroMotion';

export class NuroMotion {
  constructor(onStepEvent) {
    this.onStepEvent = onStepEvent;
    this.core = new CoreNuroMotion();
    this.isTracking = false;
    this.isDemoMode = false;
    this.videoElement = null;

    this.core.onMovementEvent((ev) => {
      if (this.onStepEvent) {
        this.onStepEvent(ev);
      }
    });
  }

  async startCamera(videoElement, targetBpm = 54) {
    this.videoElement = videoElement;
    this.isTracking = true;
    this.isDemoMode = false;

    const ok = await this.core.start({
      videoElement,
      targetBpm,
      useMic: true,
    });

    if (!ok) {
      this.startDemoMode(targetBpm);
    }
  }

  startDemoMode(targetBpm = 54) {
    this.isTracking = true;
    this.isDemoMode = true;
    this.core.startDemo(targetBpm);
  }

  stop() {
    this.isTracking = false;
    this.core.stop();
  }

  pause() {
    this.core.pause();
  }

  resume() {
    this.core.resume();
  }

  registerBeat(timestamp) {
    this.core.registerBeat(timestamp);
  }
}


export class NuroSync {
  constructor(toleranceMs = null, initialBpm = 60) {
    this.currentBpm = initialBpm;
    this.customTolerance = toleranceMs;
    this.toleranceMs = toleranceMs !== null ? toleranceMs : this.calculateDynamicTolerance(this.currentBpm);
    this.beatTimestamps = [];
    this.errorHistory = []; // list of timing errors in ms
    this.scoreHistory = []; // list of per-event sync scores
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

  recordBeat(timestamp) {
    this.beatTimestamps.push(timestamp);
    // Keep only recent 20 beats
    if (this.beatTimestamps.length > 20) {
      this.beatTimestamps.shift();
    }
  }

  evaluateStep(stepTimestamp) {
    if (this.beatTimestamps.length === 0) {
      return {
        valid: false,
        timingErrorMs: null,
        syncScore: null,
        rhythmAlignmentScore: null,
        isSynchronized: false,
        phase: 'NO_DATA'
      };
    }

    let minDiff = Infinity;
    // Find nearest beat timestamp
    for (const bTime of this.beatTimestamps) {
      const diff = Math.abs(stepTimestamp - bTime);
      if (diff < minDiff) {
        minDiff = diff;
      }
    }

    // Check upcoming projected beat if event anticipates next beat
    const lastRecorded = this.beatTimestamps[this.beatTimestamps.length - 1];
    if (stepTimestamp > lastRecorded) {
      const beatPeriod = 60.0 / this.currentBpm;
      const cycles = Math.max(1, Math.round((stepTimestamp - lastRecorded) / beatPeriod));
      const projected = lastRecorded + cycles * beatPeriod;
      const projDiff = Math.abs(stepTimestamp - projected);
      if (projDiff < minDiff) {
        minDiff = projDiff;
      }
    }

    const timingErrorMs = Math.round(minDiff * 1000);
    this.errorHistory.push(timingErrorMs);
    if (this.errorHistory.length > 30) this.errorHistory.shift();

    // Deterministic sync score formula based on dynamic beat tolerance
    // Error 0ms -> 100%, Error >= toleranceMs -> 0%
    const score = Math.max(0, Math.min(100, Math.round(100 * (1 - timingErrorMs / this.toleranceMs))));
    const isSynchronized = timingErrorMs <= this.toleranceMs;

    this.scoreHistory.push(score);
    if (this.scoreHistory.length > 30) this.scoreHistory.shift();

    if (this.rhythmAlignmentScore === null) {
      this.rhythmAlignmentScore = score;
    } else {
      this.rhythmAlignmentScore = Math.round(0.3 * score + 0.7 * this.rhythmAlignmentScore);
    }

    return {
      valid: true,
      timingErrorMs,
      syncScore: score,
      rhythmAlignmentScore: this.rhythmAlignmentScore,
      isSynchronized
    };
  }

  getAverageSync() {
    if (!this.scoreHistory || this.scoreHistory.length === 0) return null;
    const avgScore = this.scoreHistory.reduce((a, b) => a + b, 0) / this.scoreHistory.length;
    return Math.round(avgScore);
  }

  reset() {
    this.beatTimestamps = [];
    this.errorHistory = [];
    this.scoreHistory = [];
    this.rhythmAlignmentScore = null;
  }
}

export class AdaptationEngine {
  constructor(minSafeBpm = 45, maxSafeBpm = 72) {
    this.minSafeBpm = minSafeBpm;
    this.maxSafeBpm = maxSafeBpm;
    this.consecutiveHigh = 0;
    this.consecutiveLow = 0;
  }

  updateSafetyLimits(minBpm, maxBpm) {
    this.minSafeBpm = minBpm;
    this.maxSafeBpm = maxBpm;
  }

  evaluateAdaptation(currentBpm, recentSyncScores, lastStepIntervalSec = 1.0) {
    if (recentSyncScores.length < 3) {
      return { adapted: false, newBpm: currentBpm, reason: "Collecting initial rhythm telemetry" };
    }

    const recent3 = recentSyncScores.slice(-3);
    const avgSync = recent3.reduce((a, b) => a + b, 0) / recent3.length;

    // Freezing of Gait check (excessive step lag > 3.5s)
    if (lastStepIntervalSec > 3.5) {
      const protectedBpm = Math.max(this.minSafeBpm, currentBpm - 2);
      return {
        adapted: true,
        newBpm: protectedBpm,
        isFreezing: true,
        reason: `Freezing episode detected (${lastStepIntervalSec.toFixed(1)}s delay). Eased tempo to safe floor.`
      };
    }

    // High Synchronization (>88%): gently increase tempo towards therapeutic target
    if (avgSync >= 88) {
      this.consecutiveHigh += 1;
      this.consecutiveLow = 0;

      if (this.consecutiveHigh >= 4) {
        this.consecutiveHigh = 0;
        if (currentBpm < this.maxSafeBpm) {
          const nextBpm = currentBpm + 1;
          return {
            adapted: true,
            newBpm: nextBpm,
            reason: `High synchronization (${Math.round(avgSync)}%). Lifting tempo +1 BPM (Doctor ceiling: ${this.maxSafeBpm} BPM).`
          };
        }
      }
    } 
    // Low Synchronization (<75%): fatigue protection, gently ease tempo down
    else if (avgSync < 75) {
      this.consecutiveLow += 1;
      this.consecutiveHigh = 0;

      if (this.consecutiveLow >= 3) {
        this.consecutiveLow = 0;
        if (currentBpm > this.minSafeBpm) {
          const nextBpm = currentBpm - 1;
          return {
            adapted: true,
            newBpm: nextBpm,
            reason: `Sync dropped to ${Math.round(avgSync)}%. Easing tempo -1 BPM for joint and fatigue safety.`
          };
        }
      }
    } else {
      this.consecutiveHigh = 0;
      this.consecutiveLow = 0;
    }

    return { adapted: false, newBpm: currentBpm, reason: "Steady cadence in harmony with rhythm" };
  }
}
