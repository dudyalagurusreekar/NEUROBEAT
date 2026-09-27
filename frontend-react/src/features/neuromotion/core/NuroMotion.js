/**
 * NuroMotion Public API
 * Unified decoupled facade for the entire NuroMotion sensing and telemetry pipeline.
 */

import { PoseEngine } from './PoseEngine';
import { POSE_CONFIG } from './PoseConfig';

export class NuroMotion {
  constructor(config = {}) {
    this.config = { ...POSE_CONFIG, ...config };
    this.engine = new PoseEngine(this.config);
  }

  /**
   * Starts camera capture and real-time MediaPipe Pose Landmarker Full processing
   * @param {Object} options
   * @param {HTMLVideoElement} options.videoElement
   * @param {number|string} [options.sessionId]
   * @param {number} [options.targetBpm=60]
   * @param {boolean} [options.useMic=true]
   */
  async start({ videoElement, sessionId = null, targetBpm = 60, useMic = true }) {
    return this.engine.startCamera(videoElement, { sessionId, targetBpm, useMic });
  }

  /**
   * Starts synthetic demo mode with simulated gait events
   * @param {number} [targetBpm=60]
   * @param {number|string} [sessionId=null]
   */
  startDemo(targetBpm = 60, sessionId = null) {
    this.engine.startDemoMode({ targetBpm, sessionId });
  }

  pause() {
    this.engine.pause();
  }

  resume() {
    this.engine.resume();
  }

  stop() {
    this.engine.stop();
  }

  reset() {
    this.engine.reset();
  }

  setTargetBpm(bpm) {
    this.engine.setTargetBpm(bpm);
  }

  getState() {
    return {
      isRunning: this.engine.isRunning,
      isPaused: this.engine.isPaused,
      isDemoMode: this.engine.isDemoMode,
      diagnostics: this.engine.getDiagnostics(),
    };
  }

  /**
   * Registers a metronome/audio synthesizer beat timestamp for synchronization alignment
   * @param {number} timestamp - timestamp in seconds
   */
  registerBeat(timestamp) {
    this.engine.audio.registerSynthesizedBeat(timestamp);
  }

  onMovementEvent(callback) {
    return this.engine.onMovementEvent(callback);
  }

  onMetrics(callback) {
    return this.engine.onMetrics(callback);
  }

  onPoseUpdate(callback) {
    return this.engine.onPoseUpdate(callback);
  }

  onTelemetry(callback) {
    return this.engine.onTelemetry(callback);
  }

  onError(callback) {
    return this.engine.onError(callback);
  }

  getDiagnostics() {
    return this.engine.getDiagnostics();
  }
}

// Default singleton instance for direct import
export const nuroMotionInstance = new NuroMotion();
