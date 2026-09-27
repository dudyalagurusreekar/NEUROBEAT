/**
 * Audio Engine for NeuroBeat
 * Handles beat generation and real-time tempo adjustment using Tone.js
 */

class NeuroAudioEngine {
    constructor() {
        this.synth = null;
        this.speechSynth = null;
        this.drumSynth = null;
        this.bellSynth = null;
        this.woodSynth = null;
        this.pianoSynth = null;
        this.loop = null;
        this.bpm = 60;
        this.isPlaying = false;
        this.beatCallback = null;
        this.sessionType = 'gait_trainer';
        this.soundType = 'metronome';
        this.isStarted = false; // Added to track if audio has started
        this.recentBeats = [];
        this.lastBeatTime = performance.now();
        
        // Voice detection & acoustic syllable tracking properties
        this.microphoneStream = null;
        this.voiceAudioContext = null;
        this.voiceAnalyserNode = null;
        this.voiceDataArray = null;
        this.voiceDetectionActive = false;
        this.voiceVolume = 0;
        this.ambientNoise = 0.005;
        this.isVocalizing = false;
        this.lastOnsetMs = 0;
        this.lastVoiceTime = 0;
        this.vocalEventsCount = 0;
        this.vocalCadenceSpm = 0;
        this.recentVocalIntervals = [];
        this.recentVoiceScores = [];
        this.lastVoiceTimingErrorMs = 0;
        this.voiceSyncAccuracy = 0;
        this.voiceMonitoringAnimId = null;
        this.onVocalOnsetCallback = null;
    }

    async initialize() {
        try {
            // Initialize Tone.js audio context
            await Tone.start();
            console.log('Audio context started');

            // Create metronome synth
            this.synth = new Tone.Synth({
                oscillator: { type: "triangle" },
                envelope: { attack: 0.005, decay: 0.1, sustain: 0, release: 0.1 }
            }).toDestination();

            // Create drum synth
            this.drumSynth = new Tone.MembraneSynth({
                pitchDecay: 0.05,
                octaves: 2,
                oscillator: { type: "triangle" },
                envelope: { attack: 0.001, decay: 0.4, sustain: 0.01, release: 1.4 }
            }).toDestination();

            // Create soft bell synth
            this.bellSynth = new Tone.Synth({
                oscillator: { type: "sine" },
                envelope: { attack: 0.02, decay: 0.3, sustain: 0.1, release: 0.8 }
            }).toDestination();

            // Create wooden block synth
            this.woodSynth = new Tone.NoiseSynth({
                noise: { type: "brown" },
                envelope: { attack: 0.001, decay: 0.1, sustain: 0, release: 0.05 }
            }).toDestination();

            // Create piano synth
            this.pianoSynth = new Tone.Synth({
                oscillator: { type: "triangle" },
                envelope: { attack: 0.008, decay: 0.2, sustain: 0.3, release: 1.2 }
            }).toDestination();

            // Create transport loop
            this.setupLoop();

            return true;
        } catch (error) {
            console.error('Failed to initialize audio:', error);
            return false;
        }
    }

    async initializeVoiceDetection() {
        if (this.voiceDetectionActive && this.voiceAnalyserNode) {
            return true;
        }

        try {
            let stream = null;
            try {
                // Request microphone permission with noise suppression
                stream = await navigator.mediaDevices.getUserMedia({
                    audio: {
                        echoCancellation: true,
                        noiseSuppression: true,
                        autoGainControl: true
                    }
                });
            } catch (constraintErr) {
                console.warn('[AUDIO] Detailed audio constraints failed, trying basic audio: true', constraintErr);
                stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            }
            console.log('[AUDIO] Microphone stream access granted');
            this.microphoneStream = stream;

            const AudioContextClass = window.AudioContext || window.webkitAudioContext;
            if (!this.voiceAudioContext || this.voiceAudioContext.state === 'closed') {
                this.voiceAudioContext = new AudioContextClass();
            }
            if (this.voiceAudioContext.state === 'suspended') {
                await this.voiceAudioContext.resume();
            }

            const source = this.voiceAudioContext.createMediaStreamSource(stream);
            this.voiceAnalyserNode = this.voiceAudioContext.createAnalyser();
            this.voiceAnalyserNode.fftSize = 1024;
            this.voiceAnalyserNode.smoothingTimeConstant = 0.2;
            source.connect(this.voiceAnalyserNode);

            this.voiceDataArray = new Float32Array(this.voiceAnalyserNode.fftSize);
            this.voiceDetectionActive = true;

            // Start vocal syllable monitoring
            this.startVoiceMonitoring();

            return true;
        } catch (error) {
            console.warn('[AUDIO] Voice detection disabled - continuing without microphone:', error);
            this.voiceDetectionActive = false;
            return false;
        }
    }

    startVoiceMonitoring() {
        if (!this.voiceAnalyserNode || !this.voiceDataArray) return;
        this.voiceDetectionActive = true;

        const analyzeVoice = () => {
            if (!this.voiceDetectionActive || !this.voiceAnalyserNode) return;

            this.voiceAnalyserNode.getFloatTimeDomainData(this.voiceDataArray);

            // Compute true RMS energy
            let sumSquares = 0;
            for (let i = 0; i < this.voiceDataArray.length; i++) {
                const sample = this.voiceDataArray[i];
                sumSquares += sample * sample;
            }
            const rms = Math.sqrt(sumSquares / this.voiceDataArray.length);
            this.voiceVolume = rms;

            const nowMs = performance.now();

            // Track ambient noise floor when quiet
            if (!this.isVocalizing) {
                this.ambientNoise = 0.98 * this.ambientNoise + 0.02 * rms;
            }
            // Dynamic threshold: well above ambient background noise
            const threshold = Math.max(0.015, this.ambientNoise * 2.5 + 0.010);

            // Rising edge: Vocal syllable onset detection (minimum 180ms refractory debounce)
            if (rms > threshold && !this.isVocalizing && (nowMs - this.lastOnsetMs) > 180) {
                this.isVocalizing = true;
                const prevOnset = this.lastOnsetMs;
                this.lastOnsetMs = nowMs;
                this.lastVoiceTime = nowMs;
                this.vocalEventsCount++;

                // Track syllable cadence (SPM = Syllables Per Minute)
                if (prevOnset > 0) {
                    const intervalMs = nowMs - prevOnset;
                    if (intervalMs >= 180 && intervalMs <= 3000) {
                        this.recentVocalIntervals.push(intervalMs);
                        if (this.recentVocalIntervals.length > 10) this.recentVocalIntervals.shift();
                        const meanInterval = this.recentVocalIntervals.reduce((a, b) => a + b, 0) / this.recentVocalIntervals.length;
                        this.vocalCadenceSpm = Math.round(60000 / meanInterval);
                    }
                }

                // Deterministic synchronization against rhythmic metronome beat
                const nearestBeat = this.getNearestBeatTimestamp(nowMs);
                const timingErrorMs = Math.round(Math.abs(nowMs - nearestBeat));
                const signedErrorMs = Math.round(nowMs - nearestBeat);
                this.lastVoiceTimingErrorMs = timingErrorMs;

                // Dynamic beat tolerance: 20% of beat interval (60000 / BPM * 0.20)
                const currentBpm = (this.bpm && this.bpm > 0) ? this.bpm : 60;
                const toleranceMs = (60000.0 / currentBpm) * 0.20;
                const instantScore = Math.max(0, Math.min(100, Math.round(100 * (1 - timingErrorMs / toleranceMs))));

                this.recentVoiceScores.push(instantScore);
                if (this.recentVoiceScores.length > 20) this.recentVoiceScores.shift();
                this.voiceSyncAccuracy = Math.round(this.recentVoiceScores.reduce((a, b) => a + b, 0) / this.recentVoiceScores.length);

                // Notify callback for visual animations or recording
                if (typeof this.onVocalOnsetCallback === 'function') {
                    try {
                        this.onVocalOnsetCallback({
                            timestamp: nowMs,
                            timingErrorMs,
                            signedErrorMs,
                            score: instantScore,
                            averageScore: this.voiceSyncAccuracy,
                            cadence: this.vocalCadenceSpm || currentBpm,
                            count: this.vocalEventsCount
                        });
                    } catch (e) {
                        console.warn('[VOICE_CALLBACK_ERR]', e);
                    }
                }
            } else if ((rms < threshold * 0.65 && (nowMs - this.lastOnsetMs) > 120) || (nowMs - this.lastOnsetMs) > 750) {
                // Falling edge or max utterance duration: Reset vocalizing flag for next syllable
                this.isVocalizing = false;
            }

            this.voiceMonitoringAnimId = requestAnimationFrame(analyzeVoice);
        };

        if (this.voiceMonitoringAnimId) cancelAnimationFrame(this.voiceMonitoringAnimId);
        this.voiceMonitoringAnimId = requestAnimationFrame(analyzeVoice);
    }

    getVoiceSyncAccuracy() {
        if (this.vocalEventsCount === 0 || this.recentVoiceScores.length === 0) {
            return 0;
        }
        return Math.round(this.voiceSyncAccuracy);
    }

    getVocalCadence() {
        if (this.vocalEventsCount < 2 || !this.vocalCadenceSpm) {
            return 0;
        }
        return this.vocalCadenceSpm;
    }

    getVocalCount() {
        return this.vocalEventsCount;
    }

    isVoiceActive() {
        const nowMs = performance.now();
        return (nowMs - this.lastVoiceTime) < 1200;
    }

    getVoiceVolume() {
        return this.voiceVolume || 0;
    }

    setOnVocalOnsetCallback(callback) {
        this.onVocalOnsetCallback = callback;
    }

    stopVoiceMonitoring() {
        this.voiceDetectionActive = false;
        if (this.voiceMonitoringAnimId) {
            cancelAnimationFrame(this.voiceMonitoringAnimId);
            this.voiceMonitoringAnimId = null;
        }
        if (this.microphoneStream) {
            try {
                this.microphoneStream.getTracks().forEach(track => track.stop());
            } catch (e) {}
                this.microphoneStream = null;
        }
    }

    setupLoop() {
        // Set up the transport loop for beats
        Tone.Transport.scheduleRepeat((time) => {
            // Guard: Absolutely do not play audio if engine is not playing or session is not running!
            if (!this.isPlaying || !this.isStarted || (typeof sessionData !== 'undefined' && !sessionData.isRunning)) {
                return;
            }

            // Play appropriate sound based on sound type and session type
            switch (this.soundType) {
                case 'drum':
                    this.drumSynth.triggerAttackRelease("C2", "16n", time);
                    break;
                case 'soft_bell':
                    this.bellSynth.triggerAttackRelease("C6", "8n", time);
                    break;
                case 'wooden_block':
                    this.woodSynth.triggerAttackRelease("8n", time);
                    break;
                case 'piano':
                    this.pianoSynth.triggerAttackRelease("C4", "8n", time);
                    break;
                default: // metronome
                    this.synth.triggerAttackRelease("C5", "8n", time);
                    break;
            }

            // Record beat timestamp for kinematic synchronization
            const beatNow = performance.now();
            this.lastBeatTime = beatNow;
            this.recentBeats.push(beatNow);
            if (this.recentBeats.length > 60) this.recentBeats.shift();

            // Record beat timestamp in NuroSync
            if (window.nuroSync && typeof window.nuroSync.recordBeat === 'function') {
                window.nuroSync.recordBeat(beatNow / 1000);
            }

            // Trigger beat visual callback if set
            if (this.beatCallback) {
                Tone.Draw.schedule(() => {
                    this.beatCallback();
                }, time);
            }
        }, "4n"); // Quarter note intervals
    }

    getNearestBeatTimestamp(timestampMs) {
        const t = timestampMs || performance.now();
        const currentBpm = (this.bpm && this.bpm > 0) ? this.bpm : 60;
        const beatPeriodMs = 60000.0 / currentBpm;

        if (!this.recentBeats || this.recentBeats.length === 0) {
            if (this.lastBeatTime) {
                const elapsed = t - this.lastBeatTime;
                return this.lastBeatTime + Math.round(elapsed / beatPeriodMs) * beatPeriodMs;
            }
            return t;
        }

        let nearest = this.recentBeats[0];
        let minDiff = Math.abs(t - nearest);
        for (let i = 1; i < this.recentBeats.length; i++) {
            const diff = Math.abs(t - this.recentBeats[i]);
            if (diff < minDiff) {
                minDiff = diff;
                nearest = this.recentBeats[i];
            }
        }

        // Project upcoming beat from the latest recorded beat (handles anticipation of next beat)
        const lastRecorded = this.recentBeats[this.recentBeats.length - 1];
        if (t > lastRecorded) {
            const cycles = Math.max(1, Math.round((t - lastRecorded) / beatPeriodMs));
            const projected = lastRecorded + cycles * beatPeriodMs;
            const projDiff = Math.abs(t - projected);
            if (projDiff < minDiff) {
                nearest = projected;
                minDiff = projDiff;
            }
        }

        return nearest;
    }

    setBPM(bpm) {
        // Different BPM ranges for different session types
        if (this.sessionType === 'speech_rhythm') {
            this.bpm = Math.max(80, Math.min(bpm, 180)); // Speech: 80-180 SPM
        } else {
            this.bpm = Math.max(40, Math.min(bpm, 200)); // Gait: 40-200 BPM
        }
        Tone.Transport.bpm.value = this.bpm;
        console.log(`BPM set to: ${this.bpm}`);
    }

    start() {
        if (!this.isPlaying) {
            Tone.Transport.start();
            this.isPlaying = true;
            this.isStarted = true; // Mark as started
            console.log('Audio engine started');
        }
    }

    stop() {
        this.isPlaying = false;
        this.isStarted = false;
        this.stopVoiceMonitoring();
        try {
            if (typeof Tone !== 'undefined' && Tone.Transport) {
                Tone.Transport.stop();
                Tone.Transport.position = 0;
            }
        } catch (e) {
            console.warn('[AUDIO] Error stopping Tone.Transport:', e);
        }
        try {
            if (this.synth && typeof this.synth.triggerRelease === 'function') this.synth.triggerRelease();
            if (this.drumSynth && typeof this.drumSynth.triggerRelease === 'function') this.drumSynth.triggerRelease();
            if (this.bellSynth && typeof this.bellSynth.triggerRelease === 'function') this.bellSynth.triggerRelease();
            if (this.pianoSynth && typeof this.pianoSynth.triggerRelease === 'function') this.pianoSynth.triggerRelease();
        } catch (e) {
            console.warn('[AUDIO] Error releasing synths:', e);
        }
        console.log('Audio engine stopped');
    }

    pause() {
        if (this.isPlaying) {
            Tone.Transport.pause();
            this.isPlaying = false;
            console.log('Audio engine paused');
        }
    }

    resume() {
        if (!this.isPlaying) {
            Tone.Transport.start();
            this.isPlaying = true;
            console.log('Audio engine resumed');
        }
    }

    setBeatCallback(callback) {
        this.beatCallback = callback;
    }

    // Alternative beat types
    setMetronomeSound() {
        this.synth.dispose();
        this.synth = new Tone.Synth({
            oscillator: { type: "triangle" },
            envelope: {
                attack: 0.001,
                decay: 0.1,
                sustain: 0,
                release: 0.05
            }
        }).toDestination();
    }

    setSoftBeatSound() {
        this.synth.dispose();
        this.synth = new Tone.Synth({
            oscillator: { type: "sine" },
            envelope: {
                attack: 0.02,
                decay: 0.3,
                sustain: 0,
                release: 0.2
            }
        }).toDestination();
    }

    // Volume control
    setVolume(volume) {
        // Volume range: 0 to 1
        const dbVolume = Tone.gainToDb(Math.max(0.001, volume));
        this.synth.volume.value = dbVolume;
    }

    setSessionType(sessionType) {
        this.sessionType = sessionType;
        console.log(`Session type set to: ${sessionType}`);
    }

    setSoundType(soundType) {
        this.soundType = soundType;
        console.log(`Sound type set to: ${soundType}`);
    }

    // Cleanup
    dispose() {
        // Stop voice detection
        this.voiceDetectionActive = false;
        
        if (this.microphone) {
            this.microphone.close();
            this.microphone.dispose();
        }
        
        if (this.voiceAnalyzer) {
            this.voiceAnalyzer.dispose();
        }
        
        if (this.synth) {
            this.synth.dispose();
        }
        if (this.speechSynth) {
            this.speechSynth.dispose();
        }
        if (this.drumSynth) {
            this.drumSynth.dispose();
        }
        if (this.bellSynth) {
            this.bellSynth.dispose();
        }
        if (this.woodSynth) {
            this.woodSynth.dispose();
        }
        if (this.pianoSynth) {
            this.pianoSynth.dispose();
        }
        
        Tone.Transport.cancel();
        this.isPlaying = false;
        this.isStarted = false;
        console.log('Audio engine disposed');
    }
}

// Global audio engine instance
let audioEngine = null;
let currentBPM = 60; // Initialize currentBPM
let currentSoundType = 'metronome'; // Initialize currentSoundType

// Initialize audio engine
async function initializeAudio() {
    audioEngine = new NeuroAudioEngine();
    const initialized = await audioEngine.initialize();

    if (!initialized) {
        console.error('Failed to initialize audio engine');
        return false;
    }

    // Set up beat visualization callback
    audioEngine.setBeatCallback(triggerBeatVisualization);

    return true;
}

// Start audio with initial BPM
function startAudioEngine(bpm = 60) {
    if (!audioEngine) {
        console.error('Audio engine not initialized');
        return;
    }

    currentBPM = bpm; // Set initial BPM
    audioEngine.setBPM(currentBPM);
    audioEngine.start();
}

// Stop audio engine
function stopAudioEngine() {
    if (audioEngine) {
        audioEngine.stop();
    }
    try {
        if (typeof Tone !== 'undefined' && Tone.Transport) {
            Tone.Transport.stop();
            Tone.Transport.position = 0;
        }
    } catch (e) {}
    if (typeof window !== 'undefined' && window.activeAiAudioTrack) {
        try {
            window.activeAiAudioTrack.pause();
            window.activeAiAudioTrack.currentTime = 0;
        } catch (e) {}
        window.activeAiAudioTrack = null;
    }
}

// Adjust tempo in real-time
function adjustAudioTempo(newBPM) {
    if (audioEngine && audioEngine.isStarted) {
        currentBPM = Math.max(40, Math.min(200, newBPM));
        console.log(`BPM adjusted to: ${currentBPM}`);

        // The tempo will be updated in the next beat cycle
        return true;
    }
    return false;
}

// Change beat sound type
function changeSoundType(soundType) {
    if (!audioEngine || !audioEngine.isStarted) return;

    currentSoundType = soundType;
    audioEngine.setSoundType(soundType); // Ensure audioEngine's soundType is updated

    // Update oscillator settings based on sound type
    const soundConfig = getSoundConfig(soundType);

    // Assuming beatOscillator is accessible or managed within audioEngine
    // For this example, we'll assume we are directly manipulating the synth sound
    // This part might need adjustment based on how beatOscillator is actually implemented/accessed.
    // If audioEngine manages its synths internally, we should call a method on audioEngine
    // For now, let's use the existing setMetronomeSound and setSoftBeatSound as examples
    // and assume a way to set other types.

    // A more robust approach would be to have a method in NeuroAudioEngine to set the sound
    // For example: audioEngine.setBeatSound(soundConfig.waveType, soundConfig.frequency);

    // Based on the original code, it seems to re-create the synth. Let's adapt that.
    if (soundConfig.waveType === 'sawtooth' && soundType === 'drum') {
        audioEngine.drumSynth.set({
            oscillator: { type: soundConfig.waveType },
            envelope: { attack: 0.001, decay: 0.4, sustain: 0.01, release: 1.4 }
        });
    } else if (soundConfig.waveType === 'sine' && soundType === 'soft_bell') {
        audioEngine.bellSynth.set({
            oscillator: { type: soundConfig.waveType },
            envelope: { attack: 0.02, decay: 0.3, sustain: 0.1, release: 0.8 }
        });
    } else if (soundConfig.waveType === 'brown' && soundType === 'wooden_block') {
        audioEngine.woodSynth.set({
            noise: { type: "brown" }, // Assuming brown noise for wooden block
            envelope: { attack: 0.001, decay: 0.1, sustain: 0, release: 0.05 }
        });
    } else if (soundConfig.waveType === 'triangle' && soundType === 'piano') {
        audioEngine.pianoSynth.set({
            oscillator: { type: soundConfig.waveType },
            envelope: { attack: 0.008, decay: 0.2, sustain: 0.3, release: 1.2 }
        });
    } else { // metronome
        audioEngine.setMetronomeSound(); // Re-initialize synth for metronome
    }

    console.log(`Sound type changed to: ${soundType}`);
}

// Get sound configuration
function getSoundConfig(soundType) {
    const soundConfigs = {
        'metronome': { waveType: 'square', frequency: 800 },
        'drum': { waveType: 'sawtooth', frequency: 100 }, // Frequency might not be directly applicable to MembraneSynth
        'soft_bell': { waveType: 'sine', frequency: 1200 },
        'wooden_block': { waveType: 'triangle', frequency: 400 }, // Noise synths don't typically have a 'type' like oscillators, but we can map it.
        'piano': { waveType: 'triangle', frequency: 523.25 } // C5
    };

    return soundConfigs[soundType] || soundConfigs['metronome'];
}

let beatVisualizationCount = 0;
function triggerBeatVisualization() {
    const beatIndicator = document.getElementById('beatIndicator');
    const beatVisualizer = document.getElementById('beatVisualizer');
    const syllablePrompt = document.getElementById('speechSyllablePrompt');

    if (beatIndicator && beatVisualizer) {
        // Scale animation for beat indicator
        beatIndicator.style.transform = 'scale(1.25)';
        beatVisualizer.style.borderColor = '#01aac5';

        // Reset after short duration
        setTimeout(() => {
            beatIndicator.style.transform = 'scale(0.8)';
            beatVisualizer.style.borderColor = 'var(--neuro-primary-border, #e2e8f0)';
        }, 110);
    }

    if (syllablePrompt) {
        syllablePrompt.style.transform = 'scale(1.08)';
        syllablePrompt.style.color = '#38bdf8';
        setTimeout(() => {
            syllablePrompt.style.transform = 'scale(1.0)';
            syllablePrompt.style.color = '#00e5ff';
        }, 120);

        // Sequentially pulse individual syllable targets (0, 1, 2, 3)
        const sylIndex = beatVisualizationCount % 4;
        beatVisualizationCount++;
        const targetSyl = document.getElementById(`syl_${sylIndex}`);
        if (targetSyl) {
            targetSyl.style.opacity = '1.0';
            targetSyl.style.textDecoration = 'underline';
            for (let i = 0; i < 4; i++) {
                if (i !== sylIndex) {
                    const other = document.getElementById(`syl_${i}`);
                    if (other) {
                        other.style.opacity = '0.6';
                        other.style.textDecoration = 'none';
                    }
                }
            }
        }
    }
}

// Audio preference settings
function setAudioPreference(type) {
    if (!audioEngine) return;

    switch (type) {
        case 'metronome':
            audioEngine.setSoundType('metronome'); // Explicitly set sound type
            audioEngine.setMetronomeSound();
            break;
        case 'soft':
            audioEngine.setSoundType('soft_bell'); // Explicitly set sound type
            audioEngine.setSoftBeatSound();
            break;
        case 'drum':
            audioEngine.setSoundType('drum');
            break;
        case 'wooden_block':
            audioEngine.setSoundType('wooden_block');
            break;
        case 'piano':
            audioEngine.setSoundType('piano');
            break;
        default:
            // Default sine wave sound
            audioEngine.setSoundType('metronome');
            audioEngine.setMetronomeSound();
            break;
    }
}

// Volume control
function setAudioVolume(volume) {
    if (audioEngine) {
        audioEngine.setVolume(volume);
    }
}

// Cleanup function
function cleanupAudio() {
    if (audioEngine) {
        audioEngine.dispose();
        audioEngine = null;
    }
}

// Clean up when leaving page
window.addEventListener('beforeunload', cleanupAudio);

window.getNearestBeatTimestamp = function(timestampMs) {
    if (audioEngine && typeof audioEngine.getNearestBeatTimestamp === 'function') {
        return audioEngine.getNearestBeatTimestamp(timestampMs);
    }
    return timestampMs || performance.now();
};