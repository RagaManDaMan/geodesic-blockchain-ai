/**
 * diagnostic_listener.js — Generative audio driven by Geodesic AI simulation state.
 *
 * Six-voice polyphonic architecture (Bach cantata structure):
 *   V1 Soprano   — best routing path, bright, leading
 *   V2 Alto      — second routing path, lyrical
 *   V3 Tenor     — third routing path, purposeful
 *   V4 Bass      — fourth routing path, grounded
 *   V5 Continuo  — ComCoin supply signal, always present
 *   V6 Ripieno   — curl field aggregate, harmonic colouring only
 *
 * Harmonic minor scale with icosahedral geometry tuning.
 * All audio via Web Audio API — no external libraries.
 *
 * Public API:
 *   const dl = new DiagnosticListener();
 *   await dl.init(audioStatesObject);
 *   dl.play(targetDurationMs);
 *   dl.stop();
 */

/* =========================================================================
   VOICE — a single audio voice with oscillator → gain → reverb → panner
   ========================================================================= */

class Voice {
  constructor(ctx, masterGain, { type = 'triangle', pan = 0, baseGain = 0.12 } = {}) {
    this.ctx       = ctx;
    this.active    = false;
    this.baseGain  = baseGain;
    this.currentFreq = 0;

    // Oscillator
    this.osc  = ctx.createOscillator();
    this.osc.type = type;
    this.osc.frequency.value = 220;

    // Gain (per-voice volume)
    this.gain = ctx.createGain();
    this.gain.gain.value = 0;          // silent until activated

    // Stereo panner
    this.panner = ctx.createStereoPanner();
    this.panner.pan.value = pan;

    // Simple delay (for latency spatialization)
    this.delay = ctx.createDelay(2.0);
    this.delay.delayTime.value = 0;

    // Signal chain: osc → gain → delay → panner → master
    this.osc.connect(this.gain);
    this.gain.connect(this.delay);
    this.delay.connect(this.panner);
    this.panner.connect(masterGain);

    this.osc.start();
  }

  setFrequency(freq, rampTime = 0.15) {
    if (freq <= 0) return;
    this.currentFreq = freq;
    this.osc.frequency.linearRampToValueAtTime(
      freq, this.ctx.currentTime + rampTime
    );
  }

  setGain(vol, rampTime = 0.1) {
    this.gain.gain.linearRampToValueAtTime(
      Math.max(0, vol), this.ctx.currentTime + rampTime
    );
  }

  setDelay(seconds) {
    const clamped = Math.max(0, Math.min(0.5, seconds));
    this.delay.delayTime.linearRampToValueAtTime(
      clamped, this.ctx.currentTime + 0.1
    );
  }

  activate(freq, rampTime = 0.3) {
    this.active = true;
    this.setFrequency(freq, rampTime);
    this.setGain(this.baseGain, rampTime);
  }

  deactivate(rampTime = 0.5) {
    this.active = false;
    this.setGain(0, rampTime);
  }

  destroy() {
    try { this.osc.stop(); } catch (_) {}
    this.osc.disconnect();
    this.gain.disconnect();
    this.delay.disconnect();
    this.panner.disconnect();
  }
}


/* =========================================================================
   RIPIENO VOICE (V6) — harmonic tension via beating clusters
   ========================================================================= */

class RipienoVoice {
  constructor(ctx, masterGain) {
    this.ctx = ctx;
    this.baseGain = 0.06;

    // Two oscillators that beat against each other
    this.osc1 = ctx.createOscillator();
    this.osc2 = ctx.createOscillator();
    this.osc1.type = 'sine';
    this.osc2.type = 'sine';
    this.osc1.frequency.value = 220;
    this.osc2.frequency.value = 220;

    this.gain = ctx.createGain();
    this.gain.gain.value = 0;

    this.osc1.connect(this.gain);
    this.osc2.connect(this.gain);
    this.gain.connect(masterGain);

    this.osc1.start();
    this.osc2.start();
  }

  /**
   * Set tension from curl magnitude.
   *   curl < 0.5  → silent or tonic (consonance)
   *   curl 0.5–1.5 → 7th/2nd interval (mild tension)
   *   curl > 1.5  → semitone cluster (roughness from beating)
   */
  setTension(curlMag, refFreq, rampTime = 0.2) {
    const t = this.ctx.currentTime + rampTime;
    if (curlMag < 0.3) {
      // Silent — consonance
      this.gain.gain.linearRampToValueAtTime(0, t);
      return;
    }

    let f1 = refFreq;
    let f2;

    if (curlMag < 1.0) {
      // Minor 7th interval — mild tension
      f2 = refFreq * (2 ** (10 / 12));
      this.gain.gain.linearRampToValueAtTime(this.baseGain * 0.4, t);
    } else if (curlMag < 1.5) {
      // Major 2nd — more tension
      f2 = refFreq * (2 ** (2 / 12));
      this.gain.gain.linearRampToValueAtTime(this.baseGain * 0.7, t);
    } else {
      // Semitone cluster — maximum roughness (beating at ~10-30 Hz)
      f2 = refFreq * (2 ** (1 / 12));
      this.gain.gain.linearRampToValueAtTime(this.baseGain, t);
    }

    this.osc1.frequency.linearRampToValueAtTime(f1, t);
    this.osc2.frequency.linearRampToValueAtTime(f2, t);
  }

  destroy() {
    try { this.osc1.stop(); } catch (_) {}
    try { this.osc2.stop(); } catch (_) {}
    this.osc1.disconnect();
    this.osc2.disconnect();
    this.gain.disconnect();
  }
}


/* =========================================================================
   STATE INTERPOLATOR — continuous morphing between discrete snapshots
   ========================================================================= */

class StateInterpolator {
  constructor(audioData, targetDurationMs) {
    this.states          = audioData.states;
    this.nodeFrequencies = audioData.node_frequencies;
    this.nSteps          = audioData.n_steps;
    this.targetMs        = targetDurationMs;
    this.startTime       = null;
  }

  start(audioTime) {
    this.startTime = audioTime;
  }

  /** Return interpolated state at the given audio timestamp, or null if done. */
  getStateAt(audioTime) {
    if (this.startTime === null) return null;
    const elapsed  = (audioTime - this.startTime) * 1000;  // ms
    const progress = elapsed / this.targetMs;               // 0..1

    if (progress >= 1.0) return null;  // playback finished

    const rawIdx = progress * (this.states.length - 1);
    const n      = Math.floor(rawIdx);
    const alpha  = rawIdx - n;

    if (n >= this.states.length - 1) return this.states[this.states.length - 1];

    return this._interpolate(this.states[n], this.states[n + 1], alpha);
  }

  /** Progress fraction 0..1 */
  getProgress(audioTime) {
    if (!this.startTime) return 0;
    return Math.min(1, (audioTime - this.startTime) * 1000 / this.targetMs);
  }

  _interpolate(a, b, alpha) {
    const result = {};
    for (const key of Object.keys(a)) {
      const va = a[key], vb = b[key];
      if (typeof va === 'number' && typeof vb === 'number') {
        result[key] = va * (1 - alpha) + vb * alpha;
      } else if (Array.isArray(va) && Array.isArray(vb) &&
                 va.length > 0 && typeof va[0] === 'number') {
        // Numeric array — element-wise interpolation
        result[key] = va.map((v, i) => v * (1 - alpha) + (vb[i] ?? v) * alpha);
      } else {
        // Non-numeric (paths, etc.) — take nearer snapshot
        result[key] = alpha < 0.5 ? va : vb;
      }
    }
    return result;
  }
}


/* =========================================================================
   RHYTHM SCHEDULER — PIPPET-inspired timing with jitter control
   ========================================================================= */

class RhythmScheduler {
  constructor(ctx) {
    this.ctx      = ctx;
    this.baseBPM  = 52;        // Bach cantata tempo
    this.jitterMs = 60;        // current timing jitter (ms)
    this.nextBeat = 0;
  }

  start(audioTime) {
    this.nextBeat = audioTime;
  }

  /** Advance to next beat, return its scheduled time. */
  tick() {
    const beatDur = 60 / this.baseBPM;
    const jitter  = (Math.random() * 2 - 1) * (this.jitterMs / 1000);
    this.nextBeat += beatDur + jitter;
    return this.nextBeat;
  }

  /**
   * Tighten rhythm as redemption window approaches.
   *   proximity 1.0  → far from window → jitter 60ms (rubato)
   *   proximity 0.0  → at window → jitter 2ms (driving)
   */
  setRedemptionProximity(proximity) {
    this.jitterMs = 2 + proximity * 58;   // 2..60 ms
  }
}


/* =========================================================================
   VOICE-LEADING — parsimonious pitch movement, no parallel fifths
   ========================================================================= */

function findNearestPitch(currentFreq, targetFreqs, usedFreqs) {
  let best     = targetFreqs[0];
  let bestDist = Infinity;

  for (const f of targetFreqs) {
    if (usedFreqs.has(f)) continue;                     // no unison collisions
    const dist = Math.abs(Math.log2(f / currentFreq));  // interval in octaves
    if (dist < bestDist) {
      bestDist = dist;
      best     = f;
    }
  }
  return best;
}

function hasParallelFifth(freq1old, freq1new, freq2old, freq2new) {
  if (!freq1old || !freq2old) return false;
  const interval_old = Math.abs(12 * Math.log2(freq1old / freq2old)) % 12;
  const interval_new = Math.abs(12 * Math.log2(freq1new / freq2new)) % 12;
  // A perfect fifth is ~7 semitones
  return Math.abs(interval_old - 7) < 0.5 && Math.abs(interval_new - 7) < 0.5;
}


/* =========================================================================
   DIAGNOSTIC LISTENER — main controller
   ========================================================================= */

class DiagnosticListener {
  constructor() {
    this.ctx           = null;
    this.masterGain    = null;
    this.voices        = [];     // V1–V4 melodic
    this.continuo      = null;   // V5
    this.ripieno       = null;   // V6
    this.interpolator  = null;
    this.rhythm        = null;
    this.timer         = null;
    this.playing       = false;
    this.cadencing     = false;
    this.audioData     = null;

    // Callbacks for UI
    this.onProgress    = null;   // (progress, elapsed, voiceLevels) => {}
    this.onStateChange = null;   // ('playing'|'cadence'|'stopped') => {}
  }

  /** Initialize with audio data from /api/results → audio_states */
  async init(audioData) {
    this.audioData = audioData;
    if (!audioData || !audioData.states || audioData.states.length === 0) {
      throw new Error('No audio states available. Run the simulation first.');
    }
  }

  /** Start playback, time-stretched to targetDurationMs (default 10 min). */
  play(targetDurationMs = 600000) {
    if (this.playing) return;
    this.playing   = true;
    this.cadencing = false;

    // Create AudioContext on user gesture
    this.ctx = new (window.AudioContext || window.webkitAudioContext)();

    // Master gain — controlled by net divergence
    this.masterGain = this.ctx.createGain();
    this.masterGain.gain.value = 0.3;
    this.masterGain.connect(this.ctx.destination);

    // Create voices
    const pans = [-0.7, -0.3, 0.3, 0.7];    // spread across stereo field
    const types = ['triangle', 'triangle', 'sawtooth', 'sawtooth'];
    this.voices = pans.map((pan, i) =>
      new Voice(this.ctx, this.masterGain, {
        type: types[i], pan, baseGain: 0.10 - i * 0.01
      })
    );

    // V5 Continuo — always present, low register
    this.continuo = new Voice(this.ctx, this.masterGain, {
      type: 'sine', pan: 0, baseGain: 0.08
    });

    // V6 Ripieno — harmonic texture
    this.ripieno = new RipienoVoice(this.ctx, this.masterGain);

    // Interpolator
    this.interpolator = new StateInterpolator(this.audioData, targetDurationMs);
    this.interpolator.start(this.ctx.currentTime);

    // Rhythm
    this.rhythm = new RhythmScheduler(this.ctx);
    this.rhythm.start(this.ctx.currentTime);

    // Start continuo on the tonic (lowest node frequency)
    const tonicFreq = Math.min(...this.audioData.node_frequencies.filter(f => f > 0));
    this.continuo.activate(tonicFreq, 1.0);

    if (this.onStateChange) this.onStateChange('playing');

    // Start render loop
    this._renderLoop();
  }

  stop() {
    this.playing = false;
    clearTimeout(this.timer);

    if (this.voices) this.voices.forEach(v => v.destroy());
    if (this.continuo) this.continuo.destroy();
    if (this.ripieno) this.ripieno.destroy();

    if (this.ctx && this.ctx.state !== 'closed') {
      this.ctx.close();
    }

    this.voices   = [];
    this.continuo = null;
    this.ripieno  = null;
    this.ctx      = null;

    if (this.onStateChange) this.onStateChange('stopped');
  }

  /** Get current voice amplitude levels for UI meters. */
  getVoiceLevels() {
    const levels = [];
    for (const v of this.voices) {
      levels.push(v.active ? v.gain.gain.value / v.baseGain : 0);
    }
    levels.push(this.continuo?.active ? this.continuo.gain.gain.value / this.continuo.baseGain : 0);
    levels.push(this.ripieno ? this.ripieno.gain.gain.value / this.ripieno.baseGain : 0);
    return levels;  // [V1, V2, V3, V4, V5, V6]
  }

  // ─── Private ─────────────────────────────────────────────────────────

  _renderLoop() {
    if (!this.playing || !this.ctx) return;

    const now   = this.ctx.currentTime;
    const state = this.interpolator.getStateAt(now);

    if (!state) {
      this._triggerCadence();
      return;
    }

    this._updateVoices(state);
    this._updateContinuo(state);
    this._updateRipieno(state);
    this._updateMasterGain(state);
    this._updateRhythm(state);

    // Progress callback for UI
    if (this.onProgress) {
      const progress = this.interpolator.getProgress(now);
      const elapsed  = now - this.interpolator.startTime;
      this.onProgress(progress, elapsed, this.getVoiceLevels());
    }

    // 50ms interval — decoupled from visual frame rate
    this.timer = setTimeout(() => this._renderLoop(), 50);
  }

  _updateVoices(state) {
    const nodeFreqs = this.audioData.node_frequencies;
    const paths     = state.routing_paths || [];
    const qConf     = state.q_confidence  || [];
    const epsilon   = state.epsilon ?? 0.2;
    const nodeDivs  = state.node_divergence || [];

    const usedFreqs = new Set();
    const prevFreqs = this.voices.map(v => v.currentFreq);

    for (let i = 0; i < this.voices.length; i++) {
      const voice = this.voices[i];
      const path  = paths[i];

      if (!path || path.length < 2) {
        voice.deactivate();
        continue;
      }

      // Build target frequency sequence from path nodes
      const targetFreqs = path.map(n => nodeFreqs[n]).filter(f => f > 0);
      if (targetFreqs.length === 0) { voice.deactivate(); continue; }

      // Pick current note: cycle through path at rhythm-dependent rate
      // Use epsilon for ornamental variation
      const beatIdx = Math.floor(
        (this.ctx.currentTime * (this.rhythm.baseBPM / 60)) % targetFreqs.length
      );
      let targetFreq = targetFreqs[beatIdx];

      // Epsilon → ornamentation: high epsilon = random leaps
      if (epsilon > 0.15 && Math.random() < epsilon * 0.3) {
        // Ornamental passing tone — adjacent scale degree
        const offset = Math.random() > 0.5 ? 2 ** (2/12) : 2 ** (-2/12);
        targetFreq *= offset;
      }

      // Voice pitch drift from node divergence (§5.7)
      const nodeIdx = path[beatIdx] ?? path[0];
      const nodeDiv = nodeDivs[nodeIdx] ?? 0;
      if (Math.abs(nodeDiv) > 0.5) {
        // Sink → pitch drops, Source → pitch rises
        targetFreq *= 2 ** (nodeDiv * 0.01);   // subtle: ±~12 cents per div unit
      }

      // Parsimonious voice-leading
      const nearest = findNearestPitch(
        voice.currentFreq || targetFreq, [targetFreq], usedFreqs
      );

      // Check parallel fifths with adjacent voice
      if (i > 0 && i < 3) {
        if (hasParallelFifth(prevFreqs[i-1], this.voices[i-1].currentFreq,
                             prevFreqs[i], nearest)) {
          // Resolve: shift up one scale degree (~2 semitones)
          const resolved = nearest * (2 ** (2/12));
          usedFreqs.add(resolved);
          voice.activate(resolved, 0.15);
          continue;
        }
      }

      usedFreqs.add(nearest);

      // Q-confidence → volume + sustain
      const conf = qConf[i] ?? 0.5;
      const vol  = voice.baseGain * (0.4 + conf * 0.6);

      // Ramp time: high epsilon → short (recitative), low → long (aria)
      const ramp = 0.05 + (1 - epsilon) * 0.25;

      voice.activate(nearest, ramp);
      voice.setGain(vol, ramp);

      // Latency → delay (spatialization)
      const pathLats = state.path_latencies?.[i] || [];
      const maxLat   = pathLats.length > 0 ? Math.max(...pathLats) : 0;
      voice.setDelay(maxLat * 0.08);   // scale to 0–0.4s
    }
  }

  _updateContinuo(state) {
    // V5 tracks ComCoin supply signal — always present
    const nodeFreqs = this.audioData.node_frequencies;
    const netDiv    = state.net_divergence ?? 0;

    // Continuo frequency: tonic, shifted by net divergence
    const tonicFreq = Math.min(...nodeFreqs.filter(f => f > 0));
    const shifted   = tonicFreq * (2 ** (netDiv * 0.005));

    this.continuo.setFrequency(shifted, 0.5);   // slow movement

    // Volume: steady, slight swell when divergence active
    const vol = this.continuo.baseGain * (0.7 + Math.abs(netDiv) * 0.06);
    this.continuo.setGain(Math.min(vol, this.continuo.baseGain * 1.2), 0.3);
  }

  _updateRipieno(state) {
    // V6 harmonic tension from curl field
    const meanCurl = state.mean_curl ?? 0;

    // Reference frequency: midpoint of active melodic voices
    const activeFreqs = this.voices
      .filter(v => v.active)
      .map(v => v.currentFreq)
      .filter(f => f > 0);
    const refFreq = activeFreqs.length > 0
      ? activeFreqs.reduce((a, b) => a + b) / activeFreqs.length
      : 440;

    this.ripieno.setTension(meanCurl, refFreq, 0.3);
  }

  _updateMasterGain(state) {
    // Net divergence → master amplitude
    const netDiv     = state.net_divergence ?? 0;
    const activeFlows = state.active_flows ?? 0;

    // Base: 0.25, swell with positive divergence (network loading)
    let gain = 0.25 + Math.max(0, netDiv) * 0.02;
    // Reduce with negative divergence (network draining)
    gain += Math.min(0, netDiv) * 0.015;
    // More voices = thicker texture = slightly louder
    gain += Math.min(activeFlows, 50) * 0.001;

    gain = Math.max(0.08, Math.min(0.5, gain));
    this.masterGain.gain.linearRampToValueAtTime(
      gain, this.ctx.currentTime + 0.3
    );
  }

  _updateRhythm(state) {
    const proximity = state.redemption_proximity ?? 1.0;
    this.rhythm.setRedemptionProximity(proximity);
  }

  /** Cadence: converge all voices to tonic over ~20 seconds, then silence. */
  _triggerCadence() {
    if (this.cadencing) return;
    this.cadencing = true;
    if (this.onStateChange) this.onStateChange('cadence');

    const nodeFreqs = this.audioData.node_frequencies;
    const tonic     = Math.min(...nodeFreqs.filter(f => f > 0));
    const dominant  = tonic * (2 ** (7/12));   // perfect fifth above

    const now = this.ctx.currentTime;

    // 1. Melodic voices (V1–V4): hold current note, then converge to tonic
    this.voices.forEach((v, i) => {
      if (!v.active) return;
      // Hold for 2s, then converge over 8s
      const startConverge = now + 2;
      const endConverge   = now + 10;
      v.osc.frequency.linearRampToValueAtTime(v.currentFreq, startConverge);
      v.osc.frequency.linearRampToValueAtTime(
        tonic * (2 ** Math.floor(i / 2)),  // spread across octaves
        endConverge
      );
      // Fade out V1–V4 at 13s
      v.gain.gain.linearRampToValueAtTime(v.gain.gain.value, now + 10);
      v.gain.gain.linearRampToValueAtTime(0, now + 13);
    });

    // 2. Ripieno: resolve to consonance (dominant → silence)
    this.ripieno.osc1.frequency.linearRampToValueAtTime(dominant, now + 4);
    this.ripieno.osc2.frequency.linearRampToValueAtTime(dominant, now + 4);
    this.ripieno.gain.gain.linearRampToValueAtTime(0, now + 6);

    // 3. Bass (V4) → dominant briefly
    if (this.voices[3]?.active) {
      this.voices[3].osc.frequency.linearRampToValueAtTime(dominant, now + 8);
    }

    // 4. Continuo → tonic, hold alone
    this.continuo.osc.frequency.linearRampToValueAtTime(tonic, now + 5);
    this.continuo.gain.gain.linearRampToValueAtTime(this.continuo.baseGain, now + 13);

    // 5. All voices unison on tonic at 13s for 3s
    this.continuo.gain.gain.linearRampToValueAtTime(this.continuo.baseGain * 0.6, now + 16);

    // 6. Continuo alone for 2s, then silence
    this.continuo.gain.gain.linearRampToValueAtTime(this.continuo.baseGain * 0.4, now + 18);
    this.continuo.gain.gain.linearRampToValueAtTime(0, now + 20);

    // 7. Stop everything after cadence completes
    setTimeout(() => {
      this.playing = false;
      if (this.onStateChange) this.onStateChange('stopped');

      // Clean up
      this.voices.forEach(v => v.destroy());
      if (this.continuo) this.continuo.destroy();
      if (this.ripieno)  this.ripieno.destroy();
      if (this.ctx && this.ctx.state !== 'closed') this.ctx.close();
      this.voices   = [];
      this.continuo = null;
      this.ripieno  = null;
      this.ctx      = null;
    }, 21000);

    // Progress updates during cadence
    const cadenceUpdate = () => {
      if (!this.playing) return;
      if (this.onProgress) {
        this.onProgress(1.0, 0, this.getVoiceLevels());
      }
      setTimeout(cadenceUpdate, 100);
    };
    cadenceUpdate();
  }
}

// Export for use in index.html
if (typeof window !== 'undefined') {
  window.DiagnosticListener = DiagnosticListener;
}
