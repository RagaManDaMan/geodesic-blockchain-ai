/**
 * diagnostic_listener.js — Generative audio driven by Geodesic AI simulation state.
 *
 * Six-voice polyphonic architecture (Bach cantata structure):
 *   V1 Soprano   — best routing path, bright, leading
 *   V2 Alto      — second routing path, lyrical
 *   V3 Tenor     — third routing path, purposeful
 *   V4 Bass      — fourth routing path, grounded
 *   V5 Continuo  — ComCoin supply signal, always present, last to fade
 *   V6 Ripieno   — curl field aggregate, harmonic colouring only
 *
 * Harmonic minor scale with icosahedral geometry tuning.
 * All audio via Web Audio API — no external libraries.
 *
 * Design principles:
 *   - Sine waves only. Warmth, not harshness.
 *   - Notes have real durations (1–4 seconds). Not every-frame updates.
 *   - Voices enter one by one. Continuo enters first, alone.
 *   - Maximum 3–4 simultaneous melodic voices. Density follows sim load.
 *   - Consonant intervals preferred (octaves, fifths, thirds).
 *   - All parameter changes via linearRampToValueAtTime. No clicks.
 */

/* =========================================================================
   CONSTANTS
   ========================================================================= */

const BASE_BPM       = 52;         // Bach cantata tempo
const BEAT_SEC       = 60 / BASE_BPM;
const MAX_GAIN       = 0.12;       // master ceiling — gentle
const VOICE_GAIN     = 0.06;       // per-voice max
const CONTINUO_GAIN  = 0.05;
const RIPIENO_GAIN   = 0.03;
const RENDER_MS      = 50;         // render loop interval (decoupled from frame rate)
const ENTRY_FRACTION = [0, 0.08, 0.15, 0.25];  // when V1–V4 may enter (fraction of total)

// Consonant interval ratios for snapping
const CONSONANCES = [1, 2, 3/2, 4/3, 5/4, 5/3, 6/5, 8/5];


/* =========================================================================
   HELPERS
   ========================================================================= */

/** Snap freq to nearest consonant interval relative to a reference. */
function snapToConsonance(freq, refFreq) {
  if (!refFreq || refFreq <= 0) return freq;
  let bestFreq = freq;
  let bestDist = Infinity;
  // Try each consonance in nearby octaves
  for (const ratio of CONSONANCES) {
    for (let oct = -1; oct <= 2; oct++) {
      const candidate = refFreq * ratio * (2 ** oct);
      const dist = Math.abs(Math.log2(candidate / freq));
      if (dist < bestDist) {
        bestDist = dist;
        bestFreq = candidate;
      }
    }
  }
  return bestFreq;
}

/** Clamp value to [lo, hi]. */
function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

/** Generate a synthetic reverb impulse response. */
function makeReverb(ctx, seconds) {
  seconds = clamp(seconds, 0.3, 3.0);
  const len = Math.floor(seconds * ctx.sampleRate);
  const buf = ctx.createBuffer(2, len, ctx.sampleRate);
  for (let ch = 0; ch < 2; ch++) {
    const d = buf.getChannelData(ch);
    for (let i = 0; i < len; i++) {
      d[i] = (Math.random() * 2 - 1) * Math.exp(-4 * i / len);
    }
  }
  return buf;
}


/* =========================================================================
   VOICE — a single sine-wave voice that plays discrete notes
   ========================================================================= */

class Voice {
  constructor(ctx, dest, { pan = 0, gain = VOICE_GAIN } = {}) {
    this.ctx       = ctx;
    this.maxGain   = gain;
    this.active    = false;
    this.noteEnd   = 0;          // audioTime when current note expires
    this.freq      = 0;
    this.entered   = false;      // has this voice ever sounded?

    this.osc    = ctx.createOscillator();
    this.osc.type = 'sine';
    this.osc.frequency.value = 220;

    this.env    = ctx.createGain();
    this.env.gain.value = 0;     // start silent

    this.panner = ctx.createStereoPanner();
    this.panner.pan.value = pan;

    this.osc.connect(this.env);
    this.env.connect(this.panner);
    this.panner.connect(dest);
    this.osc.start();
  }

  /** Play a note: ramp to freq over attack, hold for duration, then release. */
  playNote(freq, duration, velocity = 1.0) {
    const now     = this.ctx.currentTime;
    const attack  = Math.min(0.3, duration * 0.15);
    const release = Math.min(0.5, duration * 0.2);
    const sustain = duration - attack - release;
    const vol     = this.maxGain * clamp(velocity, 0.15, 1.0);

    this.freq    = freq;
    this.active  = true;
    this.entered = true;
    this.noteEnd = now + duration;

    // Frequency: always ramp (never jump)
    this.osc.frequency.cancelScheduledValues(now);
    this.osc.frequency.setValueAtTime(this.osc.frequency.value, now);
    this.osc.frequency.linearRampToValueAtTime(freq, now + attack);

    // Envelope: attack → sustain → release
    this.env.gain.cancelScheduledValues(now);
    this.env.gain.setValueAtTime(this.env.gain.value, now);
    this.env.gain.linearRampToValueAtTime(vol, now + attack);
    this.env.gain.setValueAtTime(vol, now + attack + sustain);
    this.env.gain.linearRampToValueAtTime(0, now + duration);
  }

  /** Is the voice currently between notes (silent, ready for next)? */
  isIdle() {
    return this.ctx.currentTime >= this.noteEnd;
  }

  /** Gentle fade to silence. */
  silence(ramp = 0.8) {
    const now = this.ctx.currentTime;
    this.env.gain.cancelScheduledValues(now);
    this.env.gain.setValueAtTime(this.env.gain.value, now);
    this.env.gain.linearRampToValueAtTime(0, now + ramp);
    this.active  = false;
    this.noteEnd = now;
  }

  destroy() {
    try { this.osc.stop(); } catch (_) {}
    this.osc.disconnect();
    this.env.disconnect();
    this.panner.disconnect();
  }
}


/* =========================================================================
   RIPIENO (V6) — two detuned sines for harmonic tension / beating
   ========================================================================= */

class RipienoVoice {
  constructor(ctx, dest) {
    this.ctx = ctx;
    this.osc1 = ctx.createOscillator(); this.osc1.type = 'sine';
    this.osc2 = ctx.createOscillator(); this.osc2.type = 'sine';
    this.env  = ctx.createGain();       this.env.gain.value = 0;

    this.osc1.connect(this.env);
    this.osc2.connect(this.env);
    this.env.connect(dest);
    this.osc1.start();
    this.osc2.start();
  }

  /**
   * curlMag 0–2+.
   *   < 0.4 → silent (consonance)
   *   0.4–1.2 → minor 7th dyad, gentle
   *   > 1.2 → semitone cluster, audible roughness
   */
  update(curlMag, refFreq) {
    const t = this.ctx.currentTime + 0.4;   // slow ramp
    if (curlMag < 0.4 || !refFreq) {
      this.env.gain.linearRampToValueAtTime(0, t);
      return;
    }
    let f2;
    let vol;
    if (curlMag < 1.2) {
      f2  = refFreq * (2 ** (10/12));        // minor 7th
      vol = RIPIENO_GAIN * 0.4;
    } else {
      f2  = refFreq * (2 ** (1/12));         // semitone — beating
      vol = RIPIENO_GAIN * Math.min(1, (curlMag - 1) * 0.5);
    }
    this.osc1.frequency.linearRampToValueAtTime(refFreq, t);
    this.osc2.frequency.linearRampToValueAtTime(f2, t);
    this.env.gain.linearRampToValueAtTime(vol, t);
  }

  destroy() {
    try { this.osc1.stop(); } catch (_) {}
    try { this.osc2.stop(); } catch (_) {}
    this.osc1.disconnect(); this.osc2.disconnect(); this.env.disconnect();
  }
}


/* =========================================================================
   STATE INTERPOLATOR — continuous morphing between discrete snapshots
   ========================================================================= */

class StateInterpolator {
  constructor(audioData, targetMs) {
    this.states   = audioData.states;
    this.freqs    = audioData.node_frequencies;
    this.targetMs = targetMs;
    this.start    = null;
  }

  begin(t) { this.start = t; }

  at(t) {
    if (this.start === null) return null;
    const progress = ((t - this.start) * 1000) / this.targetMs;
    if (progress >= 1) return null;
    const raw = progress * (this.states.length - 1);
    const n   = Math.floor(raw);
    const a   = raw - n;
    if (n >= this.states.length - 1) return this.states[this.states.length - 1];
    return this._lerp(this.states[n], this.states[n+1], a);
  }

  progress(t) {
    if (!this.start) return 0;
    return clamp(((t - this.start) * 1000) / this.targetMs, 0, 1);
  }

  _lerp(a, b, t) {
    const r = {};
    for (const k of Object.keys(a)) {
      const va = a[k], vb = b[k];
      if (typeof va === 'number' && typeof vb === 'number') {
        r[k] = va + (vb - va) * t;
      } else if (Array.isArray(va) && Array.isArray(vb) && typeof va[0] === 'number') {
        r[k] = va.map((v, i) => v + ((vb[i] ?? v) - v) * t);
      } else {
        r[k] = t < 0.5 ? va : vb;
      }
    }
    return r;
  }
}


/* =========================================================================
   DIAGNOSTIC LISTENER — main controller
   ========================================================================= */

class DiagnosticListener {
  constructor() {
    this.ctx        = null;
    this.master     = null;
    this.reverb     = null;
    this.reverbGain = null;
    this.dry        = null;
    this.voices     = [];         // V1–V4
    this.continuo   = null;       // V5
    this.ripieno    = null;       // V6
    this.interp     = null;
    this.timer      = null;
    this.playing    = false;
    this.cadencing  = false;
    this.data       = null;
    this.beatClock  = 0;          // next scheduled beat time
    this.noteIndex  = [0,0,0,0];  // where each voice is in its path

    this.onProgress    = null;
    this.onStateChange = null;
  }

  async init(audioData) {
    this.data = audioData;
    if (!audioData?.states?.length) throw new Error('No audio states.');
  }

  play(targetMs = 600000) {
    if (this.playing) return;
    this.playing   = true;
    this.cadencing = false;
    this.noteIndex = [0,0,0,0];

    // ── Audio graph ──────────────────────────────────
    this.ctx = new (window.AudioContext || window.webkitAudioContext)();
    const ctx = this.ctx;

    // Master gain (gentle ceiling)
    this.master = ctx.createGain();
    this.master.gain.value = MAX_GAIN;
    this.master.connect(ctx.destination);

    // Send bus: dry + reverb
    this.dry = ctx.createGain();
    this.dry.gain.value = 0.75;
    this.dry.connect(this.master);

    this.reverbGain = ctx.createGain();
    this.reverbGain.gain.value = 0.25;
    this.reverb = ctx.createConvolver();
    this.reverb.buffer = makeReverb(ctx, 2.0);
    this.reverb.connect(this.reverbGain);
    this.reverbGain.connect(this.master);

    // Voice destination: dry + reverb send
    const vocDest = ctx.createGain();
    vocDest.gain.value = 1;
    vocDest.connect(this.dry);
    vocDest.connect(this.reverb);

    // Create voices (all sine, spread stereo)
    const pans = [-0.6, -0.2, 0.2, 0.6];
    this.voices = pans.map((p, i) =>
      new Voice(ctx, vocDest, { pan: p, gain: VOICE_GAIN * (1 - i * 0.08) })
    );

    // V5 Continuo — centre, always present
    this.continuo = new Voice(ctx, vocDest, { pan: 0, gain: CONTINUO_GAIN });

    // V6 Ripieno — harmonic texture
    this.ripieno = new RipienoVoice(ctx, vocDest);

    // Interpolator
    this.interp = new StateInterpolator(this.data, targetMs);
    this.interp.begin(ctx.currentTime);

    // Beat clock
    this.beatClock = ctx.currentTime;

    // Start continuo on tonic — alone
    const tonic = this._tonic();
    this.continuo.playNote(tonic, BEAT_SEC * 4, 0.6);

    if (this.onStateChange) this.onStateChange('playing');
    this._loop();
  }

  stop() {
    this.playing = false;
    clearTimeout(this.timer);
    this._cleanup();
    if (this.onStateChange) this.onStateChange('stopped');
  }

  getVoiceLevels() {
    const g = v => (v && v.active) ? clamp(v.env.gain.value / v.maxGain, 0, 1) : 0;
    return [
      ...this.voices.map(g),
      g(this.continuo),
      this.ripieno ? clamp(this.ripieno.env.gain.value / RIPIENO_GAIN, 0, 1) : 0,
    ];
  }

  // ─── Private ──────────────────────────────────────────

  _tonic() {
    const fs = this.data.node_frequencies.filter(f => f > 100 && f < 300);
    return fs.length ? Math.min(...fs) : 220;
  }

  _loop() {
    if (!this.playing || !this.ctx) return;
    const now   = this.ctx.currentTime;
    const state = this.interp.at(now);

    if (!state) { this._cadence(); return; }

    const progress = this.interp.progress(now);

    // ── Schedule notes on the beat ─────────────────────
    if (now >= this.beatClock) {
      this._scheduleNotes(state, progress);
      // Advance beat clock (with slight jitter from redemption proximity)
      const prox   = state.redemption_proximity ?? 1;
      const jitter = (Math.random() * 2 - 1) * prox * 0.06;
      this.beatClock = now + BEAT_SEC + jitter;
    }

    // ── Continuous updates (slow, not every-frame pitch jumps) ──
    this._updateRipieno(state);
    this._updateMaster(state);

    // Progress callback
    if (this.onProgress) {
      this.onProgress(progress, now - this.interp.start, this.getVoiceLevels());
    }

    this.timer = setTimeout(() => this._loop(), RENDER_MS);
  }

  _scheduleNotes(state, progress) {
    const freqs = this.data.node_frequencies;
    const paths = state.routing_paths || [];
    const eps   = state.epsilon ?? 0.2;
    const qConf = state.q_confidence || [];
    const flows = state.active_flows ?? 0;

    // How many melodic voices should be active? Based on flow density.
    const density = clamp(flows / 30, 0, 1);
    const maxVoices = Math.max(1, Math.round(1 + density * 3));  // 1–4

    // Continuo (V5): always plays. Slow-moving tonic/dominant.
    if (this.continuo.isIdle()) {
      const tonic    = this._tonic();
      const netDiv   = state.net_divergence ?? 0;
      // Shift toward dominant when divergence positive, sub-dominant when negative
      const shift    = netDiv > 0.5 ? (2 ** (7/12)) : netDiv < -0.5 ? (2 ** (5/12)) : 1;
      const cFreq    = tonic * shift;
      const dur      = BEAT_SEC * (3 + Math.random() * 2);  // 3–5 beats
      this.continuo.playNote(cFreq, dur, 0.5 + Math.abs(netDiv) * 0.05);
    }

    // Melodic voices (V1–V4): enter gradually, play path notes
    for (let i = 0; i < 4; i++) {
      // Don't enter before scheduled fraction of playback
      if (progress < ENTRY_FRACTION[i]) continue;
      // Don't exceed density-based voice limit
      if (i >= maxVoices) {
        if (this.voices[i].active) this.voices[i].silence(1.0);
        continue;
      }

      const voice = this.voices[i];
      if (!voice.isIdle()) continue;  // still sustaining a note — wait

      const path = paths[i];
      if (!path || path.length < 2) {
        voice.silence(0.5);
        continue;
      }

      // Advance through the path one note at a time
      const idx    = this.noteIndex[i] % path.length;
      const node   = path[idx];
      let   freq   = freqs[node] || 220;

      // Snap to consonance relative to continuo
      if (this.continuo.freq > 0) {
        freq = snapToConsonance(freq, this.continuo.freq);
      }

      // Node divergence → subtle pitch drift
      const nodeDiv = (state.node_divergence || [])[node] || 0;
      if (Math.abs(nodeDiv) > 0.3) {
        freq *= 2 ** (nodeDiv * 0.008);   // very subtle: ~10 cents/unit
      }

      // Epsilon → note character
      //   high eps: short notes, lower velocity (exploratory, recitative)
      //   low eps:  long notes, full velocity (confident, aria)
      const conf     = qConf[i] ?? 0.5;
      const baseDur  = BEAT_SEC * (1.5 + (1 - eps) * 2.5);  // 1.5–4 beats
      const dur      = baseDur * (0.7 + conf * 0.6);
      const velocity = 0.3 + (1 - eps) * 0.4 + conf * 0.3;

      // Ornament: at high epsilon, occasionally skip a scale degree
      if (eps > 0.2 && Math.random() < eps * 0.2) {
        this.noteIndex[i] += 2;  // skip ahead — ornamental leap
      } else {
        this.noteIndex[i] += 1;
      }

      voice.playNote(freq, dur, clamp(velocity, 0.2, 0.9));
    }
  }

  _updateRipieno(state) {
    const curl = state.mean_curl ?? 0;
    // Reference: average of sounding voices
    const sounding = this.voices.filter(v => v.active).map(v => v.freq).filter(f => f > 0);
    const ref = sounding.length
      ? sounding.reduce((a,b) => a+b) / sounding.length
      : this.continuo?.freq || 440;
    this.ripieno.update(curl, ref);
  }

  _updateMaster(state) {
    const div   = state.net_divergence ?? 0;
    const flows = state.active_flows ?? 0;
    let g = MAX_GAIN;
    g += clamp(div, -2, 2) * 0.008;           // swell/recede with divergence
    g += clamp(flows, 0, 40) * 0.0005;        // thicken with traffic
    g = clamp(g, 0.04, 0.18);
    this.master.gain.linearRampToValueAtTime(g, this.ctx.currentTime + 0.5);
  }

  /* ── CADENCE — converge to tonic over ~20s, continuo last ────── */

  _cadence() {
    if (this.cadencing) return;
    this.cadencing = true;
    if (this.onStateChange) this.onStateChange('cadence');

    const now   = this.ctx.currentTime;
    const tonic = this._tonic();
    const dom   = tonic * (2 ** (7/12));

    // V1–V4: converge stepwise to tonic, then fade
    this.voices.forEach((v, i) => {
      if (!v.entered) return;
      const arrival = now + 2 + i * 1.5;
      const target  = tonic * (2 ** Math.floor(i/2));
      v.osc.frequency.cancelScheduledValues(now);
      v.osc.frequency.setValueAtTime(v.osc.frequency.value, now);
      v.osc.frequency.linearRampToValueAtTime(target, arrival);
      v.env.gain.cancelScheduledValues(now);
      v.env.gain.setValueAtTime(v.env.gain.value, now);
      v.env.gain.linearRampToValueAtTime(VOICE_GAIN * 0.5, arrival);
      v.env.gain.linearRampToValueAtTime(0, now + 12);
    });

    // V6 Ripieno: resolve dissonance → silence
    this.ripieno.osc1.frequency.linearRampToValueAtTime(dom, now + 3);
    this.ripieno.osc2.frequency.linearRampToValueAtTime(dom, now + 3);
    this.ripieno.env.gain.linearRampToValueAtTime(0, now + 5);

    // V5 Continuo: hold tonic alone after others fade
    this.continuo.osc.frequency.cancelScheduledValues(now);
    this.continuo.osc.frequency.setValueAtTime(this.continuo.osc.frequency.value, now);
    this.continuo.osc.frequency.linearRampToValueAtTime(tonic, now + 4);
    this.continuo.env.gain.cancelScheduledValues(now);
    this.continuo.env.gain.setValueAtTime(this.continuo.env.gain.value, now);
    this.continuo.env.gain.linearRampToValueAtTime(CONTINUO_GAIN * 0.7, now + 12);
    // Alone for 4 seconds
    this.continuo.env.gain.linearRampToValueAtTime(CONTINUO_GAIN * 0.4, now + 16);
    // Then silence
    this.continuo.env.gain.linearRampToValueAtTime(0, now + 19);

    // Clean up after cadence
    setTimeout(() => {
      this.playing = false;
      this._cleanup();
      if (this.onStateChange) this.onStateChange('stopped');
    }, 20000);

    // Keep progress updates alive during cadence
    const tick = () => {
      if (!this.playing) return;
      if (this.onProgress) this.onProgress(1, 0, this.getVoiceLevels());
      setTimeout(tick, 150);
    };
    tick();
  }

  _cleanup() {
    clearTimeout(this.timer);
    this.voices.forEach(v => v.destroy());
    if (this.continuo) this.continuo.destroy();
    if (this.ripieno)  this.ripieno.destroy();
    if (this.reverb)   this.reverb.disconnect();
    if (this.ctx && this.ctx.state !== 'closed') this.ctx.close();
    this.voices = []; this.continuo = null; this.ripieno = null; this.ctx = null;
  }
}

if (typeof window !== 'undefined') {
  window.DiagnosticListener = DiagnosticListener;
}
