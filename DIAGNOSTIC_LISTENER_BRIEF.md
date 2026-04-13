# DIAGNOSTIC_LISTENER_BRIEF.md
# The Geodesic Diagnostic Listener
**Module:** `diagnostic_listener.js` (browser) + `audio_extractor.py` (pipeline)
**Output:** Real-time generative audio in the browser, triggered from the GUI
**Last updated:** 2026-04-13
**North star:** A Bach cantata. Legible complexity. Every voice knows what it is doing.

---

## 1. Concept

The Diagnostic Listener is the Geodesic AI's sense of self, made audible.

It is not a data sonification in the conventional sense — not a chart
translated into beeps. It is a continuous generative composition whose
harmonic language, voice structure, and rhythmic behaviour are determined
entirely by the live state of the simulation.

At any moment, a listener should be able to answer three questions
without looking at the screen:

1. Is the network healthy or stressed?
2. Is the AI learning or has it settled?
3. Is something unusual happening?

Everything in this brief serves those three questions.
If a sound does not help answer one of them, it does not belong.

---

## 2. The Buffering Decision

The simulation may complete in as little as 26 seconds (default params).
Real-time audio over 26 seconds does not give the listener time to orient.
Therefore:

```
If simulation_duration < 180 seconds (3 minutes):
    Buffer all state snapshots first, then play back time-stretched.

If simulation_duration >= 180 seconds:
    Stream audio live, step by step, as simulation runs.
```

**Default playback target: 10 minutes.**
User-adjustable in the GUI: slider "Playback duration: 5 — 20 min".

**Time-stretch ratio:**
`stretch = target_duration_seconds / (N_steps * step_duration_seconds)`

States are NOT played as discrete snapshots. The audio renderer
interpolates continuously between step N and step N+1. At any moment
t between steps, all audio parameters are a weighted blend:
`param(t) = param[n] * (1-alpha) + param[n+1] * alpha`
where `alpha = fractional position between steps`.

This means 200 simulation steps become 10 minutes of continuously
morphing audio with no audible discontinuities or jumps.

---

## 3. Voice Architecture — The Cantata Structure

**Six voices maximum.** Bach's typical polyphonic ceiling.
If more than six routing regions are active, group by mesh geography.

| Voice | Name | Register | Source signal | Instrument character |
|-------|------|----------|--------------|---------------------|
| V1 | Soprano | High | Best-latency routing path | Bright, leading |
| V2 | Alto | Mid-high | Second routing path | Lyrical, sustained |
| V3 | Tenor | Mid | Third routing path | Melodic, purposeful |
| V4 | Bass | Low-mid | Fourth routing path | Grounded, slower moving |
| V5 | Continuo | Low | ComCoin supply signal | Always present, foundational |
| V6 | Ripieno | All registers | Curl field aggregate | Harmonic colouring only — no independent melody |

**V5 (Continuo) never stops.** It is the basso continuo.
It may slow to near-stillness but it does not go silent.
It is the last voice to fade at the cadence.

**V6 (Ripieno) has no melodic role.** It adds harmonic tension
(dissonance) when curl is high, and releases to consonance when curl
is low. It is heard as the harmonic texture around the melodic voices,
not as a melody itself.

**Voice entry order:**
- Continuo enters first, alone, on the tonic (Sa). Always.
- Voices enter one by one as routing paths become active.
- A voice goes silent when its routing region has no active flows.
- Re-entry is treated as a new melodic statement, not a resumption.

---

## 4. Harmonic Language — Icosahedral Tuning

**Do not use equal temperament.**
**Do not use 432 Hz or Golden Ratio tuning claims.**

Use the actual geometry of the icosahedral mesh.

### Node-to-frequency mapping

```python
def node_to_frequency(verts, node_idx, base_freq=220.0):
    """
    Map a mesh vertex's 3D position to a frequency.
    
    Elevation (y-axis, -1 to +1) → octave position
    Azimuth (xz-plane angle, -π to +π) → scale degree within Bhairavi
    Proximity to nearest icosahedron vertex → microtonal inflection ±30 cents
    
    Adjacent nodes on the mesh produce frequencies within 50 cents of each
    other — they sound consonant. Distant nodes produce larger intervals.
    The mesh topology IS the harmonic space.
    """
    x, y, z = verts[node_idx]
    
    # Elevation → octave (3 octaves total, from 220 to 880 Hz)
    octave_position = (y + 1) / 2          # 0 to 1
    octave_freq = base_freq * (2 ** (octave_position * 2))  # 220 to 880
    
    # Azimuth → Bhairavi scale degree
    BHAIRAVI = [0, 1, 3, 5, 7, 8, 10]      # semitone intervals
    azimuth = np.arctan2(z, x)              # -π to π
    degree_idx = int((azimuth + np.pi) / (2*np.pi) * len(BHAIRAVI))
    degree_idx = degree_idx % len(BHAIRAVI)
    semitone_offset = BHAIRAVI[degree_idx]
    
    # Scale degree frequency
    scale_freq = octave_freq * (2 ** (semitone_offset / 12))
    
    # Microtonal inflection: distance to nearest base icosahedron vertex
    # Nodes close to base vertices → pure (0 cent inflection)
    # Nodes far from base vertices → up to ±30 cents inflection
    # This creates the microtonal richness — not random, but geometric
    ico_verts = get_base_icosahedron_vertices()
    distances = [np.linalg.norm(np.array([x,y,z]) - v) for v in ico_verts]
    proximity = min(distances)              # 0 = at a base vertex
    max_proximity = 0.8                     # approximate max distance
    cent_inflection = (proximity / max_proximity) * 30  # 0 to 30 cents
    # Sign of inflection: positive in northern hemisphere, negative in south
    cent_inflection *= np.sign(y) if y != 0 else 1
    
    microtonal_ratio = 2 ** (cent_inflection / 1200)
    return scale_freq * microtonal_ratio
```

### Harmonic rules for voice-leading

- Each voice moves to the nearest available pitch in the next state.
  Parsimonious voice-leading: minimum pitch distance between steps.
- No two voices may be on the same frequency simultaneously.
  If collision: the lower-priority voice moves up one scale degree.
- Voices may sustain a pitch across multiple steps (held note).
  Sustain probability increases with Q-value confidence.
- Forbidden intervals: no parallel fifths between V1-V2 or V2-V3.
  (Bach's rule. Maintain it.)

---

## 5. State-to-Sound Mappings

### 5.1 Routing paths → Melodic voices (V1–V4)

Each active routing path produces a sequence of nodes visited.
Each node has a frequency from the icosahedral tuning above.
The voice plays these frequencies in sequence, with duration
proportional to the edge's base_latency (longer latency = longer note).

```
note_duration = base_latency[edge] * time_stretch_factor
                clamped to [0.1s, 2.0s]
```

High Q-value confidence → longer note duration, higher velocity.
Low Q-value / exploration → shorter, softer, more ornamental.

### 5.2 Epsilon → Ornamentation

```
epsilon = 0.4 (high exploration):
  Voices add mordents, passing tones, unpredictable leaps.
  Rhythm is free, rubato, speech-like (recitative character).

epsilon = 0.05 (exploiting):
  Voices are clean, purposeful, melodically resolved.
  Rhythm is metered, predictable (aria character).

Transition between these is gradual — not a sudden switch.
```

### 5.3 Stokes curl → Harmonic tension (V6 Ripieno)

```
curl < 0.5:   V6 is silent or sustains the tonic. Consonance.
curl 0.5-1.5: V6 adds a sustained 7th or 2nd above the bass. Mild tension.
curl > 1.5:   V6 adds cluster tones — sustained semitone pairs above
              the most active melodic voice. Audible roughness.
              This is controlled dissonance, not noise.
              It wants to resolve — and does, when curl drops.

The roughness comes from amplitude modulation at the beating frequency
between the cluster tones, not from distortion or noise synthesis.
This is psychoacoustically accurate: the listener feels it as tension.
```

### 5.4 ComCoin divergence → Master amplitude (crescendo/decrescendo)

```
net_divergence > 0 (network loading, minting):
  Master gain rises. More voices become active. Texture thickens.
  Spectral centroid rises — voices move to higher registers.

net_divergence < 0 (network draining, burning):
  Master gain falls. Voices drop out one by one (bass last).
  Spectral centroid falls — voices move to lower registers.

net_divergence ≈ 0 (equilibrium):
  Steady dynamic. This is the healthy cantata state.
```

### 5.5 Edge latency → Reverb and delay per voice

```
For each active voice:
  reverb_time = avg_latency_on_path * 0.5   (seconds, clamped 0.1-3.0)
  delay_time  = max_latency_on_path * 0.1   (seconds, clamped 0.0-0.5)

High-latency paths: the voice sounds like it is in a large stone church.
Low-latency paths: the voice sounds close, dry, intimate.
The listener can spatially locate slow parts of the network by ear.
```

### 5.6 ComCoin redemption window → Rhythmic tightening (PIPPET)

The simulation runs on a 30-step cycle. Steps 22-25 are the
redemption window. As the window approaches, the rhythmic grid tightens.

```
Steps 0-15:   Timing jitter ±60ms. Loose, rubato, free.
Steps 15-21:  Timing jitter decreasing linearly to ±10ms. Anticipatory.
Steps 22-25:  Timing jitter ±2ms. Tightly quantized. Driving.
              Bass voice (V4) becomes more prominent. Rhythmic cadence.
Steps 26-29:  Timing jitter relaxes back to ±60ms. Release.
```

This creates a felt periodicity — the listener begins to anticipate
the redemption window without being told it is coming.

### 5.7 Green flow divergence → Voice dropping

```
Sink nodes (persistent negative divergence):
  The voice representing that mesh region drops in pitch over time.
  Metaphor: the voice is losing energy.

Source nodes (persistent positive divergence):
  The voice rises in pitch, becomes more agitated.
  Metaphor: the voice is accumulating pressure.

Balanced nodes:
  Voice maintains stable pitch center.
```

### 5.8 Genetic algorithm fitness → Tonal clarity

```
Low fitness (early GA or poor params):
  Modal ambiguity — Bhairavi and its relative modes alternate.
  The tonal centre is unclear.

High fitness (evolved params):
  Clear modal identity — Bhairavi asserted consistently.
  All voices agree on the tonal centre.
  This is the moment of maximum musical coherence.
```

---

## 6. The Cadence — How it Ends

When the last state snapshot is reached, the music does not fade or cut.
It finds a cadence. This is non-negotiable.

**Cadence procedure (over ~20 seconds):**

1. V1-V4 melodic voices complete their current phrase and hold their
   last note. No new notes are triggered.
2. V6 (Ripieno) releases any dissonance and resolves to the fifth (Pa).
3. V4 (Bass) moves to Pa (dominant).
4. V5 (Continuo) moves to Sa (tonic).
5. V1-V3 converge stepwise toward Sa over 10 seconds.
6. All voices hold Sa in unison for 3 seconds.
7. V1-V4 release. V5 holds Sa alone for 2 seconds. Then silence.

The final sound is Sa on the continuo. Alone. Then nothing.
This is the AI's last statement of its identity.

---

## 7. Implementation — Web Audio API

All audio runs in the browser. No external audio libraries.
No server-side audio generation.

### 7.1 New files

```
frontend/
  diagnostic_listener.js   ← all audio logic
  index.html               ← add Listen button + playback controls
                              (modify existing file)
```

### 7.2 Core audio objects

```javascript
// One per voice (V1-V6)
class Voice {
  constructor(ctx, voiceId) {
    this.oscillator = ctx.createOscillator();
    this.gain       = ctx.createGain();
    this.reverb     = ctx.createConvolver();   // impulse response
    this.delay      = ctx.createDelay(2.0);
    this.panner     = ctx.createStereoPanner(); // spatial positioning
    
    // Signal chain: osc → gain → reverb → delay → panner → master
    this.oscillator.connect(this.gain);
    this.gain.connect(this.reverb);
    this.reverb.connect(this.delay);
    this.delay.connect(this.panner);
    this.panner.connect(masterGain);
  }
  
  setFrequency(freq, rampTime) {
    // Always use linearRampToValueAtTime — never set value directly.
    // This prevents clicks and ensures smooth voice-leading.
    this.oscillator.frequency.linearRampToValueAtTime(
      freq, ctx.currentTime + rampTime
    );
  }
  
  setReverbTime(seconds) {
    // Generate impulse response for the reverb time
    this.reverb.buffer = generateImpulseResponse(ctx, seconds);
  }
}

// Master gain — controlled by divergence
const masterGain = ctx.createGain();
masterGain.connect(ctx.destination);

// Rhythm scheduler — PIPPET-inspired
class RhythmScheduler {
  constructor(ctx) {
    this.ctx         = ctx;
    this.jitter_ms   = 60;    // current timing jitter
    this.base_bpm    = 52;    // Bach cantata tempo
    this.next_beat   = ctx.currentTime;
  }
  
  tick() {
    const jitter = (Math.random() * 2 - 1) * (this.jitter_ms / 1000);
    this.next_beat += (60 / this.base_bpm) + jitter;
    return this.next_beat;
  }
  
  setRedemptionProximity(steps_until_window) {
    // Map 0-22 steps to 2-60ms jitter
    this.jitter_ms = Math.max(2, Math.min(60,
      (steps_until_window / 22) * 60
    ));
  }
}
```

### 7.3 State interpolator

```javascript
class StateInterpolator {
  constructor(snapshots, targetDurationMs) {
    this.snapshots     = snapshots;   // array of state objects from sim
    this.targetMs      = targetDurationMs;
    this.stretchRatio  = targetDurationMs / (snapshots.length * stepDurationMs);
    this.startTime     = null;
  }
  
  getStateAt(audioTime) {
    // Which two snapshots bracket this time?
    const elapsed    = audioTime - this.startTime;
    const progress   = elapsed / this.targetMs;           // 0 to 1
    const rawIndex   = progress * (this.snapshots.length - 1);
    const n          = Math.floor(rawIndex);
    const alpha      = rawIndex - n;
    
    // Interpolate all numeric fields between snapshot[n] and snapshot[n+1]
    return interpolate(this.snapshots[n], this.snapshots[n+1], alpha);
  }
}

function interpolate(a, b, alpha) {
  // Deep interpolation of all numeric fields.
  // Arrays (curl field, path nodes) are element-wise interpolated.
  // Non-numeric fields take the value of whichever snapshot is closer.
  const result = {};
  for (const key of Object.keys(a)) {
    if (typeof a[key] === 'number') {
      result[key] = a[key] * (1-alpha) + b[key] * alpha;
    } else if (Array.isArray(a[key])) {
      result[key] = a[key].map((v,i) => v*(1-alpha) + (b[key][i]||v)*alpha);
    } else {
      result[key] = alpha < 0.5 ? a[key] : b[key];
    }
  }
  return result;
}
```

### 7.4 Impulse response generation (reverb)

```javascript
function generateImpulseResponse(ctx, reverbTime) {
  // Synthetic impulse response — no audio files needed
  const sampleRate = ctx.sampleRate;
  const length     = Math.floor(reverbTime * sampleRate);
  const buffer     = ctx.createBuffer(2, length, sampleRate);
  
  for (let ch = 0; ch < 2; ch++) {
    const data = buffer.getChannelData(ch);
    for (let i = 0; i < length; i++) {
      // Exponential decay with white noise
      data[i] = (Math.random() * 2 - 1) * Math.exp(-3 * i / length);
    }
  }
  return buffer;
}
```

### 7.5 Audio state format (from audio_extractor.py)

```python
# Added to pipeline.py's per-step output
# audio_extractor.py exports this dict for each simulation step

def extract_audio_state(step, mesh_state, dqn_state, traffic_state,
                        stokes_state, green_state, comcoin_state):
    return {
        "step":               step,
        
        # Voice sources — up to 4 active routing paths
        # Each path is a list of node indices visited
        "routing_paths":      get_top_paths(traffic_state, n=4),
        
        # Node frequencies — precomputed from mesh geometry
        # Array of 162 floats, one per node
        "node_frequencies":   precomputed_node_freqs,
        
        # Q-value confidence per active path (0-1)
        "q_confidence":       get_q_confidences(dqn_state, n=4),
        
        # Epsilon (0-1) — controls ornamentation density
        "epsilon":            dqn_state["epsilon"],
        
        # Curl field — one float per mesh face (320 values)
        # Used for V6 harmonic tension
        "curl_field":         stokes_state["curl_per_face"],
        "mean_curl":          stokes_state["mean_curl"],
        
        # Net divergence — controls master amplitude
        "net_divergence":     comcoin_state["ema_divergence"],
        
        # Edge latencies for active paths — controls reverb per voice
        "path_latencies":     get_path_latencies(traffic_state, n=4),
        
        # Green flow — sink/source intensity per node
        "node_divergence":    green_state["div_per_node"],
        
        # ComCoin redemption proximity (0 = at window, 1 = far away)
        "redemption_proximity": comcoin_state["steps_until_window"] / 22.0,
        
        # GA fitness (0-1 normalised) — controls tonal clarity
        "ga_fitness_norm":    genetic_state.get("best_fitness_norm", 0.5),
        
        # Active flow count — controls polyphonic density
        "active_flows":       traffic_state["active_flows"],
    }
```

---

## 8. GUI Integration

### 8.1 New controls in index.html

Add to the centre column, below the Run button:

```
┌──────────────────────────────────────┐
│  DIAGNOSTIC LISTENER                  │
│                                       │
│  [▶ Listen]  [■ Stop]                │
│                                       │
│  Playback duration:                   │
│  [────────●──────────] 10 min        │
│   5 min              20 min          │
│                                       │
│  ● Buffering...  / ● Playing  3:42   │
│                                       │
│  Voice activity:                      │
│  V1 ████████░░  V2 ██████░░░░        │
│  V3 ████░░░░░░  V4 ██░░░░░░░░        │
│  V5 ██████████  V6 ████░░░░░░        │
└──────────────────────────────────────┘
```

States:
- Before first run: "Run the simulation first"
- After run completes: "▶ Listen" button active
- During buffering: "● Buffering simulation state..."
- During playback: "■ Stop" active, elapsed time shown, voice meters live
- During cadence: "Resolving..." (last 20 seconds)

### 8.2 Listen button behaviour

1. Fetch `/api/results` — get the stored state snapshots
2. If no results: show "Run the simulation first"
3. Calculate stretch ratio from target duration and N steps
4. Initialise StateInterpolator with snapshots and target duration
5. Create AudioContext (suspended until user gesture)
6. Resume AudioContext on Listen click (browser security requirement)
7. Start the render loop

### 8.3 The render loop

```javascript
function renderLoop() {
  const now   = audioCtx.currentTime;
  const state = interpolator.getStateAt(now);
  
  if (!state) {
    triggerCadence();
    return;
  }
  
  // Update all voices from interpolated state
  updateVoices(state);
  updateRipieno(state);
  updateMasterGain(state);
  updateRhythm(state);
  
  // Schedule next frame (not requestAnimationFrame — use setTimeout
  // with 50ms interval to avoid audio glitches from frame-rate variation)
  setTimeout(renderLoop, 50);
}
```

---

## 9. What success sounds like

Under healthy conditions (default params, system at steady state):

Three or four independent melodic lines moving in Bhairavi mode,
each at its own pace, occasionally converging on shared pitches
before diverging again. A slow-moving bass that rarely changes.
A subtle harmonic colouring — neither fully consonant nor dissonant.
Rhythmic pulse that is felt but not rigid. Ornamental figures in
the upper voices when epsilon was high during training.

When you hear this and it sounds like a Bach cantata — sparse,
polyphonic, purposeful, modal — the implementation is correct.

When you hear roughness in the harmony, the network had congestion.
When you hear the rhythm tighten, a redemption window was near.
When you hear a voice drop out, a mesh region went quiet.
When all voices converge to unison at the end, the AI has said
everything it has to say.

---

## 10. Implementation notes for Claude Code

- All audio in the browser. No server-side audio. No audio files.
- Use Web Audio API only — no Tone.js, no external libraries.
- The AudioContext must be created/resumed on a user gesture
  (browser security). The Listen button IS that gesture.
- Use linearRampToValueAtTime for ALL parameter changes.
  Never set .value directly on an AudioParam — it causes clicks.
- The render loop uses setTimeout(50ms), not requestAnimationFrame.
  Audio scheduling must be decoupled from visual frame rate.
- Generate reverb impulse responses synthetically (see section 7.4).
  Do not fetch audio files.
- The cadence (section 6) is mandatory. Do not skip it.
- Test with Chrome on macOS first. Safari has Web Audio quirks.
- audio_extractor.py must not break the existing pipeline.
  It is additive only — a new function called at the end of each
  run_*() wrapper in pipeline.py.
- The audio state is stored in geodesic_results.json under
  "audio_states": [...] — an array of one dict per simulation step.
  This is the only addition to the existing JSON structure.
- If audio_states is missing from results (older run), the Listen
  button shows: "Re-run the simulation to enable audio."

---

## 11. What comes after this

The Diagnostic Listener is the real-time (buffered) version.

After this is working, a separate module — the Geodesic Symphony
(SYMPHONY_BRIEF.md) — takes the same audio_states data and renders
a composed MIDI file with dramatic arc across five movements.
They share the audio_extractor.py output format.
They are different consumers of the same data.

The symphony is the AI's autobiography.
The listener is the AI's voice.

---

*Prepared in Claude Chat on 2026-04-13*
*Read alongside PROJECT_CONTEXT.md and SYMPHONY_BRIEF.md*
*Implements after: dirac_validator.py (optional), or in parallel*
