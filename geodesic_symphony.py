"""
geodesic_symphony.py — Multi-track MIDI composition engine.

Sonifies the Geodesic Blockchain AI simulation across five movements:

  I.   Genesis     — 12 icosahedron vertices, widely spaced, no pulse
  II.  Awakening   — Subdivision 12→42→162, textures enter
  III. Learning    — DQN training curve, alapana-style exploration
  IV.  Congestion  — M/M/1 collapse, Stokes vortex, recovery
  V.   Epiphany    — Steady state, ritardando, drone fades last

Harmonic world: Bhairavi raga (C Db Eb F G Ab Bb)
Output: geodesic_symphony.mid  +  geodesic_symphony_score.txt

See SYMPHONY_BRIEF.md for full composition specification.

Dependencies: mido (pip install mido)
"""

import numpy as np
import random
from collections import defaultdict
from mido import MidiFile, MidiTrack, Message, MetaMessage, bpm2tempo

from geodesic_overlay import build_icosahedron, subdivide, mesh_to_graph
from dqn_router import assign_link_params

# ---------------------------------------------------------------------------
# Constants — Harmonic world
# ---------------------------------------------------------------------------

SCALE_ROOT    = 60                      # Sa = Middle C
SCALE_DEGREES = [0, 1, 3, 5, 7, 8, 10] # Bhairavi: C Db Eb F G Ab Bb
OCTAVE_RANGE  = (3, 6)

PPQ = 480  # ticks per quarter note

# MIDI channels
CH_MELODY   = 0
CH_COUNTER  = 1
CH_HARMONY  = 2
CH_DRONE    = 3
CH_ORNAMENT = 4
CH_PULSE    = 9   # GM drums channel
CH_BASS     = 5

# GM patches (0-indexed, as sent in program_change)
PROG_SITAR   = 103  # Track 1 Melody
PROG_FLUTE   = 72   # Track 2 Countermelody
PROG_STRINGS = 47   # Track 3 Harmony
PROG_PAD     = 91   # Track 4 Drone  (Pad 4 Choir → warm sustain)
PROG_CRYSTAL = 97   # Track 5 Ornament
PROG_CELLO   = 41   # Track 7 Bass

# Drum notes used for Pulse track
DRUM_KICK    = 36
DRUM_SNARE   = 38
DRUM_CLOSED_HAT = 42
DRUM_OPEN_HAT   = 46
DRUM_TOM_HI  = 48
DRUM_TOM_LO  = 45
DRUM_CRASH   = 49


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def B(beats: float) -> int:
    """Beats to ticks."""
    return int(beats * PPQ)


def pos_to_pitch(pos, scale_degrees=SCALE_DEGREES, root=SCALE_ROOT,
                 octave_range=OCTAVE_RANGE) -> int:
    """Map a 3-D unit-sphere position to a MIDI pitch in Bhairavi.

    y-coordinate   → octave (low hemisphere = low octave)
    atan2(z, x)    → scale degree within that octave
    """
    y      = float(pos[1])
    angle  = float(np.arctan2(pos[2], pos[0]))
    octave = int(np.interp(y, [-1.0, 1.0], octave_range))
    deg_i  = int(np.clip(
        np.interp(angle, [-np.pi, np.pi], [0, len(scale_degrees) - 1]),
        0, len(scale_degrees) - 1
    ))
    root_pc = root % 12
    return int(np.clip((octave + 1) * 12 + root_pc + scale_degrees[deg_i], 24, 108))


def node_to_pitch(idx, verts, **kw) -> int:
    return pos_to_pitch(verts[idx], **kw)


def reward_to_dur_beats(reward: float) -> float:
    """Map DQN reward to note duration in beats (brief §5)."""
    if reward < -30:   return 0.125   # sixteenth
    if reward < -10:   return 0.25    # eighth
    if reward < 5:     return 0.5     # quarter
    if reward < 12:    return 1.0     # half
    return 2.0                        # whole / fermata


def curl_to_intervals(mean_curl: float):
    """Curl magnitude → Bhairavi harmony intervals above Sa (brief §5)."""
    if mean_curl < 0.5:   return [0, 7]       # Sa Pa  (open fifth)
    if mean_curl < 1.5:   return [0, 2, 10]   # Sa Re Ni
    return [0, 1, 3]                          # Sa Reb Gab  (tight cluster)


def bhairavi_pitch(degree_idx: int, octave: int = 4) -> int:
    """Return an absolute MIDI pitch for a Bhairavi scale degree."""
    root_pc = SCALE_ROOT % 12
    return int(np.clip((octave + 1) * 12 + root_pc + SCALE_DEGREES[degree_idx % 7], 24, 108))


def sa(octave: int = 3) -> int:
    """Sa (root) in the given octave."""
    return (octave + 1) * 12 + (SCALE_ROOT % 12)


# ---------------------------------------------------------------------------
# EventList — accumulate notes in absolute ticks, convert at end
# ---------------------------------------------------------------------------

class EventList:
    """Collect (abs_tick, Message) pairs and flush to a MidiTrack."""

    def __init__(self):
        self._events: list = []

    def note(self, tick: int, dur: int, pitch: int, vel: int,
             channel: int, humanize: bool = False):
        pitch = int(np.clip(pitch, 0, 127))
        vel   = int(np.clip(vel,   1, 127))
        if humanize:
            tick = max(0, tick + random.randint(-15, 15))
        self._events.append((tick,       Message('note_on',  channel=channel,
                                                 note=pitch, velocity=vel,  time=0)))
        self._events.append((tick + dur, Message('note_off', channel=channel,
                                                 note=pitch, velocity=0,    time=0)))

    def prog(self, tick: int, channel: int, program: int):
        self._events.append((tick, Message('program_change',
                                           channel=channel, program=program, time=0)))

    def to_track(self, name: str) -> MidiTrack:
        track = MidiTrack()
        safe  = name.encode('latin-1', errors='replace').decode('latin-1')
        track.append(MetaMessage('track_name', name=safe, time=0))
        evs   = sorted(self._events, key=lambda x: x[0])
        prev  = 0
        for abs_tick, msg in evs:
            delta = max(0, abs_tick - prev)
            track.append(msg.copy(time=delta))
            prev = abs_tick
        return track

    @property
    def note_count(self) -> int:
        return sum(1 for _, m in self._events
                   if isinstance(m, Message) and m.type == 'note_on' and m.velocity > 0)


class TempoTrack:
    """Accumulate tempo and marker events for the Structure track (track 0)."""

    def __init__(self):
        self._events: list = []

    def tempo(self, tick: int, bpm: float):
        self._events.append((tick, MetaMessage('set_tempo',
                                               tempo=bpm2tempo(bpm), time=0)))

    def marker(self, tick: int, text: str):
        safe = text.encode('latin-1', errors='replace').decode('latin-1')
        self._events.append((tick, MetaMessage('marker', text=safe, time=0)))

    def to_track(self) -> MidiTrack:
        track = MidiTrack()
        track.append(MetaMessage('track_name', name='Structure', time=0))  # ascii safe
        evs   = sorted(self._events, key=lambda x: x[0])
        prev  = 0
        for abs_tick, msg in evs:
            delta = max(0, abs_tick - prev)
            track.append(msg.copy(time=delta))
            prev = abs_tick
        return track


# ---------------------------------------------------------------------------
# Simulation data helpers
# ---------------------------------------------------------------------------

def synthesize_dqn_curve(n: int = 800):
    """Logistic DQN learning curve matching dqn_router.py trajectory.

    Returns (rewards, successes, epsilons) lists of length n.
    Range: reward -63 → +15, success 0.13 → 1.0  (per PROJECT_CONTEXT §6)
    """
    rng      = np.random.RandomState(42)
    rewards, successes, epsilons = [], [], []
    eps = 0.4
    for ep in range(n):
        succ  = 1.0 / (1.0 + np.exp(-0.025 * (ep - 120)))
        noise = rng.normal(0, max(3.0 * (1 - succ), 0.4))
        rew   = -63.0 + succ * 78.0 + noise
        rewards.append(float(rew))
        successes.append(float(succ))
        epsilons.append(float(eps))
        eps = max(0.05, eps * 0.997)
    return rewards, successes, epsilons


def get_curl_field(G, faces, n_steps: int = 25):
    """Run DirectedFlowSim or fall back to PROJECT_CONTEXT mock values."""
    try:
        from stokes_curl import DirectedFlowSim
        sim  = DirectedFlowSim(G, faces, arrival_rate=4.0,
                               flow_duration=10.0, capacity_scale=0.5)
        hist = sim.run(n_steps=n_steps, verbose=False)
        n_avg = min(10, len(hist))
        avg_curl = {i: float(np.mean([hist[-j][i] for j in range(1, n_avg + 1)]))
                    for i in range(len(faces))}
        mc = max(abs(v) for v in avg_curl.values())
        print(f'  [curl] {len(faces)} faces  max|curl|={mc:.2f}  (live sim)')
        return avg_curl
    except Exception as exc:
        print(f'  [curl] MOCK from PROJECT_CONTEXT.md  (reason: {exc})')
        rng = np.random.RandomState(42)
        curl = {}
        for i in range(len(faces)):
            if rng.random() < 85 / 320:          # 85 vortex faces per context doc
                curl[i] = float(rng.uniform(1.0, 3.2) * rng.choice([-1, 1]))
            else:
                curl[i] = float(rng.uniform(-0.8, 0.8))
        return curl


def get_traffic_util(G, verts, n_steps: int = 40):
    """Run lightweight TrafficSimulator or return mock utilisation sequence."""
    try:
        from traffic_sim import TrafficSimulator
        sim  = TrafficSimulator(G, verts, arrival_rate=3.0,
                                flow_duration=8.0, capacity_scale=0.1)
        hist = sim.run(n_steps=n_steps, n_test_pairs=0, verbose=False)
        utils = [(m['avg_util'], m['max_util'], m['active_flows']) for m in hist]
        print(f'  [traffic] {n_steps} steps simulated (live)')
        return utils
    except Exception as exc:
        print(f'  [traffic] MOCK utilisation sequence  (reason: {exc})')
        rng   = np.random.RandomState(42)
        utils = []
        u = 0.05
        for _ in range(n_steps):
            u = float(np.clip(u + rng.normal(0.02, 0.05), 0.01, 0.95))
            utils.append((u, min(u * 1.8, 0.99), int(u * 80)))
        return utils


# ---------------------------------------------------------------------------
# Movement I — Genesis  (~75 seconds, free time)
# ---------------------------------------------------------------------------

def movement_i(tempo_t: TempoTrack, harmony: EventList, drone: EventList,
               verts_base: np.ndarray, tick0: int = 0) -> int:
    """12 icosahedron vertices sounding one by one.

    Only Harmony (Track 3) and Drone (Track 4) are active.
    No rhythm, no pulse. Feels like the mesh being inscribed on the sphere.
    """
    rng = random.Random(1)
    cursor = tick0

    tempo_t.marker(cursor, 'Movement I: Genesis')
    harmony.prog(cursor, CH_HARMONY, PROG_STRINGS)
    drone.prog(cursor, CH_DRONE, PROG_PAD)

    # Sa drone — very long, soft, enters first
    drone.note(cursor, B(10), sa(2), 52, CH_DRONE)

    for i, pos in enumerate(verts_base):
        bpm = rng.randint(40, 70)
        tempo_t.tempo(cursor, bpm)

        h_pitch = pos_to_pitch(pos, octave_range=(3, 5))
        d_pitch = sa(2)   # drone always Sa

        # Silence between vertices: 4-8 seconds expressed as beats
        if i > 0:
            sil_beats = rng.uniform(3.5, 7.0)
            cursor   += B(sil_beats)

        dur_beats = rng.uniform(3.0, 6.0)
        vel_h     = rng.randint(42, 65)

        harmony.note(cursor, B(dur_beats), h_pitch, vel_h, CH_HARMONY)

        # Re-trigger drone before it decays (every 3 vertices)
        if i % 3 == 0:
            drone.note(cursor, B(dur_beats * 2.5), d_pitch, 55, CH_DRONE)

        cursor += B(dur_beats)

    cursor += B(5)   # trailing silence
    tempo_t.marker(cursor, 'End Movement I')
    return cursor


# ---------------------------------------------------------------------------
# Movement II — Awakening  (~105 seconds)
# ---------------------------------------------------------------------------

def movement_ii(tempo_t: TempoTrack, harmony: EventList, drone: EventList,
                ornament: EventList, pulse: EventList,
                verts_42: np.ndarray, verts_162: np.ndarray,
                tick0: int = 0) -> int:
    """Two subdivision events (12→42, 42→162) each trigger a texture wave.

    Track 5 (Ornament) enters with rapid figures.
    Track 6 (Pulse) enters sparse and irregular.
    Track 4 (Drone) establishes Sa.
    """
    rng     = random.Random(2)
    cursor  = tick0
    BASE_BPM = 52.0

    tempo_t.marker(cursor, 'Movement II: Awakening')
    tempo_t.tempo(cursor, BASE_BPM)

    ornament.prog(cursor, CH_ORNAMENT, PROG_CRYSTAL)
    pulse.prog(cursor,    CH_PULSE,    0)     # channel 9 ignores prog; harmless

    # --- Drone re-establishes Sa ---
    drone.note(cursor, B(40), sa(2), 58, CH_DRONE)

    # --- Subdivision 1: 12 → 42  (30 new nodes) ---
    tempo_t.marker(cursor, 'Subdivision: 12 to 42 nodes')
    tempo_t.tempo(cursor, 46.0)

    # Sample 20 of the 42 nodes as slow harmonic chords
    idxs_42 = list(range(min(42, len(verts_42))))
    rng.shuffle(idxs_42)
    t = cursor
    for j, idx in enumerate(idxs_42[:20]):
        h_pitch = node_to_pitch(idx, verts_42, octave_range=(3, 5))
        vel     = rng.randint(38, 58)
        gap     = B(rng.uniform(1.5, 3.5))
        harmony.note(t, B(rng.uniform(1.5, 3.0)), h_pitch, vel, CH_HARMONY)
        # Ornament: short high figures, fires on ~40% of events (epsilon=0.4)
        if rng.random() < 0.4:
            o_pitch = pos_to_pitch(verts_42[idx], octave_range=(5, 6))
            for k in range(rng.randint(2, 5)):
                ornament.note(t + k * B(0.125),
                              B(0.1), o_pitch + rng.choice(SCALE_DEGREES),
                              rng.randint(40, 65), CH_ORNAMENT)
        t += gap

    sd1_end = t
    cursor  = max(cursor + B(30), sd1_end)

    # Brief silence between subdivisions
    cursor += B(4)
    tempo_t.tempo(cursor, 52.0)

    # --- Subdivision 2: 42 → 162  (120 new nodes) ---
    tempo_t.marker(cursor, 'Subdivision: 42 to 162 nodes')
    tempo_t.tempo(cursor, 58.0)

    # Denser: more harmony notes, pulse enters
    idxs_162 = list(range(min(162, len(verts_162))))
    rng.shuffle(idxs_162)
    t = cursor
    beat_grid = 0
    for j, idx in enumerate(idxs_162[:28]):
        h_pitch = node_to_pitch(idx, verts_162, octave_range=(3, 5))
        vel     = rng.randint(45, 68)
        gap     = B(rng.uniform(1.0, 2.5))
        harmony.note(t, B(rng.uniform(1.0, 2.5)), h_pitch, vel, CH_HARMONY)

        # Ornament: more active now (more nodes = more exploration)
        if rng.random() < 0.55:
            o_pitch = pos_to_pitch(verts_162[idx], octave_range=(5, 6))
            for k in range(rng.randint(3, 7)):
                ornament.note(t + k * B(0.0833),
                              B(0.08), o_pitch + rng.choice(SCALE_DEGREES),
                              rng.randint(38, 62), CH_ORNAMENT)

        # Sparse pulse: irregular kick/tom, not every beat
        if beat_grid % 3 == 0 and rng.random() < 0.5:
            pulse.note(t, B(0.3), DRUM_KICK, rng.randint(50, 75), CH_PULSE)
        if beat_grid % 5 == 1 and rng.random() < 0.4:
            pulse.note(t + B(0.5), B(0.2), DRUM_TOM_LO, rng.randint(40, 60), CH_PULSE)

        t         += gap
        beat_grid += 1

    cursor = max(cursor + B(38), t)

    # Drone retrigger at end of movement
    drone.note(cursor - B(6), B(12), sa(2), 60, CH_DRONE)

    cursor += B(4)
    tempo_t.marker(cursor, 'End Movement II')
    return cursor


# ---------------------------------------------------------------------------
# Movement III — Learning  (~210 beats, alapana style)
# ---------------------------------------------------------------------------

def movement_iii(tempo_t: TempoTrack,
                 melody: EventList, counter: EventList,
                 harmony: EventList, drone: EventList,
                 ornament: EventList, pulse: EventList,
                 rewards: list, successes: list, epsilons: list,
                 faces: list, curl_field: dict,
                 tick0: int = 0) -> int:
    """DQN training curve → melodic development.

    Mood: carnatic alapana — unhurried, exploratory. Melody enters alone.
    Two silence windows where Tracks 3-6 drop out completely.
    Episode 300 = moment of arrival. Countermelody enters ~episode 400.
    """
    rng    = random.Random(3)
    cursor = tick0
    BASE_BPM = 52.0

    tempo_t.marker(cursor, 'Movement III: Learning (alapana)')
    tempo_t.tempo(cursor, BASE_BPM)

    melody.prog(cursor,   CH_MELODY,   PROG_SITAR)
    counter.prog(cursor,  CH_COUNTER,  PROG_FLUTE)

    # Mean curl across all faces (for harmony voicing)
    all_curl = [abs(v) for v in curl_field.values()]
    mean_curl_global = float(np.mean(all_curl))

    # Drone holds throughout (never silent)
    drone.note(cursor, B(220), sa(2), 55, CH_DRONE)

    # -----------------------------------------------------------------------
    # We compress 800 episodes into ~200 beats.
    # Episode groups of 8 → 100 "moments".
    # -----------------------------------------------------------------------
    N_MOMENTS = 100
    EP_PER_MOMENT = len(rewards) // N_MOMENTS

    # Silence windows (tracks 3-6 out): moments 8-22 and 42-55
    SILENCE_A = (8, 22)
    SILENCE_B = (42, 55)

    # Episode 300 → moment index ≈ 37
    ARRIVAL_MOMENT = 37
    # Countermelody enters ≈ episode 400 → moment 50
    COUNTER_START  = 50

    # Startup: pure melody + drone, no harmony/ornament/pulse yet
    t = cursor

    for moment in range(N_MOMENTS):
        ep_start = moment * EP_PER_MOMENT
        ep_end   = ep_start + EP_PER_MOMENT
        rew_avg  = float(np.mean(rewards[ep_start:ep_end]))
        suc_avg  = float(np.mean(successes[ep_start:ep_end]))
        eps_avg  = float(np.mean(epsilons[ep_start:ep_end]))

        in_silence = (SILENCE_A[0] <= moment <= SILENCE_A[1] or
                      SILENCE_B[0] <= moment <= SILENCE_B[1])
        is_arrival = (moment == ARRIVAL_MOMENT)

        # ---- Tempo ----
        if is_arrival:
            tempo_t.tempo(t, 20.0)   # fermata at arrival
        elif rew_avg > 5:
            tempo_t.tempo(t, BASE_BPM)
        elif rew_avg < -40:
            tempo_t.tempo(t, BASE_BPM * 0.85)   # slower when lost

        # ---- Melody (Track 1) ----
        # Degree walks through Bhairavi driven by reward
        norm_r    = float(np.clip((rew_avg + 63) / 78.0, 0.0, 1.0))
        deg_idx   = int(norm_r * (len(SCALE_DEGREES) - 1))
        m_pitch   = bhairavi_pitch(deg_idx, octave=4 if rew_avg > 0 else 3)
        dur_beats = reward_to_dur_beats(rew_avg)
        vel       = int(np.clip(35 + suc_avg * 80, 35, 115))
        humanize  = eps_avg > 0.15   # humanise while exploring

        # Early episodes: fragmentary — sometimes skip a note
        if rew_avg < -30 and rng.random() < 0.4:
            pass   # silence: no note
        else:
            melody.note(t, B(dur_beats * 0.92), m_pitch, vel, CH_MELODY,
                        humanize=humanize)

        # Arrival: hold the note + add octave overtone
        if is_arrival:
            melody.note(t, B(4), m_pitch + 12, vel - 10, CH_MELODY)
            tempo_t.tempo(t + B(4.5), BASE_BPM)

        # ---- Countermelody (Track 2) — enters at moment 50 ----
        if moment >= COUNTER_START and not in_silence:
            # Canon offset: countermelody echoes melody at 1.5 beats later
            c_pitch = bhairavi_pitch((deg_idx + 2) % 7, octave=5)
            if rng.random() < 0.65:
                counter.note(t + B(1.5), B(dur_beats * 0.88),
                             c_pitch, max(vel - 15, 30), CH_COUNTER,
                             humanize=True)

        # ---- Harmony (Track 3) — silent in silence windows ----
        if not in_silence and moment >= 5 and moment % 4 == 0:
            ivs = curl_to_intervals(mean_curl_global)
            h_root = bhairavi_pitch(0, octave=3)   # Sa in octave 3
            for iv in ivs:
                h_vel = int(np.clip(30 + suc_avg * 40, 30, 70))
                harmony.note(t, B(3.8), h_root + iv, h_vel, CH_HARMONY)

        # ---- Ornament (Track 5) — density tracks epsilon, silent in windows ----
        if not in_silence and moment >= 10 and rng.random() < eps_avg:
            o_pitch = bhairavi_pitch((deg_idx + 4) % 7, octave=5)
            n_graces = int(np.clip(eps_avg * 8, 1, 7))
            for k in range(n_graces):
                ornament.note(t + k * B(0.1),
                              B(0.08),
                              o_pitch + rng.choice(SCALE_DEGREES),
                              rng.randint(35, 60), CH_ORNAMENT)

        # ---- Pulse (Track 6) — sparse, enters ~moment 15 ----
        if not in_silence and moment >= 15:
            p_density = suc_avg * 0.6
            if rng.random() < p_density:
                pulse.note(t, B(0.3), DRUM_TOM_HI, rng.randint(45, 75), CH_PULSE)
            if rng.random() < p_density * 0.5:
                pulse.note(t + B(0.5), B(0.2), DRUM_TOM_LO,
                           rng.randint(38, 60), CH_PULSE)

        # Advance time
        beat_step = max(dur_beats, 0.5) + rng.uniform(0.2, 0.8)
        t        += B(beat_step)

    cursor = t + B(6)
    # Drone retrigger for continuity
    drone.note(cursor - B(10), B(25), sa(2), 52, CH_DRONE)

    tempo_t.marker(cursor, 'End Movement III')
    return cursor


# ---------------------------------------------------------------------------
# Movement IV — Congestion  (~150 beats, dramatic centre)
# ---------------------------------------------------------------------------

def movement_iv(tempo_t: TempoTrack,
                melody: EventList, counter: EventList,
                harmony: EventList, drone: EventList,
                ornament: EventList, pulse: EventList, bass: EventList,
                curl_field: dict, traffic_utils: list,
                tick0: int = 0) -> int:
    """
    Phase A: DQN-base collapses under congestion. Tempo rises to 76 BPM.
             Harmony thickens to dissonance. Pulse becomes dense.
             Bass (Cello) enters.
             Track 5: high-curl faces generate ornamental vortex figures.
    Pivot:   Complete silence. One beat at 999 BPM. Single low drone note.
    Phase B: DQN-cong retrains. Recovery. Tempo falls to 58 BPM.
             Harmony opens. Melody re-enters.
    """
    rng    = random.Random(4)
    cursor = tick0

    tempo_t.marker(cursor, 'Movement IV: Congestion')
    tempo_t.tempo(cursor, 60.0)

    bass.prog(cursor, CH_BASS, PROG_CELLO)
    bass.prog(cursor, CH_BASS, PROG_CELLO)

    # Total traffic data: first half = Phase A, second half = Phase B
    n_utils = len(traffic_utils)
    mid_pt  = n_utils // 2

    # ----- Phase A: Collapse -----
    tempo_t.marker(cursor, 'Phase A: Collapse')
    t = cursor

    for i, (avg_u, max_u, n_flows) in enumerate(traffic_utils[:mid_pt]):
        prog     = i / max(mid_pt - 1, 1)    # 0 → 1 as congestion builds
        bpm      = 60.0 + prog * 16.0         # 60 → 76 BPM
        tempo_t.tempo(t, bpm)

        # Melody fragments: shorter notes, lower velocity, occasional gaps
        if rng.random() > 0.35:
            m_deg   = rng.randint(0, 6)
            m_pitch = bhairavi_pitch(m_deg, octave=3)
            m_vel   = int(np.clip(40 + (1 - avg_u) * 50, 30, 90))
            melody.note(t, B(0.25 + rng.random() * 0.25),
                        m_pitch, m_vel, CH_MELODY, humanize=True)

        # Harmony: dissonance tracks utilisation
        if i % 3 == 0:
            cur_val = abs(rng.choice(list(curl_field.values())))
            ivs     = curl_to_intervals(cur_val * (1 + avg_u))
            h_root  = bhairavi_pitch(0, octave=3)
            h_vel   = int(np.clip(45 + avg_u * 55, 45, 100))
            for iv in ivs:
                harmony.note(t, B(2.8), h_root + iv, h_vel, CH_HARMONY)

        # Pulse: dense and irregular
        n_hits = max(1, int(n_flows / 12))
        for j in range(min(n_hits, 8)):
            sub    = j / max(n_hits, 1)
            p_note = DRUM_KICK if j == 0 else (DRUM_SNARE if j == 2 else DRUM_TOM_HI)
            p_vel  = int(np.clip(60 + avg_u * 50, 60, 118))
            pulse.note(t + B(sub), B(0.25), p_note, p_vel, CH_PULSE)

        if max_u > 0.65:
            pulse.note(t + B(0.5), B(0.2), DRUM_CRASH, 100, CH_PULSE)

        # Hi-hat fills as congestion rises
        n_hats = 4 if avg_u < 0.5 else 8
        for j in range(n_hats):
            pulse.note(t + B(j / n_hats), B(0.08), DRUM_CLOSED_HAT,
                       rng.randint(35, 60), CH_PULSE)

        # Bass: heavy downbeats
        bass.note(t, B(0.8), sa(1) + rng.choice([0, 5, 7]),
                  int(np.clip(55 + avg_u * 50, 55, 110)), CH_BASS)

        # Stokes ornament: high-curl faces → vortex figures
        high_curl_faces = [i for i, v in curl_field.items() if abs(v) > 1.5]
        if high_curl_faces and rng.random() < 0.45:
            n_circ = rng.randint(4, 8)
            for k in range(n_circ):
                o_deg   = (k * 2) % 7
                o_pitch = bhairavi_pitch(o_deg, octave=5)
                ornament.note(t + k * B(0.08), B(0.06),
                              o_pitch, rng.randint(38, 65), CH_ORNAMENT)

        t += B(1.0)

    # ----- Pivot: complete silence -----
    pivot_tick = t
    tempo_t.marker(pivot_tick, 'PIVOT: Silence — DQN-cong retraining')
    tempo_t.tempo(pivot_tick, 999.0)   # one bar flies by instantly
    t += B(1.0)                        # one beat of absolute silence

    # Single low drone note — the rebirth
    tempo_t.tempo(t, 58.0)
    drone.note(t, B(4), sa(1), 60, CH_DRONE)
    t += B(5)

    # ----- Phase B: Recovery -----
    tempo_t.marker(t, 'Phase B: Recovery')

    for i, (avg_u, max_u, n_flows) in enumerate(traffic_utils[mid_pt:]):
        prog     = i / max(len(traffic_utils[mid_pt:]) - 1, 1)
        consonance = prog   # 0=still congested, 1=recovered

        # Melody returns, transformed — longer notes, higher register
        if rng.random() > 0.2:
            m_deg   = int(consonance * 4)   # walks toward Pa (Sa G Pa)
            m_pitch = bhairavi_pitch(m_deg, octave=4)
            m_vel   = int(np.clip(55 + consonance * 40, 55, 95))
            dur     = 0.5 + consonance * 1.0
            melody.note(t, B(dur * 0.9), m_pitch, m_vel, CH_MELODY, humanize=True)

        # Countermelody re-enters midway through recovery
        if consonance > 0.4 and rng.random() < 0.5:
            c_pitch = bhairavi_pitch(int(consonance * 4 + 2) % 7, octave=5)
            counter.note(t + B(0.75), B(0.6), c_pitch,
                         rng.randint(45, 70), CH_COUNTER, humanize=True)

        # Harmony opens toward consonance
        if i % 4 == 0:
            curl_level = 1.5 * (1 - consonance)
            ivs        = curl_to_intervals(curl_level)
            h_root     = bhairavi_pitch(0, octave=3)
            h_vel      = int(np.clip(38 + consonance * 35, 38, 73))
            for iv in ivs:
                harmony.note(t, B(3.5), h_root + iv, h_vel, CH_HARMONY)

        # Pulse settles
        pulse.note(t,        B(0.35), DRUM_KICK,  int(70 + consonance * 10), CH_PULSE)
        pulse.note(t + B(1), B(0.25), DRUM_SNARE, 65, CH_PULSE)
        n_hats = 4 if consonance < 0.5 else 2
        for j in range(n_hats):
            pulse.note(t + B(j / n_hats), B(0.06), DRUM_CLOSED_HAT,
                       rng.randint(30, 50), CH_PULSE)

        # Bass tapers off
        if rng.random() < (1.0 - consonance * 0.5):
            bass.note(t, B(0.7), sa(1), int(60 - consonance * 20), CH_BASS)

        t += B(1.0)

    cursor = t + B(4)
    drone.note(cursor - B(6), B(14), sa(2), 55, CH_DRONE)

    tempo_t.marker(cursor, 'End Movement IV')
    return cursor


# ---------------------------------------------------------------------------
# Movement V — Epiphany  (~80 beats, gradual ritardando)
# ---------------------------------------------------------------------------

def movement_v(tempo_t: TempoTrack,
               melody: EventList, counter: EventList,
               harmony: EventList, drone: EventList,
               ornament: EventList, pulse: EventList, bass: EventList,
               verts: np.ndarray,
               tick0: int = 0) -> int:
    """
    Steady state. Pulse becomes most prominent. Melody returns for one final
    phrase, then a fifth higher. All tracks decay. Drone is last.
    Ritardando: 52 → 38 BPM over ~80 beats.
    No Western cadential resolution — it simply becomes still.
    """
    rng    = random.Random(5)
    cursor = tick0
    TOTAL  = 80

    tempo_t.marker(cursor, 'Movement V: Epiphany')

    # Gradual tempo map: write a tempo event every 8 beats
    for step in range(TOTAL // 8 + 1):
        prog = step / (TOTAL // 8)
        bpm  = 52.0 - prog * 14.0    # 52 → 38
        tempo_t.tempo(cursor + B(step * 8), bpm)

    # Drone — holds until near the very end, then fades last
    drone.note(cursor, B(TOTAL + 12), sa(2), 60, CH_DRONE)

    # Decay schedule: how many beats each track is still active
    DECAY = {
        'bass':     40,
        'pulse':    55,
        'ornament': 35,
        'counter':  62,
        'harmony':  68,
        'melody':   72,
    }

    t = cursor

    # ---- Final pulse: tani avartanam — most prominent element ----
    for beat in range(TOTAL):
        pos = beat / TOTAL

        if beat < DECAY['pulse']:
            p_vel = int(np.clip(75 - pos * 30, 45, 85))
            pulse.note(t,        B(0.4), DRUM_KICK,  p_vel, CH_PULSE)
            pulse.note(t + B(1), B(0.3), DRUM_SNARE, p_vel - 10, CH_PULSE)
            n_hats = max(1, 4 - int(pos * 3))
            for j in range(n_hats):
                pulse.note(t + B(j / n_hats), B(0.07), DRUM_CLOSED_HAT,
                           rng.randint(28, 48), CH_PULSE)

        # Ornament: fades early
        if beat < DECAY['ornament'] and rng.random() < 0.3:
            o_pitch = bhairavi_pitch(rng.randint(0, 6), octave=5)
            ornament.note(t, B(0.12), o_pitch, rng.randint(30, 50), CH_ORNAMENT)

        t += B(1.0)

    # ---- Final melody phrase: one phrase, then a fifth higher ----
    t_mel = cursor + B(2)
    for phrase in range(2):
        pitch_offset = 0 if phrase == 0 else 7   # perfect fifth up
        phrase_vel   = 78 if phrase == 0 else 65
        for step, deg in enumerate([0, 2, 4, 3, 1, 0]):
            mel_pitch = bhairavi_pitch(deg, octave=4) + pitch_offset
            dur_b     = [1.0, 0.5, 0.75, 0.5, 0.75, 2.0][step]
            m_vel     = int(phrase_vel - step * 4)
            if t_mel + B(dur_b) < cursor + B(DECAY['melody']):
                melody.note(t_mel, B(dur_b * 0.9), mel_pitch, m_vel,
                            CH_MELODY, humanize=True)
            t_mel += B(dur_b + 0.15)
        t_mel += B(3)   # pause between phrases

    # ---- Countermelody: echoes, then fades ----
    t_cnt = cursor + B(5)
    while t_cnt < cursor + B(DECAY['counter']):
        c_deg   = rng.randint(0, 6)
        c_pitch = bhairavi_pitch(c_deg, octave=5)
        fade    = 1.0 - (t_cnt - cursor) / B(DECAY['counter'])
        c_vel   = int(np.clip(50 * fade, 12, 50))
        counter.note(t_cnt, B(rng.uniform(0.5, 1.5) * 0.9),
                     c_pitch, c_vel, CH_COUNTER, humanize=True)
        t_cnt  += B(rng.uniform(1.5, 4.0))

    # ---- Harmony: sparse chords, open 5ths only ----
    t_h = cursor
    while t_h < cursor + B(DECAY['harmony']):
        fade   = 1.0 - (t_h - cursor) / B(DECAY['harmony'])
        h_vel  = int(np.clip(45 * fade, 10, 45))
        h_root = bhairavi_pitch(0, octave=3)
        harmony.note(t_h, B(5.5), h_root,     h_vel, CH_HARMONY)
        harmony.note(t_h, B(5.5), h_root + 7, h_vel, CH_HARMONY)
        t_h += B(rng.uniform(6.0, 10.0))

    # ---- Bass: mint/burn divergence signal (brief §5) ----
    # Mint = ascending 3-note figure, Burn = descending
    t_b = cursor
    while t_b < cursor + B(DECAY['bass']):
        fade   = 1.0 - (t_b - cursor) / B(DECAY['bass'])
        b_vel  = int(np.clip(65 * fade, 15, 65))
        dir_   = rng.choice([1, -1])
        for k in range(3):
            note = sa(1) + dir_ * k * 5
            bass.note(t_b + B(k * 0.4), B(0.35),
                      int(np.clip(note, 24, 72)), b_vel, CH_BASS)
        t_b += B(rng.uniform(4.0, 8.0))

    # Trailing silence: drone fades alone for ~12 beats
    end_tick = cursor + B(TOTAL + 2)
    tempo_t.marker(end_tick, 'End Movement V: Sa holds until silence')
    return end_tick + B(14)


# ---------------------------------------------------------------------------
# Score text file
# ---------------------------------------------------------------------------

def write_score_txt(movements: dict, tracks: dict, total_ticks: int,
                    filename: str = 'geodesic_symphony_score.txt'):
    """Write movement boundaries, note counts, and key events to a text file."""
    ppq  = PPQ
    # Estimate duration at 52 BPM average
    avg_tempo = bpm2tempo(52.0)
    total_sec = total_ticks * avg_tempo / (ppq * 1_000_000)

    with open(filename, 'w') as f:
        f.write('GEODESIC SYMPHONY — Score Summary\n')
        f.write('=' * 60 + '\n\n')
        f.write(f'Generated by geodesic_symphony.py\n')
        f.write(f'Harmonic world: Bhairavi (C Db Eb F G Ab Bb)\n')
        f.write(f'Ticks per beat: {PPQ}\n')
        f.write(f'Estimated total duration: {total_sec/60:.1f} min ({total_sec:.0f} s)\n\n')

        f.write('MOVEMENTS\n')
        f.write('-' * 40 + '\n')
        for name, (t_start, t_end, bpm) in movements.items():
            sec_s = t_start * bpm2tempo(bpm) / (PPQ * 1_000_000)
            sec_e = t_end   * bpm2tempo(bpm) / (PPQ * 1_000_000)
            dur   = sec_e - sec_s
            f.write(f'{name:<20}  start={t_start:>8}t  '
                    f'end={t_end:>8}t  tempo={bpm:.0f}bpm  '
                    f'~{dur:.0f}s\n')

        f.write('\nTRACKS\n')
        f.write('-' * 40 + '\n')
        for name, count in tracks.items():
            f.write(f'{name:<20}  {count:>5} notes\n')

        f.write('\nSCALE REFERENCE\n')
        f.write('-' * 40 + '\n')
        names = ['Sa', 'Reb', 'Gab', 'Ma', 'Pa', 'Dhab', 'Nib']
        for nm, iv in zip(names, SCALE_DEGREES):
            f.write(f'  {nm:<6} = MIDI {SCALE_ROOT + iv}  '
                    f'({["C","C#","D","Eb","E","F","F#","G","Ab","A","Bb","B"][iv]})\n')

        f.write('\nCOMPOSER NOTES\n')
        f.write('-' * 40 + '\n')
        f.write('Track 1 (Melody) and Track 2 (Countermelody) carry ±15-tick\n')
        f.write('timing humanisation at 480 PPQ. All patches are GM suggestions;\n')
        f.write('reassign freely in the DAW. The drone (Track 4) never stops.\n')

    print(f'Score summary written to {filename}')


# ---------------------------------------------------------------------------
# Main compose entry point
# ---------------------------------------------------------------------------

def compose(output_mid: str = 'geodesic_symphony.mid',
            output_txt: str = 'geodesic_symphony_score.txt') -> None:

    print('=' * 62)
    print('GEODESIC SYMPHONY')
    print('Five movements · Bhairavi · ~8-12 min')
    print('=' * 62)

    # --- Build geodesic mesh ---
    print('\nBuilding geodesic mesh...')
    verts_base, faces_base = build_icosahedron()
    verts_42,   faces_42   = subdivide(verts_base, faces_base)
    verts_162,  faces_162  = subdivide(verts_42,   faces_42)

    verts_base = np.array(verts_base)
    verts_42   = np.array(verts_42)
    verts_162  = np.array(verts_162)

    G = mesh_to_graph(verts_162, faces_162)
    G = assign_link_params(G)
    print(f'  {len(verts_162)} nodes | {len(faces_162)} faces | {G.number_of_edges()} edges')

    # --- Simulation data ---
    print('\nSynthesising DQN training curve...')
    rewards, successes, epsilons = synthesize_dqn_curve(800)

    print('\nComputing Stokes curl field...')
    curl_field = get_curl_field(G, faces_162, n_steps=25)

    print('\nSimulating traffic...')
    traffic_utils = get_traffic_util(G, verts_162, n_steps=40)

    # --- MIDI setup ---
    # type=1: multi-track synchronous
    mid = MidiFile(type=1, ticks_per_beat=PPQ)

    # 8 tracks per brief; track 0 = Structure (tempo/markers)
    tempo_t  = TempoTrack()

    melody   = EventList()
    counter  = EventList()
    harmony  = EventList()
    drone    = EventList()
    ornament = EventList()
    pulse    = EventList()
    bass     = EventList()

    # --- Compose movements ---
    movement_bounds = {}
    t = 0

    print('\n--- Movement I: Genesis ---')
    t0 = t
    t  = movement_i(tempo_t, harmony, drone, verts_base, tick0=t)
    movement_bounds['I. Genesis'] = (t0, t, 52.0)
    print(f'  ends at tick {t}')

    print('\n--- Movement II: Awakening ---')
    t0 = t
    t  = movement_ii(tempo_t, harmony, drone, ornament, pulse,
                     verts_42, verts_162, tick0=t)
    movement_bounds['II. Awakening'] = (t0, t, 52.0)
    print(f'  ends at tick {t}')

    print('\n--- Movement III: Learning ---')
    t0 = t
    t  = movement_iii(tempo_t, melody, counter, harmony, drone,
                      ornament, pulse, rewards, successes, epsilons,
                      faces_162, curl_field, tick0=t)
    movement_bounds['III. Learning'] = (t0, t, 52.0)
    print(f'  ends at tick {t}')

    print('\n--- Movement IV: Congestion ---')
    t0 = t
    t  = movement_iv(tempo_t, melody, counter, harmony, drone,
                     ornament, pulse, bass,
                     curl_field, traffic_utils, tick0=t)
    movement_bounds['IV. Congestion'] = (t0, t, 68.0)
    print(f'  ends at tick {t}')

    print('\n--- Movement V: Epiphany ---')
    t0 = t
    t  = movement_v(tempo_t, melody, counter, harmony, drone,
                    ornament, pulse, bass, verts_162, tick0=t)
    movement_bounds['V. Epiphany'] = (t0, t, 45.0)
    print(f'  ends at tick {t}')

    total_ticks = t

    # --- Build MidiFile ---
    track_defs = [
        (tempo_t.to_track(),              'Structure'),
        (melody.to_track('Melody'),       'Melody'),
        (counter.to_track('Countermelody'), 'Countermelody'),
        (harmony.to_track('Harmony'),     'Harmony'),
        (drone.to_track('Drone'),         'Drone'),
        (ornament.to_track('Ornament'),   'Ornament'),
        (pulse.to_track('Pulse'),         'Pulse'),
        (bass.to_track('Bass'),           'Bass'),
    ]

    for trk, _ in track_defs:
        mid.tracks.append(trk)

    # --- Save ---
    mid.save(output_mid)
    print(f'\nSaved: {output_mid}')

    # --- Score text ---
    track_note_counts = {
        'Melody':          melody.note_count,
        'Countermelody':   counter.note_count,
        'Harmony':         harmony.note_count,
        'Drone':           drone.note_count,
        'Ornament':        ornament.note_count,
        'Pulse':           pulse.note_count,
        'Bass':            bass.note_count,
    }
    write_score_txt(movement_bounds, track_note_counts,
                    total_ticks, output_txt)

    # --- Summary ---
    print('\n' + '=' * 62)
    print('Symphony complete.')
    print(f'\n  {"Track":<20} {"Notes":>6}')
    print(f'  {"-"*28}')
    for name, cnt in track_note_counts.items():
        print(f'  {name:<20} {cnt:>6}')
    total_notes = sum(track_note_counts.values())
    print(f'  {"TOTAL":<20} {total_notes:>6}')
    print(f'\n  Total ticks : {total_ticks:,}  ({total_ticks/PPQ:.0f} beats)')
    approx_min = total_ticks / PPQ / 52 * 60 / 60
    print(f'  Approx time : {approx_min:.1f} min at avg 52 BPM')
    print(f'  MIDI file   : {output_mid}')
    print(f'  Score text  : {output_txt}')
    print('=' * 62)


if __name__ == '__main__':
    compose()
