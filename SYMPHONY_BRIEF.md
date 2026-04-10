# Geodesic Symphony — notes on what we built

**File:** `geodesic_symphony.py`  
**Output:** `geodesic_symphony.mid` + `geodesic_symphony_score.txt`  
**Last updated:** 2026-04-08 (updated to reflect actual implementation)

---

## What this actually is

It's a composition engine, not a visualiser. It takes all the live simulation
data — the mesh geometry, the DQN training curve, the congestion numbers,
the Stokes curl field — and turns it into a ~11 minute MIDI file with 8 named
tracks you can pull into a DAW and work with.

Three things are happening simultaneously in the music:
- a packet finding its way across the geodesic mesh
- an RL agent slowly figuring out what it's doing
- a network choking on itself and then recovering

It came out to 1,209 notes across 581 beats. Runs in about 30 seconds
including the live traffic and curl sims.

---

## The scale — Bhairavi throughout

Everything is in Bhairavi. No exceptions.

```
Sa   Reb  Gab  Ma   Pa   Dhab  Nib
C    Db   Eb   F    G    Ab    Bb
60   61   63   65   67   68    70
```

It spans carnatic, blues, and folk naturally, which is the point.
Root is middle C (60) but you can move it without touching the generation logic —
just change `SCALE_ROOT`.

```python
SCALE_ROOT    = 60
SCALE_DEGREES = [0, 1, 3, 5, 7, 8, 10]
OCTAVE_RANGE  = (3, 6)
```

The pitch mapping uses the y-coordinate of each mesh vertex for octave (low
hemisphere = low register) and the azimuthal angle for scale degree. This
means physically close nodes on the sphere tend to play nearby notes, which
gives the Genesis movement a nice slowly-unfolding quality.

---

## Five movements

### I — Genesis (~2 min, free time)

Just the 12 base icosahedron vertices. That's it. They sound one at a time
with 4-8 seconds of silence between them — literally the mesh being
inscribed on the sphere. Tempo wanders between 40 and 70 BPM, one tempo per
vertex.

Only Harmony and Drone are active here. Nothing else. No pulse, no melody,
no ornament. The silence is structural.

### II — Awakening (~2 min)

The two subdivision events (12→42, 42→162 nodes) each trigger a wave of
new texture. More voices enter as the mesh gets denser. The Ornament track
comes in here — rapid, scattered figures, high register, no coherent melody.
That's the DQN at epsilon=0.4, just thrashing around.

Pulse enters too, sparse and irregular. Drone settles on Sa.

### III — Learning (~4 min, this is the long one)

This is supposed to feel like a carnatic alapana — unhurried, exploratory,
melody finding its way. It maps 800 training episodes (compressed to ~100
musical moments) across roughly 200 beats.

Melody enters alone, just it and the drone, for the first stretch. Early on
it's fragmentary — short notes, low velocity, sometimes just silence (the
agent failing to reach the target). As the reward climbs, notes get longer
and more purposeful.

Two silence windows where tracks 3-6 drop out completely:
- moments 8-22 (early exploration phase)  
- moments 42-55 (mid-training plateau)

Just melody and drone during these. This is important — without them it
sounds like background music, not an alapana.

Episode 300 (moment 37 in our compressed timeline) is the arrival — tempo
drops to 20 BPM for a fermata, melody holds a long note, octave overtone
sounds. Then back to 52 BPM.

Countermelody enters around episode 400 (moment 50), echoing melody at 1.5
beats offset. Canon-ish.

### IV — Congestion (~45 seconds, dramatic centre)

Has two phases with a silence pivot between them.

**Phase A** — DQN-base collapsing. Tempo climbs from 60 to 76 BPM.
Melody fragments. Harmony thickens — the curl-to-intervals function is
mapping the live Stokes field to chord clusters (open 5th when quiet, tritone
cluster when rho > 0.6). Pulse gets dense and irregular. Bass (Cello) enters
for the first time, playing heavy downbeats.

The ornament track is doing the vortex figures here — high-curl faces in the
Stokes field trigger little circling gamaka-like ornaments in the Crystal
register.

**Pivot** — complete silence. One beat at 999 BPM so it flies by
invisibly. Then a single low Sa on the drone. The retraining moment.

**Phase B** — DQN-cong recovering. Tempo drops to 58 BPM. Melody re-enters
longer and higher. Harmony opens back toward consonance as the congestion
clears. Pulse settles.

### V — Epiphany (~2 min with ritardando)

Tempo walks from 52 down to 38 BPM over 80 beats via tempo events every 8
beats. Gradual.

Melody plays its last phrase — 6 notes, then the same phrase a perfect fifth
higher, then fades. That's all it gets.

Tracks decay in this order: bass (40 beats), ornament (35), pulse (55),
countermelody (62), harmony (68), melody (72). Drone holds to the very end
then just stops.

No resolution chord. It just goes quiet.

The bass in this movement plays the mint/burn divergence figures — ascending
3-note figure for mint, descending for burn. At equilibrium you get silence,
which is right.

---

## The 8 tracks

| # | Name | GM suggestion | What it does | When it plays |
|---|------|---------------|--------------|---------------|
| 1 | Structure | — | Tempo map, movement markers | always |
| 2 | Melody | 104 Sitar | DQN reward → pitch/duration, humanised | III IV V |
| 3 | Countermelody | 73 Flute | Canon echo of melody, humanised | III IV V |
| 4 | Harmony | 48 Strings | Curl field → chord voicings | I–V |
| 5 | Drone | 92 Pad | Sa pedal, never stops, retriggered | I–V |
| 6 | Ornament | 98 Crystal | Epsilon decay + vortex figures | II–V |
| 7 | Pulse | 116 Taiko (ch 9) | Congestion events → rhythm | II–V |
| 8 | Bass | 42 Cello | Mint/burn divergence figures | IV V |

Logic auto-assigned: Dark Abyss, Piccolo, Timpani, Classic Choir, Distant
Air, SoCal, Violas. Reassign everything in the DAW — the GM patches are
just suggestions.

---

## How the mappings work

**Node → pitch**

y-coordinate of the vertex → octave (south pole = octave 3, north = octave 5)  
atan2(z, x) → scale degree in Bhairavi

```python
def pos_to_pitch(pos, octave_range=(3, 6)):
    y      = pos[1]
    angle  = np.arctan2(pos[2], pos[0])
    octave = int(np.interp(y, [-1.0, 1.0], octave_range))
    deg_i  = int(np.interp(angle, [-np.pi, np.pi], [0, 6]))
    return (octave + 1) * 12 + (SCALE_ROOT % 12) + SCALE_DEGREES[deg_i]
```

Note: the original brief had `root + octave * 12 + degree` which gives
out-of-range pitches. The implementation uses `(octave+1)*12 + root_pc + degree`
which is the standard MIDI formula. Same musical result, correct numbers.

**Reward → note duration**

| reward | duration |
|--------|----------|
| < −30  | sixteenth |
| −30 to −10 | eighth |
| −10 to +5  | quarter |
| +5 to +12  | half |
| > +12  | whole |

**Curl → harmony**

| mean \|curl\| | voicing |
|--------------|---------|
| < 0.5 | Sa Pa (open fifth) |
| 0.5–1.5 | Sa Re Ni (added 2nd/7th) |
| > 1.5 | Sa Reb Gab (tight cluster) |

**Epsilon → ornament density**

The ornament fires with probability = epsilon. So at 0.4 it's playing on
40% of moments. By the end of training it's barely firing.

---

## Timing

- Base: 52 BPM throughout except where noted
- Movement I: per-vertex tempo, 40–70 BPM range, free time feel
- Movement III arrival (ep 300): drops to 20 BPM for the fermata
- Movement IV Phase A: climbs 60→76 BPM with congestion
- Movement IV pivot: 999 BPM for one beat (silence trick)
- Movement IV Phase B: 58 BPM
- Movement V: 52→38 BPM ritardando over 80 beats

---

## Dependencies

```bash
pip install mido       # that's it
```

`pretty_midi` isn't needed — we use `mido` directly for tick-level control.
`midiutil` is installed from an earlier attempt but not used here.

The stokes curl and traffic sims run lightweight (25 and 40 steps). If either
fails to import it falls back to mock data from the PROJECT_CONTEXT.md numbers
and logs what it's doing. It won't fail silently.

---

## Notes for working with it in the DAW

Melody and Countermelody have ±15 tick timing humanisation at 480 PPQ baked
in — they're not perfectly quantised and that's intentional.

Velocity varies meaningfully: high Q-value confidence plays louder, exploration
phases play softer. Don't normalise velocity or you'll lose the dynamics.

The drone is the anchor. Don't mute it. It's retriggered every 3 vertices in
Movement I and periodically through the rest of the piece to avoid decay gaps.

Leave space. The silences in Movement III especially are load-bearing. If you
layer something on top of them you'll turn the alapana into wallpaper.

The Pulse track is on GM channel 9 (drums). In Logic it showed up as SoCal.
You'll want to reassign that but the rhythmic data is all there.

---

*Updated 2026-04-08 after implementation. The original brief was written as
a spec before the code existed — this version reflects what actually runs.*
