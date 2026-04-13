"""
audio_extractor.py — Extract per-step audio state from pipeline module results.

Produces an array of dicts (one per simulation step) that the browser-side
diagnostic_listener.js consumes for generative audio.  Additive to the
existing pipeline — nothing is modified, only read.

Uses harmonic minor scale with icosahedral geometry tuning.
"""

import math
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Harmonic minor scale (semitone intervals from root)
# ---------------------------------------------------------------------------
HARMONIC_MINOR = [0, 2, 3, 5, 7, 8, 11]          # 7 degrees per octave


def _base_icosahedron_vertices() -> np.ndarray:
    """Return the 12 vertices of a unit-sphere icosahedron."""
    phi = (1 + math.sqrt(5)) / 2
    verts = [
        (-1,  phi, 0), ( 1,  phi, 0), (-1, -phi, 0), ( 1, -phi, 0),
        ( 0, -1,  phi), ( 0,  1,  phi), ( 0, -1, -phi), ( 0,  1, -phi),
        ( phi, 0, -1), ( phi, 0,  1), (-phi, 0, -1), (-phi, 0,  1),
    ]
    arr = np.array(verts, dtype=float)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / norms


ICO_BASE_VERTS = _base_icosahedron_vertices()


# ---------------------------------------------------------------------------
# Node → Frequency mapping  (icosahedral tuning, harmonic minor)
# ---------------------------------------------------------------------------

def precompute_node_frequencies(
    verts: List[Tuple[float, float, float]],
    base_freq: float = 220.0,
) -> List[float]:
    """Map each mesh vertex to a frequency.

    Elevation (y) → octave position  (3 octaves: base_freq → base_freq*4)
    Azimuth (xz)  → scale degree within harmonic minor
    Proximity to base-icosahedron vertex → microtonal inflection ±30 cents
    """
    freqs: List[float] = []
    for x, y, z in verts:
        # --- octave from elevation ---
        octave_t = (y + 1.0) / 2.0                              # 0..1
        octave_freq = base_freq * (2.0 ** (octave_t * 2.0))     # 3 octaves

        # --- scale degree from azimuth ---
        azimuth = math.atan2(z, x)                               # -π..π
        idx = int((azimuth + math.pi) / (2 * math.pi) * len(HARMONIC_MINOR))
        idx = idx % len(HARMONIC_MINOR)
        semitone = HARMONIC_MINOR[idx]
        scale_freq = octave_freq * (2.0 ** (semitone / 12.0))

        # --- microtonal inflection from proximity to base ico vertex ---
        dists = np.linalg.norm(ICO_BASE_VERTS - np.array([x, y, z]), axis=1)
        proximity = float(np.min(dists))
        max_prox  = 0.8
        cents     = (proximity / max_prox) * 30.0                # 0..30 cents
        cents    *= (1.0 if y >= 0 else -1.0)
        micro     = 2.0 ** (cents / 1200.0)

        freqs.append(round(scale_freq * micro, 4))
    return freqs


# ---------------------------------------------------------------------------
# Build per-step audio states from pipeline results
# ---------------------------------------------------------------------------

def _pick_diverse_pairs(G, n: int = 4, seed: int = 42) -> List[Tuple[int, int]]:
    """Select n source-target pairs spread across the mesh."""
    rng = random.Random(seed)
    nodes = sorted(G.nodes())
    # Pick nodes roughly evenly spaced by index
    step = max(1, len(nodes) // (n * 2 + 1))
    sources = [nodes[step * (2 * i + 1)] for i in range(n)]
    targets = [nodes[step * (2 * i + 2)] for i in range(n)]
    # Ensure no source == target
    for i in range(n):
        if sources[i] == targets[i]:
            targets[i] = nodes[(step * (2 * i + 2) + step // 2) % len(nodes)]
    return list(zip(sources, targets))


def _route_path(G, router, src: int, tgt: int) -> List[int]:
    """Route src→tgt using DQN router; fall back to hop-shortest."""
    import networkx as nx
    if router is not None:
        try:
            path, reached = router.route(src, tgt, max_steps=30)
            if reached and len(path) > 1:
                return path
        except Exception:
            pass
    try:
        return nx.shortest_path(G, src, tgt, weight=None)
    except Exception:
        return [src, tgt]


def build_audio_states(
    upstream: Dict[str, Any],
    params:   Dict[str, Any],
) -> Optional[List[Dict[str, Any]]]:
    """Build array of per-step audio state dicts from pipeline upstream.

    Returns None if required modules (mesh) haven't run.
    Gracefully handles any missing optional module.
    """
    # ── Mesh (required) ─────────────────────────────────────────────────
    mesh = upstream.get("mesh")
    if not mesh:
        return None
    G     = mesh["_G"]
    verts = mesh["_verts"]
    faces = mesh["_faces"]

    node_freqs = precompute_node_frequencies(verts)
    n_steps    = int(params.get("sim_steps", 200))

    # ── DQN router (optional) ──────────────────────────────────────────
    dqn    = upstream.get("dqn", {})
    router = dqn.get("_router")
    epsilon_final = getattr(router, "epsilon", 0.05) if router else 0.4

    # Pre-compute 4 melodic paths (fixed for this run)
    pairs = _pick_diverse_pairs(G, n=4)
    routing_paths = [_route_path(G, router, s, t) for s, t in pairs]

    # Path latencies from the mesh edge weights
    def _path_lat(path):
        lats = []
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            edge = G[u][v] if G.has_edge(u, v) else {}
            lats.append(edge.get("base_latency", 0.1))
        return lats

    path_latencies = [_path_lat(p) for p in routing_paths]

    # Q-confidence proxy: longer path = lower confidence, normalize 0..1
    max_len = max((len(p) for p in routing_paths), default=1)
    q_confidence = [round(1.0 - (len(p) - 1) / max(max_len, 1), 3)
                    for p in routing_paths]

    # ── Traffic history (optional) ─────────────────────────────────────
    traffic = upstream.get("traffic", {})
    traffic_hist = traffic.get("_history", [])

    # ── Stokes curl (optional) ─────────────────────────────────────────
    stokes = upstream.get("stokes", {})
    stokes_sim = stokes.get("_sim")
    curl_hist = getattr(stokes_sim, "curl_history", []) if stokes_sim else []

    # ── Green divergence (optional) ────────────────────────────────────
    green = upstream.get("green", {})
    green_sim = green.get("_sim")
    div_hist = getattr(green_sim, "div_history", []) if green_sim else []

    # ── ComCoin (optional) ─────────────────────────────────────────────
    comcoin  = upstream.get("comcoin", {})
    ema_div  = comcoin.get("_ema_div", [])
    raw_div  = comcoin.get("_raw_div", [])

    # ── Genetic (optional) ─────────────────────────────────────────────
    genetic = upstream.get("genetic", {})
    ga_fitness = genetic.get("results", {}).get("best_fitness", 0.5)
    # Normalize roughly to 0..1
    ga_fitness_norm = min(1.0, max(0.0, ga_fitness))

    # ── Build step-by-step states ──────────────────────────────────────
    audio_states: List[Dict[str, Any]] = []

    for step in range(n_steps):
        state: Dict[str, Any] = {"step": step}

        # Routing paths (constant for all steps — melodic material)
        state["routing_paths"]  = routing_paths
        state["path_latencies"] = path_latencies
        state["q_confidence"]   = q_confidence

        # Epsilon: simulate decay from epsilon_start to epsilon_final
        eps_start = float(params.get("epsilon_start", 0.4))
        if n_steps > 1:
            decay_t = step / (n_steps - 1)
            state["epsilon"] = round(eps_start + (epsilon_final - eps_start) * decay_t, 4)
        else:
            state["epsilon"] = eps_start

        # Curl field (320 floats for V6 ripieno)
        if step < len(curl_hist):
            cf = curl_hist[step]
            state["curl_field"] = [round(cf.get(f, 0.0), 4) for f in range(len(faces))]
            state["mean_curl"]  = round(float(np.mean([abs(v) for v in cf.values()])), 4) if cf else 0.0
        else:
            state["curl_field"] = [0.0] * len(faces)
            state["mean_curl"]  = 0.0

        # Node divergence (162 floats for voice pitch drift)
        if step < len(div_hist):
            dv = div_hist[step]
            state["node_divergence"] = [round(dv.get(n, 0.0), 4) for n in range(len(verts))]
        else:
            state["node_divergence"] = [0.0] * len(verts)

        # Net divergence (scalar — master amplitude)
        state["net_divergence"] = round(ema_div[step], 4) if step < len(ema_div) else 0.0

        # Active flows + utilisation
        if step < len(traffic_hist):
            th = traffic_hist[step]
            state["active_flows"] = th.get("active_flows", 0)
            state["avg_util"]     = round(th.get("avg_util", 0.0), 4)
        else:
            state["active_flows"] = 0
            state["avg_util"]     = 0.0

        # ComCoin redemption proximity (30-step cycle, window at steps 22-25)
        cycle_pos = step % 30
        if cycle_pos < 22:
            state["redemption_proximity"] = round((22 - cycle_pos) / 22.0, 4)
        elif cycle_pos <= 25:
            state["redemption_proximity"] = 0.0   # in the window
        else:
            state["redemption_proximity"] = round((30 - cycle_pos + 22) / 22.0, 4)

        # GA fitness (constant across steps)
        state["ga_fitness_norm"] = round(ga_fitness_norm, 4)

        audio_states.append(state)

    # ── Metadata (constant, not per-step) ──────────────────────────────
    return {
        "node_frequencies": node_freqs,
        "n_nodes":          len(verts),
        "n_faces":          len(faces),
        "n_steps":          n_steps,
        "base_freq":        220.0,
        "scale":            "harmonic_minor",
        "states":           audio_states,
    }
