"""
green_flow.py — Green's Theorem flow-balancing and divergence-aware routing.

Patent §D: "For each mesh cell, ∮(P dx + Q dy) = ∬(∂Q/∂x − ∂P/∂y) dA.
            Detect sink/source nodes from net flow divergence; adjust link
            weights until balanced."

INTUITION (no maths background needed):
  Stokes detected traffic *spinning* around triangles (rotational congestion).
  Green detects traffic *piling up* or *draining* at nodes (divergence congestion).

  At steady state a healthy router node should have equal inflow and outflow —
  what arrives should leave.  If a node consistently receives more traffic than
  it forwards, it is a SINK (packets getting dropped there).  If it generates
  more than it absorbs, it is a SOURCE (spamming the network).

  Green's Theorem formalises this:
      ∮ boundary flows = ∬ divergence inside
  i.e., the net flux crossing a region's boundary equals the total
  "source minus sink" inside it.  We use this both as a consistency check
  and as the signal for re-routing.

  div(v) = outflow(v) − inflow(v)
    > 0  → source (generating more than absorbing)
    < 0  → sink   (absorbing more than forwarding)
    ≈ 0  → balanced

  Mitigation: penalise edges that lead INTO sinks or OUT OF sources so
  that routers naturally route around pathological nodes.

COMPARISON (4 strategies):
  1. Hop-SP     — hop-minimising Dijkstra (baseline)
  2. Curl-aware — Stokes penalty only (imported from stokes_curl.py)
  3. Green-aware — divergence penalty only (new)
  4. Combined   — both penalties simultaneously (key result)

OUTPUT:  green_flow_results.png  (4-panel plot)
METRICS: improvement vs hop-SP for each strategy.
"""

import matplotlib
matplotlib.use('Agg')   # non-interactive backend — no display required
import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

from geodesic_overlay import build_icosahedron, subdivide, mesh_to_graph
from dqn_router import assign_link_params
from stokes_curl import (
    DirectedFlowSim,
    build_edge_to_faces,
    build_curl_weights,
    route_hop_sp,
    route_curl_aware,
)

random.seed(42)
np.random.seed(42)


# ---------------------------------------------------------------------------
# Extended simulator — tracks divergence history alongside curl
# ---------------------------------------------------------------------------

class GreenFlowSim(DirectedFlowSim):
    """Subclass of DirectedFlowSim that also records per-step node divergence.

    Divergence at node v:
        div(v) = Σ outflows(v→u)  −  Σ inflows(u→v)

    At steady state a balanced node has div ≈ 0.
    Persistent div < 0 → sink (dropping packets).
    Persistent div > 0 → source (spamming the network).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.div_history: List[Dict[int, float]] = []

    def _compute_divergence_now(self) -> Dict[int, float]:
        """Compute divergence for every node from current directed_load."""
        divs: Dict[int, float] = {}
        for v in self.G.nodes():
            out = sum(self.directed_load[(v, u)] for u in self.G.neighbors(v))
            inn = sum(self.directed_load[(u, v)] for u in self.G.neighbors(v))
            divs[v] = out - inn
        return divs

    def run(self, n_steps: int = 100, verbose: bool = True):
        """Run simulation, recording both curl_history and div_history.

        Returns:
            (curl_history, div_history) — one dict per step, keyed by face/node index.
        """
        print(f"Running Green+Stokes simulation: {n_steps} steps, "
              f"arrival_rate={self.arrival_rate:.1f}, "
              f"flow_duration={self.flow_duration:.1f}")

        for i in range(n_steps):
            # Parent step handles flow injection/expiry and appends to curl_history
            curl = super().step()

            # Compute and record divergence from the updated directed_load
            div = self._compute_divergence_now()
            self.div_history.append(div)

            if verbose and (i + 1) % 20 == 0:
                div_vals = list(div.values())
                curl_vals = list(curl.values())
                print(f"  Step {i+1:3d}: "
                      f"flows={len(self.active_flows):3d}, "
                      f"mean|div|={np.mean(np.abs(div_vals)):.2f}, "
                      f"max|div|={np.max(np.abs(div_vals)):.2f}, "
                      f"sinks(div<-1)={sum(1 for d in div_vals if d < -1)}, "
                      f"sources(div>1)={sum(1 for d in div_vals if d > 1)}, "
                      f"mean|curl|={np.mean(np.abs(curl_vals)):.2f}")

        return self.curl_history, self.div_history


# ---------------------------------------------------------------------------
# Green's Theorem functions
# ---------------------------------------------------------------------------

def compute_node_divergence(
    G: nx.Graph,
    directed_flows: Dict[Tuple[int, int], float],
) -> Dict[int, float]:
    """Compute net divergence for each node from directed flow counts.

    div(v) = Σ_{u ∈ N(v)} directed_flows[(v,u)]   (outflows)
           − Σ_{u ∈ N(v)} directed_flows[(u,v)]   (inflows)

    Positive → source, Negative → sink, Zero → balanced.
    """
    divs: Dict[int, float] = {}
    for v in G.nodes():
        out = sum(directed_flows[(v, u)] for u in G.neighbors(v))
        inn = sum(directed_flows[(u, v)] for u in G.neighbors(v))
        divs[v] = out - inn
    return divs


def green_boundary_integral(
    G: nx.Graph,
    directed_flows: Dict[Tuple[int, int], float],
    node_subset: Set[int],
) -> float:
    """Compute the net flux crossing the boundary of node_subset outward.

    By Green's Theorem this should equal Σ div(v) for v ∈ node_subset.
    Use this to verify that compute_node_divergence is consistent.

    boundary_integral = Σ_{u∈subset, v∉subset} [flow(u→v) − flow(v→u)]
    """
    boundary_flux = 0.0
    for u in node_subset:
        for v in G.neighbors(u):
            if v not in node_subset:
                boundary_flux += (directed_flows[(u, v)]
                                  - directed_flows[(v, u)])
    return boundary_flux


def green_aware_weights(
    G: nx.Graph,
    divs: Dict[int, float],
    sink_penalty: float = 5.0,
    source_penalty: float = 3.0,
    div_threshold: float = 2.0,
) -> Dict[Tuple[int, int], float]:
    """Build divergence-penalised routing weights.

    Logic:
      • Edge (u,v) becomes expensive if v is a heavy sink (div[v] << 0):
        routing more traffic into v compounds its congestion.
      • Edge (u,v) becomes expensive if u is a heavy source (div[u] >> 0):
        flowing out of a source amplifies its imbalance downstream.
      • Symmetric: the same penalty applies in both directions (undirected routing).

    weight(u,v) = base_latency
                + sink_penalty   * max(0, −div[u] − threshold)
                + sink_penalty   * max(0, −div[v] − threshold)
                + source_penalty * max(0,  div[u] − threshold)
                + source_penalty * max(0,  div[v] − threshold)
    """
    weights: Dict[Tuple[int, int], float] = {}
    for u, v in G.edges():
        base = G[u][v]['base_latency']
        pen_u = (sink_penalty   * max(0.0, -divs[u] - div_threshold) +
                 source_penalty * max(0.0,  divs[u] - div_threshold))
        pen_v = (sink_penalty   * max(0.0, -divs[v] - div_threshold) +
                 source_penalty * max(0.0,  divs[v] - div_threshold))
        total = base + pen_u + pen_v
        weights[(u, v)] = total
        weights[(v, u)] = total
    return weights


def build_combined_weights(
    G: nx.Graph,
    curl_weights: Dict[Tuple[int, int], float],
    green_weights: Dict[Tuple[int, int], float],
) -> Dict[Tuple[int, int], float]:
    """Combine curl and green penalties using max, not sum.

    Additive combination leads to penalty stacking: edges near both a curl
    vortex AND a divergence node get an astronomically high combined penalty,
    forcing extreme detours that are actually worse in effective latency.

    Using max avoids double-counting: an edge is penalised by whichever
    signal is the most urgent, but not both simultaneously.

    combined(u,v) = base_latency + max(curl_penalty, green_penalty)

    An edge that is problematic in EITHER dimension is avoided; an edge
    that is problematic in BOTH is penalised only once (by the larger).
    """
    combined: Dict[Tuple[int, int], float] = {}
    for u, v in G.edges():
        base = G[u][v]['base_latency']
        curl_pen  = curl_weights.get((u, v), base) - base
        green_pen = green_weights.get((u, v), base) - base
        combined[(u, v)] = base + max(curl_pen, green_pen)
        combined[(v, u)] = base + max(curl_pen, green_pen)
    return combined


# ---------------------------------------------------------------------------
# Routing helpers
# ---------------------------------------------------------------------------

def route_green_aware(
    G: nx.Graph,
    s: int,
    t: int,
    green_weights: Dict[Tuple[int, int], float],
) -> Optional[list]:
    """Dijkstra using green-penalised weights."""
    try:
        return nx.shortest_path(
            G, s, t,
            weight=lambda u, v, _: green_weights.get((u, v), G[u][v]['base_latency']),
        )
    except nx.NetworkXNoPath:
        return None


def route_combined(
    G: nx.Graph,
    s: int,
    t: int,
    combined_weights: Dict[Tuple[int, int], float],
) -> Optional[list]:
    """Dijkstra using combined curl+green weights."""
    try:
        return nx.shortest_path(
            G, s, t,
            weight=lambda u, v, _: combined_weights.get((u, v), G[u][v]['base_latency']),
        )
    except nx.NetworkXNoPath:
        return None


# ---------------------------------------------------------------------------
# 4-way comparison
# ---------------------------------------------------------------------------

def run_comparison_4way(
    G: nx.Graph,
    sim: GreenFlowSim,
    avg_curl: Dict[int, float],
    avg_divs: Dict[int, float],
    edge_to_faces: Dict[Tuple[int, int], List[int]],
    n_pairs: int = 60,
    curl_threshold: float = 1.0,
    curl_penalty: float = 8.0,
    sink_penalty: float = 5.0,
    source_penalty: float = 3.0,
    div_threshold: float = 2.0,
) -> List[Dict]:
    """Compare all four routing strategies on n_pairs random source-destination pairs.

    All latencies measured with M/M/1 effective latency (congestion-aware).
    """
    # Build weight maps
    curl_weights  = build_curl_weights(G, avg_curl, edge_to_faces,
                                       curl_threshold, curl_penalty)
    green_weights = green_aware_weights(G, avg_divs,
                                        sink_penalty, source_penalty, div_threshold)
    comb_weights  = build_combined_weights(G, curl_weights, green_weights)

    nodes   = list(G.nodes())
    results = []

    def path_avg_div(path: list) -> float:
        """Mean |div| across all nodes on the path."""
        return float(np.mean([abs(avg_divs.get(v, 0.0)) for v in path]))

    attempts = 0
    while len(results) < n_pairs and attempts < n_pairs * 5:
        attempts += 1
        s, t = random.sample(nodes, 2)

        sp   = route_hop_sp(G, s, t)
        cap  = route_curl_aware(G, s, t, curl_weights)
        gp   = route_green_aware(G, s, t, green_weights)
        comb = route_combined(G, s, t, comb_weights)

        if any(p is None or len(p) < 3 for p in [sp, cap, gp, comb]):
            continue

        results.append({
            'sp_lat':   sim.path_latency(sp),
            'curl_lat': sim.path_latency(cap),
            'green_lat':sim.path_latency(gp),
            'comb_lat': sim.path_latency(comb),
            'sp_div':   path_avg_div(sp),
            'curl_div': path_avg_div(cap),
            'green_div':path_avg_div(gp),
            'comb_div': path_avg_div(comb),
            'sp_hops':   len(sp)   - 1,
            'comb_hops': len(comb) - 1,
        })

    print(f"  Compared {len(results)} pairs ({attempts} attempts)")
    return results


# ---------------------------------------------------------------------------
# Green's theorem consistency check
# ---------------------------------------------------------------------------

def verify_green_theorem(G, sim, avg_divs, n_samples=10):
    """Verify boundary_integral ≈ Σ div for random node subsets.

    In a discrete setting with integer flow counts this is exact.
    With floating-point and time-averaged divs it should be very close.
    """
    print("\nGreen's Theorem verification (boundary flux ≈ Σ divergence):")
    nodes = list(G.nodes())
    for i in range(n_samples):
        subset_size = random.randint(5, 30)
        subset = set(random.sample(nodes, subset_size))
        boundary = green_boundary_integral(G, sim.directed_load, subset)
        internal = sum(avg_divs[v] for v in subset)
        err = abs(boundary - internal)
        print(f"  Sample {i+1:2d}: subset={subset_size:3d} nodes, "
              f"boundary_flux={boundary:+.3f}, "
              f"Σdiv={internal:+.3f}, "
              f"|error|={err:.4f}")
    print("  (small errors expected from time-averaging; exact for instantaneous flows)")


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def visualize(
    verts: np.ndarray,
    G: nx.Graph,
    avg_divs: Dict[int, float],
    div_history: List[Dict[int, float]],
    results: List[Dict],
):
    fig = plt.figure(figsize=(16, 12))

    div_values = np.array([avg_divs[v] for v in range(len(verts))])
    max_abs_div = max(float(np.abs(div_values).max()), 0.1)

    # --- Panel 1: Node divergence map on 3D sphere ---
    ax1 = fig.add_subplot(221, projection='3d')
    cmap = plt.colormaps['RdBu']   # red = sink (neg), blue = source (pos)
    # Normalise so 0 maps to center (white/neutral)
    normed = (div_values + max_abs_div) / (2 * max_abs_div)
    colors = [cmap(float(n)) for n in normed]

    xs, ys, zs = verts[:, 0], verts[:, 1], verts[:, 2]
    sc = ax1.scatter(xs, ys, zs, c=div_values, cmap='RdBu',
                     vmin=-max_abs_div, vmax=max_abs_div, s=15, alpha=0.85)
    # Draw edges lightly for context
    for u, v in list(G.edges())[:200]:   # sample edges to avoid clutter
        x = [verts[u][0], verts[v][0]]
        y = [verts[u][1], verts[v][1]]
        z = [verts[u][2], verts[v][2]]
        ax1.plot(x, y, z, color='grey', alpha=0.08, linewidth=0.4)

    n_sinks   = sum(1 for d in div_values if d < -1)
    n_sources = sum(1 for d in div_values if d >  1)
    ax1.set_title(f'Node Divergence (Green\'s Theorem)\n'
                  f'Red=sink ({n_sinks}), Blue=source ({n_sources}), White=balanced')
    ax1.axis('off')
    ax1.set_box_aspect([1, 1, 1])
    plt.colorbar(sc, ax=ax1, fraction=0.025, pad=0.04, label='div(v)')

    # --- Panel 2: Divergence magnitude over simulation time ---
    ax2 = fig.add_subplot(222)
    mean_abs = [np.mean(np.abs(list(d.values()))) for d in div_history]
    max_abs  = [np.max(np.abs(list(d.values()))) for d in div_history]
    n_sinks_t  = [sum(1 for dv in d.values() if dv < -1) for d in div_history]
    steps = range(1, len(div_history) + 1)
    ax2.plot(steps, mean_abs, color='steelblue', label='Mean |div|')
    ax2.plot(steps, max_abs, color='tomato', linestyle='--', label='Max |div|')
    ax2.fill_between(steps, mean_abs, alpha=0.15, color='steelblue')
    ax2_r = ax2.twinx()
    ax2_r.plot(steps, n_sinks_t, color='darkorange', alpha=0.5,
               linewidth=0.8, label='Sink count (|div|>1)')
    ax2_r.set_ylabel('Sink node count', color='darkorange')
    ax2_r.tick_params(axis='y', labelcolor='darkorange')
    ax2.set_title('Divergence Magnitude Over Time\n'
                  '(grows as flows build, settles at steady state)')
    ax2.set_xlabel('Simulation step')
    ax2.set_ylabel('|div|')
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_r.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=8)
    ax2.set_ylim(bottom=0)

    # --- Panel 3: Per-pair latency comparison (4 strategies) ---
    ax3 = fig.add_subplot(223)
    if results:
        # Sort pairs by hop-SP latency for a clean display
        results_sorted = sorted(results, key=lambda r: r['sp_lat'])
        x = range(len(results_sorted))
        strategies = [
            ('sp_lat',    'Hop-SP',      'gray',    'o'),
            ('curl_lat',  'Curl-aware',  'royalblue','s'),
            ('green_lat', 'Green-aware', 'forestgreen','^'),
            ('comb_lat',  'Combined',    'crimson',  'D'),
        ]
        for key, label, color, marker in strategies:
            lats = [r[key] for r in results_sorted]
            ax3.plot(x, lats, marker=marker, color=color, ms=3,
                     alpha=0.75, linewidth=0.8, label=label)

        # Compute aggregate improvements (ratio-of-means)
        sp_mean    = np.mean([r['sp_lat']    for r in results])
        curl_mean  = np.mean([r['curl_lat']  for r in results])
        green_mean = np.mean([r['green_lat'] for r in results])
        comb_mean  = np.mean([r['comb_lat']  for r in results])
        curl_imp  = (sp_mean - curl_mean)  / sp_mean * 100
        green_imp = (sp_mean - green_mean) / sp_mean * 100
        comb_imp  = (sp_mean - comb_mean)  / sp_mean * 100
        ax3.set_title(f'Congested Latency: 4-Strategy Comparison\n'
                      f'Curl: {curl_imp:+.1f}%  '
                      f'Green: {green_imp:+.1f}%  '
                      f'Combined: {comb_imp:+.1f}%  vs Hop-SP')
        ax3.set_xlabel('Test pair (sorted by Hop-SP latency)')
        ax3.set_ylabel('Effective (M/M/1) path latency')
        ax3.legend(fontsize=8)

    # --- Panel 4: Divergence exposure vs path latency scatter ---
    ax4 = fig.add_subplot(224)
    if results:
        scatter_cfg = [
            ('sp_div',    'sp_lat',    'gray',       'o', 'Hop-SP',     0.45),
            ('green_div', 'green_lat', 'forestgreen','^', 'Green-aware',0.55),
            ('comb_div',  'comb_lat',  'crimson',    'D', 'Combined',   0.65),
        ]
        for xk, yk, color, marker, label, alpha in scatter_cfg:
            ax4.scatter(
                [r[xk] for r in results],
                [r[yk] for r in results],
                color=color, marker=marker, alpha=alpha, s=25, label=label, zorder=3,
            )
        ax4.set_title('Divergence Exposure vs Path Latency\n'
                      '(green/combined paths shift left = fewer sink/source hops)')
        ax4.set_xlabel('Mean |div| of nodes on path')
        ax4.set_ylabel('Effective path latency')
        ax4.legend(fontsize=8)

    plt.suptitle("Geodesic Blockchain — Green's Theorem Flow Balancing",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('green_flow_results.png', dpi=150)
    plt.close()
    print('\nPlot saved to green_flow_results.png')


# ---------------------------------------------------------------------------
# Summary printout
# ---------------------------------------------------------------------------

def print_summary(results: List[Dict], avg_divs: Dict[int, float], n_avg: int):
    print(f'\n{"="*65}')
    print(f'Green Flow Results — {len(results)} pairs, div averaged over last {n_avg} steps')
    print(f'{"="*65}')

    sp_mean    = np.mean([r['sp_lat']    for r in results])
    curl_mean  = np.mean([r['curl_lat']  for r in results])
    green_mean = np.mean([r['green_lat'] for r in results])
    comb_mean  = np.mean([r['comb_lat']  for r in results])

    curl_imp  = (sp_mean - curl_mean)  / sp_mean * 100
    green_imp = (sp_mean - green_mean) / sp_mean * 100
    comb_imp  = (sp_mean - comb_mean)  / sp_mean * 100

    print(f'{"Strategy":<20} {"Avg latency":>14} {"vs Hop-SP":>12}')
    print(f'{"-"*47}')
    print(f'{"Hop-SP":<20} {sp_mean:>14.4f} {"(baseline)":>12}')
    print(f'{"Curl-aware SP":<20} {curl_mean:>14.4f} {curl_imp:>+11.1f}%')
    print(f'{"Green-aware SP":<20} {green_mean:>14.4f} {green_imp:>+11.1f}%')
    print(f'{"Combined SP":<20} {comb_mean:>14.4f} {comb_imp:>+11.1f}%')
    print(f'{"="*65}')

    # Node divergence stats
    div_vals = list(avg_divs.values())
    n_sinks   = sum(1 for d in div_vals if d < -1)
    n_sources = sum(1 for d in div_vals if d >  1)
    n_balanced= len(div_vals) - n_sinks - n_sources
    print(f'\nNode divergence stats (avg last {n_avg} steps):')
    print(f'  Total nodes  : {len(div_vals)}')
    print(f'  Sinks   (div < −1): {n_sinks}')
    print(f'  Sources (div >  1): {n_sources}')
    print(f'  Balanced            : {n_balanced}')
    print(f'  Mean |div| : {np.mean(np.abs(div_vals)):.3f}')
    print(f'  Max  |div| : {np.max(np.abs(div_vals)):.3f}')
    print(f'{"="*65}')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # 1. Build geodesic mesh (same parameters as all prior modules)
    print("Building mesh...")
    verts, faces = build_icosahedron()
    verts, faces = subdivide(verts, faces)
    verts, faces = subdivide(verts, faces)
    G     = mesh_to_graph(verts, faces)
    G     = assign_link_params(G)
    verts = np.array(verts)
    print(f"  {len(verts)} nodes | {len(faces)} faces | {G.number_of_edges()} edges")

    # 2. Build edge→face lookup (used for curl weights)
    edge_to_faces = build_edge_to_faces(faces)

    # 3. Run extended simulation (100 steps, arrival_rate=5, same as stokes_curl)
    sim = GreenFlowSim(G, faces, arrival_rate=5.0,
                       flow_duration=10.0, capacity_scale=0.5)
    curl_history, div_history = sim.run(n_steps=100, verbose=True)

    # 4. Time-average last 20 steps for stable signals
    n_avg = min(20, len(curl_history))

    avg_curl: Dict[int, float] = {
        i: float(np.mean([curl_history[-j][i] for j in range(1, n_avg + 1)]))
        for i in range(len(faces))
    }
    avg_divs: Dict[int, float] = {
        v: float(np.mean([div_history[-j][v] for j in range(1, n_avg + 1)]))
        for v in G.nodes()
    }

    abs_div = [abs(d) for d in avg_divs.values()]
    print(f"\nSteady-state divergence (avg last {n_avg} steps):")
    print(f"  Mean |div| : {np.mean(abs_div):.3f}")
    print(f"  Max  |div| : {np.max(abs_div):.3f}")
    print(f"  Sinks  (div < −1): {sum(1 for d in avg_divs.values() if d < -1)}")
    print(f"  Sources (div >  1): {sum(1 for d in avg_divs.values() if d >  1)}")

    # 5. Verify Green's Theorem consistency on random subsets
    verify_green_theorem(G, sim, avg_divs, n_samples=8)

    # 6. Run 4-way routing comparison
    print(f"\nRunning 4-way routing comparison (60 pairs)...")
    # Calibrate div_threshold to actual divergence range.
    # With mean|div|≈0.38 and max|div|≈1.75 we use the 70th-percentile
    # of |div| so the penalty activates on the most imbalanced ~30% of nodes.
    abs_div_vals = sorted(abs(d) for d in avg_divs.values())
    div_threshold_calibrated = float(np.percentile(abs_div_vals, 70))
    print(f"\nCalibrated div_threshold (70th percentile |div|): {div_threshold_calibrated:.3f}")

    results = run_comparison_4way(
        G, sim, avg_curl, avg_divs, edge_to_faces,
        n_pairs=60,
        curl_threshold=1.0, curl_penalty=8.0,
        sink_penalty=5.0, source_penalty=3.0,
        div_threshold=div_threshold_calibrated,
    )

    # 7. Print summary and visualise
    print_summary(results, avg_divs, n_avg)
    visualize(verts, G, avg_divs, div_history, results)
