"""
stokes_curl.py — Stokes's Theorem curl detection and mitigation.

Patent §C: "For each face S, compute ∮∂S F·dr.
            If curl exceeds threshold, reroute along adjacent lower-curl faces."

INTUITION (no maths background needed):
  Imagine traffic flowing along the edges of every triangle on the mesh.
  If more traffic circulates *around* a triangle than flows *through* it,
  that triangle is a congestion vortex — like a roundabout that's seized up.
  Stokes's Theorem gives us a single number per triangle that measures
  this "spin".  We call it curl.

  curl(face) = ∮ F·dr
             = net_flow(a→b) + net_flow(b→c) + net_flow(c→a)

  net_flow(u→v) = flows_going_u_to_v  −  flows_going_v_to_u

  High |curl| → vortex.  Mitigation → penalise those edges so routers
  pick paths that bypass the congested triangles.

WHY DIRECTED FLOWS MATTER:
  traffic_sim.py tracks *how many* flows use an edge (load).
  Here we track *which direction* each flow travels (directed load).
  The difference: two flows going opposite ways on the same edge cancel
  out (no vortex), whereas two flows in the same direction reinforce it.

OUTPUT:  stokes_curl_results.png  (4-panel plot)
METRICS: hop-SP vs curl-aware SP latency improvement at steady-state traffic.
"""

import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

from geodesic_overlay import build_icosahedron, subdivide, mesh_to_graph
from dqn_router import assign_link_params

random.seed(42)
np.random.seed(42)

# ---------------------------------------------------------------------------
# Directed-flow simulation
# ---------------------------------------------------------------------------

class DirectedFlowSim:
    """Traffic simulator that tracks asymmetric (directed) edge flows.

    Unlike traffic_sim.py (which tracks total load), we record separately
    how many active flows travel u→v vs v→u on every edge.  This lets us
    compute the Stokes circulation integral per face.

    Background traffic is routed hop-minimising (shortest path).
    """

    def __init__(
        self,
        G: nx.Graph,
        faces: list,
        arrival_rate: float = 4.0,   # Poisson mean new flows per step
        flow_duration: float = 10.0, # exponential mean lifetime (steps)
        capacity_scale: float = 0.5, # M/M/1 capacity = scale*avg_degree/base
    ):
        self.G             = G
        self.faces         = faces
        self.arrival_rate  = arrival_rate
        self.flow_duration = flow_duration

        # directed_load[(u,v)] = active flows going strictly u → v
        self.directed_load: Dict[Tuple[int,int], float] = defaultdict(float)

        # Link capacity for congested latency (M/M/1)
        self.capacity_scale = capacity_scale
        avg_deg = np.mean([d for _, d in G.degree()])
        self.link_cap: Dict[Tuple[int,int], float] = {
            (min(u,v), max(u,v)): capacity_scale * avg_deg / max(G[u][v]['base_latency'], 0.1)
            for u, v in G.edges()
        }

        self.active_flows: List[Tuple[list, float]] = []  # (path, remaining)
        self.time_step  = 0
        self.curl_history: List[Dict[int, float]] = []

    # --- flow management ---

    def _route_sp(self, s: int, t: int) -> Optional[list]:
        try:
            return nx.shortest_path(self.G, s, t, weight=None)
        except nx.NetworkXNoPath:
            return None

    def _inject(self, path: list, duration: float):
        for i in range(len(path) - 1):
            self.directed_load[(path[i], path[i+1])] += 1.0
        self.active_flows.append((path, duration))

    def _expire(self, dt: float):
        alive = []
        for path, rem in self.active_flows:
            rem -= dt
            if rem <= 0:
                for i in range(len(path) - 1):
                    key = (path[i], path[i+1])
                    self.directed_load[key] = max(0.0, self.directed_load[key] - 1.0)
            else:
                alive.append((path, rem))
        self.active_flows = alive

    # --- Stokes curl ---

    def compute_curl(self) -> Dict[int, float]:
        """Return curl for every face.

        curl(face [a,b,c]) = net_flow(a→b) + net_flow(b→c) + net_flow(c→a)
        net_flow(u→v)      = directed_load[(u,v)] − directed_load[(v,u)]

        Interpretation:
          curl ≈ 0  → balanced bidirectional traffic on this triangle
          curl >> 0 → traffic circulating counter-clockwise (vortex)
          curl << 0 → traffic circulating clockwise (vortex, opposite spin)
        """
        curl: Dict[int, float] = {}
        for idx, face in enumerate(self.faces):
            a, b, c = face
            def net(u: int, v: int) -> float:
                return self.directed_load[(u, v)] - self.directed_load[(v, u)]
            curl[idx] = net(a, b) + net(b, c) + net(c, a)
        return curl

    # --- effective (congested) latency using M/M/1 ---

    def effective_latency(self, u: int, v: int) -> float:
        """M/M/1 latency accounting for directed traffic load."""
        base = self.G[u][v]['base_latency']
        load = self.directed_load[(u, v)] + self.directed_load[(v, u)]
        cap  = self.link_cap[(min(u,v), max(u,v))]
        rho  = min(load / max(cap, 1e-6), 0.95)
        eff  = base / (1.0 - rho)
        return max(0.01, eff + np.random.normal(0, 0.02))

    def path_latency(self, path: list) -> float:
        if path is None or len(path) < 2:
            return float('inf')
        return sum(self.effective_latency(path[i], path[i+1])
                   for i in range(len(path)-1))

    # --- simulation loop ---

    def step(self, dt: float = 1.0) -> Dict[int, float]:
        self.time_step += 1
        self._expire(dt)

        nodes = list(self.G.nodes())
        for _ in range(np.random.poisson(self.arrival_rate)):
            s, t = random.sample(nodes, 2)
            path = self._route_sp(s, t)
            if path:
                self._inject(path, np.random.exponential(self.flow_duration))

        curl = self.compute_curl()
        self.curl_history.append(curl)
        return curl

    def run(self, n_steps: int = 100, verbose: bool = True) -> List[Dict[int, float]]:
        """Run for n_steps, return curl history."""
        print(f"Running directed-flow simulation: {n_steps} steps, "
              f"arrival_rate={self.arrival_rate:.1f}, "
              f"flow_duration={self.flow_duration:.1f}")
        for i in range(n_steps):
            curl = self.step()
            if verbose and (i + 1) % 20 == 0:
                vals = list(curl.values())
                print(f"  Step {i+1:3d}: "
                      f"flows={len(self.active_flows):3d}, "
                      f"mean|curl|={np.mean(np.abs(vals)):.2f}, "
                      f"max|curl|={np.max(np.abs(vals)):.2f}, "
                      f"vortex_faces(|c|>2)={sum(1 for v in vals if abs(v) > 2)}")
        return self.curl_history


# ---------------------------------------------------------------------------
# Curl-aware routing
# ---------------------------------------------------------------------------

def build_edge_to_faces(faces: list) -> Dict[Tuple[int,int], List[int]]:
    """Map each undirected edge → list of face indices that contain it.

    Each edge in the triangular mesh belongs to exactly 2 faces
    (except boundary edges, which belong to 1).
    """
    e2f: Dict[Tuple[int,int], List[int]] = defaultdict(list)
    for idx, face in enumerate(faces):
        a, b, c = face
        for u, v in [(a, b), (b, c), (c, a)]:
            e2f[(min(u, v), max(u, v))].append(idx)
    return e2f


def build_curl_weights(
    G: nx.Graph,
    curl: Dict[int, float],
    edge_to_faces: Dict[Tuple[int,int], List[int]],
    threshold: float = 1.0,
    penalty: float = 8.0,
) -> Dict[Tuple[int,int], float]:
    """Compute curl-penalised routing weights for every edge.

    weight(u,v) = base_latency
                + penalty × max(0, max_|curl|_of_adjacent_faces − threshold)

    Edges on high-curl triangles become expensive, forcing routers
    to detour around congestion vortices.
    """
    weights: Dict[Tuple[int,int], float] = {}
    for u, v in G.edges():
        base = G[u][v]['base_latency']
        key  = (min(u, v), max(u, v))
        adj_curls = [abs(curl[fi]) for fi in edge_to_faces.get(key, [])]
        max_curl   = max(adj_curls) if adj_curls else 0.0
        pen        = penalty * max(0.0, max_curl - threshold)
        weights[(u, v)] = base + pen
        weights[(v, u)] = base + pen
    return weights


def route_hop_sp(G: nx.Graph, s: int, t: int) -> Optional[list]:
    """Standard hop-minimising shortest path."""
    try:
        return nx.shortest_path(G, s, t, weight=None)
    except nx.NetworkXNoPath:
        return None


def route_curl_aware(
    G: nx.Graph,
    s: int,
    t: int,
    curl_weights: Dict[Tuple[int,int], float],
) -> Optional[list]:
    """Dijkstra using curl-penalised edge weights."""
    try:
        return nx.shortest_path(
            G, s, t,
            weight=lambda u, v, _: curl_weights.get((u, v), G[u][v]['base_latency']),
        )
    except nx.NetworkXNoPath:
        return None


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def run_comparison(
    G: nx.Graph,
    sim: DirectedFlowSim,
    curl: Dict[int, float],
    edge_to_faces: Dict[Tuple[int,int], List[int]],
    n_pairs: int = 60,
    threshold: float = 1.0,
    penalty: float = 8.0,
) -> List[Dict]:
    """Compare hop-SP vs curl-aware SP on n_pairs random source-dest pairs.

    Latency is measured using M/M/1 effective latency (congestion-aware),
    so avoiding high-curl (high-load) edges concretely reduces measured latency.
    """
    curl_weights = build_curl_weights(G, curl, edge_to_faces, threshold, penalty)
    nodes        = list(G.nodes())
    results      = []

    def path_max_curl(path: list) -> float:
        """Largest |curl| of any face adjacent to edges on this path."""
        mc = 0.0
        for i in range(len(path) - 1):
            key = (min(path[i], path[i+1]), max(path[i], path[i+1]))
            for fi in edge_to_faces.get(key, []):
                mc = max(mc, abs(curl[fi]))
        return mc

    attempts = 0
    while len(results) < n_pairs and attempts < n_pairs * 4:
        attempts += 1
        s, t = random.sample(nodes, 2)

        sp  = route_hop_sp(G, s, t)
        cap = route_curl_aware(G, s, t, curl_weights)
        if sp is None or cap is None or len(sp) < 3:
            continue

        results.append({
            'sp_hops':  len(sp)  - 1,
            'cap_hops': len(cap) - 1,
            'sp_lat':   sim.path_latency(sp),
            'cap_lat':  sim.path_latency(cap),
            'sp_curl':  path_max_curl(sp),
            'cap_curl': path_max_curl(cap),
        })

    print(f"  Compared {len(results)} pairs ({attempts} attempts)")
    return results


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def visualize(
    verts: np.ndarray,
    faces: list,
    curl: Dict[int, float],
    results: List[Dict],
    curl_history: List[Dict[int, float]],
):
    fig = plt.figure(figsize=(16, 12))

    curl_vals = np.array([abs(curl[i]) for i in range(len(faces))])
    max_c = float(curl_vals.max()) if curl_vals.max() > 0 else 1.0

    # --- Top-left: 3D mesh coloured by |curl| ---
    ax1 = fig.add_subplot(221, projection='3d')
    cmap        = cm.get_cmap('RdYlGn_r')   # red = high curl, green = low
    face_colors = [cmap(min(curl_vals[i] / max_c, 1.0)) for i in range(len(faces))]

    poly = Poly3DCollection(
        [[verts[f[0]], verts[f[1]], verts[f[2]]] for f in faces],
        alpha=0.75,
        facecolors=face_colors,
        edgecolors='none',
    )
    ax1.add_collection3d(poly)
    ax1.scatter(*verts.T, color='black', s=2, alpha=0.4)
    ax1.set_title(f'Stokes Curl per Face  (max|curl|={max_c:.1f})\n'
                  f'Red = congestion vortex,  Green = balanced flow')
    ax1.axis('off')
    ax1.set_box_aspect([1, 1, 1])

    sm = cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, max_c))
    sm.set_array([])
    plt.colorbar(sm, ax=ax1, fraction=0.025, pad=0.04, label='|curl|')

    # --- Top-right: curl magnitude over simulation time ---
    ax2 = fig.add_subplot(222)
    mean_abs = [np.mean(np.abs(list(c.values()))) for c in curl_history]
    max_abs  = [np.max(np.abs(list(c.values()))) for c in curl_history]
    steps    = range(1, len(curl_history) + 1)
    ax2.plot(steps, mean_abs, color='steelblue', label='Mean |curl|')
    ax2.plot(steps, max_abs,  color='tomato', linestyle='--', label='Max |curl|')
    ax2.fill_between(steps, mean_abs, alpha=0.15, color='steelblue')
    ax2.set_title('Curl Magnitude Over Time\n'
                  '(grows as traffic builds, plateaus at steady state)')
    ax2.set_xlabel('Simulation step')
    ax2.set_ylabel('|curl|')
    ax2.legend()
    ax2.set_ylim(bottom=0)

    # --- Bottom-left: per-pair latency comparison ---
    ax3 = fig.add_subplot(223)
    if results:
        x      = range(len(results))
        sp_lat = [r['sp_lat']  for r in results]
        ca_lat = [r['cap_lat'] for r in results]
        ax3.plot(x, sp_lat, 'o-', color='gray',    ms=3, alpha=0.8, label='Hop-SP')
        ax3.plot(x, ca_lat, 's-', color='crimson', ms=3, alpha=0.8, label='Curl-aware SP')
        avg_imp = np.mean([(s - c) / s * 100 for s, c in zip(sp_lat, ca_lat)])
        ax3.set_title(f'Congested Latency: Hop-SP vs Curl-Aware SP\n'
                      f'Avg improvement: {avg_imp:+.1f}%')
        ax3.set_xlabel('Test pair')
        ax3.set_ylabel('Effective (M/M/1) path latency')
        ax3.legend()
    else:
        ax3.text(0.5, 0.5, 'No data', ha='center', va='center',
                 transform=ax3.transAxes)

    # --- Bottom-right: curl exposure vs latency scatter ---
    ax4 = fig.add_subplot(224)
    if results:
        ax4.scatter([r['sp_curl']  for r in results],
                    [r['sp_lat']   for r in results],
                    color='gray',    alpha=0.55, s=30, label='Hop-SP', zorder=2)
        ax4.scatter([r['cap_curl'] for r in results],
                    [r['cap_lat']  for r in results],
                    color='crimson', alpha=0.55, s=30, label='Curl-aware SP', zorder=3)
        ax4.set_title('Curl Exposure vs Path Latency\n'
                      '(curl-aware paths shift left = avoid vortices)')
        ax4.set_xlabel('Max |curl| on path')
        ax4.set_ylabel('Effective path latency')
        ax4.legend()

    plt.suptitle("Geodesic Blockchain — Stokes's Theorem Curl Mitigation",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('stokes_curl_results.png', dpi=150)
    plt.show()
    print('\nPlot saved to stokes_curl_results.png')

    # Print summary table
    if results:
        print(f'\n{"="*55}')
        print(f'Results across {len(results)} pairs')
        print(f'{"="*55}')
        sp_hops  = np.mean([r['sp_hops']  for r in results])
        ca_hops  = np.mean([r['cap_hops'] for r in results])
        sp_lat   = np.mean([r['sp_lat']   for r in results])
        ca_lat   = np.mean([r['cap_lat']  for r in results])
        sp_curl  = np.mean([r['sp_curl']  for r in results])
        ca_curl  = np.mean([r['cap_curl'] for r in results])
        avg_imp  = np.mean([(r['sp_lat'] - r['cap_lat']) / r['sp_lat'] * 100
                            for r in results])
        print(f'{"Metric":<25} {"Hop-SP":>10} {"Curl-aware":>10}')
        print(f'{"-"*47}')
        print(f'{"Avg hops":<25} {sp_hops:>10.1f} {ca_hops:>10.1f}')
        print(f'{"Avg latency":<25} {sp_lat:>10.3f} {ca_lat:>10.3f}')
        print(f'{"Avg max curl on path":<25} {sp_curl:>10.2f} {ca_curl:>10.2f}')
        print(f'{"Latency improvement":<25} {"":>10} {avg_imp:>+9.1f}%')
        print(f'{"="*55}')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # 1. Build geodesic mesh (2 subdivisions = 162 nodes, 320 faces)
    print("Building mesh...")
    verts, faces = build_icosahedron()
    verts, faces = subdivide(verts, faces)
    verts, faces = subdivide(verts, faces)
    G     = mesh_to_graph(verts, faces)
    G     = assign_link_params(G)
    verts = np.array(verts)
    print(f"  {len(verts)} nodes | {len(faces)} faces | {G.number_of_edges()} edges")

    # 2. Build edge→face lookup (one-time, used throughout)
    edge_to_faces = build_edge_to_faces(faces)

    # 3. Run directed-flow simulation to reach steady-state curl
    #    Higher arrival_rate and longer flow_duration → more pronounced vortices
    sim = DirectedFlowSim(G, faces, arrival_rate=5.0,
                          flow_duration=10.0, capacity_scale=0.5)
    curl_history = sim.run(n_steps=100, verbose=True)

    # 4. Time-average last 20 steps of curl for a stable signal
    n_avg    = min(20, len(curl_history))
    avg_curl = {i: float(np.mean([curl_history[-j][i] for j in range(1, n_avg+1)]))
                for i in range(len(faces))}

    abs_curls = [abs(v) for v in avg_curl.values()]
    print(f"\nSteady-state curl (avg last {n_avg} steps):")
    print(f"  Mean |curl| : {np.mean(abs_curls):.3f}")
    print(f"  Max  |curl| : {np.max(abs_curls):.3f}")
    print(f"  Vortex faces (|curl| > 1): {sum(1 for v in abs_curls if v > 1)}")
    print(f"  Vortex faces (|curl| > 2): {sum(1 for v in abs_curls if v > 2)}")

    # 5. Compare hop-SP vs curl-aware SP on 60 random pairs
    print(f"\nRunning routing comparison (60 pairs)...")
    results = run_comparison(G, sim, avg_curl, edge_to_faces,
                             n_pairs=60, threshold=1.0, penalty=8.0)

    # 6. Visualise
    visualize(verts, faces, avg_curl, results, curl_history)
