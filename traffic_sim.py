"""
traffic_sim.py — Dynamic traffic simulation layer for geodesic blockchain routing.

Three routing strategies compared under live congestion:

  1. Hop-SP        — hop-minimizing Dijkstra (classic baseline)
  2. Latency-SP    — Dijkstra weighted by *current* effective latency (oracle)
  3. DQN-base      — geometry-trained DQN from dqn_router.py (no congestion features)
  4. DQN-cong      — retrained DQN with link utilization as 11th input feature

Congestion model: M/M/1 queuing,  effective_latency = base / (1 - rho)
  rho = 0   →  base latency
  rho = 0.5 →  2x base
  rho = 0.9 →  10x base   (capped at rho=0.95 → 20x)

capacity_scale=0.1 calibrated so arrival_rate sweep 0.5→8 drives avg util 0.1→0.7.
"""

import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, defaultdict
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

from geodesic_overlay import build_icosahedron, subdivide, mesh_to_graph
from dqn_router import DQNRouter, assign_link_params

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# ---------------------------------------------------------------------------
# M/M/1 queuing congestion model
# ---------------------------------------------------------------------------

_RHO_CAP = 0.95  # cap: 20x multiplier at saturation

def mm1_latency(base: float, load: float, capacity: float) -> float:
    """M/M/1 mean service time: base / (1 - rho), capped at rho=0.95."""
    rho = min(load / max(capacity, 1e-6), _RHO_CAP)
    return base / (1.0 - rho)


# ---------------------------------------------------------------------------
# Flow dataclass
# ---------------------------------------------------------------------------

@dataclass
class Flow:
    source:    int
    target:    int
    path:      List[int]
    remaining: float
    router:    str = "background"


# ---------------------------------------------------------------------------
# Part B — Congestion-aware DQN (11-dim features)
# ---------------------------------------------------------------------------

class QNetwork11(nn.Module):
    """Q-network with 11 inputs: geometry (9) + effective latency (1) + rho (1)."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(11, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x)


def make_features_cong(verts, u, target, v, eff_latency, rho):
    """11-dim feature: pos_u (3) + pos_target (3) + pos_v (3) + latency (1) + rho (1)."""
    return torch.FloatTensor(
        np.concatenate([verts[u], verts[target], verts[v], [eff_latency], [rho]])
    )


class DQNRouterCongestion:
    """DQN router trained with live congestion state (rho) as an 11th feature.

    Trains alongside a background TrafficSimulator so it sees realistic load
    distributions during learning. At routing time, queries the eval sim's
    current link loads to make congestion-aware decisions.
    """

    def __init__(self, G, verts, training_sim, alpha=0.001, gamma=0.95, epsilon=0.4):
        self.G            = G
        self.verts        = verts
        self.training_sim = training_sim
        self.gamma        = gamma
        self.epsilon      = epsilon

        self.q_net      = QNetwork11()
        self.target_net = QNetwork11()
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer  = optim.Adam(self.q_net.parameters(), lr=alpha)
        self.buffer     = deque(maxlen=15000)
        self.loss_fn    = nn.MSELoss()

        self.episode_rewards = []
        self.success_rate    = []

    def _rho(self, sim, u, v):
        return sim._load(u, v) / max(sim._cap(u, v), 1e-6)

    def get_q(self, u, target, v, sim):
        lat  = sim.get_effective_latency(u, v)
        rho  = self._rho(sim, u, v)
        feat = make_features_cong(self.verts, u, target, v, lat, rho)
        with torch.no_grad():
            return self.q_net(feat.unsqueeze(0)).item()

    def select_next(self, u, target, visited, sim):
        neighbors = [n for n in self.G.neighbors(u) if n not in visited]
        if not neighbors:
            return None
        if random.random() < self.epsilon:
            return random.choice(neighbors)
        return max(neighbors, key=lambda v: self.get_q(u, target, v, sim))

    def _compute_reward(self, u, v, target, latency, distances):
        reward   = -latency
        progress = distances[u].get(target, 999) - distances[v].get(target, 999)
        reward  += 20.0 if v == target else progress * 2.0
        return reward

    def _train_step(self, batch_size=64):
        if len(self.buffer) < batch_size:
            return
        batch = random.sample(self.buffer, batch_size)
        for feat, reward, next_feats, done in batch:
            current_q = self.q_net(feat.unsqueeze(0))
            if done or not next_feats:
                target_q = torch.FloatTensor([[reward]])
            else:
                with torch.no_grad():
                    best = max(self.target_net(f.unsqueeze(0)).item()
                               for f in next_feats)
                    target_q = torch.FloatTensor([[reward + self.gamma * best]])
            loss = self.loss_fn(current_q, target_q)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def train(self, distances, episodes=800, steps_per_episode=25,
              target_update_freq=20):
        nodes = list(self.G.nodes())
        sim   = self.training_sim
        print(f"Training congestion-aware DQN: {episodes} episodes with live traffic...")

        for ep in range(episodes):
            # Advance background traffic so training sees realistic link loads
            sim.step(dqn_router=None, cong_router=None, n_test_pairs=0)

            source = random.choice(nodes)
            target = random.choice(nodes)
            while target == source:
                target = random.choice(nodes)

            u       = source
            visited = {u}
            total_reward = 0
            reached = False

            for _ in range(steps_per_episode):
                if u == target:
                    reached = True
                    break
                v = self.select_next(u, target, visited, sim)
                if v is None:
                    break

                lat    = sim.get_effective_latency(u, v)
                rho    = self._rho(sim, u, v)
                reward = self._compute_reward(u, v, target, lat, distances)
                done   = (v == target)

                feat       = make_features_cong(self.verts, u, target, v, lat, rho)
                next_nbrs  = [n for n in self.G.neighbors(v) if n not in visited]
                next_feats = [
                    make_features_cong(
                        self.verts, v, target, w,
                        sim.get_effective_latency(v, w),
                        self._rho(sim, v, w),
                    )
                    for w in next_nbrs
                ]

                self.buffer.append((feat, reward, next_feats, done))
                self._train_step(batch_size=64)

                total_reward += reward
                visited.add(v)
                u = v

            self.episode_rewards.append(total_reward)
            self.success_rate.append(1 if reached else 0)

            if (ep + 1) % target_update_freq == 0:
                self.target_net.load_state_dict(self.q_net.state_dict())
            self.epsilon = max(0.05, self.epsilon * 0.997)

            if (ep + 1) % 100 == 0:
                sr  = np.mean(self.success_rate[-100:]) * 100
                avg = np.mean(self.episode_rewards[-100:])
                print(f"  Episode {ep+1:4d}: reward = {avg:6.2f}, "
                      f"success = {sr:.0f}%, epsilon = {self.epsilon:.3f}")

    def route(self, source, target, sim, max_steps=30):
        path    = [source]
        visited = {source}
        u       = source
        for _ in range(max_steps):
            if u == target:
                break
            neighbors = [n for n in self.G.neighbors(u) if n not in visited]
            if not neighbors:
                break
            v = max(neighbors, key=lambda n: self.get_q(u, target, n, sim))
            path.append(v)
            visited.add(v)
            u = v
        return path, u == target


# ---------------------------------------------------------------------------
# TrafficSimulator
# ---------------------------------------------------------------------------

class TrafficSimulator:
    """Discrete-time traffic simulator on a geodesic mesh graph.

    Each time step:
      1. Age active flows, expire finished ones (freeing link load).
      2. Generate Poisson background flows routed via hop-SP.
      3. Sample test pairs and measure congested latency for all four strategies.
    """

    def __init__(
        self,
        G: nx.Graph,
        verts: np.ndarray,
        arrival_rate: float = 2.0,
        flow_duration: float = 8.0,
        capacity_scale: float = 0.1,
    ):
        self.G            = G
        self.verts        = verts
        self.arrival_rate = arrival_rate
        self.flow_duration = flow_duration

        self.link_load: Dict[Tuple[int, int], float] = defaultdict(float)
        self.link_capacity: Dict[Tuple[int, int], float] = {}
        self._init_capacities(capacity_scale)

        self.active_flows: List[Flow] = []
        self.time_step = 0
        self.history: List[Dict] = []

    def _init_capacities(self, scale: float):
        avg_degree = np.mean([d for _, d in self.G.degree()])
        for u, v in self.G.edges():
            base = self.G[u][v]['base_latency']
            cap  = scale * avg_degree / max(base, 0.1)
            self.link_capacity[(min(u, v), max(u, v))] = cap

    def _cap(self, u, v):
        return self.link_capacity[(min(u, v), max(u, v))]

    def _load(self, u, v):
        return self.link_load[(min(u, v), max(u, v))]

    def _add_load(self, u, v, delta):
        key = (min(u, v), max(u, v))
        self.link_load[key] = max(0.0, self.link_load[key] + delta)

    def get_effective_latency(self, u: int, v: int) -> float:
        """M/M/1-congested latency with small jitter."""
        base   = self.G[u][v]['base_latency']
        eff    = mm1_latency(base, self._load(u, v), self._cap(u, v))
        jitter = np.random.normal(0, 0.02)
        return max(0.01, eff + jitter)

    # --- Routing methods ---

    def _route_hop_sp(self, s, t):
        """Hop-minimizing shortest path (ignores latency)."""
        try:
            return nx.shortest_path(self.G, s, t, weight=None)
        except nx.NetworkXNoPath:
            return None

    def _route_latency_sp(self, s, t):
        """Dijkstra weighted by *current* effective latency — online oracle."""
        try:
            return nx.shortest_path(
                self.G, s, t,
                weight=lambda u, v, _: self.get_effective_latency(u, v),
            )
        except nx.NetworkXNoPath:
            return None

    def _route_dqn_base(self, router: DQNRouter, s, t):
        if router is None:
            return None
        path, reached = router.route(s, t, max_steps=30)
        return path if reached else None

    def _route_dqn_cong(self, router: DQNRouterCongestion, s, t):
        if router is None:
            return None
        path, reached = router.route(s, t, self, max_steps=30)
        return path if reached else None

    # --- Flow management ---

    def _inject(self, path, duration, label):
        for i in range(len(path) - 1):
            self._add_load(path[i], path[i + 1], 1.0)
        self.active_flows.append(
            Flow(source=path[0], target=path[-1],
                 path=path, remaining=duration, router=label)
        )

    def _expire_flows(self, dt):
        alive = []
        for flow in self.active_flows:
            flow.remaining -= dt
            if flow.remaining <= 0:
                for i in range(len(flow.path) - 1):
                    self._add_load(flow.path[i], flow.path[i + 1], -1.0)
            else:
                alive.append(flow)
        self.active_flows = alive

    # --- Main simulation step ---

    def step(
        self,
        dqn_router: Optional[DQNRouter] = None,
        cong_router: Optional[DQNRouterCongestion] = None,
        n_test_pairs: int = 5,
        dt: float = 1.0,
    ) -> Dict:
        self.time_step += 1
        self._expire_flows(dt)

        # Background Poisson traffic (hop-SP routed)
        nodes = list(self.G.nodes())
        for _ in range(np.random.poisson(self.arrival_rate)):
            s, t = random.sample(nodes, 2)
            path = self._route_hop_sp(s, t)
            if path:
                self._inject(path, np.random.exponential(self.flow_duration),
                             "background")

        # Test pairs: compare all available strategies
        step_results = []
        any_router = dqn_router is not None or cong_router is not None
        if any_router and n_test_pairs > 0:
            for _ in range(n_test_pairs):
                s, t = random.sample(nodes, 2)
                sp = self._route_hop_sp(s, t)
                if sp is None or len(sp) - 1 > 10:
                    continue

                lsp = self._route_latency_sp(s, t)
                qp  = self._route_dqn_base(dqn_router, s, t)
                cqp = self._route_dqn_cong(cong_router, s, t)

                def path_lat(path):
                    if path is None:
                        return None
                    return sum(self.get_effective_latency(path[i], path[i+1])
                               for i in range(len(path)-1))

                sp_lat  = path_lat(sp)
                lsp_lat = path_lat(lsp)
                qp_lat  = path_lat(qp)
                cqp_lat = path_lat(cqp)

                if sp_lat is None:
                    continue

                step_results.append({
                    'sp_hops':  len(sp)-1,  'sp_lat':  sp_lat,
                    'lsp_hops': len(lsp)-1 if lsp else None, 'lsp_lat': lsp_lat,
                    'qp_hops':  len(qp)-1  if qp  else None, 'qp_lat':  qp_lat,
                    'cqp_hops': len(cqp)-1 if cqp else None, 'cqp_lat': cqp_lat,
                })

        loads = [self._load(u, v) for u, v in self.G.edges()]
        caps  = [self._cap(u, v)  for u, v in self.G.edges()]
        utils = [l / c for l, c in zip(loads, caps)]
        metrics = {
            'step':         self.time_step,
            'active_flows': len(self.active_flows),
            'avg_util':     float(np.mean(utils)),
            'max_util':     float(np.max(utils)),
            'pairs':        step_results,
        }
        self.history.append(metrics)
        return metrics

    def run(
        self,
        dqn_router=None,
        cong_router=None,
        n_steps: int = 50,
        n_test_pairs: int = 5,
        verbose: bool = True,
    ) -> List[Dict]:
        print(f"Running traffic simulation: {n_steps} steps, "
              f"arrival_rate={self.arrival_rate:.1f}")
        for i in range(n_steps):
            m = self.step(dqn_router, cong_router, n_test_pairs=n_test_pairs)
            if verbose and (i + 1) % 10 == 0:
                pairs = m['pairs']
                if pairs:
                    def imp(key):
                        vals = [(r['sp_lat'] - r[key]) / r['sp_lat'] * 100
                                for r in pairs if r[key] is not None]
                        return np.mean(vals) if vals else float('nan')
                    print(f"  Step {i+1:3d}: flows={m['active_flows']:3d}, "
                          f"util={m['avg_util']:.2f}/{m['max_util']:.2f}  "
                          f"lsp={imp('lsp_lat'):+.1f}%  "
                          f"dqn={imp('qp_lat'):+.1f}%  "
                          f"cong={imp('cqp_lat'):+.1f}%  vs hop-SP")
                else:
                    print(f"  Step {i+1:3d}: flows={m['active_flows']:3d}, "
                          f"avg_util={m['avg_util']:.2f}")
        return self.history


# ---------------------------------------------------------------------------
# Congestion sweep
# ---------------------------------------------------------------------------

def congestion_sweep(
    G, verts, dqn_router, cong_router,
    arrival_rates: List[float],
    n_steps: int = 40,
    n_test_pairs: int = 8,
    capacity_scale: float = 0.1,
) -> Dict[float, List[Dict]]:
    """Run at multiple arrival rates; returns all per-pair results per rate."""
    results = {}
    for rate in arrival_rates:
        print(f"\n--- arrival_rate={rate:.1f} ---")
        sim = TrafficSimulator(G, verts, arrival_rate=rate,
                               capacity_scale=capacity_scale)
        history = sim.run(dqn_router, cong_router, n_steps=n_steps,
                          n_test_pairs=n_test_pairs, verbose=False)
        pairs = [p for m in history for p in m['pairs']]
        results[rate] = pairs

        def avg_imp(key):
            vals = [(r['sp_lat'] - r[key]) / r['sp_lat'] * 100
                    for r in pairs if r[key] is not None]
            return f"{np.mean(vals):+.1f}%" if vals else "n/a"

        print(f"  {len(pairs)} pairs  |  "
              f"latency-SP: {avg_imp('lsp_lat')}  "
              f"DQN-base: {avg_imp('qp_lat')}  "
              f"DQN-cong: {avg_imp('cqp_lat')}")
    return results


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def visualize(history: List[Dict], sweep: Dict[float, List[Dict]]):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Colors / labels for the four strategies
    STYLES = {
        'hop-SP':   ('gray',    'o-', 'Hop-SP (baseline)'),
        'lsp':      ('green',   's-', 'Latency-SP (oracle)'),
        'dqn-base': ('navy',    '^-', 'DQN-base (geometry)'),
        'dqn-cong': ('crimson', 'D-', 'DQN-cong (congestion-aware)'),
    }

    # --- Top-left: utilisation over time ---
    ax = axes[0][0]
    steps = [m['step'] for m in history]
    ax.plot(steps, [m['avg_util'] for m in history],
            color='steelblue', label='Avg util')
    ax.plot(steps, [m['max_util'] for m in history],
            color='tomato', linestyle='--', label='Max util')
    ax.axhline(1.0, color='red', linestyle=':', alpha=0.5, label='Capacity')
    ax.set_title('Link Utilisation Over Time')
    ax.set_xlabel('Step'); ax.set_ylabel('Load / Capacity')
    ax.legend(); ax.set_ylim(bottom=0)

    # --- Top-right: active flows over time ---
    ax = axes[0][1]
    active = [m['active_flows'] for m in history]
    ax.plot(steps, active, color='navy')
    ax.fill_between(steps, active, alpha=0.15, color='navy')
    ax.set_title('Active Flows Over Time')
    ax.set_xlabel('Step'); ax.set_ylabel('Concurrent flows')

    # --- Bottom-left: per-pair latency at nominal load ---
    ax = axes[1][0]
    all_pairs = [p for m in history for p in m['pairs']]
    if all_pairs:
        x = range(len(all_pairs))
        ax.plot(x, [r['sp_lat']  for r in all_pairs],
                'o-', color='gray',    ms=3, label='Hop-SP')
        lsp_vals = [r['lsp_lat'] for r in all_pairs if r['lsp_lat'] is not None]
        if lsp_vals:
            ax.plot(range(len(lsp_vals)), lsp_vals,
                    's-', color='green',   ms=3, label='Latency-SP')
        qp_vals = [r['qp_lat'] for r in all_pairs if r['qp_lat'] is not None]
        if qp_vals:
            ax.plot(range(len(qp_vals)), qp_vals,
                    '^-', color='navy',    ms=3, label='DQN-base')
        cqp_vals = [r['cqp_lat'] for r in all_pairs if r['cqp_lat'] is not None]
        if cqp_vals:
            ax.plot(range(len(cqp_vals)), cqp_vals,
                    'D-', color='crimson', ms=3, label='DQN-cong')
        ax.set_title('Per-pair Latency Under Live Traffic')
        ax.set_xlabel('Test pair'); ax.set_ylabel('Congested path latency')
        ax.legend(fontsize=8)

    # --- Bottom-right: improvement vs congestion level (grouped bars) ---
    ax = axes[1][1]
    rates = sorted(sweep.keys())
    keys  = [('lsp_lat', 'Latency-SP', 'green'),
             ('qp_lat',  'DQN-base',   'navy'),
             ('cqp_lat', 'DQN-cong',   'crimson')]
    n_keys = len(keys)
    x      = np.arange(len(rates))
    width  = 0.25

    for i, (key, label, color) in enumerate(keys):
        imps = []
        for rate in rates:
            pairs = sweep[rate]
            vals  = [(r['sp_lat'] - r[key]) / r['sp_lat'] * 100
                     for r in pairs if r[key] is not None]
            imps.append(np.mean(vals) if vals else 0.0)
        bars = ax.bar(x + (i - 1) * width, imps, width,
                      label=label, color=color, alpha=0.8)
        for bar, v in zip(bars, imps):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5 * np.sign(v),
                    f'{v:+.0f}%', ha='center', va='bottom', fontsize=7)

    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([str(r) for r in rates])
    ax.set_title('Latency Improvement vs Hop-SP\nBy Congestion Level')
    ax.set_xlabel('Arrival rate (flows/step)')
    ax.set_ylabel('Improvement over hop-SP (%)')
    ax.legend(fontsize=8)

    plt.suptitle('Geodesic Blockchain Routing — 4-Strategy Congestion Study',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('traffic_sim_results.png', dpi=150)
    plt.show()
    print("\nPlot saved to traffic_sim_results.png")

    # Summary table
    print("\n=== Summary: avg improvement over hop-SP ===")
    print(f"{'Rate':>6}  {'util':>6}  {'Lat-SP':>8}  {'DQN-base':>9}  {'DQN-cong':>9}")
    print("-" * 50)
    for rate in rates:
        pairs = sweep[rate]
        def pct(key):
            vals = [(r['sp_lat'] - r[key]) / r['sp_lat'] * 100
                    for r in pairs if r[key] is not None]
            return f"{np.mean(vals):+.1f}%" if vals else "  n/a"
        print(f"{rate:>6.1f}  {'':>6}  {pct('lsp_lat'):>8}  "
              f"{pct('qp_lat'):>9}  {pct('cqp_lat'):>9}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # 1. Build mesh
    print("Building mesh...")
    verts, faces = build_icosahedron()
    verts, faces = subdivide(verts, faces)
    verts, faces = subdivide(verts, faces)
    G     = mesh_to_graph(verts, faces)
    G     = assign_link_params(G)
    verts = np.array(verts)

    print("Precomputing distances...")
    distances = dict(nx.all_pairs_shortest_path_length(G))

    # 2. Train geometry-only DQN (as before)
    print("\n=== Training DQN-base (geometry features, no congestion) ===")
    dqn_base = DQNRouter(G, verts, alpha=0.001, gamma=0.95, epsilon=0.4)
    dqn_base.train(distances, episodes=800, steps_per_episode=25)

    # 3. Train congestion-aware DQN with a background sim running alongside
    print("\n=== Training DQN-cong (11-dim features, live traffic) ===")
    training_sim = TrafficSimulator(G, verts, arrival_rate=3.0,
                                    flow_duration=8.0, capacity_scale=0.1)
    dqn_cong = DQNRouterCongestion(G, verts, training_sim,
                                   alpha=0.001, gamma=0.95, epsilon=0.4)
    dqn_cong.train(distances, episodes=800, steps_per_episode=25)

    # 4. Nominal-load evaluation (arrival_rate=2)
    print("\n=== Nominal traffic simulation (arrival_rate=2) ===")
    eval_sim = TrafficSimulator(G, verts, arrival_rate=2.0,
                                flow_duration=8.0, capacity_scale=0.1)
    history = eval_sim.run(dqn_base, dqn_cong, n_steps=50, n_test_pairs=6)

    # 5. Congestion sweep
    print("\n=== Congestion sweep ===")
    sweep = congestion_sweep(
        G, verts, dqn_base, dqn_cong,
        arrival_rates=[0.5, 1.0, 2.0, 4.0, 8.0],
        n_steps=40,
        n_test_pairs=8,
        capacity_scale=0.1,
    )

    # 6. Visualise
    visualize(history, sweep)
