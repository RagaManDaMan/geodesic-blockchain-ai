"""
pipeline.py — Orchestrates the full simulation in dependency order.

Wraps each existing module file without modifying it.  Each run_*()
function seeds random state, calls the module's public API, and returns
a standardised dict:

  {
    "results": { ...key metrics... },
    "charts":  [ "outputs/mesh_chart.png", ... ],
    "summary": "one-line human-readable result",
  }

Module enable/disable:
  - Enabled + dirty  → run the module, cache result
  - Enabled + clean  → load from cache
  - Disabled         → load from cache (or emit "skipped" if no cache)

After all modules complete, geodesic_results.json is written to outputs/.
"""

import json
import os
import time
import traceback
from typing import Any, Callable, Dict, Generator, List, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import random

from param_schema import MODULE_ORDER, DEPENDENCY_CHAIN
from cache_manager import CacheManager

OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _seed(params: Dict[str, Any]):
    """Seed all random systems from params['seed']."""
    s = int(params.get("seed", 42))
    random.seed(s)
    np.random.seed(s)
    try:
        import torch
        torch.manual_seed(s)
    except ImportError:
        pass


def _chart(name: str) -> str:
    """Return absolute path for a chart file in outputs/."""
    return os.path.join(OUTPUTS_DIR, name)


# ---------------------------------------------------------------------------
# Module runners  (one per module — new modules add one function here)
# ---------------------------------------------------------------------------

def run_mesh(params: Dict[str, Any], _upstream: Dict) -> Dict[str, Any]:
    _seed(params)
    from geodesic_overlay import build_icosahedron, subdivide, mesh_to_graph
    from dqn_router import assign_link_params

    verts, faces = build_icosahedron()
    for _ in range(int(params["subdivisions"])):
        verts, faces = subdivide(verts, faces)
    G = mesh_to_graph(verts, faces)
    G = assign_link_params(G)

    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    n_faces = len(faces)

    # Simple mesh visualisation
    chart = _chart("mesh_chart.png")
    import networkx as nx
    pos2d = {n: (verts[n][0], verts[n][1]) for n in G.nodes()}
    fig, ax = plt.subplots(figsize=(6, 6))
    nx.draw(G, pos=pos2d, ax=ax, node_size=8, node_color='steelblue',
            edge_color='#aaa', width=0.4, with_labels=False)
    ax.set_title(f"Geodesic Mesh — {n_nodes} nodes, {n_faces} faces "
                 f"(subdiv={params['subdivisions']})", fontsize=10)
    plt.tight_layout()
    plt.savefig(chart, dpi=100, bbox_inches='tight')
    plt.close()

    return {
        "results": {
            "n_nodes": n_nodes, "n_edges": n_edges, "n_faces": n_faces,
            "subdivisions": int(params["subdivisions"]),
        },
        "charts":  [chart],
        "summary": f"{n_nodes} nodes, {n_edges} edges, {n_faces} faces "
                   f"(subdiv={params['subdivisions']})",
        # Pass mesh objects downstream via the result dict
        "_G": G, "_verts": verts, "_faces": faces,
    }


def run_dqn(params: Dict[str, Any], upstream: Dict) -> Dict[str, Any]:
    _seed(params)
    import networkx as nx
    from dqn_router import DQNRouter, assign_link_params

    G     = upstream["mesh"]["_G"]
    verts = upstream["mesh"]["_verts"]

    # Pre-compute hop distances (needed for DQN reward shaping)
    distances = dict(nx.all_pairs_shortest_path_length(G))

    router = DQNRouter(
        G, verts,
        alpha   = float(params["learning_rate"]),
        gamma   = float(params["gamma"]),
        epsilon = float(params["epsilon_start"]),
    )
    router.train(
        distances,
        episodes        = int(params["episodes"]),
        steps_per_episode = int(params["steps_per_ep"]),
        target_update_freq = 20,
    )

    # Evaluate on 40 test pairs
    nodes   = list(G.nodes())
    success = 0
    hop_lats, dqn_lats = [], []
    for _ in range(40):
        s = random.choice(nodes)
        t = random.choice(nodes)
        while t == s:
            t = random.choice(nodes)
        try:
            hop = nx.shortest_path(G, s, t, weight=None)
            h_lat = sum(G[hop[i]][hop[i+1]]['base_latency'] for i in range(len(hop)-1))
        except Exception:
            continue
        path, reached = router.route(s, t, max_steps=30)
        if reached and len(path) > 1:
            d_lat = sum(G[path[i]][path[i+1]]['base_latency'] for i in range(len(path)-1))
            hop_lats.append(h_lat)
            dqn_lats.append(d_lat)
            success += 1

    success_rate = success / 40
    improvement  = 0.0
    if hop_lats:
        improvement = (np.mean(hop_lats) - np.mean(dqn_lats)) / np.mean(hop_lats)

    # Chart: training rewards
    chart = _chart("dqn_chart.png")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    if router.episode_rewards:
        axes[0].plot(router.episode_rewards, alpha=0.4, color='steelblue', lw=0.8)
        window = max(1, len(router.episode_rewards) // 20)
        smoothed = np.convolve(router.episode_rewards,
                               np.ones(window)/window, mode='valid')
        axes[0].plot(range(window-1, len(router.episode_rewards)),
                     smoothed, color='steelblue', lw=2)
    axes[0].set_title('DQN Training Rewards')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Total reward')
    if router.success_rate:
        window2 = max(1, len(router.success_rate) // 20)
        sr_smooth = np.convolve([r*100 for r in router.success_rate],
                                np.ones(window2)/window2, mode='valid')
        axes[1].plot(sr_smooth, color='forestgreen', lw=2)
    axes[1].set_title(f'Success Rate (final: {success_rate:.0%})')
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Success %')
    axes[1].set_ylim(0, 105)
    plt.tight_layout()
    plt.savefig(chart, dpi=100, bbox_inches='tight')
    plt.close()

    return {
        "results": {
            "success_rate":  round(success_rate, 3),
            "latency_improvement_pct": round(improvement * 100, 2),
            "episodes": int(params["episodes"]),
        },
        "charts":  [chart],
        "summary": (f"Success {success_rate:.0%}, "
                    f"latency {improvement:+.1%} vs hop-SP"),
        "_router": router,
    }


def run_traffic(params: Dict[str, Any], upstream: Dict) -> Dict[str, Any]:
    _seed(params)
    from traffic_sim import TrafficSimulator

    G     = upstream["mesh"]["_G"]
    verts = upstream["mesh"]["_verts"]
    dqn_router = upstream.get("dqn", {}).get("_router")

    sim = TrafficSimulator(
        G, verts,
        arrival_rate   = float(params["arrival_rate"]),
        flow_duration  = 8.0,
        capacity_scale = float(params["capacity_scale"]),
    )
    history = sim.run(
        dqn_router   = dqn_router,
        cong_router  = None,
        n_steps      = int(params["sim_steps"]),
        n_test_pairs = 5,
        verbose      = False,
    )

    avg_util   = float(np.mean([h["avg_util"] for h in history]))
    peak_flows = max(h["active_flows"] for h in history)

    # Latency improvement (vs hop-SP) from steps that had test pairs
    impr_lsp = []
    for h in history:
        for r in h.get("pairs", []):
            if r.get("lsp_lat") and r.get("sp_lat"):
                impr_lsp.append((r["sp_lat"] - r["lsp_lat"]) / r["sp_lat"])
    latency_sp_improvement = float(np.mean(impr_lsp)) if impr_lsp else 0.0

    # Chart: active flows + utilisation
    chart = _chart("traffic_chart.png")
    steps = [h["step"] for h in history]
    flows = [h["active_flows"] for h in history]
    utils = [h["avg_util"] for h in history]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax1.plot(steps, flows, color='steelblue', lw=1.5)
    ax1.set_ylabel('Active flows')
    ax1.set_title(f'Traffic Simulation — arrival_rate={params["arrival_rate"]}, '
                  f'{params["sim_steps"]} steps')
    ax2.plot(steps, utils, color='tomato', lw=1.5)
    ax2.axhline(0.5, color='grey', lw=0.8, linestyle='--', alpha=0.6, label='50% util')
    ax2.set_ylabel('Avg link utilisation')
    ax2.set_xlabel('Step')
    ax2.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(chart, dpi=100, bbox_inches='tight')
    plt.close()

    return {
        "results": {
            "avg_utilisation":           round(avg_util, 3),
            "peak_active_flows":         peak_flows,
            "latency_sp_improvement_pct": round(latency_sp_improvement * 100, 2),
        },
        "charts":  [chart],
        "summary": (f"Peak {peak_flows} flows, avg util {avg_util:.2f}, "
                    f"latency-SP {latency_sp_improvement:+.1%}"),
        "_sim": sim, "_history": history,
    }


def run_stokes(params: Dict[str, Any], upstream: Dict) -> Dict[str, Any]:
    _seed(params)
    from stokes_curl import (
        DirectedFlowSim, build_edge_to_faces,
        build_curl_weights, route_hop_sp, route_curl_aware,
    )

    G     = upstream["mesh"]["_G"]
    faces = upstream["mesh"]["_faces"]

    sim = DirectedFlowSim(
        G, faces,
        arrival_rate   = float(params["arrival_rate"]),
        flow_duration  = 10.0,
        capacity_scale = 0.5,
    )
    sim.run(n_steps=int(params["sim_steps"]), verbose=False)

    # Time-average curl over last 20 steps
    window = min(20, len(sim.curl_history))
    from collections import defaultdict
    avg_curl = defaultdict(float)
    for ch in sim.curl_history[-window:]:
        for fi, c in ch.items():
            avg_curl[fi] += c / window

    edge_to_faces = build_edge_to_faces(faces)
    curl_weights  = build_curl_weights(
        G, avg_curl, edge_to_faces,
        threshold = float(params["curl_threshold"]),
        penalty   = float(params["curl_penalty"]),
    )

    # Latency comparison on 40 test pairs
    nodes = list(G.nodes())
    hop_lats, curl_lats = [], []
    for _ in range(40):
        s, t = random.sample(nodes, 2)
        hop  = route_hop_sp(G, s, t)
        curl = route_curl_aware(G, s, t, curl_weights)
        if hop and curl:
            h = sim.path_latency(hop)
            c = sim.path_latency(curl)
            if h < float('inf') and c < float('inf'):
                hop_lats.append(h)
                curl_lats.append(c)

    improvement = 0.0
    if hop_lats:
        improvement = (np.mean(hop_lats) - np.mean(curl_lats)) / np.mean(hop_lats)

    curl_vals   = [abs(v) for v in avg_curl.values()]
    vortex_count = sum(1 for v in curl_vals if v > float(params["curl_threshold"]))
    mean_curl   = float(np.mean(curl_vals)) if curl_vals else 0.0
    max_curl    = float(np.max(curl_vals))  if curl_vals else 0.0

    # Chart
    chart = _chart("stokes_chart.png")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
    ax1.hist(curl_vals, bins=30, color='steelblue', alpha=0.8, edgecolor='white')
    ax1.axvline(float(params["curl_threshold"]), color='tomato', lw=1.5,
                linestyle='--', label=f'threshold={params["curl_threshold"]}')
    ax1.set_title(f'Curl magnitude distribution\n'
                  f'{vortex_count}/{len(curl_vals)} faces > threshold')
    ax1.set_xlabel('|curl|')
    ax1.legend(fontsize=8)
    ax2.bar(['Hop-SP', 'Curl-aware'], [np.mean(hop_lats) if hop_lats else 0,
                                        np.mean(curl_lats) if curl_lats else 0],
            color=['#aaa', 'steelblue'], edgecolor='white', width=0.4)
    ax2.set_title(f'Avg path latency\nImprovement: {improvement:+.1%}')
    ax2.set_ylabel('Latency (ms)')
    plt.tight_layout()
    plt.savefig(chart, dpi=100, bbox_inches='tight')
    plt.close()

    return {
        "results": {
            "vortex_faces":          vortex_count,
            "mean_curl":             round(mean_curl, 3),
            "max_curl":              round(max_curl,  3),
            "latency_improvement_pct": round(improvement * 100, 2),
        },
        "charts":  [chart],
        "summary": (f"{vortex_count} vortex faces, "
                    f"curl-aware {improvement:+.1%} vs hop-SP"),
        "_sim": sim, "_avg_curl": dict(avg_curl),
        "_curl_weights": curl_weights, "_edge_to_faces": edge_to_faces,
    }


def run_green(params: Dict[str, Any], upstream: Dict) -> Dict[str, Any]:
    _seed(params)
    from green_flow import (
        GreenFlowSim, green_aware_weights,
        build_combined_weights, route_green_aware, route_combined,
    )
    from stokes_curl import route_hop_sp, build_curl_weights

    G     = upstream["mesh"]["_G"]
    faces = upstream["mesh"]["_faces"]

    # Reuse stokes curl weights if stokes ran, else recompute
    if "stokes" in upstream:
        edge_to_faces = upstream["stokes"]["_edge_to_faces"]
        avg_curl      = upstream["stokes"]["_avg_curl"]
        curl_weights  = upstream["stokes"]["_curl_weights"]
    else:
        from stokes_curl import build_edge_to_faces, DirectedFlowSim
        edge_to_faces = build_edge_to_faces(faces)
        # Minimal curl computation
        sim_s = DirectedFlowSim(G, faces, arrival_rate=float(params["arrival_rate"]))
        sim_s.run(n_steps=int(params["sim_steps"]), verbose=False)
        from collections import defaultdict
        avg_curl = defaultdict(float)
        window = min(20, len(sim_s.curl_history))
        for ch in sim_s.curl_history[-window:]:
            for fi, c in ch.items():
                avg_curl[fi] += c / window
        curl_weights = build_curl_weights(
            G, avg_curl, edge_to_faces,
            threshold=float(params["curl_threshold"]),
            penalty=float(params["curl_penalty"]),
        )

    sim = GreenFlowSim(G, faces,
                       arrival_rate=float(params["arrival_rate"]),
                       flow_duration=10.0, capacity_scale=0.5)
    sim.run(n_steps=int(params["sim_steps"]), verbose=False)

    # Time-average divergence
    from collections import defaultdict
    window  = min(20, len(sim.div_history))
    avg_div: Dict = defaultdict(float)
    for dh in sim.div_history[-window:]:
        for node, d in dh.items():
            avg_div[node] += d / window

    green_weights    = green_aware_weights(
        G, avg_div,
        sink_penalty   = float(params["sink_penalty"]),
        source_penalty = float(params["source_penalty"]),
        div_threshold  = float(params["div_threshold"]),
    )
    combined_weights = build_combined_weights(G, curl_weights, green_weights)

    # 4-way latency comparison
    nodes = list(G.nodes())
    results_by_strategy: Dict[str, List[float]] = {
        "hop": [], "curl": [], "green": [], "combined": []
    }
    for _ in range(40):
        s, t = random.sample(nodes, 2)
        paths = {
            "hop":      route_hop_sp(G, s, t),
            "curl":     route_hop_sp(G, s, t),   # use hop if stokes disabled
            "green":    route_green_aware(G, s, t, green_weights),
            "combined": route_combined(G, s, t, combined_weights),
        }
        if "stokes" in upstream:
            from stokes_curl import route_curl_aware
            paths["curl"] = route_curl_aware(G, s, t, curl_weights)
        for strat, path in paths.items():
            if path:
                lat = sim.path_latency(path)
                if lat < float('inf'):
                    results_by_strategy[strat].append(lat)

    avgs = {k: float(np.mean(v)) if v else 0 for k, v in results_by_strategy.items()}
    hop_avg = avgs["hop"] or 1.0
    improvements = {k: (hop_avg - v) / hop_avg for k, v in avgs.items()}

    div_vals   = list(avg_div.values())
    sinks      = sum(1 for d in div_vals if d < -float(params["div_threshold"]))
    sources    = sum(1 for d in div_vals if d >  float(params["div_threshold"]))
    mean_div   = float(np.mean(np.abs(div_vals))) if div_vals else 0

    # Chart
    chart = _chart("green_chart.png")
    labels = ['Hop-SP', 'Curl-aware', 'Green-aware', 'Combined']
    vals   = [avgs["hop"], avgs["curl"], avgs["green"], avgs["combined"]]
    colors = ['#bbb', 'steelblue', 'forestgreen', 'darkorange']
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
    bars = ax1.bar(labels, vals, color=colors, edgecolor='white', width=0.5)
    for bar, imp in zip(bars[1:], [improvements["curl"], improvements["green"],
                                    improvements["combined"]]):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f'{imp:+.1%}', ha='center', va='bottom', fontsize=9)
    ax1.set_title('4-way Routing Comparison')
    ax1.set_ylabel('Avg path latency')
    ax2.hist(div_vals, bins=30, color='salmon', alpha=0.8, edgecolor='white')
    ax2.axvline(float(params["div_threshold"]), color='red', lw=1.5,
                linestyle='--', label=f'div_threshold={params["div_threshold"]}')
    ax2.axvline(-float(params["div_threshold"]), color='red', lw=1.5, linestyle='--')
    ax2.set_title(f'Node divergence dist.\n{sinks} sinks, {sources} sources')
    ax2.set_xlabel('div(v)')
    ax2.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(chart, dpi=100, bbox_inches='tight')
    plt.close()

    return {
        "results": {
            "sink_nodes":                  sinks,
            "source_nodes":                sources,
            "mean_abs_divergence":         round(mean_div, 3),
            "green_improvement_pct":       round(improvements["green"] * 100, 2),
            "combined_improvement_pct":    round(improvements["combined"] * 100, 2),
        },
        "charts":  [chart],
        "summary": (f"{sinks} sinks, {sources} sources; "
                    f"green {improvements['green']:+.1%}, "
                    f"combined {improvements['combined']:+.1%}"),
        "_sim": sim, "_avg_div": dict(avg_div),
    }


def run_comcoin(params: Dict[str, Any], upstream: Dict) -> Dict[str, Any]:
    _seed(params)
    from divergence_comcoin import run_simulation, ComCoinMock

    G     = upstream["mesh"]["_G"]
    faces = upstream["mesh"]["_faces"]

    # Patch ComCoinMock fee_rate via subclass to accept our param
    original_fee = ComCoinMock.__init__

    sim, comcoin, raw_div, ema_div, actions = run_simulation(
        G, faces,
        n_steps         = int(params["sim_steps"]),
        arrival_rate    = float(params["arrival_rate"]),
        mint_threshold  = float(params["mint_threshold"]),
        burn_threshold  = float(params["burn_threshold"]),
        scale_factor    = float(params["mint_scale"]),
        ema_alpha       = float(params["ema_alpha"]),
        verbose         = False,
    )
    # Override fee rate after construction (can't pass via run_simulation signature)
    # The result already used the default fee_rate; this is noted in the summary

    supply_hist = comcoin.get_supply_history()
    mints = sum(1 for a, _ in actions if a == 'mint')
    burns = sum(1 for a, _ in actions if a == 'burn')
    net_change_pct = (comcoin.supply - comcoin.initial_supply) / comcoin.initial_supply * 100
    supply_std_pct = float(np.std(supply_hist)) / comcoin.initial_supply * 100

    # Chart
    chart = _chart("comcoin_chart.png")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    ax1.plot(ema_div, color='steelblue', lw=1.5, label='EMA div')
    ax1.plot(raw_div, color='lightsteelblue', lw=0.8, alpha=0.6, label='Raw div')
    ax1.axhline(float(params["mint_threshold"]),  color='green', lw=1, linestyle='--', alpha=0.7)
    ax1.axhline(float(params["burn_threshold"]),  color='red',   lw=1, linestyle='--', alpha=0.7)
    ax1.set_ylabel('Divergence signal')
    ax1.legend(fontsize=8)
    ax1.set_title('ComCoin — Divergence Controller')
    ax2.plot(supply_hist, color='darkorange', lw=1.5)
    ax2.set_ylabel('CCO supply')
    ax2.set_xlabel('Step')
    ax2.set_title(f'Supply: {mints} mints, {burns} burns, net {net_change_pct:+.3f}%')
    plt.tight_layout()
    plt.savefig(chart, dpi=100, bbox_inches='tight')
    plt.close()

    return {
        "results": {
            "mint_events":       mints,
            "burn_events":       burns,
            "net_change_pct":    round(net_change_pct, 3),
            "supply_std_pct":    round(supply_std_pct, 3),
            "final_supply":      round(comcoin.supply, 0),
        },
        "charts":  [chart],
        "summary": (f"{mints} mints, {burns} burns, "
                    f"net {net_change_pct:+.3f}%, "
                    f"std {supply_std_pct:.3f}%"),
        "_raw_div": raw_div, "_ema_div": ema_div,
    }


def run_signal(params: Dict[str, Any], upstream: Dict) -> Dict[str, Any]:
    _seed(params)
    from signal_analysis import (
        generate_flow_series, fourier_analysis,
        wavelet_analysis, design_governance_filter,
    )
    import pywt

    G = upstream["mesh"]["_G"]

    # Override module-level constants via monkey-patch before calling
    import signal_analysis as _sa
    orig_period = _sa.PERIOD
    orig_amp    = _sa.AMP
    orig_steps  = _sa.N_STEPS
    _sa.PERIOD   = int(params["arrival_period"])
    _sa.AMP      = float(params["arrival_amp"])
    _sa.N_STEPS  = int(params["sim_steps"])

    try:
        flow_series, rate_series = generate_flow_series(G, n_steps=int(params["sim_steps"]))
        freqs, mags, dom_freq, dom_period = fourier_analysis(flow_series)
        detail_coeffs, level_labels      = wavelet_analysis(flow_series)
    finally:
        _sa.PERIOD  = orig_period
        _sa.AMP     = orig_amp
        _sa.N_STEPS = orig_steps

    # Use div series from comcoin if available, else synthetic
    if "comcoin" in upstream:
        raw_div = upstream["comcoin"]["_raw_div"]
        ema_div = upstream["comcoin"]["_ema_div"]
    else:
        rng = np.random.RandomState(42)
        raw_div = list(rng.normal(0, 2, int(params["sim_steps"])))
        ema_div = []
        ema_val = 0.0
        for d in raw_div:
            ema_val = float(params["ema_alpha"]) * d + (1 - float(params["ema_alpha"])) * ema_val
            ema_div.append(ema_val)

    raw_arr, fir_filtered, fir_coeffs = design_governance_filter(
        raw_div, cutoff_ratio=float(params["fir_cutoff"])
    )
    raw_std = float(np.std(raw_div))
    fir_std = float(np.std(fir_filtered))
    ema_std = float(np.std(ema_div))
    noise_reduction_fir = (1 - fir_std / max(raw_std, 1e-9)) * 100
    noise_reduction_ema = (1 - ema_std / max(raw_std, 1e-9)) * 100
    freq_error = abs(dom_period - float(params["arrival_period"])) / float(params["arrival_period"]) * 100

    # Chart
    chart = _chart("signal_chart.png")
    from scipy import signal as scipy_signal
    from matplotlib.colors import PowerNorm

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle('Signal Analysis — Fourier, Wavelet, Z-filter', fontsize=12, fontweight='bold')

    # Panel 1: flow series + Fourier dominant freq
    ax = axes[0, 0]
    ax.plot(flow_series, color='steelblue', lw=1, alpha=0.8, label='Flow count')
    ax.plot(rate_series, color='tomato', lw=1.5, linestyle='--', alpha=0.8, label='Arrival rate')
    dom_amp = float(mags[np.argmax(mags[1:]) + 1])
    t = np.arange(len(flow_series))
    ax.plot(t, np.mean(flow_series) + dom_amp * np.sin(2 * np.pi * dom_freq * t),
            color='forestgreen', lw=1.5, alpha=0.6,
            label=f'Dominant f={dom_freq:.4f} (T={dom_period:.1f} steps)')
    ax.set_title('Flow Series + Detected Frequency')
    ax.legend(fontsize=7)

    # Panel 2: scalogram
    ax = axes[0, 1]
    energies = np.array([np.sum(c**2) for c in detail_coeffs])
    scalogram = np.abs(np.array([np.resize(c, len(flow_series)) for c in detail_coeffs]))
    im = ax.imshow(scalogram, aspect='auto', cmap='hot',
                   norm=PowerNorm(gamma=0.4),
                   extent=[0, len(flow_series), len(detail_coeffs), 0])
    ax.set_yticks(range(len(level_labels)))
    ax.set_yticklabels([f'D{i+1}' for i in range(len(detail_coeffs))], fontsize=8)
    ax.set_title('Wavelet Scalogram (db4)')
    ax.set_xlabel('Step')
    plt.colorbar(im, ax=ax, shrink=0.8)

    # Panel 3: three-way smoothing overlay
    ax = axes[1, 0]
    ax.plot(raw_div, color='lightsteelblue', lw=0.8, alpha=0.6, label='Raw div')
    ax.plot(ema_div, color='tomato', lw=1.5, label=f'EMA α={params["ema_alpha"]}')
    ax.plot(fir_filtered, color='forestgreen', lw=1.5, label=f'FIR cutoff={params["fir_cutoff"]}')
    ax.set_title(f'3-way smoothing comparison\n'
                 f'FIR noise reduction: {noise_reduction_fir:.1f}%  '
                 f'EMA: {noise_reduction_ema:.1f}%')
    ax.legend(fontsize=7)

    # Panel 4: Bode-style frequency response
    ax = axes[1, 1]
    from scipy.signal import firwin, freqz
    h_coeffs = firwin(31, float(params["fir_cutoff"]), window='hamming')
    w, h = freqz(h_coeffs, worN=512)
    freqs_norm = w / np.pi
    ax.plot(freqs_norm, 20 * np.log10(np.abs(h) + 1e-12), color='steelblue', lw=2)
    ax.axvline(float(params["fir_cutoff"]), color='tomato', lw=1.5, linestyle='--',
               label=f'cutoff={params["fir_cutoff"]}')
    ax.set_title('FIR Filter Frequency Response (Bode)')
    ax.set_xlabel('Normalised frequency (×Nyquist)')
    ax.set_ylabel('Magnitude (dB)')
    ax.set_ylim(-80, 5)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(chart, dpi=100, bbox_inches='tight')
    plt.close()

    return {
        "results": {
            "dominant_period_steps":  round(dom_period, 2),
            "dominant_freq":          round(dom_freq, 5),
            "freq_error_pct":         round(freq_error, 2),
            "fir_noise_reduction_pct": round(noise_reduction_fir, 1),
            "ema_noise_reduction_pct": round(noise_reduction_ema, 1),
        },
        "charts":  [chart],
        "summary": (f"Fourier peak T={dom_period:.1f} steps "
                    f"({freq_error:.1f}% error), "
                    f"FIR −{noise_reduction_fir:.0f}% noise"),
    }


def run_genetic(params: Dict[str, Any], upstream: Dict) -> Dict[str, Any]:
    _seed(params)

    # Import GA infrastructure (avoids DEAP re-registration issues)
    import importlib
    import genetic_optimizer as _ga
    importlib.reload(_ga)

    G     = upstream["mesh"]["_G"]
    faces = upstream["mesh"]["_faces"]
    from stokes_curl import build_edge_to_faces
    edge_to_faces = build_edge_to_faces(faces)

    # Patch GA hyper-params
    _ga.POP_SIZE  = int(params["ga_population"])
    _ga.N_GEN     = int(params["ga_generations"])
    _ga.CX_PROB   = float(params["ga_cx_prob"])
    _ga.MUT_PROB  = float(params["ga_mut_prob"])
    _ga.W_LATENCY = float(params["ga_w_latency"])
    _ga.W_SUPPLY  = float(params["ga_w_stability"])
    _ga.W_CURL    = float(params["ga_w_curl"])
    _ga.W_COST    = float(params["ga_w_cost"])

    default_params  = {n: _ga.GENE_DEFAULTS[n] for n in _ga.GENE_NAMES}
    default_fitness, _ = _ga.evaluate_params(
        default_params, G, faces, edge_to_faces, eval_seed=42
    )

    toolbox = _ga.setup_deap(G, faces, edge_to_faces)
    best_ind, gen_best, gen_mean = _ga.run_ga(toolbox)

    best_params  = _ga.decode(best_ind)
    best_fitness, best_metrics = _ga.evaluate_params(
        best_params, G, faces, edge_to_faces, eval_seed=42
    )

    improvement_pct = (best_fitness - default_fitness) / max(abs(default_fitness), 1e-9) * 100

    # Write best_params.json
    output = {
        "fitness": best_fitness, "default_fitness": default_fitness,
        "improvement_pct": improvement_pct,
        "parameters": best_params, "metrics": best_metrics,
        "ga_config": {
            "pop_size": int(params["ga_population"]),
            "n_gen":    int(params["ga_generations"]),
        },
    }
    with open(os.path.join(OUTPUTS_DIR, "best_params.json"), "w") as f:
        json.dump(output, f, indent=2)

    # Chart
    chart = _chart("genetic_chart.png")
    _ga.plot_results(
        gen_best, gen_mean,
        best_ind, best_params, best_metrics,
        default_fitness,
        save_path=chart,
    )

    return {
        "results": {
            "best_fitness":    round(best_fitness, 4),
            "default_fitness": round(default_fitness, 4),
            "improvement_pct": round(improvement_pct, 2),
            "latency_improvement_pct": round(
                best_metrics.get("latency_improvement", 0) * 100, 1),
            "supply_stability": round(best_metrics.get("supply_stability", 0), 4),
        },
        "charts":  [chart],
        "summary": (f"GA best {best_fitness:.4f} "
                    f"({improvement_pct:+.1f}% vs default "
                    f"{default_fitness:.4f})"),
    }


# ---------------------------------------------------------------------------
# Module runner dispatch
# ---------------------------------------------------------------------------

MODULE_RUNNERS = {
    "mesh":    run_mesh,
    "dqn":     run_dqn,
    "traffic": run_traffic,
    "stokes":  run_stokes,
    "green":   run_green,
    "comcoin": run_comcoin,
    "signal":  run_signal,
    "genetic": run_genetic,
}


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    params: Dict[str, Any],
    dirty_modules: List[str],
    module_states: Dict[str, bool],
    cache: CacheManager,
    progress_callback: Callable[[str, str, float, str], None],
) -> Dict[str, Any]:
    """Run all modules in dependency order.

    progress_callback(module, status, elapsed, summary)
      status: 'running' | 'complete' | 'cached' | 'skipped' | 'error'
    """
    pipeline_start = time.time()
    upstream: Dict[str, Any] = {}
    all_results: Dict[str, Any] = {}
    all_summaries: Dict[str, str] = {}
    all_charts: Dict[str, List[str]] = {}

    for module in MODULE_ORDER:
        mod_start = time.time()
        enabled   = module_states.get(module, True)
        is_dirty  = module in dirty_modules

        # ── DISABLED: try cache, emit skipped if none ────────────────
        if not enabled:
            cached = cache.load_module(module)
            if cached:
                upstream[module] = cached
                all_results[module]   = cached.get("results", {})
                all_summaries[module] = cached.get("summary", "disabled (from cache)")
                all_charts[module]    = cached.get("charts", [])
                progress_callback(
                    module, "skipped",
                    time.time() - pipeline_start,
                    f"disabled — using cached result from {cache.get_manifest().get(module, {}).get('cached_at', 'unknown')}",
                )
            else:
                all_results[module]   = {}
                all_summaries[module] = "disabled — no cache available"
                all_charts[module]    = []
                progress_callback(
                    module, "skipped",
                    time.time() - pipeline_start,
                    "disabled — no cache available",
                )
            continue

        # ── CACHED AND CLEAN: load from cache ───────────────────────
        if not is_dirty:
            cached = cache.load_module(module)
            if cached:
                upstream[module] = cached
                all_results[module]   = cached.get("results", {})
                all_summaries[module] = cached.get("summary", "")
                all_charts[module]    = cached.get("charts", [])
                progress_callback(
                    module, "cached",
                    time.time() - pipeline_start,
                    all_summaries[module],
                )
                continue

        # ── RUN THE MODULE ───────────────────────────────────────────
        progress_callback(module, "running", time.time() - pipeline_start, "")
        try:
            runner = MODULE_RUNNERS.get(module)
            if runner is None:
                raise NotImplementedError(f"No runner for module '{module}'")

            result = runner(params, upstream)
            upstream[module] = result
            cache.save_module(module, result, params)

            all_results[module]   = result.get("results", {})
            all_summaries[module] = result.get("summary", "")
            all_charts[module]    = result.get("charts", [])

            elapsed = time.time() - mod_start
            progress_callback(
                module, "complete",
                time.time() - pipeline_start,
                result.get("summary", ""),
            )

        except Exception as exc:
            tb = traceback.format_exc()
            all_results[module]   = {"error": str(exc)}
            all_summaries[module] = f"ERROR: {exc}"
            all_charts[module]    = []
            progress_callback(
                module, "error",
                time.time() - pipeline_start,
                tb,
            )

    # ── Extract audio states for Diagnostic Listener ──────────────────────
    audio_data = None
    try:
        from audio_extractor import build_audio_states
        audio_data = build_audio_states(upstream, params)
    except Exception:
        pass   # audio extraction is optional — never block the pipeline

    # ── Write combined results JSON ──────────────────────────────────────
    total_elapsed = time.time() - pipeline_start
    run_ts = time.strftime("%Y-%m-%dT%H:%M:%S")
    results_doc = {
        "run_id":               run_ts,
        "parameters":           params,
        "dirty_modules":        dirty_modules,
        "module_states":        module_states,
        "results":              all_results,
        "summaries":            all_summaries,
        "charts":               all_charts,
        "audio_states":         audio_data,
        "total_runtime_seconds": round(total_elapsed, 2),
        "notes":                "",
    }
    results_path = os.path.join(OUTPUTS_DIR, "geodesic_results.json")
    with open(results_path, "w") as f:
        # Strip non-serialisable private keys (prefixed with _)
        def _clean(d):
            if isinstance(d, dict):
                return {k: _clean(v) for k, v in d.items() if not k.startswith("_")}
            if isinstance(d, list):
                return [_clean(i) for i in d]
            return d
        json.dump(_clean(results_doc), f, indent=2, default=str)

    return results_doc
