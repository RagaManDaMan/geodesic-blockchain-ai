"""
genetic_optimizer.py — Genetic Algorithm parameter evolution.

Patent §I: "Continuously evolve routing, fee, and filter hyperparameters
            using a genetic algorithm to maximise network utility."

INTUITION (no maths background needed):
  Every module below this one had parameters chosen by hand:
    - curl / green penalty factors and thresholds
    - ComCoin fee rate, mint/burn scale factor
    - DQN learning rate, discount factor, exploration rate
    - FIR filter cutoff ratio

  This module treats the entire system as a black box and evolves all
  8 parameters simultaneously.  Each candidate parameter set is a
  "chromosome" of 8 genes.  The GA runs 20 generations of 30 chromosomes,
  scoring each candidate with a fast 50-step simulation and selecting
  survivors by tournament.  Crossover and mutation explore the space.

  After 600 evaluations the best parameter set is written to
  best_params.json so other modules can load it.

CHROMOSOME (8 genes — all stored as normalised floats in [0, 1]):

  Gene            Real range          What it controls
  ─────────────   ──────────────────  ─────────────────────────────────────
  dqn_alpha       [0.001,  0.500]     DQN learning rate
  dqn_gamma       [0.800,  0.990]     DQN discount factor
  dqn_epsilon     [0.100,  0.500]     Initial DQN exploration rate (ε)
  curl_penalty    [1.0,   20.0]       Stokes curl penalty factor
  green_penalty   [1.0,   15.0]       Green divergence penalty (sink/source)
  fee_rate        [1e-8,   1e-6]      ComCoin fee rate per step
  mint_scale      [1.0,   50.0]       Scale factor for mint/burn (×100 in sim)
  filter_cutoff   [0.05,   0.40]      FIR low-pass cutoff ratio

NOTE: DQN retraining (200+ episodes × 600 evaluations) is too expensive for
      GA search.  The DQN genes (alpha, gamma, epsilon) ride along in the
      chromosome so best_params.json is complete for a future retraining run.
      Fitness is evaluated using curl-aware + green-aware routing, which
      captures the same latency signal without needing a trained network.

FITNESS FUNCTION (higher = better):
  fitness = w1 * latency_improvement      (w1=0.35)
           + w2 * supply_stability        (w2=0.40)
           + w3 * curl_reduction          (w3=0.15)
           - w4 * compute_cost            (w4=0.10)

  latency_improvement = (hop_lat − combined_lat) / hop_lat
  supply_stability    = 1 − supply_std / initial_supply
  curl_reduction      = (hop_curl_exp − combined_curl_exp) / hop_curl_exp
  compute_cost        = mean |gene_normalised − default_normalised|

OUTPUT:  genetic_optimizer_results.png, best_params.json
"""

import matplotlib
matplotlib.use('Agg')    # non-interactive backend

import json
import time
import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from deap import base, creator, tools, algorithms

from geodesic_overlay import build_icosahedron, subdivide, mesh_to_graph
from dqn_router import assign_link_params
from stokes_curl import DirectedFlowSim, build_edge_to_faces, build_curl_weights
from green_flow import GreenFlowSim, green_aware_weights, build_combined_weights
from divergence_comcoin import ComCoinMock, divergence_controller

# ---------------------------------------------------------------------------
# Chromosome definition
# ---------------------------------------------------------------------------

GENE_NAMES = [
    'dqn_alpha',
    'dqn_gamma',
    'dqn_epsilon',
    'curl_penalty',
    'green_penalty',
    'fee_rate',
    'mint_scale',
    'filter_cutoff',
]

# Real-value bounds for each gene
GENE_BOUNDS: Dict[str, Tuple[float, float]] = {
    'dqn_alpha':     (0.001,  0.500),
    'dqn_gamma':     (0.800,  0.990),
    'dqn_epsilon':   (0.100,  0.500),
    'curl_penalty':  (1.0,   20.0),
    'green_penalty': (1.0,   15.0),
    'fee_rate':      (1e-8,   1e-6),
    'mint_scale':    (1.0,   50.0),
    'filter_cutoff': (0.050,  0.400),
}

# Hand-tuned defaults used throughout the project (reference / baseline)
GENE_DEFAULTS: Dict[str, float] = {
    'dqn_alpha':     0.001,
    'dqn_gamma':     0.990,
    'dqn_epsilon':   0.300,
    'curl_penalty':  8.000,
    'green_penalty': 5.000,
    'fee_rate':      1.157e-7,   # natural 1%/day (1 step ≈ 1 s)
    'mint_scale':    5.0,        # × 100 internally → scale_factor = 500
    'filter_cutoff': 0.200,
}

# Fitness weights (revised 2026-04-08 per design-decisions handoff)
W_LATENCY = 0.35
W_SUPPLY  = 0.40
W_CURL    = 0.15
W_COST    = 0.10

# GA hyper-parameters (per spec)
POP_SIZE  = 30
N_GEN     = 20
CX_PROB   = 0.50
MUT_PROB  = 0.20

# Per-evaluation simulation parameters
N_STEPS   = 50    # simulation steps (keep fast: ~50 ms per eval)
N_TEST    = 20    # routing test pairs per evaluation
ARRIVAL   = 4.0   # Poisson arrival rate for mini-sim

N_GENES   = len(GENE_NAMES)

# ---------------------------------------------------------------------------
# Gene coding helpers
# ---------------------------------------------------------------------------

def decode(individual: List[float]) -> Dict[str, float]:
    """Map normalised [0,1] chromosome → real parameter values."""
    params: Dict[str, float] = {}
    for i, name in enumerate(GENE_NAMES):
        lo, hi = GENE_BOUNDS[name]
        params[name] = lo + float(individual[i]) * (hi - lo)
    return params


def encode_defaults() -> List[float]:
    """Encode hand-tuned defaults as a normalised [0,1] individual."""
    ind = []
    for name in GENE_NAMES:
        lo, hi = GENE_BOUNDS[name]
        val = GENE_DEFAULTS[name]
        ind.append(max(0.0, min(1.0, (val - lo) / (hi - lo))))
    return ind


# ---------------------------------------------------------------------------
# Shared mesh (built once outside the fitness loop)
# ---------------------------------------------------------------------------

def build_mesh() -> Tuple[nx.Graph, list]:
    """Construct the 162-node geodesic mesh (2-level icosahedron)."""
    verts, faces = build_icosahedron()
    for _ in range(2):
        verts, faces = subdivide(verts, faces)
    G = mesh_to_graph(verts, faces)
    G = assign_link_params(G)
    return G, faces


# ---------------------------------------------------------------------------
# Fitness evaluation
# ---------------------------------------------------------------------------

def evaluate_params(
    params: Dict[str, float],
    G: nx.Graph,
    faces: list,
    edge_to_faces: Dict[Tuple[int, int], List[int]],
    eval_seed: int = 42,
) -> Tuple[float, Dict[str, float]]:
    """Score one parameter set via a fast 50-step mini-simulation.

    Returns (fitness_scalar, metrics_dict).
    """
    # Seed both random systems so each evaluation is fully reproducible.
    # Using the global numpy seed (not just a local RandomState) ensures
    # that GreenFlowSim's internal np.random.poisson calls are also seeded.
    np.random.seed(eval_seed)
    rng = np.random.RandomState(eval_seed)
    random.seed(eval_seed)

    nodes = list(G.nodes())

    # ── 1. Run directed-flow simulation (silent) ───────────────────────────
    sim = GreenFlowSim(G, faces, arrival_rate=ARRIVAL,
                       flow_duration=10.0, capacity_scale=0.5)

    for _ in range(N_STEPS):
        curl_now = sim.step()                       # inherits DirectedFlowSim.step()
        div_now  = sim._compute_divergence_now()
        sim.div_history.append(div_now)

    # ── 2. Time-average curl and divergence over last 20 steps ─────────────
    window = min(20, len(sim.curl_history))

    avg_curl: Dict[int, float] = defaultdict(float)
    for ch in sim.curl_history[-window:]:
        for face_idx, c in ch.items():
            avg_curl[face_idx] += c / window

    avg_div: Dict[int, float] = defaultdict(float)
    for dh in sim.div_history[-window:]:
        for node, d in dh.items():
            avg_div[node] += d / window

    # ── 3. Build penalised routing weights ─────────────────────────────────
    curl_weights = build_curl_weights(
        G, avg_curl, edge_to_faces,
        threshold=1.0,
        penalty=params['curl_penalty'],
    )
    green_weights = green_aware_weights(
        G, avg_div,
        sink_penalty=params['green_penalty'],
        source_penalty=params['green_penalty'] * 0.6,
        div_threshold=0.5,
    )
    combined_weights = build_combined_weights(G, curl_weights, green_weights)

    # ── 4. Route N_TEST pairs — compare hop-SP vs combined ─────────────────
    raw_pairs = [rng.choice(len(nodes), 2, replace=False) for _ in range(N_TEST * 2)]
    test_pairs = [(nodes[int(a)], nodes[int(b)])
                  for a, b in raw_pairs if a != b][:N_TEST]

    hop_lats:      List[float] = []
    comb_lats:     List[float] = []
    hop_curl_exp:  List[float] = []
    comb_curl_exp: List[float] = []

    def _curl_exposure(path: list) -> float:
        """Mean |curl| of faces adjacent to each edge on the path."""
        if len(path) < 2:
            return 0.0
        total = 0.0
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            adj = edge_to_faces.get((min(u, v), max(u, v)), [])
            if adj:
                total += float(np.mean([abs(avg_curl[f]) for f in adj]))
        return total / (len(path) - 1)

    for s, t in test_pairs:
        try:
            hop_path = nx.shortest_path(G, s, t, weight=None)
        except nx.NetworkXNoPath:
            continue
        try:
            comb_path = nx.shortest_path(
                G, s, t,
                weight=lambda u, v, _: combined_weights.get(
                    (u, v), G[u][v]['base_latency']
                ),
            )
        except nx.NetworkXNoPath:
            comb_path = hop_path

        h_lat = sum(sim.effective_latency(hop_path[i],  hop_path[i + 1])
                    for i in range(len(hop_path)  - 1))
        c_lat = sum(sim.effective_latency(comb_path[i], comb_path[i + 1])
                    for i in range(len(comb_path) - 1))

        hop_lats.append(h_lat)
        comb_lats.append(c_lat)
        hop_curl_exp.append(_curl_exposure(hop_path))
        comb_curl_exp.append(_curl_exposure(comb_path))

    if not hop_lats:
        return 0.0, {}

    avg_hop   = float(np.mean(hop_lats))
    avg_comb  = float(np.mean(comb_lats))
    latency_improvement = max(0.0, (avg_hop - avg_comb) / max(avg_hop, 1e-9))

    avg_hop_curl  = float(np.mean(hop_curl_exp))
    avg_comb_curl = float(np.mean(comb_curl_exp))
    curl_reduction = max(0.0,
        (avg_hop_curl - avg_comb_curl) / max(avg_hop_curl, 1e-9))

    # ── 5. ComCoin supply stability ─────────────────────────────────────────
    cc = ComCoinMock(initial_supply=1_000_000)
    cc.fee_rate_per_step = params['fee_rate']        # gene range is per-step
    scale_factor = params['mint_scale'] * 100.0       # gene (1,50) → sf (100,5000)

    ema_val = 0.0
    for step in range(N_STEPS):
        # Synthetic divergence: normal(0,5) gives σ_EMA≈2.2 so controller
        # fires regularly at threshold=±1.5, making supply_stability informative.
        raw_div = float(rng.normal(0, 5.0))
        ema_val = 0.2 * raw_div + 0.8 * ema_val
        divergence_controller(ema_val, cc, step,
                               mint_threshold=1.5,
                               burn_threshold=-1.5,
                               scale_factor=scale_factor)
        cc.tick(step)

    supply_std      = float(np.std(cc.supply_history))
    supply_stability = max(0.0, 1.0 - supply_std / cc.initial_supply)

    # ── 6. Compute-cost proxy ───────────────────────────────────────────────
    # Mean absolute deviation of normalised gene values from hand-tuned defaults.
    # Penalises extreme solutions and keeps the GA anchored near known-good space.
    defaults_norm = encode_defaults()
    cost_devs = []
    for i, name in enumerate(GENE_NAMES):
        lo, hi = GENE_BOUNDS[name]
        val_norm = (params[name] - lo) / (hi - lo)
        cost_devs.append(abs(val_norm - defaults_norm[i]))
    compute_cost = float(np.mean(cost_devs))

    # ── 7. Combine ──────────────────────────────────────────────────────────
    fitness = (W_LATENCY * latency_improvement
             + W_SUPPLY  * supply_stability
             + W_CURL    * curl_reduction
             - W_COST    * compute_cost)

    metrics = {
        'latency_improvement': latency_improvement,
        'supply_stability':    supply_stability,
        'curl_reduction':      curl_reduction,
        'compute_cost':        compute_cost,
        'avg_hop_latency':     avg_hop,
        'avg_comb_latency':    avg_comb,
        'supply_std':          supply_std,
    }

    return fitness, metrics


# ---------------------------------------------------------------------------
# DEAP GA setup
# ---------------------------------------------------------------------------

def setup_deap(G: nx.Graph, faces: list,
               edge_to_faces: Dict[Tuple[int, int], List[int]]) -> base.Toolbox:
    """Register DEAP creators and operators."""

    # Guard against repeated module-level creator registration
    if not hasattr(creator, 'FitnessMax'):
        creator.create('FitnessMax', base.Fitness, weights=(1.0,))
    if not hasattr(creator, 'Individual'):
        creator.create('Individual', list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    # Each gene is a uniform float in [0, 1]
    toolbox.register('gene', random.uniform, 0.0, 1.0)
    toolbox.register('individual', tools.initRepeat,
                     creator.Individual, toolbox.gene, n=N_GENES)
    toolbox.register('population', tools.initRepeat,
                     list, toolbox.individual)

    def _eval(ind):
        params = decode(ind)
        fitness, _ = evaluate_params(params, G, faces, edge_to_faces, eval_seed=42)
        return (fitness,)

    toolbox.register('evaluate', _eval)
    toolbox.register('mate',   tools.cxBlend,       alpha=0.5)
    toolbox.register('mutate', tools.mutGaussian,   mu=0.0, sigma=0.1, indpb=0.2)
    toolbox.register('select', tools.selTournament, tournsize=3)

    return toolbox


# ---------------------------------------------------------------------------
# GA loop with progress reporting
# ---------------------------------------------------------------------------

def run_ga(
    toolbox: base.Toolbox,
    pop_size: int  = POP_SIZE,
    n_gen: int     = N_GEN,
    cx_prob: float = CX_PROB,
    mut_prob: float = MUT_PROB,
) -> Tuple[list, List[float], List[float]]:
    """Run the GA and return (best_individual, gen_best_fitness, gen_mean_fitness).

    Progress is printed every 5 generations.
    """
    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values[0]
                              if ind.fitness.valid else float('nan'))
    stats.register('best', lambda vals: max(v for v in vals if not np.isnan(v)))
    stats.register('mean', lambda vals: float(np.nanmean(vals)))

    gen_best:  List[float] = []
    gen_mean:  List[float] = []

    print(f"\n{'='*60}")
    print(f"Genetic Optimizer — {pop_size} individuals × {n_gen} generations")
    print(f"  Total evaluations: {pop_size * (n_gen + 1)}")
    print(f"  Fitness: {W_LATENCY}×latency + {W_SUPPLY}×stability "
          f"+ {W_CURL}×curl − {W_COST}×cost")
    print(f"{'='*60}\n")

    # Evaluate initial population
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses   = list(map(toolbox.evaluate, invalid_ind))
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
    hof.update(pop)

    record = stats.compile(pop)
    gen_best.append(record['best'])
    gen_mean.append(record['mean'])
    print(f"Gen  0 | best={record['best']:.4f}  mean={record['mean']:.4f}")

    for gen in range(1, n_gen + 1):
        # Selection
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))

        # Crossover
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cx_prob:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        # Mutation — clamp genes to [0, 1] after mutation
        for mutant in offspring:
            if random.random() < mut_prob:
                toolbox.mutate(mutant)
                for i in range(len(mutant)):
                    mutant[i] = max(0.0, min(1.0, mutant[i]))
                del mutant.fitness.values

        # Evaluate offspring with invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses   = list(map(toolbox.evaluate, invalid_ind))
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        pop[:] = offspring
        hof.update(pop)

        record = stats.compile(pop)
        gen_best.append(record['best'])
        gen_mean.append(record['mean'])

        if gen % 5 == 0 or gen == n_gen:
            print(f"Gen {gen:2d} | best={record['best']:.4f}  "
                  f"mean={record['mean']:.4f}  "
                  f"evals={pop_size + gen * len(invalid_ind)}")

    return list(hof[0]), gen_best, gen_mean


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_results(
    gen_best:    List[float],
    gen_mean:    List[float],
    best_ind:    List[float],
    best_params: Dict[str, float],
    best_metrics: Dict[str, float],
    default_fitness: float,
    save_path: str = 'genetic_optimizer_results.png',
):
    """2-panel results plot."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        'Genetic Algorithm Parameter Evolution\n'
        f'Patent §I — {POP_SIZE} individuals × {N_GEN} generations, '
        f'8-gene chromosome',
        fontsize=13, fontweight='bold',
    )

    # ── Panel 1: Fitness over generations ──────────────────────────────────
    ax1 = axes[0]
    gens = list(range(len(gen_best)))
    ax1.plot(gens, gen_best, color='steelblue', lw=2.5, label='Best fitness')
    ax1.plot(gens, gen_mean, color='tomato',    lw=1.5,
             linestyle='--', alpha=0.8, label='Mean fitness')
    ax1.axhline(default_fitness, color='forestgreen', lw=1.5,
                linestyle=':', label=f'Default params ({default_fitness:.4f})')
    ax1.fill_between(gens, gen_mean, gen_best, alpha=0.15, color='steelblue')
    ax1.set_xlabel('Generation', fontsize=11)
    ax1.set_ylabel('Fitness score', fontsize=11)
    ax1.set_title('Fitness Evolution', fontsize=11)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    final_improvement = (gen_best[-1] - default_fitness) / max(abs(default_fitness), 1e-9) * 100
    ax1.annotate(
        f'+{final_improvement:.1f}% vs default',
        xy=(len(gen_best)-1, gen_best[-1]),
        xytext=(-40, -18), textcoords='offset points',
        fontsize=9, color='steelblue',
        arrowprops=dict(arrowstyle='->', color='steelblue', lw=1),
    )

    # ── Panel 2: Best chromosome vs defaults ───────────────────────────────
    ax2 = axes[1]
    defaults_norm = encode_defaults()
    best_norm     = best_ind

    x       = np.arange(N_GENES)
    bar_w   = 0.38
    colors  = ['steelblue' if b > d else 'tomato'
               for b, d in zip(best_norm, defaults_norm)]

    bars = ax2.bar(x - bar_w/2, best_norm,   bar_w, color=colors,
                   alpha=0.85, label='Best (GA)')
    ax2.bar(x + bar_w/2, defaults_norm, bar_w, color='silver',
            alpha=0.85, label='Default (hand-tuned)')
    ax2.axhline(0.5, color='grey', lw=0.8, linestyle=':', alpha=0.5)

    # Annotate real values on best bars
    for i, (bar, val) in enumerate(zip(bars, best_norm)):
        lo, hi = GENE_BOUNDS[GENE_NAMES[i]]
        real   = lo + val * (hi - lo)
        # Format according to magnitude
        if abs(real) < 1e-4:
            label = f'{real:.2e}'
        elif abs(real) >= 10:
            label = f'{real:.1f}'
        else:
            label = f'{real:.3f}'
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 label, ha='center', va='bottom', fontsize=6.5, rotation=45)

    ax2.set_xticks(x)
    ax2.set_xticklabels(GENE_NAMES, rotation=35, ha='right', fontsize=9)
    ax2.set_ylabel('Normalised gene value [0, 1]', fontsize=11)
    ax2.set_title('Best Chromosome vs Hand-Tuned Defaults', fontsize=11)
    ax2.legend(fontsize=9)
    ax2.set_ylim(0, 1.25)
    ax2.grid(True, axis='y', alpha=0.3)

    # Metrics annotation in top-right
    if best_metrics:
        txt = (
            f"Latency impr: {best_metrics.get('latency_improvement',0):.1%}\n"
            f"Supply stab:  {best_metrics.get('supply_stability',0):.4f}\n"
            f"Curl reduc:   {best_metrics.get('curl_reduction',0):.1%}\n"
            f"Compute cost: {best_metrics.get('compute_cost',0):.3f}"
        )
        ax2.text(0.98, 0.97, txt,
                 transform=ax2.transAxes, ha='right', va='top',
                 fontsize=8.5, family='monospace',
                 bbox=dict(boxstyle='round,pad=0.4', fc='#f0f4f8', ec='#aab'))

    plt.tight_layout()
    plt.savefig(save_path, dpi=140, bbox_inches='tight')
    plt.close()
    print(f"\nPlot saved → {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    random.seed(42)
    np.random.seed(42)

    t_start = time.time()

    # ── Build mesh once ────────────────────────────────────────────────────
    print("Building geodesic mesh (162 nodes, 320 faces)…")
    G, faces = build_mesh()
    edge_to_faces = build_edge_to_faces(faces)
    print(f"  Nodes={G.number_of_nodes()}, "
          f"Edges={G.number_of_edges()}, "
          f"Faces={len(faces)}")

    # ── Evaluate default parameter set as baseline ─────────────────────────
    print("\nEvaluating hand-tuned defaults (baseline)…")
    default_params  = {n: GENE_DEFAULTS[n] for n in GENE_NAMES}
    default_fitness, default_metrics = evaluate_params(
        default_params, G, faces, edge_to_faces, eval_seed=42
    )
    print(f"  Default fitness = {default_fitness:.4f}")
    print(f"  Latency improvement = "
          f"{default_metrics.get('latency_improvement', 0):.1%}")
    print(f"  Supply stability    = "
          f"{default_metrics.get('supply_stability', 0):.4f}")
    print(f"  Curl reduction      = "
          f"{default_metrics.get('curl_reduction', 0):.1%}")

    # ── Set up and run GA ──────────────────────────────────────────────────
    toolbox = setup_deap(G, faces, edge_to_faces)
    best_ind, gen_best, gen_mean = run_ga(toolbox)

    t_elapsed = time.time() - t_start

    # ── Decode and fully evaluate best individual ──────────────────────────
    best_params  = decode(best_ind)
    best_fitness, best_metrics = evaluate_params(
        best_params, G, faces, edge_to_faces, eval_seed=42
    )

    # ── Print summary ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"OPTIMISATION COMPLETE  ({t_elapsed/60:.1f} min)")
    print(f"{'='*60}")
    print(f"  Best fitness:    {best_fitness:.4f}  "
          f"(default: {default_fitness:.4f}, "
          f"improvement: {(best_fitness-default_fitness)/max(abs(default_fitness),1e-9)*100:+.1f}%)")
    print(f"\n  Best parameter set:")
    for name in GENE_NAMES:
        lo, hi = GENE_BOUNDS[name]
        val    = best_params[name]
        def_   = GENE_DEFAULTS[name]
        delta  = (val - def_) / max(abs(def_), 1e-12) * 100
        print(f"    {name:15s}  {val:.5g}  (default {def_:.5g}, {delta:+.0f}%)")
    print(f"\n  Metrics breakdown:")
    for k, v in best_metrics.items():
        print(f"    {k:22s} = {v:.4f}")

    # ── Save best_params.json ──────────────────────────────────────────────
    output = {
        'fitness':        best_fitness,
        'default_fitness': default_fitness,
        'improvement_pct': (best_fitness - default_fitness)
                           / max(abs(default_fitness), 1e-9) * 100,
        'parameters':     best_params,
        'metrics':        best_metrics,
        'ga_config': {
            'pop_size':  POP_SIZE,
            'n_gen':     N_GEN,
            'cx_prob':   CX_PROB,
            'mut_prob':  MUT_PROB,
            'n_steps':   N_STEPS,
            'n_test':    N_TEST,
        },
        'fitness_weights': {
            'latency':  W_LATENCY,
            'supply':   W_SUPPLY,
            'curl':     W_CURL,
            'cost':     W_COST,
        },
        'runtime_minutes': t_elapsed / 60.0,
    }
    with open('best_params.json', 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nbest_params.json written ✓")

    # ── Plot ───────────────────────────────────────────────────────────────
    plot_results(
        gen_best, gen_mean,
        best_ind, best_params, best_metrics,
        default_fitness,
        save_path='genetic_optimizer_results.png',
    )

    # ── Top-3 genes that moved most ────────────────────────────────────────
    defaults_norm = encode_defaults()
    movements = []
    for i, name in enumerate(GENE_NAMES):
        lo, hi = GENE_BOUNDS[name]
        best_n = (best_params[name] - lo) / (hi - lo)
        movements.append((abs(best_n - defaults_norm[i]), name,
                          best_params[name], GENE_DEFAULTS[name]))
    movements.sort(reverse=True)
    print(f"\n  Top-3 genes that changed most from defaults:")
    for delta_norm, name, best_v, def_v in movements[:3]:
        print(f"    {name}: {def_v:.5g} → {best_v:.5g}  "
              f"(Δnorm={delta_norm:.3f})")
