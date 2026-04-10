# NEXT_STEPS.md — Geodesic Blockchain AI
**Last updated:** 2026-04-08
**Prepared for:** Claude Code implementation sessions
**Read alongside:** PROJECT_CONTEXT.md

This document covers all remaining modules in build order.
Work through them sequentially — each builds on the previous.
Do not skip ahead. Genetic optimizer must come last.

---

## How to use this document

At the start of each Claude Code session:
1. Read PROJECT_CONTEXT.md (project state and results so far)
2. Read this file (what to build next and why)
3. Implement the next unchecked module
4. Update PROJECT_CONTEXT.md with results before ending the session

Current status:
- [x] geodesic_overlay.py
- [x] qlearning_v2.py
- [x] dqn_router.py
- [x] traffic_sim.py
- [x] stokes_curl.py
- [x] green_flow.py          ← completed 2026-04-08
- [x] divergence_comcoin.py  ← completed 2026-04-08
- [x] signal_analysis.py     ← completed 2026-04-08
- [x] genetic_optimizer.py  ← completed 2026-04-08
- [ ] dirac_validator.py     ← START HERE (lowest priority, do last)

---

## Module 1: green_flow.py
**Patent claim:** §D — Green's Theorem flow balancing
**Builds on:** stokes_curl.py (same directed flow infrastructure)
**New dependency:** none

### What it does and why

Stokes detects *rotational* congestion — traffic spinning around a face.
Green detects *divergence* congestion — traffic accumulating at a node
(sink) or being generated faster than it can be routed (source).

Green's Theorem states: what flows in across a closed boundary equals
what diverges inside it. Applied to the mesh: for each node, the net
flow arriving across its edges should equal zero at steady state.
If a node has persistent net inflow, it is a sink (dropping packets).
If persistent net outflow, it is a source (generating more than it routes).

Together Stokes + Green give a complete local picture:
- Stokes: is traffic going in circles? (rotational problem)
- Green: is traffic piling up or draining? (divergence problem)

### Implementation spec

```python
# green_flow.py

# Reuse the same TrafficSimulator and directed flow tracking
# from stokes_curl.py. The flow data structure is identical.
# directed_flows[(u,v)] = count of flows routed u->v

def compute_node_divergence(G, directed_flows):
    """
    For each node, compute net divergence:
    div(v) = sum of outflows from v - sum of inflows to v
    Positive div: source node (generating more than absorbing)
    Negative div: sink node (absorbing more than generating)
    Near zero: balanced
    """

def green_boundary_integral(G, directed_flows, node_subset):
    """
    For a subset of nodes (a 'cell'), compute the boundary integral:
    sum of net flows crossing the boundary of the cell.
    Should equal the sum of divergences inside by Green's theorem.
    Use this to verify the divergence computation is consistent.
    """

def green_aware_weights(G, directed_flows, base_weights,
                        sink_penalty=5.0, source_penalty=3.0,
                        div_threshold=2.0):
    """
    Adjust edge weights to route around sink and source nodes.
    Edges leading INTO a sink node get penalised (avoid adding load).
    Edges leading OUT OF a source node get penalised (avoid amplifying).
    Returns updated weight dict for use in Dijkstra routing.
    """
```

### Simulation and comparison

Run the same setup as stokes_curl.py:
- 162 nodes, 320 faces, 480 edges
- arrival_rate = 5 flows/step, 100 simulation steps
- Compare: Hop-SP vs Curl-aware SP vs Green-aware SP vs Combined SP
  (Combined = curl penalty + green penalty applied simultaneously)

The combined router is the key result — it should outperform either
theorem applied alone because it handles both rotational and divergence
congestion.

### Visualisation (4-panel, match stokes_curl style)

Panel 1: Node divergence map — colour nodes by div(v) on the 3D sphere.
         Red = sink, blue = source, green = balanced.
Panel 2: Divergence magnitude over simulation time (mean and max).
Panel 3: Per-pair latency — Hop-SP vs Green-aware SP vs Combined SP.
Panel 4: Scatter — node divergence magnitude vs path latency.
         Points should shift left (lower divergence on path) for Green-aware.

### Expected results

Green-aware alone: expect +8 to +14% latency improvement over hop-SP.
Combined (curl + green): expect +20 to +28% — the main result to report.
Key output: green_flow_results.png

### Update PROJECT_CONTEXT.md with:
- Actual improvement numbers
- Number of sink/source nodes detected
- Combined improvement figure

---

## Module 2: divergence_comcoin.py
**Patent claim:** §E — Divergence Theorem + ComCoin tokenomics
**Builds on:** green_flow.py (node divergence signal), traffic_sim.py
**New dependency:** none (pure Python mock — no Solidity needed)

### What it does and why

This module connects network health to token economics.
The Divergence Theorem says: the total flux diverging out of a volume
equals the net source/sink inside it. Applied to ComCoin:

- If the network has net positive divergence (more transactions being
  generated than routed to completion), the system is under-supplied.
  Response: mint ComCoin to increase liquidity.
- If net negative divergence (transactions draining, demand falling),
  the system is over-supplied. Response: burn ComCoin.

This is what makes ComCoin's supply elastic to network demand,
rather than fixed like Bitcoin or manually adjusted like a stablecoin.

### ComCoin mock — features to implement

All in Python. No Solidity. Mock the contract behaviour only.

```python
class ComCoinMock:
    def __init__(self, initial_supply=1_000_000):
        self.supply = initial_supply
        self.fee_rate_per_second = 1.157e-7   # ~0.01% per day
        self.redemption_window = (22, 25)      # days 22-25 of 30-day cycle
        self.min_lot = 1000                    # minimum redemption size
        self.price = 80.0                      # mock oil price USD/barrel
        self.balances = {}                     # address -> balance
        self.last_fee_time = {}               # address -> timestamp

    def accrue_fees(self, address, current_time):
        """ fee = balance * rate * delta_t — burns on every transfer """

    def mint(self, amount, reason="divergence"):
        """ increase supply, log reason """

    def burn(self, amount, reason="divergence"):
        """ decrease supply, log reason """

    def redeem(self, address, amount, current_day):
        """ enforce redemption window and min_lot """

    def oracle_price(self):
        """ mock Chainlink — random walk around $80/barrel """

    def get_supply_history(self):
        """ return time series of supply for plotting """
```

### Divergence controller

```python
def divergence_controller(network_divergence, comcoin, step,
                           mint_threshold=5.0, burn_threshold=-5.0,
                           scale_factor=10.0):
    """
    network_divergence: scalar, total net divergence across all nodes
    Positive -> mint, negative -> burn, near zero -> do nothing.
    scale_factor: how many CCO to mint/burn per unit of divergence.
    Log every mint/burn event with step, amount, reason, new supply.
    """
```

### Simulation

Run for 200 steps at arrival_rate = 4 (moderate-high congestion).
Track supply, price, mint/burn events, and fee accrual simultaneously.
The supply should oscillate around its initial value at steady state
if the controller is well-calibrated.

### Visualisation (4-panel)

Panel 1: Network divergence over time (the input signal).
Panel 2: ComCoin supply over time — shows mint/burn response.
Panel 3: Mint/burn event log — bar chart of event sizes over time.
Panel 4: Fee accrual rate over time — shows time-decay burn in action.

Key output: divergence_comcoin_results.png

### Update PROJECT_CONTEXT.md with:
- Total mint events, total burn events, net supply change
- Supply volatility (std dev as % of initial supply)
- Calibration verdict: does supply stabilise?

---

## Module 3: signal_analysis.py
**Patent claims:** §F (Fourier + Wavelet) and §G (Z-transform)
**Builds on:** traffic_sim.py flow history, stokes_curl.py curl history
**New dependencies:** scipy (already installed), pywt (pip install PyWavelets)

### What it does and why

This module analyses the time-series data already produced by the
simulation — it does not generate new simulation data, it illuminates
what the existing data contains.

Three tools, three questions:

**Fourier transform:** Does the network have periodic behaviour?
For example, does congestion spike every N steps because of how
flows are generated? A dominant frequency here suggests the network
has a natural rhythm that governance parameters could align with.

**Wavelet transform:** When do bursts happen and at what scale?
Fourier tells you the frequencies present overall. Wavelet tells you
*when* each frequency is active. This catches transient events —
a brief congestion storm, a routing oscillation — that Fourier would
average away. Use PyWavelets (pywt), Daubechies-4 wavelet.

**Z-transform (discrete-time filter):** How should governance parameters
be updated smoothly? If the fee rate is updated every step based on
raw divergence signal, it will thrash. The Z-transform designs a
low-pass filter (IIR or FIR) that smooths the update signal, preventing
the fee rate from oscillating. This is the same principle as the
smoothing filters in network hardware — your hardware background applies.

```python
def fourier_analysis(flow_series):
    """ FFT of per-step flow count. Return dominant frequencies. """

def wavelet_analysis(flow_series, wavelet='db4', levels=5):
    """ Multi-level DWT decomposition. Return coefficients and plot. """

def design_governance_filter(signal, cutoff_ratio=0.1):
    """
    Design a low-pass FIR filter using scipy.signal.firwin.
    cutoff_ratio: fraction of Nyquist frequency to pass through.
    Apply to fee_rate update signal from divergence_comcoin.py.
    Return: raw signal, filtered signal, filter coefficients.
    """

def apply_z_filter(signal, filter_coeffs):
    """ Apply the designed filter using scipy.signal.lfilter """
```

### Input data

Generate fresh time-series data by running an internal simulation at the
top of signal_analysis.py (do NOT import from other modules at runtime —
just replicate the minimal simulation loop).

**Flow series (Fourier + Wavelet input):**
Use a sinusoidally varying arrival rate to inject a known dominant frequency:

    arrival_rate(step) = 4 + 2 * sin(2π * step / 20)

This creates a clear peak at frequency 1/20 steps that Fourier must detect.
Run for 200 steps. Track per-step active flow count as the primary series.

**Divergence series (Z-filter input):**
Import `run_simulation` from divergence_comcoin.py and use the returned
`raw_div_history` and `ema_div_history` (both 200 steps, arrival_rate=4).
The Z-filter should be applied to `raw_div_history`; overlay the
`ema_div_history` as a comparison to show both smoothing approaches.

### Visualisation (4-panel)

Panel 1: Raw flow time series + dominant Fourier frequencies marked.
Panel 2: Wavelet scalogram — time on x-axis, scale on y-axis,
         colour = coefficient magnitude. Standard wavelet plot.
Panel 3: Three-way overlay — raw_div_history (noisy), ema_div_history (EMA α=0.2),
         and Z-filter output (FIR low-pass). Shows the trade-off between EMA
         (simple, causal, slight lag) and FIR (linear phase, better freq separation).
Panel 4: Frequency response of the designed FIR filter (Bode-style plot).

Key output: signal_analysis_results.png

### What success looks like

- Fourier: at least one dominant frequency identified with a clear peak
- Wavelet: visible banding in the scalogram showing burst structure
- Z-filter: filtered signal tracks the raw signal's trend but removes
  high-frequency noise. Quantify: std dev of filtered signal should be
  30-60% of raw signal std dev.

### Update PROJECT_CONTEXT.md with:
- Dominant frequency found (in steps)
- Filter cutoff chosen and resulting noise reduction %

---

## Module 4: genetic_optimizer.py
**Patent claim:** §I — Genetic Algorithm parameter evolution
**Builds on:** ALL previous modules — this is the top-level wrapper
**New dependency:** deap (already installed)

### What it does and why

Every module below this one has hyperparameters that were chosen
by hand: DQN learning rate, epsilon decay, curl penalty threshold,
fee rate, mint/burn scale factor, filter cutoff. The genetic optimizer
finds better values for all of them simultaneously by treating the
entire system as a black box and evolving the parameter set.

This is the patent's claim of "continuous self-optimisation."
In practice, one run of the GA is expensive (each fitness evaluation
requires a full simulation), so this is not truly continuous —
it is a periodic optimisation pass that produces a better parameter set.

### Chromosome definition

```python
# 8 genes, all normalised to [0, 1] internally, mapped to real ranges

CHROMOSOME = {
    'dqn_alpha':        (0.001, 0.5),    # DQN learning rate
    'dqn_gamma':        (0.8,   0.99),   # DQN discount factor
    'dqn_epsilon':      (0.1,   0.5),    # Initial exploration rate
    'curl_penalty':     (1.0,   20.0),   # Stokes curl penalty factor
    'green_penalty':    (1.0,   15.0),   # Green divergence penalty
    'fee_rate':         (1e-8,  1e-6),   # ComCoin fee rate per second
    'mint_scale':       (1.0,   50.0),   # CCO minted per divergence unit
    'filter_cutoff':    (0.05,  0.4),    # Z-filter cutoff ratio
}
```

### Fitness function

```python
def evaluate(chromosome):
    """
    Run a mini-simulation with these parameters and return fitness score.
    Keep simulation short (50 steps, arrival_rate=4) for speed.
    
    Fitness = w1 * latency_improvement
            + w2 * supply_stability      (1 - supply_std/initial_supply)
            + w3 * curl_reduction        (reduction in avg curl exposure)
            - w4 * compute_cost          (proxy: training episodes used)
    
    Weights: w1=0.35, w2=0.40, w3=0.15, w4=0.10
    # Revised 2026-04-08: supply stability weight raised (0.3→0.4) because
    # ComCoin parameters required manual retuning; latency weight slightly
    # reduced (0.4→0.35); curl weight reduced (0.2→0.15) to match observed
    # lower contribution of curl signal vs divergence signal.
    Higher fitness = better parameter set.
    Returns: (fitness_score,)  — DEAP requires a tuple
    """
```

### GA setup

```python
# Population and evolution parameters
POP_SIZE  = 30     # keep small — each eval is expensive
N_GEN     = 20     # 20 generations
CX_PROB   = 0.5    # crossover probability
MUT_PROB  = 0.2    # mutation probability

# DEAP operators
toolbox.register("mate",   tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)
```

### Output

Print the best chromosome found and its fitness score.
Show improvement over the hand-tuned defaults used so far.
Save the best parameters to `best_params.json` — other modules
can optionally load this file to use evolved parameters.

### Visualisation (2-panel — keep simple)

Panel 1: Fitness over generations — best and mean fitness per generation.
         Should show clear improvement trend.
Panel 2: Best chromosome as a bar chart — normalised gene values,
         with the hand-tuned default shown as a horizontal line per bar.
         Shows which parameters the GA moved most.

Key output: genetic_optimizer_results.png, best_params.json

### Runtime warning

With POP_SIZE=30 and N_GEN=20, this is 600 fitness evaluations.
Each evaluation runs a 50-step simulation. Expect 5-15 minutes.
Add a progress bar (tqdm or simple print every 5 generations).
Do not increase population or generations without testing first.

### Update PROJECT_CONTEXT.md with:
- Best fitness score vs default parameter fitness score
- Which genes changed most (top 3)
- Runtime on this machine

---

## Module 5: dirac_validator.py
**Patent claim:** §7 — Dirac-inspired transaction validator
**Builds on:** Nothing in the existing simulation directly
**Priority:** Lowest — implement only after all above are complete

### What it does and why

The Dirac equation describes how a quantum particle's state evolves
as a 4-component spinor via a 4x4 matrix operator. The patent borrows
this mathematical structure — not the physics — to model how a
transaction's validation priority evolves as it propagates through
the network.

Each transaction has 4 properties that collectively determine when
it should be validated:

```python
# Transaction state vector (the "spinor")
state = np.array([
    validity_score,    # 0-1: cryptographic validity confidence
    fee_level,         # 0-1: normalised fee relative to mempool
    age_factor,        # 0-1: how long the tx has been waiting
    dependency_depth,  # 0-1: how many unconfirmed txs this depends on
])
```

The state evolves via a Dirac-inspired 4x4 operator at each network hop.
After N hops the final state vector's norm gives a priority score.
Transactions are ordered by priority score for inclusion in the next block.

### Implementation spec

```python
def build_dirac_operator(alpha=0.3, beta=0.4, gamma_factor=0.2, delta=0.1):
    """
    Construct a 4x4 matrix inspired by Dirac's gamma matrices.
    The exact values are tunable — this is an analogy, not physics.
    The matrix should mix the 4 components so that a transaction
    with high validity AND high fee AND low age AND low dependency
    converges toward a high-norm state (high priority).
    """

def evolve_transaction(state, operator, n_hops):
    """
    Apply operator n_hops times: state = operator^n @ state
    Normalise at each step to prevent explosion.
    Return final state and priority score = np.linalg.norm(final_state)
    """

def validate_mempool(transactions, operator, n_hops=6):
    """
    transactions: list of state vectors (one per pending tx)
    Returns transactions sorted by priority score, highest first.
    """
```

### Simulation

Generate 100 mock transactions with random state vectors.
Apply the validator and show that the ordering is sensible:
high-fee, high-validity, older transactions should rank higher.
Compare to a naive fee-only ranking — the Dirac validator should
produce a different (more nuanced) ordering.

### Visualisation (2-panel)

Panel 1: Priority score distribution for the 100 transactions.
Panel 2: Scatter — fee_level vs priority_score, coloured by validity_score.
         Shows that priority is not just fee — other factors matter.

Key output: dirac_validator_results.png

### Update PROJECT_CONTEXT.md with:
- Correlation between fee_level and priority (should be high but not 1.0)
- Examples of transactions where Dirac ranking differs from fee-only ranking

---

## Notes for Claude Code

- Implement one module per session. Do not combine modules.
- Each module gets its own .py file. No monoliths.
- Match the visual style of stokes_curl_results.png (4-panel,
  matplotlib, clean axes, descriptive titles and subtitles).
- After each module, update the checklist at the top of this file
  and add results to PROJECT_CONTEXT.md.
- If a module's simulation takes more than 10 minutes, something
  is wrong — stop and optimise before continuing.
- Seed all random operations with seed=42 for reproducibility.

---

*Prepared in Claude Chat on 2026-04-08*
*Read alongside PROJECT_CONTEXT.md*
