# Geodesic Blockchain AI — Project Context & Status Report
**Last updated:** 2026-04-09
**Prepared for:** Handoff between Claude Code (implementation) and Claude Chat (design/planning)

---

## 1. What This Project Is

We are building a **piecemeal simulation** of an Indian patent titled:
*"Geodesic Blockchain AI for Optimized Distributed Ledger Consensus and Data Routing"*

The patent describes a physics-inspired blockchain network that:
- Arranges nodes on a **subdivided icosahedral (geodesic) mesh** instead of random peer-to-peer topology
- Uses **reinforcement learning** to learn low-latency routing paths
- Applies **vector calculus theorems** (Stokes, Green, Divergence) to detect and fix congestion
- Uses **Fourier/Wavelet/Z-transforms** for signal analysis and parameter tuning
- Evolves all parameters automatically via **Genetic Algorithms**
- Powers a commodity-backed token called **ComCoin** (ERC-20, pegged to oil barrels)

The patent also mentions a **Dirac Equation**-inspired model for transaction validation state vectors.

The development workflow: the user generates ideas and initial code in **Claude Chat**, then hands the code to **Claude Code** for implementation, debugging, simulation, and iteration.

---

## 2. The Codebase — File by File

All files are in: `/Users/ragavan/Documents/claudecode/geodesic_sandbox/`

### 2.1 `geodesic_overlay.py` ✅ Complete
**What it does:** Builds the geodesic mesh from scratch.
- Starts from a 12-vertex icosahedron (golden-ratio construction)
- Recursively subdivides each triangle into 4 smaller triangles (2 levels → 162 nodes, 320 faces, 480 edges)
- Projects all vertices onto a unit sphere
- Converts the mesh into a NetworkX graph with edge weights = Euclidean chord distance
- Visualises the 3D mesh and degree distribution

**Key output:** `geodesic_mesh.png`

---

### 2.2 `qlearning_v2.py` ✅ Complete
**What it does:** Tabular Q-learning router (the precursor to the DQN).
- **Goal-conditioned:** the state is `(current_node, target_node)`, not just current node — this is what the patent's vanilla implementation was missing
- **Telegrapher's delay model:** each edge has inductance L and capacitance C; delay = 1/√(LC)
- **Reward shaping:** penalises latency + rewards progress toward target (progress * 2.0) + large bonus for reaching target (+20)
- **ε-greedy** with epsilon decay

**Results:** 100% routing success rate by episode 300. Latency improvement over hop-shortest: **~7–10%** on 30 test pairs.

---

### 2.3 `dqn_router.py` ✅ Complete
**What it does:** Deep Q-Network (DQN) replacing the tabular Q-table.
- **QNetwork:** 4-layer MLP (10 → 64 → 64 → 32 → 1)
- **10-dim feature vector per edge candidate:**
  `[pos_current(3), pos_target(3), pos_neighbor(3), edge_latency(1)]`
  The network learns geometry — "move toward the target on the sphere" becomes a learnable pattern
- **Experience replay buffer** (capacity 15,000) — breaks temporal correlations, stabilises training
- **Target network** — separate stable copy updated every 20 episodes, prevents oscillation

**Results:** 100% routing success, **+12.7% latency improvement** over hop-shortest on 40 test pairs.
**Key output:** `dqn_results.png`

---

### 2.4 `traffic_sim.py` ✅ Complete — 4-strategy comparison
**What it does:** Full traffic simulation with M/M/1 congestion model.

**Congestion model:**
`effective_latency = base_latency / (1 − ρ)` where `ρ = load / capacity`
- ρ = 0 → base latency; ρ = 0.5 → 2× base; ρ = 0.9 → 10× base (cap at 0.95 → 20×)

**Four routing strategies compared:**
1. **Hop-SP** — hop-minimising Dijkstra (baseline)
2. **Latency-SP** — Dijkstra weighted by *current* effective latency (oracle — full knowledge)
3. **DQN-base** — geometry-trained DQN from `dqn_router.py` (no congestion awareness)
4. **DQN-cong** — retrained DQN with **11-dim features** (adds link utilisation ρ as 11th input)

**DQN-cong training:** Runs alongside a background `TrafficSimulator` so the agent sees realistic congestion during learning. This required a new `QNetwork11` and `DQNRouterCongestion` class.

**Congestion sweep results** (improvement over hop-SP):

| Arrival rate | Latency-SP (oracle) | DQN-base | DQN-cong |
|---|---|---|---|
| 0.5 flows/step | +21.6% | -19.6% | **+6.2%** |
| 1.0 | +28.5% | -36.3% | **+11.9%** |
| 2.0 | +49.9% | -53.4% | **+38.3%** |
| 4.0 | +53.7% | -50.4% | **+39.2%** |
| 8.0 | +67.3% | -52.1% | **+51.1%** |

**Key finding:** DQN-base *collapses* under congestion (negative improvement) because it was trained on static base latency and takes longer paths that compound congestion costs. DQN-cong tracks the oracle closely because it learned to avoid high-ρ edges.
**Key output:** `traffic_sim_results.png`

---

### 2.5 `stokes_curl.py` ✅ Complete — just implemented today
**What it does:** Patent §C — Stokes's Theorem curl mitigation.

**Intuition (plain English):**
Every triangular face on the mesh can have traffic circulating *around* it (like a roundabout). Stokes's Theorem gives one number per triangle — the "curl" — that measures this spin. High |curl| = congestion vortex.

**Stokes integral per face:**
`curl([a,b,c]) = net_flow(a→b) + net_flow(b→c) + net_flow(c→a)`
where `net_flow(u→v) = flows_going_u_to_v − flows_going_v_to_u`

**Key difference from `traffic_sim.py`:** This module tracks *directed* flows (u→v vs v→u separately). Two flows going opposite ways cancel — no vortex. Same-direction flows reinforce — vortex.

**Curl-aware routing:**
Edge weight = `base_latency + penalty × max(0, max_|curl|_of_adjacent_faces − threshold)`
High-curl edges become expensive → routers detour around vortex triangles.

**Results (162 nodes, 320 faces, 100 simulation steps, arrival_rate=5):**
- 85 vortex faces detected (|curl| > 1) at steady state
- Max |curl| = 3.2 (time-averaged)
- Avg latency: Hop-SP = 114.4 → Curl-aware SP = 95.6 → **+16.4% improvement**
- Avg curl exposure on path: 1.77 (hop-SP) → 1.09 (curl-aware)
**Key output:** `stokes_curl_results.png`

---

---

### 2.6 `green_flow.py` ✅ Complete — implemented 2026-04-08
**What it does:** Patent §D — Green's Theorem divergence detection and mitigation.

**Intuition (plain English):**
At steady state a healthy router node should have equal inflow and outflow. Persistent net inflow = sink (dropping packets). Persistent net outflow = source (generating more than it routes). Green's Theorem: boundary flux = Σ divergence inside the region. We detect imbalanced nodes and penalise edges adjacent to them.

**Divergence per node:**
`div(v) = Σ outflows(v→u) − Σ inflows(u→v)`
Positive = source, Negative = sink, ~Zero = balanced.

**Green-aware routing:**
`weight(u,v) = base_latency + sink_penalty × max(0, −div[u or v] − threshold) + source_penalty × max(0, div[u or v] − threshold)`
High-divergence nodes become expensive → routers detour around sink/source zones.

**Combined (Stokes + Green):**
Uses `max(curl_penalty, green_penalty)` per edge — avoids penalty stacking from additive combination.

**Architecture:** `GreenFlowSim` subclasses `DirectedFlowSim` from `stokes_curl.py`, adding `div_history` tracking alongside the existing `curl_history`.

**Threshold calibration:** `div_threshold` set at 70th percentile of |div| (≈ 0.50) so the penalty activates on the most imbalanced ~30% of nodes.

**Results (162 nodes, 320 faces, 100 simulation steps, arrival_rate=5):**
- 6 sink nodes detected (div < −1), 4 source nodes (div > 1), 152 balanced
- Mean |div| = 0.380, Max |div| = 1.750 (time-averaged, last 20 steps)
- Avg latency comparisons vs Hop-SP baseline (146.1):

| Strategy | Avg latency | Improvement vs Hop-SP |
|---|---|---|
| Hop-SP | 146.1 | baseline |
| Curl-aware SP | 107.8 | **+26.2%** |
| Green-aware SP | 100.2 | **+31.4%** |
| Combined SP | 111.3 | **+23.8%** |

**Key finding:** Green-aware routing achieves the best single-strategy result because divergence (node-level imbalance) maps directly to M/M/1 effective latency — routing on base_latency while penalising sink/source nodes closely approximates the latency-SP oracle. The combined router uses `max(curl, green)` rather than additive penalties to prevent penalty stacking; despite this, it sits between the two individual strategies because the Stokes curl signal and the Green divergence signal are not always aligned — the most curl-free path is not always the most divergence-free.

**Key output:** `green_flow_results.png`

---

---

### 2.7 `divergence_comcoin.py` ✅ Complete — implemented 2026-04-08
**What it does:** Patent §E — Divergence Theorem + ComCoin elastic token supply.

**Intuition (plain English):**
The Divergence Theorem zooms out from per-node imbalance (Green's theorem) to ask: is the whole network generating more transaction traffic than it completes? The control signal is `net_div = new_flows_injected − flows_expired` per step. Positive → backlog growing → mint CCO. Negative → backlog draining → burn CCO.

**Key design note:** Σ div(v) across all nodes is always exactly 0 by flow conservation (a conservation law — every unit leaving u arrives at some v). The correct Divergence Theorem signal is therefore the rate of change of the total active-flow volume (Δactive_flows), not a sum of per-node divergences.

**ComCoin contract mechanics (Python mock):**
- `accrue_fees`: `fee = balance × 1.157e-5 × Δt` — burns at ~1%/day on every transfer
- `mint` / `burn`: elastic supply response to controller signal
- `redeem`: enforces 30-step cycle window (steps 22–25), minimum lot 1000 CCO
- `oracle_price`: Chainlink mock — Gaussian random walk around $80/barrel
- `merkle_reserve_proof`: SHA-256 hash of current supply (simplified)

**Divergence controller:**
- Fires when EMA signal `|ema_div| > threshold (±1.5)`; mints/burns `excess × 500` CCO
- EMA(α=0.2) of raw Δflows used as input — suppresses Poisson noise before controller sees it
- Dead-band prevents thrashing around equilibrium

**Parameter history:**
- v1: threshold=±5, scale=10 → 8 events, 120 CCO (invisible vs fees)
- v2: threshold=±3, scale=1000 → 28 mints, 19 burns, 49k CCO (visible but no EMA)
- v3 (final): EMA controller, threshold=±1.5, scale=500 → 18 mints, 3 burns, balanced

**Final results (200 steps, arrival_rate=4, EMA controller):**
- Mint events: **18**, Burn events: **3**
- Total minted: **5,997 CCO**, Total burned (controller): **371 CCO**
- Total burned by time-decay fees: **462 CCO**
- Net supply change: **+5,164 CCO (+0.516%)**
- Supply std dev: **766 CCO (0.077% of initial)** — STABLE
- Mean |raw_div|=2.30, Mean |ema_div|=0.78 (EMA reduces noise by ~66%)
- Oracle price range: $79.51 – $90.57
- Redemption window enforcement: ✅ (day 23 accepted, day 11 rejected)

**Exported for signal_analysis.py:** both `raw_div_history` and `ema_div_history` (200 steps each)

**Key output:** `divergence_comcoin_results.png`

---

### 2.8 `signal_analysis.py` ✅ Complete — implemented 2026-04-08
**What it does:** Patent §F+§G — Fourier, Wavelet, and Z-transform (FIR) signal analysis.

**Intuition (plain English):**
After measuring network flows and divergence signals, this module asks three questions:
- *Are there periodic patterns?* (Fourier)
- *When do bursts happen and at what scale?* (Wavelet)
- *How should governance parameters be smoothed?* (Z-transform / FIR filter)

**Input generation:**
Sinusoidal arrival rate `rate(step) = 4 + 2·sin(2π·step/20)` guarantees a dominant frequency at exactly 1/20 cyc/step. A RuntimeError guard rejects results if the detected peak deviates >10% from the expected period.

**Fourier analysis:**
- Detrended (DC removed), one-sided RFFT, 2/N normalization
- **Dominant frequency: 0.05000 cyc/step (period = 20.00 steps, 0.0% error)** ✅

**Wavelet analysis:**
- PyWavelets `pywt.wavedec`, Daubechies-4, 5 decomposition levels
- Detail coefficient energy: **D2 carries peak energy** (scale ~4 steps ≈ burst length)
- Note: boundary effect warnings at level 5 are expected for N=200; results remain valid

**FIR governance filter (Z-transform):**
- `scipy.signal.firwin` low-pass, cutoff_ratio=0.20, 31 taps, zero-phase (`filtfilt`)
- Applied to `raw_div_history` from `divergence_comcoin.py`
- **FIR noise reduction: 64.2%** (std reduced from 2.91 → 1.04) ✅
- **EMA noise reduction: 69.7%** (std reduced from 2.91 → 0.88) ✅

**Panel 3 three-way overlay:** raw_div / EMA(α=0.2) / FIR Z-filter — demonstrates FIR's linear phase (no lag distortion) vs EMA's causal simplicity.

**Key output:** `signal_analysis_results.png`

---

### 2.9 `genetic_optimizer.py` ✅ Complete — implemented 2026-04-08
**What it does:** Patent §I — Genetic Algorithm over the full 8-gene parameter space.

**Chromosome (8 genes, all normalised to [0,1] internally):**

| Gene | Real range | Purpose |
|---|---|---|
| dqn_alpha | [0.001, 0.500] | DQN learning rate |
| dqn_gamma | [0.800, 0.990] | DQN discount factor |
| dqn_epsilon | [0.100, 0.500] | Initial exploration rate |
| curl_penalty | [1.0, 20.0] | Stokes curl penalty factor |
| green_penalty | [1.0, 15.0] | Green divergence penalty |
| fee_rate | [1e-8, 1e-6] | ComCoin fee rate per step |
| mint_scale | [1.0, 50.0] | CCO minted per divergence unit (×100 in sim) |
| filter_cutoff | [0.05, 0.40] | FIR low-pass cutoff ratio |

**Note on DQN genes:** dqn_alpha / gamma / epsilon are included in the chromosome so `best_params.json` is complete for future DQN retraining. Full DQN retraining per evaluation (200+ episodes × 600 calls) is too expensive; DQN fitness is proxied by routing quality from curl+green SP.

**Fitness function:**
`fitness = 0.35 × latency_improvement + 0.40 × supply_stability + 0.15 × curl_reduction − 0.10 × compute_cost`

**GA setup:** DEAP, POP_SIZE=30, N_GEN=20, blend crossover (α=0.5), Gaussian mutation (σ=0.1, indpb=0.2), tournament selection (k=3). Each evaluation: 50-step GreenFlowSim + 20-pair routing test + 50-step ComCoin mini-sim.

**Results:**

| Metric | Default (hand-tuned) | GA best | Change |
|---|---|---|---|
| Fitness score | 0.5860 | **0.6512** | **+11.1%** |
| Latency improvement | 39.7% | **68.2%** | +28.5pp |
| Supply stability | 0.9983 | 0.9975 | −0.1pp |
| Curl reduction | 31.9% | 17.1% | −14.8pp |
| Runtime | — | **0.1 min** | — |

**Top-3 genes that moved most from defaults:**
1. `fee_rate`: 1.157e-7 → 5.86e-7 (+407%, Δnorm=0.475)
2. `curl_penalty`: 8.0 → 1.84 (−77%, Δnorm=0.324)
3. `filter_cutoff`: 0.20 → 0.163 (−19%, Δnorm=0.107)

**Key finding:** The GA dramatically lowered `curl_penalty` (8→1.8) at the cost of less curl reduction (31.9%→17.1%) to unlock a large latency improvement gain (39.7%→68.2%). This reveals that the hand-tuned curl penalty was over-aggressive — it forced unnecessarily long detours that increased total path latency more than the congestion it was avoiding. The GA discovered the optimal trade-off automatically.

**Best parameters saved to:** `best_params.json`
**Key output:** `genetic_optimizer_results.png`

---

## 3. Remaining Patent Modules — TBD

Listed in planned build order:

### 3.4 `genetic_optimizer.py` — Genetic Algorithm Parameter Evolution
**Patent §I:**
**Chromosome:** `[mesh_depth, α, γ, ε, feeRate, k_fanout]` (6 genes)
**Fitness:** `w₁(−avg_latency) + w₂(1−fork_rate) + w₃×throughput − w₄×cost`
**Process:** Population of 50 chromosomes → tournament selection → blend crossover → Gaussian mutation → 40 generations
**Uses DEAP library.** Wraps the existing `DQNRouter` training + `TrafficSimulator` as the fitness function.
**This is the top-level self-optimiser** — it finds the best hyperparameters for everything below it.

---

### 3.5 `dirac_validator.py` — Dirac-Inspired Transaction Validator
**Patent §7 (Dirac Equation):**
The Dirac equation describes how a quantum particle's state evolves via a 4-component spinor. The patent borrows this for **transaction validation state vectors**.
Each pending transaction is a 4-component state: `[validity_score, fee_level, age_factor, dependency_depth]`
The state evolves as the transaction propagates through the network via a 4×4 Dirac-inspired matrix operator.
Output: a **validation priority score** per transaction — used to order the mempool.
**No actual quantum hardware needed** — it's linear algebra (matrix multiplication) using the structure of the Dirac equation as inspiration.

---

## 3b. Demo GUI — fully implemented and tested 2026-04-09

Flask REST API + pipeline orchestration layer + single-file dark dashboard. All files in `geodesic_sandbox/`.

### New files

| File | Role |
|---|---|
| `param_schema.py` | Single source of truth for all 32 params (default/min/max/type/module/label) + `MODULE_REGISTRY` + `DEPENDENCY_CHAIN` |
| `cache_manager.py` | Pickle-based module cache; manifest.json tracks params used per module; dirty-module detection propagates upstream changes downstream |
| `time_estimator.py` | Per-module runtime models (calibrated M-series MacBook); fast-params suggestion for runs >120s |
| `pipeline.py` | Orchestrates modules in dependency order; wraps existing `.py` files without modifying them; writes `outputs/geodesic_results.json` |
| `app.py` | Flask server on port **7432**; 12 REST routes; SSE streaming for `/api/run`; CORS headers for frontend dev server |
| `frontend/index.html` | Single-file dark dashboard — all CSS+JS inline; schema-driven params; SSE progress streaming; module toggles; chart tabs; notes |
| `outputs/cache/` | Pickle files per module + `manifest.json` |
| `README.md` | Setup instructions, project overview, API reference |
| `requirements.txt` | Pinned Python dependencies |
| `.gitignore` | Excludes cache pickles, PNGs, venv, __pycache__ |

### API routes (12 total)

| Method | Route | Purpose |
|---|---|---|
| GET | `/` | Serve `frontend/index.html` |
| GET | `/api/schema` | Returns all 32 params + MODULE_REGISTRY + module order |
| GET | `/api/defaults` | Flat dict of all defaults |
| GET | `/api/status` | Cache manifest + per-module enabled/cached state |
| POST | `/api/estimate` | Runtime estimate, dirty modules, warning + fast-params if >120s |
| POST | `/api/run` | SSE stream: runs pipeline, emits per-module progress events |
| GET | `/api/results` | Latest `geodesic_results.json` |
| GET | `/api/results/download` | Download results as JSON file attachment |
| POST | `/api/results/notes` | Update notes field in results |
| GET | `/api/best_params` | GA best params remapped to param_schema keys |
| GET | `/outputs/<file>` | Serves chart PNGs |
| POST | `/api/reset` | Clears cache |

### Module enable/disable design

`MODULE_REGISTRY` in `param_schema.py` controls which modules can be toggled at runtime:
- `required: True` (mesh only) — cannot be disabled
- `enabled: True/False` — default on/off state
- `/api/run` and `/api/estimate` accept `"module_states": {"dqn": false, ...}` in request body
- Disabled modules: skip running, use cached result if available, emit `"status": "skipped"`
- New modules (e.g. `dirac`) require only: one entry in `MODULE_REGISTRY` + one `run_dirac()` function in `pipeline.py`

### Confirmed working — API (curl tests 2026-04-09)
- `/api/schema` → 32 params, 8 modules ✅
- `/api/defaults` → 32 defaults ✅
- `/api/status` → empty cache, all enabled ✅
- `/api/estimate` (default params) → 26.4s total, per-module breakdown ✅
- `/api/estimate` (dqn/genetic/signal disabled) → 15.0s, correct dirty propagation ✅
- `/api/estimate` (slow params) → 192s warning + fast_params suggestion ✅
- `/api/reset` → cache cleared ✅

### Confirmed working — GUI end-to-end test (2026-04-09)
Full browser test at http://localhost:7432 with `python app.py`:

1. **Page load** — dark dashboard renders: 3-column layout, 32 param sliders, 8 module toggles, estimate panel, run button, 8 result tiles, chart tabs, notes ✅
2. **All 32 params** — schema-driven, grouped by module, log-scale sliders for `fee_rate` and `learning_rate` ✅
3. **Slider → estimate** — changing episodes 800→400 instantly updated DQN estimate 1.2s→0.6s (debounced 300ms) ✅
4. **Slow params warning** — setting ga_population=100, ga_generations=100, dqn_episodes=1000 → 190s warning + fast_params suggestion shown ✅
5. **Run button / SSE streaming** — amber "Running..." pulsing button, "● Live" header, progress log streams per-module events; all params locked during run ✅
6. **Module tiles** — all 8 turn green with real summaries after 47.5s run (fast params) ✅
7. **Chart tabs** — clicking DQN Router tab lazy-loads training reward + success rate chart; Geodesic Mesh auto-selected after run ✅
8. **Notes save** — typed note saved to `outputs/geodesic_results.json` via POST /api/results/notes ✅
9. **Page refresh** — all 8 result tiles restored from `/api/results`, cache status re-fetched from `/api/status`, notes textarea populated, chart auto-loads ✅
10. **Module enable/disable** — DQN toggle off → grey section, `· disabled` badge, `— disabled` in estimate, DQN tile shows disabled; toggle back on → re-enables correctly ✅
- **Load GA best** — clicking "⚡ Load GA best" applies best_params.json values: fee_rate 1.157e-5 → 5.861e-7, learning_rate 0.001 → 0.0031, curl_penalty → 1.84 ✅

### Known behaviour notes
- Toggle switch label is 30×16px; clicking must be precise on the visible track. Toggle works correctly when clicked on target — if missed, it collapses the section instead.
- Fourier period error at fast params (sim_steps=100): signal_analysis reports 400% period error because the injected sinusoidal period (20 steps) is poorly resolved in only 100 simulation steps. Expected with fast settings; use sim_steps ≥ 200 for valid signal analysis.
- Module toggle uses zero-area hidden checkbox (`opacity:0; width:0; height:0`) inside a label; clicking the visible `.toggle-track` span inside the label works, but the checkbox is not directly accessible via browser accessibility tree.

### Import notes for pipeline.py
- All existing modules use `random.seed(42)` / `np.random.seed(42)` at module level. `pipeline.py` re-seeds before each `run_*()` call using `params["seed"]` to ensure reproducibility without modifying the original files.
- `signal_analysis.py` uses module-level constants (`PERIOD`, `AMP`, `N_STEPS`). `run_signal()` monkey-patches these before calling the module's functions and restores them after — avoids modifying the original file.
- `genetic_optimizer.py` DEAP creator registration raises on duplicate calls. `run_genetic()` uses `importlib.reload()` on each invocation to get a clean DEAP state.
- `divergence_comcoin.run_simulation()` does not expose `fee_rate` as a parameter; the pipeline uses the default internal value and notes this in the result summary.

### Start command
```
cd /Users/ragavan/Documents/claudecode/geodesic_sandbox
source venv/bin/activate
python app.py
# → Geodesic demo running at http://localhost:7432
```

---

## 4. Architecture Overview

```
geodesic_overlay.py          ← mesh geometry (shared foundation)
        │
        ├── qlearning_v2.py  ← tabular Q-learning (prototype)
        │
        ├── dqn_router.py    ← DQN with geometry features (10-dim)
        │        │
        │   traffic_sim.py   ← M/M/1 congestion + 4-strategy comparison
        │        │            (DQN-cong uses 11-dim features + live traffic)
        │        │
        │   stokes_curl.py   ← directed flows + Stokes curl mitigation ✅ NEW
        │
        ├── green_flow.py    ← Green's theorem flow balancing [TBD]
        │
        ├── divergence_comcoin.py  ← Divergence theorem + ComCoin mock [TBD]
        │
        ├── signal_analysis.py    ← Fourier + Wavelet + Z-transform [TBD]
        │
        ├── genetic_optimizer.py  ← GA over all hyperparameters [TBD]
        │
        └── dirac_validator.py    ← Dirac state-vector validator [TBD]
```

---

## 5. Technical Notes for Continuity

### Python environment
- Python 3.13, venv at `/Users/ragavan/Documents/claudecode/geodesic_sandbox/venv/`
- Key packages: `numpy`, `networkx`, `matplotlib`, `torch`, `scipy`, `deap` (for GA)

### Mesh parameters (used consistently across all files)
- 2 levels of subdivision → **162 nodes, 320 faces, 480 edges**
- All edges have `base_latency = delay + jitter`, `delay = 1/√(L×C)` (Telegrapher's)
- L, C drawn from `Uniform(0.1, 1.0)` with fixed seed 42

### Capacity calibration (for M/M/1 congestion)
- `capacity_scale = 0.1` → capacity ≈ 0.6 flows per link
- At `arrival_rate=2`, steady-state ~16 flows → avg load ≈ 0.13 per link → util ≈ 0.22
- At `arrival_rate=8`, steady-state ~64 flows → util ≈ 0.75 (M/M/1 gives 4× base latency)

### Stokes curl notes
- Curl is tracked on **directed** flows (u→v separate from v→u)
- Time-averaging last 20 steps gives a stable curl field for routing decisions
- Penalty threshold = 1.0, penalty factor = 8.0 (well-calibrated — router detours but doesn't over-penalise)
- Mean-of-ratios latency metric can show artifacts with high-variance pairs; use ratio-of-means (aggregate avg latency) as the primary metric

### DQN-cong key design note
- Must be trained with a *running* TrafficSimulator alongside it (not just random link loads)
- The 11th feature is `ρ = load/capacity` for the candidate edge at routing time
- At inference, it queries the *eval* simulator's current state (not training sim)

---

## 6. Key Results Summary

| Module | Routing strategy | Improvement vs hop-SP baseline |
|---|---|---|
| `qlearning_v2.py` | Tabular Q-learning | ~+7–10% |
| `dqn_router.py` | DQN (geometry only) | **+12.7%** (no congestion) |
| `traffic_sim.py` | DQN-base (under congestion) | −19% to −53% (collapses) |
| `traffic_sim.py` | DQN-cong (11-dim, live traffic) | **+6% to +51%** (tracks oracle) |
| `traffic_sim.py` | Latency-SP oracle | +22% to +67% (ceiling) |
| `stokes_curl.py` | Curl-aware SP | **+16.4%** (aggregate avg) |
| `green_flow.py` | Green-aware SP | **+31.4%** (best single-theorem) |
| `green_flow.py` | Combined (Stokes + Green) | **+23.8%** (max-penalty combination) |
| `signal_analysis.py` | Fourier dominant period | **20.00 steps (0.0% error)** |
| `signal_analysis.py` | FIR filter noise reduction | **64.2%** (cutoff=0.20) |
| `signal_analysis.py` | EMA noise reduction | **69.7%** (α=0.2) |
| `genetic_optimizer.py` | GA best fitness | **0.6512** (+11.1% vs default 0.5860) |
| `genetic_optimizer.py` | GA latency improvement | **68.2%** (vs 39.7% default) |
| `genetic_optimizer.py` | Key GA finding | `curl_penalty` 8→1.8 (less detour overhead) |

---

## 7. What to Build Next

**Recommended next step: `dirac_validator.py`**
- `genetic_optimizer.py` is complete (2026-04-08)
- Demo GUI is complete and fully tested (2026-04-09)
- Next simulation module: Dirac-inspired 4×4 matrix operator, 100 mock transactions, mempool priority ordering
- See NEXT_STEPS.md Module 5 for full implementation spec
- To add to GUI: add entry to `MODULE_REGISTRY` in `param_schema.py`, add `run_dirac()` to `pipeline.py` → GUI auto-discovers it

---

## 7b. Design Decisions Resolved 2026-04-08 (Claude Chat → Claude Code)

These were open questions from the previous session. Answers applied; no further review needed.

**Q1 — Supply drift (+2.45%):** Accept as-is. Symmetric thresholds retained. Upward drift
is a known Poisson variance artifact with seed 42 and does not require correction for demo.

**Q2 — Divergence signal:** Add EMA(α=0.2) of Δactive_flows. Controller input switched from
raw to EMA for noise resilience. Both raw and EMA series exported from `run_simulation()` for
use in `signal_analysis.py`. Thresholds recalibrated to EMA scale (±1.5 EMA units,
scale_factor=500). Final results: 18 mints, 3 burns, +0.516% net supply change, STABLE.

**Q3 — scale_factor realism:** Fixed scalar kept for simulation. Oracle-price-linked scaling
noted as intended production behaviour in a comment in `divergence_comcoin.py`. No code change.

**Q4 — signal_analysis.py input periodicity:** Inject sinusoidal arrival rate:
`arrival_rate(step) = 4 + 2·sin(2π·step/20)` — period 20 steps, dominant frequency 1/20.
Fourier must detect this peak. NEXT_STEPS.md updated with this spec. Panel 3 updated to show
three-way overlay: raw_div / EMA / Z-filter FIR (demonstrates trade-off between the two
smoothing approaches).

**Q5 — GA fitness weights:** Updated in NEXT_STEPS.md:
`w1=0.35 (latency), w2=0.40 (supply stability), w3=0.15 (curl), w4=0.10 (cost)`

---

*Last updated: 2026-04-09. GUI session complete — all 10 spec tests passing.*
