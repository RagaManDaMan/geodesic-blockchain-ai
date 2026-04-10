# Geodesic Blockchain AI

A physics-inspired simulation of the patent:
*"Geodesic Blockchain AI for Optimized Distributed Ledger Consensus and Data Routing"*

Nodes arranged on a subdivided icosahedral mesh, routing optimised via deep reinforcement
learning, congestion managed with vector-calculus theorems (Stokes, Green, Divergence), and
all parameters tuned by a Genetic Algorithm. A web GUI drives the full pipeline interactively.

---

## Quick start

```bash
# 1. Clone / enter the project directory
cd geodesic_sandbox

# 2. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate          # macOS / Linux
# venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start the GUI server
python app.py

# 5. Open the dashboard
open http://localhost:7432
```

---

## Simulation modules

| File | Patent section | What it does |
|---|---|---|
| `geodesic_overlay.py` | §A — Mesh | Icosahedron → 2-level subdivision → 162-node NetworkX graph |
| `dqn_router.py` | §B — DQN | Deep Q-Network geometry router (+12.7% vs hop-shortest) |
| `traffic_sim.py` | §B — Traffic | M/M/1 congestion simulation; DQN-cong tracks oracle +6–51% |
| `stokes_curl.py` | §C — Stokes | Directed-flow curl mitigation (+16.4%) |
| `green_flow.py` | §D — Green | Node-divergence detection, Green-aware routing (+31.4%) |
| `divergence_comcoin.py` | §E — ComCoin | Divergence controller + elastic CCO token supply |
| `signal_analysis.py` | §F-G — Signal | Fourier, Wavelet, FIR/Z-transform parameter analysis |
| `genetic_optimizer.py` | §I — GA | DEAP genetic algorithm over 8 hyperparameters (+11.1% fitness) |

---

## GUI dashboard (`frontend/index.html`)

Single-page dark dashboard served by Flask at `http://localhost:7432`.

- **Parameter panel** — all 32 params, schema-driven, log-scale sliders for `fee_rate`
  and `learning_rate`
- **Module toggles** — enable/disable any module at runtime; disabled modules use cached
  results; `mesh` is always required
- **Estimate panel** — per-module runtime bars, cached/will-run/disabled status; warns and
  offers fast-param suggestions for runs projected >120 s
- **Run controls** — SSE streaming progress log with amber/green/red module tiles
- **Chart tabs** — lazy-loaded output PNGs for each module
- **Notes** — freeform run notes saved into `outputs/geodesic_results.json`
- **Load GA best** — applies `best_params.json` values back into the parameter sliders
- **Reset cache** — clears all cached module results

---

## API endpoints

| Method | Route | Description |
|---|---|---|
| `GET` | `/` | Serve dashboard |
| `GET` | `/api/schema` | All 32 params + module registry |
| `GET` | `/api/defaults` | Default parameter values |
| `GET` | `/api/status` | Cache manifest + module enabled/cached state |
| `POST` | `/api/estimate` | Runtime estimate + dirty-module list |
| `POST` | `/api/run` | SSE stream: runs pipeline, emits per-module progress |
| `GET` | `/api/results` | Latest `geodesic_results.json` |
| `GET` | `/api/results/download` | Download results as JSON attachment |
| `POST` | `/api/results/notes` | Update notes in results file |
| `GET` | `/api/best_params` | GA best params remapped to schema keys |
| `GET` | `/outputs/<file>` | Serve chart PNGs |
| `POST` | `/api/reset` | Clear cache |

---

## Project structure

```
geodesic_sandbox/
├── app.py                    Flask API server (port 7432)
├── param_schema.py           32-param schema + MODULE_REGISTRY + DEPENDENCY_CHAIN
├── cache_manager.py          Pickle cache per module; dirty-module propagation
├── time_estimator.py         Per-module runtime models; fast-params suggestion
├── pipeline.py               Module orchestration; writes outputs/geodesic_results.json
├── frontend/
│   └── index.html            Self-contained dark GUI (all CSS+JS inline)
├── outputs/
│   ├── geodesic_results.json Latest pipeline results
│   ├── best_params.json      GA-optimised hyperparameters
│   ├── cache/                Pickle files per module + manifest.json
│   └── *.png                 Module chart images
├── geodesic_overlay.py       Mesh module
├── dqn_router.py             DQN module
├── traffic_sim.py            Traffic simulation
├── stokes_curl.py            Stokes curl module
├── green_flow.py             Green flow module
├── divergence_comcoin.py     ComCoin divergence module
├── signal_analysis.py        Signal analysis module
├── genetic_optimizer.py      Genetic optimiser
├── requirements.txt
└── .gitignore
```

---

## Adding a new module

1. Write `my_module.py` with a `run_simulation()` or equivalent function.
2. Add an entry to `MODULE_REGISTRY` in `param_schema.py`:
   ```python
   "my_module": {"label": "My Module", "enabled": True, "required": False, "order": 9}
   ```
3. Add any new params to the `PARAMS` dict in `param_schema.py`.
4. Add `run_my_module()` to `pipeline.py` and append it to `MODULE_RUNNERS`.
5. Restart the server — the GUI auto-discovers the new module and params.

---

## Key results

| Module | Best result |
|---|---|
| DQN router | +12.7% latency vs hop-shortest |
| DQN-cong (traffic sim) | +51.1% at high load, tracks oracle |
| Stokes curl-aware routing | +16.4% |
| Green-aware routing | +31.4% (best single theorem) |
| FIR noise reduction | 64.2% (cutoff=0.20) |
| GA fitness improvement | +11.1% (0.5860 → 0.6512) |
| GA latency improvement | +28.5pp (39.7% → 68.2%) |
