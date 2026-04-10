"""
param_schema.py — Single source of truth for all tunable parameters.

Every parameter has: default, min, max, type, module, label, unit.
Frontend reads this to build controls; backend validates inputs against it.

MODULE_REGISTRY drives the enable/disable feature.  Each module can be
toggled at runtime.  When disabled, the pipeline skips running it and
uses its cached result instead.  Adding a new module (e.g. dirac) only
requires a new entry here plus a run_* function in pipeline.py.
"""

from typing import Any, Dict

# ---------------------------------------------------------------------------
# Parameter definitions
# ---------------------------------------------------------------------------

PARAMS: Dict[str, Dict[str, Any]] = {
    # ── MESH ──────────────────────────────────────────────────────────────
    "subdivisions": {
        "default": 2,    "min": 1,    "max": 4,     "type": "int",
        "module": "mesh",    "label": "Subdivision levels",    "unit": "",
    },
    "seed": {
        "default": 42,   "min": 0,    "max": 999,   "type": "int",
        "module": "mesh",    "label": "Random seed",           "unit": "",
    },

    # ── DQN ───────────────────────────────────────────────────────────────
    "learning_rate": {
        "default": 0.001,  "min": 0.0001, "max": 0.1,  "type": "float",
        "module": "dqn",     "label": "Learning rate (α)",     "unit": "",
    },
    "gamma": {
        "default": 0.95,   "min": 0.8,    "max": 0.99, "type": "float",
        "module": "dqn",     "label": "Discount factor (γ)",   "unit": "",
    },
    "epsilon_start": {
        "default": 0.4,    "min": 0.1,    "max": 0.8,  "type": "float",
        "module": "dqn",     "label": "Initial exploration ε", "unit": "",
    },
    "episodes": {
        "default": 800,    "min": 100,    "max": 2000, "type": "int",
        "module": "dqn",     "label": "Training episodes",     "unit": "",
    },
    "steps_per_ep": {
        "default": 25,     "min": 10,     "max": 50,   "type": "int",
        "module": "dqn",     "label": "Steps per episode",     "unit": "",
    },

    # ── TRAFFIC ───────────────────────────────────────────────────────────
    "arrival_rate": {
        "default": 4.0,    "min": 0.5,    "max": 10.0, "type": "float",
        "module": "traffic", "label": "Flow arrival rate",     "unit": "flows/step",
    },
    "sim_steps": {
        "default": 200,    "min": 50,     "max": 500,  "type": "int",
        "module": "traffic", "label": "Simulation steps",      "unit": "steps",
    },
    "capacity_scale": {
        "default": 0.1,    "min": 0.05,   "max": 0.3,  "type": "float",
        "module": "traffic", "label": "Link capacity scale",   "unit": "",
    },

    # ── STOKES ────────────────────────────────────────────────────────────
    "curl_threshold": {
        "default": 1.0,    "min": 0.5,    "max": 3.0,  "type": "float",
        "module": "stokes",  "label": "Curl threshold",        "unit": "",
    },
    "curl_penalty": {
        "default": 1.84,   "min": 0.5,    "max": 20.0, "type": "float",
        "module": "stokes",  "label": "Curl penalty factor",   "unit": "",
    },

    # ── GREEN ─────────────────────────────────────────────────────────────
    "div_threshold": {
        "default": 0.5,    "min": 0.1,    "max": 2.0,  "type": "float",
        "module": "green",   "label": "Divergence threshold",  "unit": "",
    },
    "sink_penalty": {
        "default": 5.0,    "min": 1.0,    "max": 15.0, "type": "float",
        "module": "green",   "label": "Sink node penalty",     "unit": "",
    },
    "source_penalty": {
        "default": 3.0,    "min": 1.0,    "max": 10.0, "type": "float",
        "module": "green",   "label": "Source node penalty",   "unit": "",
    },

    # ── COMCOIN ───────────────────────────────────────────────────────────
    "initial_supply": {
        "default": 1000000, "min": 100000, "max": 10000000, "type": "int",
        "module": "comcoin", "label": "Initial supply (CCO)",  "unit": "CCO",
    },
    "fee_rate": {
        "default": 1.157e-5, "min": 1e-7, "max": 1e-4, "type": "float",
        "module": "comcoin", "label": "Fee rate/step",         "unit": "/step",
    },
    "ema_alpha": {
        "default": 0.2,    "min": 0.05,   "max": 0.5,  "type": "float",
        "module": "comcoin", "label": "EMA smoothing α",       "unit": "",
    },
    "mint_threshold": {
        "default": 1.5,    "min": 0.5,    "max": 5.0,  "type": "float",
        "module": "comcoin", "label": "Mint threshold",        "unit": "EMA units",
    },
    "burn_threshold": {
        "default": -1.5,   "min": -5.0,   "max": -0.5, "type": "float",
        "module": "comcoin", "label": "Burn threshold",        "unit": "EMA units",
    },
    "mint_scale": {
        "default": 500,    "min": 50,     "max": 5000, "type": "int",
        "module": "comcoin", "label": "Mint scale factor",     "unit": "CCO/unit",
    },

    # ── SIGNAL ────────────────────────────────────────────────────────────
    "arrival_period": {
        "default": 20,     "min": 5,      "max": 50,   "type": "int",
        "module": "signal",  "label": "Arrival rate period",   "unit": "steps",
    },
    "arrival_amp": {
        "default": 2.0,    "min": 0.0,    "max": 4.0,  "type": "float",
        "module": "signal",  "label": "Arrival rate amplitude","unit": "flows",
    },
    "fir_cutoff": {
        "default": 0.20,   "min": 0.05,   "max": 0.45, "type": "float",
        "module": "signal",  "label": "FIR filter cutoff",     "unit": "×Nyquist",
    },

    # ── GENETIC ───────────────────────────────────────────────────────────
    "ga_population": {
        "default": 30,     "min": 10,     "max": 100,  "type": "int",
        "module": "genetic", "label": "GA population size",    "unit": "",
    },
    "ga_generations": {
        "default": 20,     "min": 5,      "max": 50,   "type": "int",
        "module": "genetic", "label": "GA generations",        "unit": "",
    },
    "ga_cx_prob": {
        "default": 0.5,    "min": 0.2,    "max": 0.8,  "type": "float",
        "module": "genetic", "label": "Crossover probability", "unit": "",
    },
    "ga_mut_prob": {
        "default": 0.2,    "min": 0.05,   "max": 0.5,  "type": "float",
        "module": "genetic", "label": "Mutation probability",  "unit": "",
    },
    "ga_w_latency": {
        "default": 0.35,   "min": 0.0,    "max": 1.0,  "type": "float",
        "module": "genetic", "label": "Fitness: latency wt",   "unit": "",
    },
    "ga_w_stability": {
        "default": 0.40,   "min": 0.0,    "max": 1.0,  "type": "float",
        "module": "genetic", "label": "Fitness: stability wt", "unit": "",
    },
    "ga_w_curl": {
        "default": 0.15,   "min": 0.0,    "max": 1.0,  "type": "float",
        "module": "genetic", "label": "Fitness: curl wt",      "unit": "",
    },
    "ga_w_cost": {
        "default": 0.10,   "min": 0.0,    "max": 1.0,  "type": "float",
        "module": "genetic", "label": "Fitness: cost wt",      "unit": "",
    },
}

# ---------------------------------------------------------------------------
# Module registry — controls enable/disable at runtime
# ---------------------------------------------------------------------------
# 'required': True  → cannot be disabled (everything depends on mesh)
# 'enabled':  True  → default on state
# New modules (e.g. dirac) drop in here; no other file needs to change.

MODULE_REGISTRY: Dict[str, Dict[str, Any]] = {
    "mesh":    {"label": "Geodesic Mesh",      "enabled": True,  "required": True,  "order": 0},
    "dqn":     {"label": "DQN Router",         "enabled": True,  "required": False, "order": 1},
    "traffic": {"label": "Traffic Sim",        "enabled": True,  "required": False, "order": 2},
    "stokes":  {"label": "Stokes Curl",        "enabled": True,  "required": False, "order": 3},
    "green":   {"label": "Green Flow",         "enabled": True,  "required": False, "order": 4},
    "comcoin": {"label": "ComCoin",            "enabled": True,  "required": False, "order": 5},
    "signal":  {"label": "Signal Analysis",    "enabled": True,  "required": False, "order": 6},
    "genetic": {"label": "Genetic Optimizer",  "enabled": True,  "required": False, "order": 7},
    # "dirac":   {"label": "Dirac Validator",   "enabled": False, "required": False, "order": 8},
}

# Ordered list for pipeline execution
MODULE_ORDER = sorted(MODULE_REGISTRY.keys(), key=lambda m: MODULE_REGISTRY[m]["order"])

# ---------------------------------------------------------------------------
# Dependency chain — changing an upstream module dirties all downstream ones
# ---------------------------------------------------------------------------

DEPENDENCY_CHAIN: Dict[str, list] = {
    "mesh":    ["mesh", "dqn", "traffic", "stokes", "green", "comcoin", "signal", "genetic"],
    "dqn":     ["dqn", "traffic", "stokes", "green", "comcoin", "signal", "genetic"],
    "traffic": ["traffic", "stokes", "green", "comcoin", "signal", "genetic"],
    "stokes":  ["stokes", "green", "comcoin", "signal", "genetic"],
    "green":   ["green", "comcoin", "signal", "genetic"],
    "comcoin": ["comcoin", "signal", "genetic"],
    "signal":  ["signal", "genetic"],
    "genetic": ["genetic"],
}


def get_defaults() -> Dict[str, Any]:
    """Return a flat dict of all parameter default values."""
    return {name: spec["default"] for name, spec in PARAMS.items()}


def validate_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and coerce params against schema.  Returns cleaned dict.

    Unknown keys are ignored.  Missing keys are filled with defaults.
    Values are clamped to [min, max] and cast to the declared type.
    """
    result = get_defaults()
    for name, value in params.items():
        if name not in PARAMS:
            continue
        spec = PARAMS[name]
        try:
            if spec["type"] == "int":
                value = int(round(float(value)))
            else:
                value = float(value)
            value = max(spec["min"], min(spec["max"], value))
        except (TypeError, ValueError):
            value = spec["default"]
        result[name] = value
    return result


def get_module_states(requested: Dict[str, bool]) -> Dict[str, bool]:
    """Merge requested module states with registry defaults.

    Required modules (mesh) are always forced to True regardless of request.
    Unknown module names are ignored.
    """
    states: Dict[str, bool] = {
        m: info["enabled"] for m, info in MODULE_REGISTRY.items()
    }
    for module, enabled in requested.items():
        if module in MODULE_REGISTRY:
            if MODULE_REGISTRY[module]["required"]:
                states[module] = True   # cannot be disabled
            else:
                states[module] = bool(enabled)
    return states
