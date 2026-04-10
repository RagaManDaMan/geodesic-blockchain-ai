"""
time_estimator.py — Runtime estimation before any simulation runs.

Models calibrated from M-series MacBook runs on 2026-04-08.
Returns (total_seconds, per_module_breakdown).

Fast-params suggestion: returns a reduced parameter set for runs
estimated to exceed 120 seconds.
"""

from typing import Dict, List, Tuple, Any

SLOW_THRESHOLD_SECONDS = 120   # warn and suggest fast params above this


def estimate_runtime(
    params: Dict[str, Any],
    dirty_modules: List[str],
    module_states: Dict[str, bool],
) -> Tuple[float, Dict[str, float]]:
    """Estimate wall-clock seconds for each dirty module.

    Disabled modules contribute 0 seconds (they use cache).
    Clean (cached) modules also contribute 0 seconds.
    Returns (total_seconds, {module: seconds}).
    """
    breakdown: Dict[str, float] = {}

    for module in dirty_modules:
        if not module_states.get(module, True):
            breakdown[module] = 0.0
            continue

        t = _module_estimate(module, params)
        breakdown[module] = t

    # Modules not in dirty_modules cost nothing (cached or disabled)
    total = sum(breakdown.values())
    return total, breakdown


def _module_estimate(module: str, p: Dict[str, Any]) -> float:
    """Per-module timing model."""

    if module == "mesh":
        # 0.05 * 4^subdivisions: sub=1→0.2s, 2→0.8s, 3→3.2s, 4→12.8s
        return 0.05 * (4 ** p.get("subdivisions", 2))

    elif module == "dqn":
        # (episodes/100) * 0.15 * (steps_per_ep/25)
        return (p.get("episodes", 800) / 100) * 0.15 * (p.get("steps_per_ep", 25) / 25)

    elif module == "traffic":
        # (sim_steps/100) * 0.8 * (1 + arrival_rate/10)
        return (p.get("sim_steps", 200) / 100) * 0.8 * (1 + p.get("arrival_rate", 4.0) / 10)

    elif module == "stokes":
        # (sim_steps/100) * 2.0
        return (p.get("sim_steps", 200) / 100) * 2.0

    elif module == "green":
        # (sim_steps/100) * 2.5
        return (p.get("sim_steps", 200) / 100) * 2.5

    elif module == "comcoin":
        # (sim_steps/100) * 1.5
        return (p.get("sim_steps", 200) / 100) * 1.5

    elif module == "signal":
        return 3.0   # FFT + wavelet on N=200 is fast

    elif module == "genetic":
        # ga_pop * ga_gen * 0.003 * (sim_steps/50)
        # 30*20*0.003*4 = 7.2s (matches observed ~6s)
        return (p.get("ga_population", 30)
                * p.get("ga_generations", 20)
                * 0.003
                * (p.get("sim_steps", 200) / 50))

    else:
        # Unknown future modules: 5s default placeholder
        return 5.0


def suggest_fast_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of params with expensive knobs turned down.

    Only overrides the four costly parameters; preserves all other
    user-customised values so the fast run is still representative.
    """
    fast = dict(params)
    fast["episodes"]       = min(fast.get("episodes",       800),  200)
    fast["sim_steps"]      = min(fast.get("sim_steps",      200),  100)
    fast["ga_population"]  = min(fast.get("ga_population",   30),   15)
    fast["ga_generations"] = min(fast.get("ga_generations",  20),   10)
    return fast


def build_estimate_response(
    params: Dict[str, Any],
    dirty_modules: List[str],
    cached_modules: List[str],
    module_states: Dict[str, bool],
) -> Dict[str, Any]:
    """Build the full dict returned by /api/estimate."""
    total, breakdown = estimate_runtime(params, dirty_modules, module_states)

    warning = None
    fast_params = None
    if total > SLOW_THRESHOLD_SECONDS:
        minutes = total / 60
        fast = suggest_fast_params(params)
        fast_total, _ = estimate_runtime(fast, dirty_modules, module_states)
        warning = (
            f"Estimated {minutes:.1f} min. "
            f"Consider fast params (episodes=200, sim_steps=100, "
            f"ga_population=15, ga_generations=10) → ~{fast_total:.0f}s."
        )
        fast_params = fast

    return {
        "dirty_modules":  dirty_modules,
        "cached_modules": cached_modules,
        "total_seconds":  round(total, 1),
        "breakdown":      {m: round(t, 2) for m, t in breakdown.items()},
        "warning":        warning,
        "fast_params":    fast_params,
    }
