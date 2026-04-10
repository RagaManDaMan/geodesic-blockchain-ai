"""
cache_manager.py — Pickle-based module result cache with dirty detection.

Each module's result is stored as:
  outputs/cache/{module}.pkl   — pickled result dict
  outputs/cache/manifest.json  — records params used + timestamp per module

A module is "dirty" (needs re-running) if:
  1. Its cache file does not exist, OR
  2. Any of its own parameters differ from what's in the manifest, OR
  3. Any upstream module is dirty (propagated via DEPENDENCY_CHAIN)

When a module is disabled, it is never dirty — the pipeline always uses
its cached result regardless of param changes.
"""

import json
import os
import pickle
import time
from typing import Any, Dict, List, Optional

from param_schema import DEPENDENCY_CHAIN, MODULE_ORDER, PARAMS

CACHE_DIR   = os.path.join(os.path.dirname(__file__), "outputs", "cache")
MANIFEST    = os.path.join(CACHE_DIR, "manifest.json")


class CacheManager:

    def __init__(self):
        os.makedirs(CACHE_DIR, exist_ok=True)
        self._manifest: Dict[str, Any] = self._load_manifest()

    # ------------------------------------------------------------------
    # Manifest helpers
    # ------------------------------------------------------------------

    def _load_manifest(self) -> Dict[str, Any]:
        if os.path.exists(MANIFEST):
            try:
                with open(MANIFEST) as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                pass
        return {}

    def _save_manifest(self):
        with open(MANIFEST, "w") as f:
            json.dump(self._manifest, f, indent=2)

    def get_manifest(self) -> Dict[str, Any]:
        """Return current manifest (for /api/status)."""
        return dict(self._manifest)

    # ------------------------------------------------------------------
    # Dirty detection
    # ------------------------------------------------------------------

    def _params_for_module(self, module: str, all_params: Dict[str, Any]) -> Dict[str, Any]:
        """Extract only the params that belong to this module."""
        return {
            name: all_params[name]
            for name, spec in PARAMS.items()
            if spec["module"] == module and name in all_params
        }

    def _module_self_dirty(self, module: str, current_params: Dict[str, Any]) -> bool:
        """True if this module's own params changed or its cache file is absent."""
        pkl_path = os.path.join(CACHE_DIR, f"{module}.pkl")
        if not os.path.exists(pkl_path):
            return True
        if module not in self._manifest:
            return True
        cached_params = self._manifest[module].get("params_used", {})
        current_module_params = self._params_for_module(module, current_params)
        return cached_params != current_module_params

    def get_dirty_modules(
        self,
        current_params: Dict[str, Any],
        module_states: Dict[str, bool],
    ) -> List[str]:
        """Return ordered list of modules that need re-running.

        Disabled modules are never dirty — they always use cache.
        Dirty state propagates downstream via DEPENDENCY_CHAIN.
        """
        dirty: set = set()
        for module in MODULE_ORDER:
            if not module_states.get(module, True):
                # Disabled → never dirty, skip propagation
                continue
            # Module is dirty if it's self-dirty OR any upstream is dirty
            upstream_dirty = any(
                up in dirty
                for up in MODULE_ORDER[:MODULE_ORDER.index(module)]
                if module in DEPENDENCY_CHAIN.get(up, [])
            )
            if self._module_self_dirty(module, current_params) or upstream_dirty:
                dirty.add(module)

        # Return in pipeline execution order
        return [m for m in MODULE_ORDER if m in dirty]

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def save_module(
        self,
        module: str,
        data: Dict[str, Any],
        params_used: Dict[str, Any],
    ):
        """Pickle result data and update manifest."""
        pkl_path = os.path.join(CACHE_DIR, f"{module}.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump(data, f)

        self._manifest[module] = {
            "params_used":   self._params_for_module(module, params_used),
            "cached_at":     time.strftime("%Y-%m-%dT%H:%M:%S"),
            "summary":       data.get("summary", ""),
        }
        self._save_manifest()

    def load_module(self, module: str) -> Optional[Dict[str, Any]]:
        """Load pickled result for module.  Returns None if not cached."""
        pkl_path = os.path.join(CACHE_DIR, f"{module}.pkl")
        if not os.path.exists(pkl_path):
            return None
        try:
            with open(pkl_path, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None

    def is_cached(self, module: str) -> bool:
        pkl_path = os.path.join(CACHE_DIR, f"{module}.pkl")
        return os.path.exists(pkl_path)

    def clear(self):
        """Delete all cache files and reset manifest."""
        for module in MODULE_ORDER:
            pkl_path = os.path.join(CACHE_DIR, f"{module}.pkl")
            if os.path.exists(pkl_path):
                os.remove(pkl_path)
        self._manifest = {}
        self._save_manifest()
