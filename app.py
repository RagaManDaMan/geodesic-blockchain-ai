"""
app.py — Flask API server for the Geodesic Blockchain AI demo GUI.

Routes:
  GET  /                      → serves frontend/index.html
  GET  /api/schema            → PARAMS dict + MODULE_REGISTRY
  GET  /api/defaults          → default parameter values
  GET  /api/status            → cache manifest + module enabled states
  POST /api/estimate          → runtime estimate for given params/module_states
  POST /api/run               → SSE stream: runs pipeline, emits progress events
  GET  /api/results           → latest geodesic_results.json
  GET  /api/results/download  → geodesic_results.json as file attachment
  POST /api/results/notes     → update notes field in results
  GET  /api/best_params       → best_params.json remapped to param_schema keys
  GET  /outputs/<filename>    → serve chart PNGs
  POST /api/reset             → clear cache

Start: python app.py
URL:   http://localhost:7432
"""

import json
import os
import time
import traceback
from threading import Thread, Event
from queue import Queue, Empty

from flask import Flask, Response, jsonify, request, send_file, send_from_directory, stream_with_context

from param_schema import PARAMS, MODULE_REGISTRY, get_defaults, validate_params, get_module_states
from cache_manager import CacheManager
from time_estimator import build_estimate_response
from pipeline import run_pipeline, OUTPUTS_DIR, MODULE_ORDER

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False

CACHE = CacheManager()
RESULTS_FILE  = os.path.join(OUTPUTS_DIR, "geodesic_results.json")
BEST_PARAMS_FILE = os.path.join(OUTPUTS_DIR, "best_params.json")
FRONTEND_DIR  = os.path.join(os.path.dirname(__file__), "frontend")

# Gene names from genetic_optimizer → param_schema param names
# Only fields where the meaning maps cleanly; others are skipped.
_GENE_TO_PARAM = {
    "dqn_alpha":     "learning_rate",
    "dqn_gamma":     "gamma",
    "dqn_epsilon":   "epsilon_start",
    "curl_penalty":  "curl_penalty",
    "green_penalty": "sink_penalty",
    "fee_rate":      "fee_rate",
    "filter_cutoff": "fir_cutoff",
    # mint_scale: GA gene × 100 = scale_factor ≈ param_schema mint_scale (CCO/unit)
    "mint_scale":    "mint_scale",
}

PORT = 7432


# ---------------------------------------------------------------------------
# CORS helper (allow the frontend dev server on any origin)
# ---------------------------------------------------------------------------

def _cors(response):
    response.headers["Access-Control-Allow-Origin"]  = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


@app.after_request
def after_request(response):
    return _cors(response)


@app.route("/", defaults={"path": ""}, methods=["OPTIONS"])
@app.route("/<path:path>", methods=["OPTIONS"])
def options_handler(path):
    return _cors(Response("", status=200))


# ---------------------------------------------------------------------------
# GET /api/schema
# ---------------------------------------------------------------------------

@app.route("/api/schema")
def schema():
    """Return full parameter schema + module registry.

    Frontend uses this to build sliders/inputs and the module toggle panel.
    """
    return jsonify({
        "params":   PARAMS,
        "modules":  MODULE_REGISTRY,
        "order":    MODULE_ORDER,
    })


# ---------------------------------------------------------------------------
# GET /api/defaults
# ---------------------------------------------------------------------------

@app.route("/api/defaults")
def defaults():
    """Return flat dict of all default parameter values."""
    return jsonify(get_defaults())


# ---------------------------------------------------------------------------
# GET /api/status
# ---------------------------------------------------------------------------

@app.route("/api/status")
def status():
    """Return cache manifest and per-module enabled state."""
    manifest = CACHE.get_manifest()
    module_status = {}
    for m in MODULE_ORDER:
        module_status[m] = {
            "cached":     CACHE.is_cached(m),
            "cached_at":  manifest.get(m, {}).get("cached_at"),
            "summary":    manifest.get(m, {}).get("summary", ""),
            "enabled":    MODULE_REGISTRY[m]["enabled"],
            "required":   MODULE_REGISTRY[m]["required"],
            "label":      MODULE_REGISTRY[m]["label"],
        }
    return jsonify({"modules": module_status, "manifest": manifest})


# ---------------------------------------------------------------------------
# POST /api/estimate
# ---------------------------------------------------------------------------

@app.route("/api/estimate", methods=["POST"])
def estimate():
    """Estimate runtime and identify dirty modules for a given param set.

    Request body:
      {
        "params":        { ...param overrides... },
        "module_states": { "dqn": true, "signal": false, ... }   (optional)
      }
    """
    body          = request.get_json(force=True) or {}
    raw_params    = body.get("params", {})
    raw_states    = body.get("module_states", {})

    params        = validate_params(raw_params)
    module_states = get_module_states(raw_states)

    dirty   = CACHE.get_dirty_modules(params, module_states)
    cached  = [m for m in MODULE_ORDER
               if CACHE.is_cached(m) and m not in dirty]

    response_body = build_estimate_response(params, dirty, cached, module_states)
    return jsonify(response_body)


# ---------------------------------------------------------------------------
# POST /api/run  — SSE streaming endpoint
# ---------------------------------------------------------------------------

@app.route("/api/run", methods=["POST"])
def run():
    """Run the pipeline and stream progress as Server-Sent Events.

    Request body:
      {
        "params":        { ...param overrides... },
        "module_states": { "dqn": false, ... },   (optional)
        "notes":         "optional run notes"
      }

    SSE event format:
      data: {"module": "mesh", "status": "running",  "elapsed": 0.0}
      data: {"module": "mesh", "status": "complete", "elapsed": 0.8,
              "summary": "162 nodes, ..."}
      data: {"module": "dqn",  "status": "skipped",  "elapsed": 0.9,
              "summary": "disabled — using cached result"}
      ...
      data: {"status": "done", "results_file": "outputs/geodesic_results.json",
              "total_seconds": 12.4}
      data: {"status": "error", "module": "...", "message": "traceback..."}
    """
    body          = request.get_json(force=True) or {}
    raw_params    = body.get("params", {})
    raw_states    = body.get("module_states", {})
    notes         = body.get("notes", "")

    params        = validate_params(raw_params)
    module_states = get_module_states(raw_states)
    dirty         = CACHE.get_dirty_modules(params, module_states)

    # Use a queue to pass events from the pipeline thread to the SSE generator
    event_queue: Queue = Queue()
    done_event:  Event = Event()

    def progress_callback(module: str, status: str, elapsed: float, summary: str):
        event = {
            "module":  module,
            "status":  status,
            "elapsed": round(elapsed, 2),
            "summary": summary if status != "error" else "",
        }
        if status == "error":
            event["message"] = summary   # summary carries traceback on error
        event_queue.put(event)

    def pipeline_thread():
        try:
            result = run_pipeline(
                params, dirty, module_states, CACHE, progress_callback
            )
            # Patch notes field into the written JSON
            if notes:
                try:
                    with open(RESULTS_FILE) as f:
                        doc = json.load(f)
                    doc["notes"] = notes
                    with open(RESULTS_FILE, "w") as f:
                        json.dump(doc, f, indent=2)
                except Exception:
                    pass

            event_queue.put({
                "status":        "done",
                "results_file":  RESULTS_FILE,
                "total_seconds": result.get("total_runtime_seconds", 0),
            })
        except Exception as exc:
            event_queue.put({
                "status":  "error",
                "module":  "pipeline",
                "message": traceback.format_exc(),
            })
        finally:
            done_event.set()

    thread = Thread(target=pipeline_thread, daemon=True)
    thread.start()

    def generate():
        while True:
            try:
                event = event_queue.get(timeout=0.25)
                yield f"data: {json.dumps(event)}\n\n"
                if event.get("status") in ("done", "error") and event.get("module") is None:
                    break
            except Empty:
                # Keep-alive ping so the connection doesn't time out
                if done_event.is_set() and event_queue.empty():
                    break
                yield ": ping\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control":  "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# GET /api/results
# ---------------------------------------------------------------------------

@app.route("/api/results")
def results():
    """Return the latest geodesic_results.json."""
    if not os.path.exists(RESULTS_FILE):
        return jsonify({"error": "No results yet — run the pipeline first."}), 404
    with open(RESULTS_FILE) as f:
        return jsonify(json.load(f))


# ---------------------------------------------------------------------------
# POST /api/results/notes
# ---------------------------------------------------------------------------

@app.route("/api/results/notes", methods=["POST"])
def update_notes():
    """Update the notes field in the latest results file."""
    if not os.path.exists(RESULTS_FILE):
        return jsonify({"error": "No results file found."}), 404
    body = request.get_json(force=True) or {}
    notes = body.get("notes", "")
    with open(RESULTS_FILE) as f:
        doc = json.load(f)
    doc["notes"] = notes
    with open(RESULTS_FILE, "w") as f:
        json.dump(doc, f, indent=2)
    return jsonify({"ok": True, "notes": notes})


# ---------------------------------------------------------------------------
# GET /outputs/<filename>
# ---------------------------------------------------------------------------

@app.route("/outputs/<path:filename>")
def serve_output(filename):
    """Serve chart PNGs and JSON from the outputs directory."""
    return send_from_directory(OUTPUTS_DIR, filename)


# ---------------------------------------------------------------------------
# GET /  — serve frontend
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    """Serve the single-page frontend."""
    return send_from_directory(FRONTEND_DIR, "index.html")

@app.route("/<path:filename>")
def frontend_static(filename):
    """Serve static assets from the frontend directory (JS, etc.)."""
    return send_from_directory(FRONTEND_DIR, filename)


# ---------------------------------------------------------------------------
# GET /api/best_params
# ---------------------------------------------------------------------------

@app.route("/api/best_params")
def best_params():
    """Return best_params.json remapped to param_schema keys.

    If no best_params.json exists yet (genetic module hasn't run),
    returns the schema defaults so the frontend can still function.
    """
    from param_schema import PARAMS, get_defaults
    defaults = get_defaults()

    if not os.path.exists(BEST_PARAMS_FILE):
        return jsonify({"source": "defaults", "params": defaults})

    with open(BEST_PARAMS_FILE) as f:
        bp = json.load(f)

    gene_params = bp.get("parameters", {})
    mapped: dict = {}
    for gene_name, param_name in _GENE_TO_PARAM.items():
        if gene_name not in gene_params or param_name not in PARAMS:
            continue
        raw = gene_params[gene_name]
        # mint_scale: gene value is in GA range (1–50); multiply ×100 for CCO/unit
        if gene_name == "mint_scale":
            raw = raw * 100.0
        spec = PARAMS[param_name]
        # Clamp to schema bounds
        if spec["type"] == "int":
            raw = int(round(float(raw)))
        else:
            raw = float(raw)
        raw = max(spec["min"], min(spec["max"], raw))
        mapped[param_name] = raw

    return jsonify({
        "source":          "best_params.json",
        "fitness":         bp.get("fitness"),
        "improvement_pct": bp.get("improvement_pct"),
        "params":          mapped,
    })


# ---------------------------------------------------------------------------
# GET /api/results/download
# ---------------------------------------------------------------------------

@app.route("/api/results/download")
def download_results():
    """Return geodesic_results.json as a file download attachment."""
    if not os.path.exists(RESULTS_FILE):
        return jsonify({"error": "No results yet."}), 404
    return send_file(
        RESULTS_FILE,
        mimetype="application/json",
        as_attachment=True,
        download_name="geodesic_results.json",
    )


# ---------------------------------------------------------------------------
# POST /api/reset
# ---------------------------------------------------------------------------

@app.route("/api/reset", methods=["POST"])
def reset():
    """Clear all cached module results and reset to defaults."""
    CACHE.clear()
    if os.path.exists(RESULTS_FILE):
        os.remove(RESULTS_FILE)
    return jsonify({"ok": True, "message": "Cache cleared."})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    print(f"\n{'='*50}")
    print(f"  Geodesic demo running at http://localhost:{PORT}")
    print(f"  outputs/ → {OUTPUTS_DIR}")
    print(f"{'='*50}\n")
    app.run(host="0.0.0.0", port=PORT, debug=False, threaded=True)
