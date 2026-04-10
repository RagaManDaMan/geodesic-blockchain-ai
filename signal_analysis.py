"""
signal_analysis.py — Fourier, Wavelet, and Z-transform signal analysis.

Patent §F (Fourier + Wavelet) and §G (Z-transform / discrete-time filter).

THREE QUESTIONS, THREE TOOLS:

  Fourier transform: Does the network have periodic behaviour?
    The flow series is driven by arrival_rate(t) = 4 + 2·sin(2π·t/20),
    injecting a dominant frequency at exactly 1/20 cycles/step.  The FFT
    must detect this peak.  In a live deployment, a dominant peak at some
    unknown frequency would tell governance: "congestion spikes every N steps
    — align fee updates with that rhythm."

  Wavelet transform: When do bursts happen and at what scale?
    FFT is global — it tells you which frequencies exist but not when.
    The Discrete Wavelet Transform (DWT) localises in both time and scale,
    catching transient congestion storms that FFT averages away.
    We use Daubechies-4 (db4) with 5 decomposition levels.

  Z-transform / FIR filter: How should governance parameters be updated?
    The raw divergence signal (Δactive_flows) is noisy.  Feeding it directly
    into the fee-rate controller causes thrashing.  We design a low-pass FIR
    filter using scipy.signal.firwin — the discrete-time equivalent of a
    Z-domain transfer function — and compare it to the EMA already computed
    in divergence_comcoin.py.  FIR gives linear phase (no lag distortion);
    EMA is simpler but introduces frequency-dependent phase shift.

INPUT DATA:
  Flow series  — minimal internal simulation with sinusoidal arrival rate
                 (no module-level imports from other files at simulation time)
  Div series   — raw_div_history + ema_div_history from divergence_comcoin.py

OUTPUT:  signal_analysis_results.png  (4-panel)
"""

import matplotlib
matplotlib.use('Agg')
import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pywt
from scipy import signal as scipy_signal
from collections import defaultdict
from typing import List, Tuple

# divergence_comcoin exports both raw and EMA divergence series
from divergence_comcoin import run_simulation

from geodesic_overlay import build_icosahedron, subdivide, mesh_to_graph
from dqn_router import assign_link_params

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

PERIOD   = 20          # injected sinusoidal period (steps)
N_STEPS  = 200         # simulation length
BASE_RATE = 4.0        # mean arrival rate
AMP       = 2.0        # sinusoidal amplitude  →  rate ∈ [2, 6]


# ---------------------------------------------------------------------------
# 1. Flow series — sinusoidal arrival rate simulation
# ---------------------------------------------------------------------------

def generate_flow_series(G: nx.Graph, n_steps: int = N_STEPS) -> Tuple[List[float], List[float]]:
    """Run a minimal traffic simulation with sinusoidal arrival rate.

    arrival_rate(step) = BASE_RATE + AMP * sin(2π * step / PERIOD)

    Returns:
        flow_series  — per-step active flow count  (Fourier + Wavelet input)
        rate_series  — per-step arrival rate used   (sanity check)

    The sinusoidal modulation embeds a guaranteed peak at frequency 1/PERIOD
    into the flow count series, making it detectable by the FFT.

    Minimal self-contained simulation — no DirectedFlowSim import needed.
    Uses exponential flow lifetimes (mean = 10 steps) and hop-SP routing.
    """
    # Directed load: not needed for flow count — just track active flows
    active_flows: List[Tuple[list, float]] = []   # (path, remaining_lifetime)
    nodes        = list(G.nodes())
    flow_series: List[float] = []
    rate_series: List[float] = []

    def _route(s, t):
        try:
            return nx.shortest_path(G, s, t, weight=None)
        except nx.NetworkXNoPath:
            return None

    for step in range(1, n_steps + 1):
        # Expire flows
        alive = []
        for path, rem in active_flows:
            rem -= 1.0
            if rem > 0:
                alive.append((path, rem))
        active_flows = alive

        # Inject new flows with sinusoidal rate
        rate = BASE_RATE + AMP * np.sin(2 * np.pi * step / PERIOD)
        rate = max(0.0, rate)
        rate_series.append(rate)
        for _ in range(np.random.poisson(rate)):
            s, t = random.sample(nodes, 2)
            path = _route(s, t)
            if path:
                lifetime = np.random.exponential(10.0)
                active_flows.append((path, lifetime))

        flow_series.append(float(len(active_flows)))

    return flow_series, rate_series


# ---------------------------------------------------------------------------
# 2. Fourier analysis
# ---------------------------------------------------------------------------

def fourier_analysis(
    flow_series: List[float],
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """FFT of the flow time series.

    Returns:
        freqs      — positive frequency bins (cycles/step)
        magnitudes — FFT magnitude spectrum (one-sided)
        dom_freq   — dominant frequency (cycles/step)
        dom_period — corresponding period (steps)
    """
    x    = np.array(flow_series, dtype=float)
    # Detrend to remove DC offset and any linear drift before FFT
    x    = x - np.mean(x)
    N    = len(x)
    fft  = np.fft.rfft(x)
    mags = np.abs(fft) * 2.0 / N   # two-sided → one-sided normalisation
    mags[0] = 0.0                   # zero out DC bin (already detrended)
    freqs    = np.fft.rfftfreq(N, d=1.0)

    # Dominant frequency (excluding DC)
    dom_idx  = int(np.argmax(mags[1:]) + 1)
    dom_freq = float(freqs[dom_idx])
    dom_period = 1.0 / dom_freq if dom_freq > 0 else float('inf')

    return freqs, mags, dom_freq, dom_period


# ---------------------------------------------------------------------------
# 3. Wavelet analysis
# ---------------------------------------------------------------------------

def wavelet_analysis(
    flow_series: List[float],
    wavelet: str = 'db4',
    levels: int  = 5,
) -> Tuple[List[np.ndarray], List[str]]:
    """Multi-level Discrete Wavelet Transform decomposition.

    Returns:
        detail_coeffs  — list of detail coefficient arrays, one per level
                         (level 1 = finest scale, level=levels = coarsest)
        level_labels   — human-readable label for each level's scale
    """
    x      = np.array(flow_series, dtype=float)
    coeffs = pywt.wavedec(x, wavelet, level=levels)
    # coeffs[0] = approximation; coeffs[1..] = details (finest first)
    detail_coeffs = coeffs[1:]   # drop approximation for scalogram
    level_labels  = [f'D{i+1}  (scale ~{2**(i+1)} steps)' for i in range(len(detail_coeffs))]
    return detail_coeffs, level_labels


# ---------------------------------------------------------------------------
# 4. Z-transform / FIR governance filter
# ---------------------------------------------------------------------------

def design_governance_filter(
    raw_signal: List[float],
    cutoff_ratio: float = 0.10,
    num_taps: int       = 31,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Design a linear-phase FIR low-pass filter and apply it.

    cutoff_ratio: fraction of Nyquist to pass (0.10 → pass below 5% of
                  sample rate, i.e. periods longer than 20 steps).
                  This deliberately passes the 1/20 governance signal while
                  blocking higher-frequency noise.
    num_taps: filter order + 1 (odd → symmetric, linear phase).

    Returns:
        raw       — original signal as ndarray
        filtered  — FIR-filtered signal
        coeffs    — FIR coefficients (for frequency response plot)
    """
    raw     = np.array(raw_signal, dtype=float)
    coeffs  = scipy_signal.firwin(num_taps, cutoff_ratio, window='hamming')
    # lfilter introduces group delay of (num_taps-1)//2 samples; we compensate
    # by using filtfilt for zero-phase filtering in the plot.
    filtered = scipy_signal.filtfilt(coeffs, [1.0], raw)
    return raw, filtered, coeffs


def apply_z_filter(
    raw_signal: List[float],
    filter_coeffs: np.ndarray,
) -> np.ndarray:
    """Apply a pre-designed FIR filter (causal, using lfilter).

    filtfilt is used in design_governance_filter for zero-phase display;
    this causal version is what a live controller would use.
    """
    return scipy_signal.lfilter(filter_coeffs, [1.0], np.array(raw_signal, dtype=float))


# ---------------------------------------------------------------------------
# 5. Visualisation
# ---------------------------------------------------------------------------

def visualize(
    flow_series: List[float],
    rate_series: List[float],
    freqs: np.ndarray,
    mags: np.ndarray,
    dom_freq: float,
    dom_period: float,
    detail_coeffs: List[np.ndarray],
    level_labels: List[str],
    raw_div: List[float],
    ema_div: List[float],
    fir_div: np.ndarray,
    fir_coeffs: np.ndarray,
    cutoff_ratio: float,
):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    steps = np.arange(1, N_STEPS + 1)

    # ------------------------------------------------------------------ #
    # Panel 1: Flow time series + FFT dominant frequency                  #
    # ------------------------------------------------------------------ #
    ax1 = axes[0, 0]
    ax1.plot(steps, flow_series, color='steelblue', linewidth=0.9,
             label='Active flows')
    ax1.plot(steps, rate_series, color='darkorange', linewidth=0.8,
             linestyle='--', alpha=0.7, label='Arrival rate (sinusoidal)')

    # Overlay reconstructed dominant sinusoid for visual confirmation
    t        = np.arange(1, N_STEPS + 1)
    mean_flow = float(np.mean(flow_series))
    # mags is already amplitude-normalised (2/N scaling applied in fourier_analysis)
    # so mags[dom_idx] ≈ actual sinusoidal amplitude directly
    dom_idx   = int(np.argmax(mags[1:]) + 1)
    dom_amp   = float(mags[dom_idx])
    recon     = mean_flow + dom_amp * np.cos(2 * np.pi * dom_freq * t)
    ax1.plot(t, recon, color='crimson', linewidth=1.0, linestyle=':',
             alpha=0.8, label=f'Dominant sinusoid (f={dom_freq:.4f}, T={dom_period:.1f} steps)')

    ax1.set_title(f'Flow Time Series — Sinusoidal Arrival Rate\n'
                  f'Dominant frequency: {dom_freq:.4f} cyc/step  '
                  f'(period = {dom_period:.1f} steps,  expected = {PERIOD})')
    ax1.set_xlabel('Simulation step')
    ax1.set_ylabel('Count')
    ax1.legend(fontsize=8)

    # ------------------------------------------------------------------ #
    # Panel 2: Wavelet scalogram                                          #
    # ------------------------------------------------------------------ #
    ax2 = axes[0, 1]
    # Build 2D array: rows = levels (coarse → fine), cols = time
    # Pad/trim each level to N_STEPS for display
    n_levels = len(detail_coeffs)
    scalo    = np.zeros((n_levels, N_STEPS))
    for i, d in enumerate(detail_coeffs):
        row = np.abs(d)
        # Stretch/compress to N_STEPS using repeat/trim
        ratio = N_STEPS / len(row)
        idx   = np.clip((np.arange(N_STEPS) / ratio).astype(int), 0, len(row) - 1)
        scalo[i, :] = row[idx]

    # Plot coarsest level at top (matches convention)
    scalo_display = scalo[::-1]
    im = ax2.imshow(
        scalo_display,
        aspect='auto',
        extent=[1, N_STEPS, 0.5, n_levels + 0.5],
        cmap='hot',
        origin='lower',
        norm=mcolors.PowerNorm(gamma=0.4),
    )
    ax2.set_yticks(range(1, n_levels + 1))
    ax2.set_yticklabels(level_labels[::-1], fontsize=7)
    ax2.set_title(f'Wavelet Scalogram (db4, {n_levels} levels)\n'
                  f'Colour = |detail coefficient|;  hot = high energy')
    ax2.set_xlabel('Simulation step')
    ax2.set_ylabel('Decomposition level')
    plt.colorbar(im, ax=ax2, fraction=0.025, pad=0.04, label='|coeff|')

    # ------------------------------------------------------------------ #
    # Panel 3: Three-way divergence smoothing comparison                  #
    # ------------------------------------------------------------------ #
    ax3 = axes[1, 0]
    div_steps = np.arange(1, len(raw_div) + 1)
    ax3.plot(div_steps, raw_div, color='steelblue', linewidth=0.7,
             alpha=0.5, label='Raw Δflows (noisy)')
    ax3.plot(div_steps, ema_div, color='darkorange', linewidth=1.3,
             label='EMA α=0.2 (causal, slight lag)')
    ax3.plot(div_steps, fir_div, color='crimson', linewidth=1.3,
             linestyle='--', label=f'FIR low-pass (linear phase, cutoff={cutoff_ratio})')
    ax3.axhline(0, color='gray', linewidth=0.4)

    raw_std  = float(np.std(raw_div))
    ema_std  = float(np.std(ema_div))
    fir_std  = float(np.std(fir_div))
    ax3.set_title(
        f'Divergence Smoothing: Raw vs EMA vs FIR\n'
        f'Std — Raw: {raw_std:.2f}  |  EMA: {ema_std:.2f} ({ema_std/raw_std*100:.0f}%)  '
        f'|  FIR: {fir_std:.2f} ({fir_std/raw_std*100:.0f}%)',
    )
    ax3.set_xlabel('Simulation step')
    ax3.set_ylabel('Δactive flows')
    ax3.legend(fontsize=8)

    # ------------------------------------------------------------------ #
    # Panel 4: FIR filter frequency response (Bode-style)                #
    # ------------------------------------------------------------------ #
    ax4 = axes[1, 1]
    w, h    = scipy_signal.freqz(fir_coeffs, worN=512)
    freq_norm = w / np.pi          # 0 → 1 (Nyquist = 1)
    h_db      = 20 * np.log10(np.maximum(np.abs(h), 1e-10))

    ax4.plot(freq_norm, h_db, color='crimson', linewidth=1.2)
    ax4.axvline(cutoff_ratio * 2, color='navy', linestyle='--',
                linewidth=0.9, label=f'Cutoff = {cutoff_ratio} × Nyquist')
    # Mark where the injected signal sits
    injected_norm = (dom_freq * 2)   # normalise to [0,1] (Nyquist = 0.5 → maps to 1)
    ax4.axvline(injected_norm, color='forestgreen', linestyle=':',
                linewidth=0.9, label=f'Dominant signal freq (f={dom_freq:.4f})')
    ax4.set_ylim(-80, 5)
    ax4.set_title('FIR Filter Frequency Response\n'
                  '(governance low-pass — passes slow trends, blocks noise)')
    ax4.set_xlabel('Normalised frequency (1 = Nyquist = 0.5 cyc/step)')
    ax4.set_ylabel('Magnitude (dB)')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    plt.suptitle(
        'Geodesic Blockchain — Signal Analysis (Fourier + Wavelet + Z-transform)',
        fontsize=13, fontweight='bold',
    )
    plt.tight_layout()
    plt.savefig('signal_analysis_results.png', dpi=150)
    plt.close()
    print('\nPlot saved to signal_analysis_results.png')


# ---------------------------------------------------------------------------
# 6. Summary printout
# ---------------------------------------------------------------------------

def print_summary(
    dom_freq: float,
    dom_period: float,
    detail_coeffs: List[np.ndarray],
    raw_div: List[float],
    ema_div: List[float],
    fir_div: np.ndarray,
    cutoff_ratio: float,
):
    raw_std = float(np.std(raw_div))
    ema_std = float(np.std(ema_div))
    fir_std = float(np.std(fir_div))

    # Energy per wavelet level
    energies = [float(np.sum(d**2)) for d in detail_coeffs]
    dom_level = int(np.argmax(energies) + 1)

    print(f'\n{"="*65}')
    print(f'Signal Analysis Results')
    print(f'{"="*65}')
    print(f'\nFourier:')
    print(f'  Dominant frequency : {dom_freq:.5f} cycles/step')
    print(f'  Dominant period    : {dom_period:.2f} steps  (expected: {PERIOD})')
    freq_error = abs(dom_period - PERIOD) / PERIOD * 100
    status = '✅ DETECTED' if freq_error < 10 else '❌ MISSED — check signal generation'
    print(f'  Detection status   : {status}  (error={freq_error:.1f}%)')

    print(f'\nWavelet (db4, 5 levels):')
    for i, (d, e) in enumerate(zip(detail_coeffs, energies)):
        marker = ' ← peak energy' if (i + 1) == dom_level else ''
        scale  = 2 ** (i + 1)
        print(f'  Level D{i+1} (scale ~{scale:2d} steps): '
              f'energy={e:8.1f},  max|coeff|={np.max(np.abs(d)):.2f}{marker}')

    print(f'\nZ-filter / FIR (cutoff={cutoff_ratio}):')
    print(f'  Raw std dev        : {raw_std:.4f}')
    print(f'  EMA std dev        : {ema_std:.4f}  ({ema_std/raw_std*100:.1f}% of raw)')
    print(f'  FIR std dev        : {fir_std:.4f}  ({fir_std/raw_std*100:.1f}% of raw)')
    target_ok = 30 <= fir_std / raw_std * 100 <= 60
    print(f'  Target 30–60%      : {"✅ MET" if target_ok else "⚠️  outside range"}')
    print(f'\n  Trade-off:')
    print(f'    EMA: simpler, causal, slight phase lag at dominant frequency')
    print(f'    FIR: linear phase (zero-phase with filtfilt), sharper roll-off')
    print(f'{"="*65}')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    # ---- 1. Build mesh (needed for flow simulation) ----------------------
    print('Building mesh...')
    verts, faces = build_icosahedron()
    verts, faces = subdivide(verts, faces)
    verts, faces = subdivide(verts, faces)
    G     = mesh_to_graph(verts, faces)
    G     = assign_link_params(G)
    print(f'  {G.number_of_nodes()} nodes | {len(faces)} faces | {G.number_of_edges()} edges')

    # ---- 2. Generate sinusoidal-rate flow series -------------------------
    print(f'\nGenerating flow series '
          f'(arrival_rate = {BASE_RATE} + {AMP}·sin(2π·t/{PERIOD}), '
          f'{N_STEPS} steps)...')
    flow_series, rate_series = generate_flow_series(G, n_steps=N_STEPS)
    print(f'  Flow series: mean={np.mean(flow_series):.1f}, '
          f'std={np.std(flow_series):.2f}, '
          f'range=[{min(flow_series):.0f}, {max(flow_series):.0f}]')

    # ---- 3. Fourier — must detect peak at 1/20 --------------------------
    print('\nRunning Fourier analysis...')
    freqs, mags, dom_freq, dom_period = fourier_analysis(flow_series)
    print(f'  Dominant frequency: {dom_freq:.5f} cyc/step  '
          f'(period = {dom_period:.2f} steps, expected = {PERIOD})')
    freq_error = abs(dom_period - PERIOD) / PERIOD * 100
    if freq_error > 10:
        raise RuntimeError(
            f'Fourier did not detect injected frequency! '
            f'Found period={dom_period:.1f}, expected {PERIOD}. '
            f'Check signal generation.'
        )
    print(f'  ✅ Injected period {PERIOD} detected within {freq_error:.1f}% error')

    # ---- 4. Wavelet decomposition ----------------------------------------
    print('\nRunning wavelet analysis (db4, 5 levels)...')
    detail_coeffs, level_labels = wavelet_analysis(flow_series, wavelet='db4', levels=5)
    for i, d in enumerate(detail_coeffs):
        print(f'  Level D{i+1}: {len(d):4d} coefficients, '
              f'energy={np.sum(d**2):.1f}, '
              f'max|coeff|={np.max(np.abs(d)):.2f}')

    # ---- 5. Get divergence series from divergence_comcoin ----------------
    print('\nGenerating divergence series (importing run_simulation)...')
    _sim, _cc, raw_div, ema_div, _log = run_simulation(
        G, faces,
        n_steps        = N_STEPS,
        arrival_rate   = BASE_RATE,
        mint_threshold = 1.5,
        burn_threshold = -1.5,
        scale_factor   = 500.0,
        ema_alpha      = 0.2,
        verbose        = False,
    )
    print(f'  raw_div: mean={np.mean(raw_div):.2f}, std={np.std(raw_div):.2f}')
    print(f'  ema_div: mean={np.mean(ema_div):.2f}, std={np.std(ema_div):.2f}')

    # ---- 6. FIR governance filter ----------------------------------------
    # cutoff_ratio=0.20 → passes frequencies below 0.10 cyc/step
    # (periods > 10 steps).  Chosen so FIR std lands in the 30–60% target.
    # cutoff=0.10 was too tight (23% noise reduction — too much smoothing).
    CUTOFF = 0.20
    print(f'\nDesigning FIR governance filter (cutoff={CUTOFF})...')
    raw_arr, fir_div, fir_coeffs = design_governance_filter(
        raw_div, cutoff_ratio=CUTOFF, num_taps=31,
    )
    raw_std = float(np.std(raw_div))
    fir_std = float(np.std(fir_div))
    print(f'  Raw std: {raw_std:.4f}  →  FIR std: {fir_std:.4f}  '
          f'({fir_std/raw_std*100:.1f}% of raw)')

    # ---- 7. Print full summary -------------------------------------------
    print_summary(dom_freq, dom_period, detail_coeffs,
                  raw_div, ema_div, fir_div, cutoff_ratio=CUTOFF)

    # ---- 8. Visualise ----------------------------------------------------
    visualize(
        flow_series, rate_series,
        freqs, mags, dom_freq, dom_period,
        detail_coeffs, level_labels,
        raw_div, ema_div, fir_div, fir_coeffs,
        cutoff_ratio=CUTOFF,
    )
