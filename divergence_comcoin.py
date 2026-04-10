"""
divergence_comcoin.py — Divergence Theorem + ComCoin elastic token supply.

Patent §E: "∯ F·dS = ∭ div F dV
            Total flux diverging out of the network volume equals the net
            source/sink inside it.  Use this as the control signal for
            minting or burning ComCoin."

INTUITION (no maths background needed):
  Green's Theorem (green_flow.py) looked at individual nodes — is THIS node
  a source or a sink?  The Divergence Theorem zooms all the way out and asks
  the same question about the ENTIRE network at once:

      total_divergence = Σ div(v)  over all nodes v

  If total_divergence > 0: more transaction traffic is being generated than
  is being completed (routed to destination).  The network is under-supplied.
  Response: MINT ComCoin to add liquidity.

  If total_divergence < 0: transactions are completing faster than new ones
  arrive.  The network is over-supplied.  Response: BURN ComCoin.

  Near zero: supply is in equilibrium with demand — do nothing.

  This is what makes ComCoin's supply elastic to network demand, unlike
  Bitcoin's fixed supply or a manually adjusted stablecoin.

COMCOIN MECHANICS (Python mock of the Solidity ERC-20 contract):
  • Time-decay fee: balance burns at 0.01%/day on every transfer
  • Redemption window: only days 22–25 of each 30-day cycle
  • Minimum lot: 1000 CCO per redemption
  • Merkle proof of reserve: mocked with a hash of the supply figure
  • Chainlink oracle: mocked as a random walk around $80/barrel

OUTPUT:  divergence_comcoin_results.png  (4-panel plot)
METRICS: mint/burn event counts, supply volatility, calibration verdict.
"""

import matplotlib
matplotlib.use('Agg')
import numpy as np
import random
import hashlib
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from geodesic_overlay import build_icosahedron, subdivide, mesh_to_graph
from dqn_router import assign_link_params
from green_flow import GreenFlowSim

random.seed(42)
np.random.seed(42)


# ---------------------------------------------------------------------------
# ComCoin mock contract
# ---------------------------------------------------------------------------

class ComCoinMock:
    """Python mock of the ComCoin ERC-20 contract described in the patent.

    All monetary amounts are in CCO (ComCoin units).
    Time is measured in simulation steps (1 step ≈ 1 second for fee purposes,
    but the 30-day redemption cycle is scaled to 30 simulation steps so we
    actually see a full cycle in our 200-step run).
    """

    CYCLE_LENGTH = 30          # steps per redemption cycle
    REDEMPTION_START = 22      # first day of window (inclusive)
    REDEMPTION_END   = 25      # last  day of window (inclusive)

    def __init__(self, initial_supply: float = 1_000_000.0):
        self.supply              = initial_supply
        self.initial_supply      = initial_supply
        self.fee_rate_per_step   = 1.157e-5      # ~1%/day — scaled so fees are visible in 200 steps
        self.min_lot             = 1_000.0
        self._price              = 80.0           # USD / barrel (oracle seed)
        self._price_rng          = np.random.RandomState(99)

        # address → balance
        self.balances: Dict[str, float] = {}
        # address → last step at which fees were accrued
        self.last_fee_step: Dict[str, int] = {}

        # event log: list of dicts
        self.event_log: List[Dict] = []

        # supply history (one entry per simulation step)
        self.supply_history: List[float] = []

        # fee accrual history (total CCO burned by fees per step)
        self.fee_history: List[float] = []

        # oracle price history
        self.price_history: List[float] = []

    # --- internal helpers ---

    def _log(self, step: int, event: str, amount: float, reason: str):
        self.event_log.append({
            'step': step, 'event': event,
            'amount': amount, 'reason': reason,
            'supply_after': self.supply,
        })

    # --- public contract interface ---

    def accrue_fees(self, address: str, current_step: int) -> float:
        """Burn time-decay fees on the holder's balance.

        fee = balance × fee_rate_per_step × (current_step − last_fee_step)

        Called implicitly on every transfer.  Returns the fee amount burned.
        """
        if address not in self.balances or self.balances[address] <= 0:
            return 0.0
        last = self.last_fee_step.get(address, current_step)
        dt   = max(0, current_step - last)
        if dt == 0:
            return 0.0
        fee = self.balances[address] * self.fee_rate_per_step * dt
        fee = min(fee, self.balances[address])   # can't burn more than balance
        self.balances[address]   -= fee
        self.supply              -= fee
        self.last_fee_step[address] = current_step
        return fee

    def mint(self, amount: float, step: int, reason: str = "divergence") -> float:
        """Increase total supply.  Returns actual amount minted."""
        amount = max(0.0, amount)
        self.supply += amount
        self._log(step, 'mint', amount, reason)
        return amount

    def burn(self, amount: float, step: int, reason: str = "divergence") -> float:
        """Decrease total supply (capped at current supply).  Returns actual burn."""
        amount = min(max(0.0, amount), self.supply)
        self.supply -= amount
        self._log(step, 'burn', amount, reason)
        return amount

    def redeem(self, address: str, amount: float, current_step: int) -> Tuple[bool, str]:
        """Redeem CCO for the underlying commodity.

        Enforces:
          • Redemption window: step mod CYCLE_LENGTH must be in [22, 25]
          • Minimum lot: amount >= min_lot
          • Sufficient balance

        Returns (success, message).
        """
        day_in_cycle = (current_step % self.CYCLE_LENGTH) + 1   # 1-indexed
        if not (self.REDEMPTION_START <= day_in_cycle <= self.REDEMPTION_END):
            return False, (f"Outside redemption window "
                           f"(day {day_in_cycle} of {self.CYCLE_LENGTH}-step cycle; "
                           f"window is days {self.REDEMPTION_START}–{self.REDEMPTION_END})")
        if amount < self.min_lot:
            return False, f"Below minimum lot ({amount:.0f} < {self.min_lot:.0f} CCO)"
        # Accrue fees first (transfer event)
        self.accrue_fees(address, current_step)
        bal = self.balances.get(address, 0.0)
        if amount > bal:
            return False, f"Insufficient balance ({bal:.2f} CCO < {amount:.0f} requested)"
        self.balances[address] -= amount
        self.supply            -= amount
        usd_value = amount * self.oracle_price() / 1000.0   # 1000 CCO = 1 barrel
        self._log(current_step, 'redeem', amount,
                  f"redemption by {address} — ${usd_value:.2f} USD")
        return True, f"Redeemed {amount:.0f} CCO → ${usd_value:.2f} USD"

    def oracle_price(self) -> float:
        """Mock Chainlink price feed — random walk around $80/barrel."""
        shock = self._price_rng.normal(0, 0.5)
        self._price = max(50.0, min(120.0, self._price + shock))
        return self._price

    def tick(self, step: int):
        """Call at the end of each simulation step to record state snapshots."""
        self.supply_history.append(self.supply)
        self.oracle_price()                     # advance price random walk each step
        self.price_history.append(self._price)

    def get_supply_history(self) -> List[float]:
        return self.supply_history

    def merkle_reserve_proof(self) -> str:
        """Simplified reserve proof — hash of supply rounded to nearest CCO."""
        payload = f"supply={round(self.supply)}".encode()
        return hashlib.sha256(payload).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Divergence controller
# ---------------------------------------------------------------------------

def divergence_controller(
    network_divergence: float,
    comcoin: ComCoinMock,
    step: int,
    mint_threshold: float  =  5.0,
    burn_threshold: float  = -5.0,
    scale_factor: float    = 10.0,
) -> Tuple[str, float]:
    """Elastic supply controller driven by global network divergence.

    network_divergence: EMA-smoothed Δactive_flows signal
    Positive → network generating more than routing → mint
    Negative → network routing more than generating → burn
    |divergence| < threshold → do nothing (dead-band to prevent thrashing)

    scale_factor: CCO minted/burned per unit of excess divergence.

    PRODUCTION NOTE: In a live deployment scale_factor should be tied to the
    Chainlink oracle price so that each divergence unit triggers minting of
    exactly enough CCO to cover one barrel equivalent at current market price
    (e.g. scale_factor = oracle_price() * barrels_per_div_unit / cco_per_barrel).
    The fixed scalar used here is appropriate for simulation only.

    Returns (action, amount) where action ∈ {'mint', 'burn', 'hold'}.
    """
    excess = network_divergence - mint_threshold
    deficit = burn_threshold - network_divergence   # positive when div < burn_threshold

    if network_divergence > mint_threshold:
        amount = excess * scale_factor
        comcoin.mint(amount, step, reason=f"net_div={network_divergence:+.2f}")
        return 'mint', amount

    elif network_divergence < burn_threshold:
        amount = deficit * scale_factor
        comcoin.burn(amount, step, reason=f"net_div={network_divergence:+.2f}")
        return 'burn', amount

    return 'hold', 0.0


# ---------------------------------------------------------------------------
# Full simulation
# ---------------------------------------------------------------------------

def run_simulation(
    G: nx.Graph,
    faces: list,
    n_steps: int        = 200,
    arrival_rate: float = 4.0,
    mint_threshold: float =  5.0,
    burn_threshold: float = -5.0,
    scale_factor: float   = 10.0,
    ema_alpha: float      = 0.2,
    verbose: bool       = True,
) -> Tuple[GreenFlowSim, ComCoinMock, List[float], List[float], List[Tuple[str, float]]]:
    """Run the full network + ComCoin simulation.

    Network divergence signal:
        raw_div_step = new_flows_injected − flows_expired   (per step)

    Σ div(v) across all nodes is always exactly 0 by flow conservation
    (every unit leaving u arrives at some v — a conservation law).
    The correct Divergence Theorem analogue is therefore the *rate of change*
    of the total active-flow volume:
        raw_div = d(active_flows)/dt ≈ new − expired  per step

    The EMA-smoothed version (α=0.2) is used as the controller input.
    Both raw and EMA series are returned for use in signal_analysis.py.

    Positive → backlog growing, network under-supplied → mint
    Negative → backlog draining, demand falling → burn

    Returns:
        sim              — GreenFlowSim after n_steps
        comcoin          — ComCoinMock with full event log
        raw_div_history  — per-step raw divergence signal (new − expired)
        ema_div_history  — EMA-smoothed divergence (α=ema_alpha); controller input
        action_log       — per-step (action, amount) from controller
    """
    sim     = GreenFlowSim(G, faces, arrival_rate=arrival_rate,
                           flow_duration=10.0, capacity_scale=0.5)
    comcoin = ComCoinMock(initial_supply=1_000_000)

    # Seed some balances so time-decay fees are visible from step 1
    addresses = [f"node_{i:03d}" for i in range(20)]
    for addr in addresses:
        comcoin.balances[addr]      = 10_000.0
        comcoin.last_fee_step[addr] = 0

    raw_div_history: List[float] = []
    ema_div_history: List[float] = []
    action_log: List[Tuple[str, float]] = []
    fee_step_totals: List[float] = []
    nodes = list(G.nodes())
    ema_val = 0.0   # EMA initialised to 0 (no backlog at step 0)

    print(f"\nRunning ComCoin simulation: {n_steps} steps, "
          f"arrival_rate={arrival_rate:.1f}")
    print(f"  mint_threshold={mint_threshold:+.1f}, "
          f"burn_threshold={burn_threshold:+.1f}, "
          f"scale_factor={scale_factor:.1f}, "
          f"ema_alpha={ema_alpha}")
    print(f"  Divergence signal: EMA(α={ema_alpha}) of Δactive_flows; "
          f"both raw and EMA exported for signal_analysis.py")

    for step in range(1, n_steps + 1):
        # 1. Advance one step, tracking injection / expiration counts
        sim.time_step += 1

        n_before_expire = len(sim.active_flows)
        sim._expire(1.0)
        n_expired = n_before_expire - len(sim.active_flows)

        n_before_inject = len(sim.active_flows)
        for _ in range(np.random.poisson(arrival_rate)):
            s, t = random.sample(nodes, 2)
            path = sim._route_sp(s, t)
            if path:
                sim._inject(path, np.random.exponential(sim.flow_duration))
        n_injected = len(sim.active_flows) - n_before_inject

        curl    = sim.compute_curl()
        sim.curl_history.append(curl)
        div_map = sim._compute_divergence_now()
        sim.div_history.append(div_map)

        # 2. Raw divergence: Δactive_flows (rate of change of backlog)
        raw_div = float(n_injected - n_expired)
        raw_div_history.append(raw_div)

        # 3. EMA-smoothed divergence: used as controller input and Z-filter baseline
        #    EMA_t = α * raw_t + (1 − α) * EMA_{t-1}
        ema_val = ema_alpha * raw_div + (1.0 - ema_alpha) * ema_val
        ema_div_history.append(ema_val)

        # 4. Divergence controller uses EMA signal (more stable than raw)
        action, amount = divergence_controller(
            ema_val, comcoin, step,
            mint_threshold, burn_threshold, scale_factor,
        )
        action_log.append((action, amount))

        # 5. Accrue time-decay fees on all accounts
        step_fees = sum(comcoin.accrue_fees(addr, step) for addr in addresses)
        fee_step_totals.append(step_fees)

        # 6. Snapshot ComCoin state
        comcoin.tick(step)

        if verbose and step % 40 == 0:
            mint_events = sum(1 for a, _ in action_log if a == 'mint')
            burn_events = sum(1 for a, _ in action_log if a == 'burn')
            print(f"  Step {step:3d}: "
                  f"flows={len(sim.active_flows):3d}, "
                  f"raw_div={raw_div:+.1f}, ema_div={ema_val:+.2f}, "
                  f"supply={comcoin.supply:,.0f}, "
                  f"mints={mint_events}, burns={burn_events}, "
                  f"fees={step_fees:.2f}")

    comcoin.fee_history = fee_step_totals
    return sim, comcoin, raw_div_history, ema_div_history, action_log


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def visualize(
    comcoin: ComCoinMock,
    raw_div_history: List[float],
    ema_div_history: List[float],
    action_log: List[Tuple[str, float]],
    n_steps: int,
):
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    steps = np.arange(1, n_steps + 1)

    # --- Panel 1: Network divergence over time (raw + EMA) ---
    ax1 = axes[0, 0]
    div_arr = np.array(raw_div_history)
    ema_arr = np.array(ema_div_history)
    ax1.plot(steps, div_arr, color='steelblue', linewidth=0.8, alpha=0.5, label='Raw Δflows')
    ax1.plot(steps, ema_arr, color='navy',      linewidth=1.4, label='EMA (α=0.2) — controller input')
    ax1.axhline(1.5,  color='forestgreen', linestyle='--', linewidth=0.8,
                label='Mint threshold (+1.5)')
    ax1.axhline(-1.5, color='crimson',     linestyle='--', linewidth=0.8,
                label='Burn threshold (−1.5)')
    # Shade based on EMA (the actual controller input)
    ax1.fill_between(steps, ema_arr, 0,
                     where=(ema_arr >  1.5), alpha=0.25, color='forestgreen')
    ax1.fill_between(steps, ema_arr, 0,
                     where=(ema_arr < -1.5), alpha=0.25, color='crimson')
    ax1.axhline(0, color='gray', linewidth=0.5)
    ax1.set_title('Network Divergence Signal\n'
                  '(Raw Δflows + EMA(α=0.2) smoothing; shaded = controller fires)')
    ax1.set_xlabel('Simulation step')
    ax1.set_ylabel('Δactive flows')
    ax1.legend(fontsize=8)

    # --- Panel 2: ComCoin supply over time ---
    ax2 = axes[0, 1]
    supply = np.array(comcoin.supply_history)
    ax2.plot(steps, supply, color='darkorange', linewidth=1.2, label='Supply')
    ax2.axhline(comcoin.initial_supply, color='gray', linestyle=':', linewidth=0.8,
                label=f'Initial supply ({comcoin.initial_supply:,.0f})')

    # Shade redemption windows
    for cycle_start in range(0, n_steps, ComCoinMock.CYCLE_LENGTH):
        ws = cycle_start + ComCoinMock.REDEMPTION_START
        we = cycle_start + ComCoinMock.REDEMPTION_END + 1
        ax2.axvspan(min(ws, n_steps), min(we, n_steps),
                    alpha=0.08, color='purple', label='_')
    ax2.set_title('ComCoin Supply Over Time\n'
                  '(mint/burn response to divergence; purple = redemption windows)')
    ax2.set_xlabel('Simulation step')
    ax2.set_ylabel('CCO supply')
    # Use a proxy artist for the redemption-window shading in the legend
    redemption_proxy = mpatches.Rectangle((0, 0), 1, 1,
                                           facecolor='purple', alpha=0.15,
                                           label='Redemption window')
    handles, labels = ax2.get_legend_handles_labels()
    ax2.legend(handles=handles + [redemption_proxy],
               labels=labels + ['Redemption window'], fontsize=8)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x/1e6:.3f}M'))

    # --- Panel 3: Mint/burn event bar chart ---
    ax3 = axes[1, 0]
    mint_steps   = [e['step']   for e in comcoin.event_log if e['event'] == 'mint']
    mint_amounts = [e['amount'] for e in comcoin.event_log if e['event'] == 'mint']
    burn_steps   = [e['step']   for e in comcoin.event_log if e['event'] == 'burn']
    burn_amounts = [e['amount'] for e in comcoin.event_log if e['event'] == 'burn']

    if mint_steps:
        ax3.bar(mint_steps, mint_amounts,  color='forestgreen', alpha=0.75,
                label=f'Mint ({len(mint_steps)} events)', width=1.2)
    if burn_steps:
        ax3.bar(burn_steps, [-a for a in burn_amounts], color='crimson', alpha=0.75,
                label=f'Burn ({len(burn_steps)} events)', width=1.2)

    ax3.axhline(0, color='black', linewidth=0.5)
    total_minted = sum(mint_amounts) if mint_amounts else 0
    total_burned = sum(burn_amounts) if burn_amounts else 0
    net_change   = total_minted - total_burned
    ax3.set_title(f'Mint / Burn Events\n'
                  f'Minted: {total_minted:,.0f} CCO  |  '
                  f'Burned: {total_burned:,.0f} CCO  |  '
                  f'Net: {net_change:+,.0f} CCO')
    ax3.set_xlabel('Simulation step')
    ax3.set_ylabel('CCO amount  (+ mint, − burn)')
    ax3.legend(fontsize=8)
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:,.0f}'))

    # --- Panel 4: Fee accrual over time ---
    ax4 = axes[1, 1]
    fee_arr = np.array(comcoin.fee_history)
    ax4.plot(steps, fee_arr, color='purple', linewidth=1.0, label='Fees burned (step)')
    ax4.fill_between(steps, fee_arr, alpha=0.15, color='purple')

    # Cumulative fees on a twin axis
    ax4b = ax4.twinx()
    cumfee = np.cumsum(fee_arr)
    ax4b.plot(steps, cumfee, color='indigo', linestyle='--',
              linewidth=0.9, alpha=0.7, label='Cumulative fees')
    ax4b.set_ylabel('Cumulative fees burned (CCO)', color='indigo')
    ax4b.tick_params(axis='y', labelcolor='indigo')

    total_fees = float(np.sum(fee_arr))
    ax4.set_title(f'Time-Decay Fee Accrual\n'
                  f'Total burned by fees: {total_fees:,.1f} CCO '
                  f'({total_fees / comcoin.initial_supply * 100:.3f}% of initial supply)')
    ax4.set_xlabel('Simulation step')
    ax4.set_ylabel('CCO burned (per step)', color='purple')
    ax4.tick_params(axis='y', labelcolor='purple')
    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4b.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, fontsize=8)

    plt.suptitle(
        "Geodesic Blockchain — Divergence Theorem + ComCoin Elastic Supply",
        fontsize=13, fontweight='bold',
    )
    plt.tight_layout()
    plt.savefig('divergence_comcoin_results.png', dpi=150)
    plt.close()
    print('\nPlot saved to divergence_comcoin_results.png')


# ---------------------------------------------------------------------------
# Summary printout
# ---------------------------------------------------------------------------

def print_summary(
    comcoin: ComCoinMock,
    raw_div_history: List[float],
    ema_div_history: List[float],
    action_log: List[Tuple[str, float]],
    n_steps: int,
):
    supply_arr = np.array(comcoin.supply_history)
    div_arr    = np.array(raw_div_history)
    ema_arr    = np.array(ema_div_history)

    mint_events  = [(s, a) for s, (act, a) in enumerate(action_log, 1) if act == 'mint']
    burn_events  = [(s, a) for s, (act, a) in enumerate(action_log, 1) if act == 'burn']
    total_minted = sum(a for _, a in mint_events)
    total_burned = sum(a for _, a in burn_events)
    total_fees   = sum(comcoin.fee_history)
    net_supply_change = comcoin.supply - comcoin.initial_supply
    supply_std_pct    = np.std(supply_arr) / comcoin.initial_supply * 100

    # Redemption demo
    addr = "node_000"
    ok, msg = comcoin.redeem(addr, 5_000, current_step=23)   # day 23 ∈ window
    ok2, msg2 = comcoin.redeem(addr, 5_000, current_step=10) # day 11 ∉ window

    print(f'\n{"="*65}')
    print(f'ComCoin Simulation Results — {n_steps} steps, arrival_rate=4')
    print(f'{"="*65}')
    print(f'  Mint events       : {len(mint_events)}')
    print(f'  Burn events       : {len(burn_events)}')
    print(f'  Total minted      : {total_minted:>12,.1f} CCO')
    print(f'  Total burned      : {total_burned:>12,.1f} CCO')
    print(f'  Burned by fees    : {total_fees:>12,.1f} CCO')
    print(f'  Net supply change : {net_supply_change:>+12,.1f} CCO  '
          f'({net_supply_change/comcoin.initial_supply*100:+.3f}%)')
    print(f'  Supply std dev    : {np.std(supply_arr):>12,.1f} CCO  '
          f'({supply_std_pct:.3f}% of initial)')
    print(f'  Supply range      : [{supply_arr.min():,.1f}, {supply_arr.max():,.1f}]')
    print(f'  Mean |raw_div|    : {np.mean(np.abs(div_arr)):.3f}')
    print(f'  Max  |raw_div|    : {np.max(np.abs(div_arr)):.3f}')
    print(f'  Mean |ema_div|    : {np.mean(np.abs(ema_arr)):.3f}')
    print(f'  Max  |ema_div|    : {np.max(np.abs(ema_arr)):.3f}')
    print(f'  Oracle price range: ${min(comcoin.price_history):.2f} – '
          f'${max(comcoin.price_history):.2f}')
    print(f'\n  Redemption demo:')
    print(f'    Step 23 (day 23, IN window):  {"OK" if ok else "FAIL"} — {msg}')
    print(f'    Step 10 (day 11, OUT window): {"OK" if ok2 else "FAIL"} — {msg2}')
    print(f'\n  Merkle reserve proof (current): {comcoin.merkle_reserve_proof()}')

    # Calibration verdict
    print(f'\n  Calibration verdict:')
    if supply_std_pct < 1.0:
        verdict = "STABLE — supply oscillates < 1% around initial value"
    elif supply_std_pct < 5.0:
        verdict = "MODERATE — supply fluctuates 1–5% (mint/burn thresholds may need tuning)"
    else:
        verdict = "UNSTABLE — supply drift > 5% (reduce scale_factor or tighten thresholds)"
    print(f'    {verdict}')
    print(f'{"="*65}')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # 1. Build geodesic mesh (same parameters throughout the project)
    print("Building mesh...")
    verts, faces = build_icosahedron()
    verts, faces = subdivide(verts, faces)
    verts, faces = subdivide(verts, faces)
    G     = mesh_to_graph(verts, faces)
    G     = assign_link_params(G)
    verts = np.array(verts)
    print(f"  {len(verts)} nodes | {len(faces)} faces | {G.number_of_edges()} edges")

    # 2. Run the combined network + ComCoin simulation
    #
    # Parameter rationale (revised 2026-04-08):
    #   Original params (threshold=±5, scale=10) produced only 8 controller events
    #   totalling 120 CCO — 0.012% of 1M supply, invisible against fee burn.
    #   Fix: lower threshold to ±3 (fires on ~40% of steps where |Δflows|>3)
    #   and raise scale_factor to 1000 (1000 CCO per excess unit of divergence).
    #   Controller input is now EMA(α=0.2) of raw Δflows for noise resilience.
    # Threshold calibrated to EMA signal scale (not raw signal scale).
    # Raw Δflows: mean|div|=2.3, max=9.  EMA(0.2): mean|ema|≈0.8, max≈3.2.
    # threshold=1.5 fires when EMA exceeds ~2× its mean (≈30-40% of steps).
    # scale_factor=500 → ~250 CCO per average event, ~15k CCO total (~1.5% supply).
    sim, comcoin, raw_div_history, ema_div_history, action_log = run_simulation(
        G, faces,
        n_steps         = 200,
        arrival_rate    = 4.0,
        mint_threshold  =  1.5,   # calibrated to EMA scale (was 3.0 for raw)
        burn_threshold  = -1.5,
        scale_factor    = 500.0,  # 500 CCO per excess EMA unit
        ema_alpha       = 0.2,
        verbose         = True,
    )

    # 3. Print summary stats and redemption demos
    print_summary(comcoin, raw_div_history, ema_div_history, action_log, n_steps=200)

    # 4. Visualise
    visualize(comcoin, raw_div_history, ema_div_history, action_log, n_steps=200)
