"""
Local batch runner — equivalent to running the Colab notebook cells B–F.

Edit CONFIG and the RUN_* flags below, then run from the repo root:

    python run_local.py

Available strategy keys:
    "Zero"
    "Transport-Adapted Pricing"
    "Static Median"
    "Online Competitive"
    "Smooth Tail"
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from ExpandedTimeSimulation.simulation_zefat.colab_utils import run_experiment, plot_results
from ExpandedTimeSimulation.simulation_zefat.experiments.batch_run import run_batch_for_demand

# ---------------------------------------------------------------------------
# What to run
# ---------------------------------------------------------------------------
RUN_DEMAND = True   # True → run expected-demand baseline (run_batch_for_demand)
RUN_BATCH  = False    # True → run full auction batch (run_experiment)

# ---------------------------------------------------------------------------
# Shared network + vehicle parameters
# ---------------------------------------------------------------------------
CONFIG = {
    # ── Output ──────────────────────────────────────────────────────────────
    "excel_file":       "results.csv",      # .csv = per-sheet CSVs (resumable)
                                             # .xlsx = single workbook

    # ── Network ─────────────────────────────────────────────────────────────
    "place_name":       "Ashdod, Israel",
    "graph_file":       "ashdod.gpickle",
    "max_time_slots":   60,
    "slot_seconds":     60,

    # ── Vehicles ─────────────────────────────────────────────────────────────
    "od_count_start":   10000,
    "od_count_end":     20000,
    "od_count_step":    5000,

    "vehicle_T":        100,
    "peak_slot":        15,
    "peak_sigma_start": 5.0,
    "peak_sigma_end":   5.0,
    "peak_sigma_step":  1.0,

    "time_mode":        "both",
    "arrival_percentage": 0.5,

    "alpha_lo":         0.0,
    "alpha_hi":         1.0,
    "entry_fee_lo":     0.0,
    "entry_fee_hi":     1.0,
    "lateness_fee_lo":  0.0,
    "lateness_fee_hi":  1.0,
    "reserve_lo":       30.0,
    "reserve_hi":       100.0,

    # ── Pricing ──────────────────────────────────────────────────────────────
    "vmax":             100.0,
    "r":                30,
    "smooth_tail_u0":   0.95,
    "capacity_is_hourly": True,

    # Capacity factor sweep (percentage, 100 = no change)
    "cap_factor_start": 10.0,
    "cap_factor_end":   10.0,
    "cap_factor_step":  10.0,

    # ── Path solver ──────────────────────────────────────────────────────────
    # Options: dijkstra | bidirectional_dijkstra | astar_euclidean |
    #          astar_fflb | astar_fflb_delay | astar_reverse_arrival
    "path_solver":      "astar_reverse_arrival",

    # ── Strategies ───────────────────────────────────────────────────────────
    # None → all five strategies
    "strategy_keys":    ["Zero", "Transport-Adapted Pricing"],

    # ── Run control ──────────────────────────────────────────────────────────
    "num_runs":         1,
    "base_seed":        2025,
    "verbose":          False,
    "resume":           False,
    "big_roads_only":   False,

    # ── Used by batch run when an expected-demand file exists ─────────────────
    "demand_fraction":  0.7,
}

# ---------------------------------------------------------------------------
# Demand-baseline parameters (only used when RUN_DEMAND = True)
# ---------------------------------------------------------------------------
DEMAND_CONFIG = {
    "num_runs":         3,              # demand_runs from Colab
    "od_count":         50000,          # demand_num_vehicles from Colab
    "excel_file":       "demand.csv",
    "parquet_file":     None,           # e.g. "ashdod_demand.parquet"
    "base_seed":        CONFIG["base_seed"],
    "place_name":       CONFIG["place_name"],
    "graph_file":       CONFIG["graph_file"],
    "max_time_slots":   CONFIG["max_time_slots"],
    "vmax":             CONFIG["vmax"],
    "r":                CONFIG["r"],
    "slot_seconds":     CONFIG["slot_seconds"],
    "vehicle_T":        CONFIG["vehicle_T"],
    "peak_slot":        CONFIG["peak_slot"],
    "peak_sigma":       CONFIG["peak_sigma_start"],
    "verbose":          CONFIG["verbose"],
    "resume":           CONFIG["resume"],
    "big_roads_only":   CONFIG["big_roads_only"],
}

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if RUN_DEMAND:
        print("=== Running expected-demand baseline ===")
        run_batch_for_demand(**DEMAND_CONFIG)
        print(f"Demand results written to: {DEMAND_CONFIG['excel_file']}")

    if RUN_BATCH:
        print("=== Running auction batch ===")
        excel_file = run_experiment(CONFIG)
        print(f"\nResults written to: {excel_file}")

        # Uncomment to show diagnostic plots after the run:
        # plot_results(excel_file, CONFIG)
