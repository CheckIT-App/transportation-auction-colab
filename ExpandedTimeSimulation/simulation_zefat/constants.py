# ExpandedTimeSimulation/simulation_zefat/constants.py
# ---------------------------------------------------------------------------
# Shared constants (sheet names + column names) used across analysis/plots/experiments.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Sheet names (Excel)
# ---------------------------------------------------------------------------
SHEET_VEHICLE_METRICS = "vehicle_metrics"
SHEET_EDGE_METRICS = "edge_metrics"
SHEET_VEHICLES_TABLE = "vehicles_table"
SHEET_EDGE_TIMESLICES = "edge_timeslices"
SHEET_RUN_SUMMARY = "run_summary"

# ---------------------------------------------------------------------------
# Common columns
# ---------------------------------------------------------------------------
COL_RUN = "run"
COL_STRATEGY = "strategy"
COL_N = "N"

# ---------------------------------------------------------------------------
# vehicles_table columns
# ---------------------------------------------------------------------------
COL_SERVED = "served"
COL_ALPHA = "alpha"
COL_RESERVE = "reserve"
COL_PAID_FEE = "paid_fee"
COL_SURPLUS = "surplus"
COL_ENTRY_TIME = "entry_time"
COL_EXIT_TIME = "exit_time"
COL_TRAVEL_TIME = "travel_time"
COL_ENTRY_DELAY = "entry_delay"
COL_ARRIVAL_DELAY = "arrival_delay"
COL_ENTRY_FEE = "entry_fee"
COL_LATENESS_FEE = "lateness_fee"
COL_REJECT_REASON = "reject_reason"
COL_DESIRED_ENTRY = "desired_entry"
COL_DESIRED_ARRIVAL = "desired_arrival"

# ---------------------------------------------------------------------------
# edge_metrics columns
# ---------------------------------------------------------------------------
COL_EDGE = "edge"
COL_UTILIZATION = "utilization"
COL_PRICE_INCREASE = "price_increase"
COL_PEAK_PRICE = "peak_price"

# ---------------------------------------------------------------------------
# edge_timeslices columns
# ---------------------------------------------------------------------------
COL_T = "t"
COL_UTIL = "util"
COL_PRICE = "price"

# ---------------------------------------------------------------------------
# run_summary columns
# ---------------------------------------------------------------------------
COL_SOCIAL_WELFARE = "social_welfare"
COL_AVG_TRAVEL_TIME = "avg_travel_time"
COL_TOTAL_REVENUE = "total_revenue"
COL_SERVICE_RATE = "service_rate"

# ---------------------------------------------------------------------------
# strategies
# ---------------------------------------------------------------------------
STRAT_ZERO = "Zero"
STRAT_ZERO_UNCAPPED = "Zero (Uncapped)"
STRAT_TRANSPORT_ADAPTED = "Transport-Adapted Pricing"
STRAT_STATIC_MEDIAN = "Static Median"
STRAT_ONLINE_COMPETITIVE = "Online Competitive"
STRAT_SMOOTH_TAIL = "Smooth Tail"

ALL_STRATEGIES = [
    STRAT_ZERO,
    STRAT_ZERO_UNCAPPED,
    STRAT_TRANSPORT_ADAPTED,
    STRAT_STATIC_MEDIAN,
    STRAT_ONLINE_COMPETITIVE,
    STRAT_SMOOTH_TAIL,
]
# ---------------------------------------------------------------------------