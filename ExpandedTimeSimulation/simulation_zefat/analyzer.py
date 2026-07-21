import math

import numpy as np
import pandas as pd


def compute_sw_time_only(
    vehicles,
    served_only: bool = True,
    delay_units: str = "slots",  # "slots" or "seconds"
    slot_seconds: int = 60,
    fee_per: str = "per_slot",  # "per_slot" | "per_second" | "per_minute"
    include_entry_delay: bool = True,
    include_arrival_delay: bool = True,
    travel_time_units: str = "slots",
):
    """Compute SW from time/delay terms only: SW = Σ_served [reserve - (delay_fees + travel_time)]."""

    def _to_seconds(val, units):
        """Convert delay/travel values to seconds when given in slots."""
        return float(val) * (slot_seconds if units == "slots" else 1.0)

    def _scale_for_fee(seconds):
        """Convert seconds to the units used by the fee."""
        if fee_per == "per_second":
            return seconds
        if fee_per == "per_minute":
            return seconds / 60.0
        # per_slot by default
        return seconds / slot_seconds

    sw = 0.0
    served_cnt = 0
    per_vehicle = []

    for v in vehicles:
        served = bool(v.get("served", False))

        # Delays (clamped >= 0)
        e_delay = max(0.0, float(v.get("entry_delay", 0.0)))
        l_delay = max(0.0, float(v.get("arrival_delay", 0.0)))

        # Convert delays to fee units
        e_units = _scale_for_fee(_to_seconds(e_delay, delay_units))
        l_units = _scale_for_fee(_to_seconds(l_delay, delay_units))

        entry_fee = float(v.get("entry_fee", 0.0))
        lateness_fee = float(v.get("lateness_fee", 0.0))

        delay_term = 0.0
        if include_entry_delay:
            delay_term += entry_fee * e_units
        if include_arrival_delay:
            delay_term += lateness_fee * l_units

        # Travel time (fallback to exit-entry if present)
        ttime = v.get("travel_time")
        if ttime is None and ("entry_time" in v and "exit_time" in v):
            try:
                ttime = float(v["exit_time"]) - float(v["entry_time"])
            except Exception:
                ttime = None

        # Treat NaN/None as 0
        try:
            ttime = float(ttime)
            if math.isnan(ttime):
                ttime = 0.0
        except Exception:
            ttime = 0.0

        # Normalize travel time to the same unit family as delay_units when needed
        if travel_time_units != delay_units:
            t_sec = _to_seconds(ttime, travel_time_units)
            ttime = t_sec / (slot_seconds if delay_units == "slots" else 1.0)

        reserve = float(v.get("reserve", 0.0))
        cost_i = delay_term + ttime

        if served:
            contrib = reserve - cost_i
            served_cnt += 1
        elif not served_only:
            contrib = 0.0
        else:
            contrib = 0.0

        sw += contrib
        per_vehicle.append(
            {
                "vid": v.get("vid"),
                "served": served,
                "reserve": reserve,
                "entry_delay_units": e_units,
                "lateness_delay_units": l_units,
                "delay_term": delay_term,
                "travel_time": ttime,
                "contrib": contrib,
            }
        )

    return {
        "social_welfare": float(sw),
        "served_count": int(served_cnt),
        "total_count": len(vehicles),
        "per_vehicle": per_vehicle,
    }


def compute_metrics(vehicles):
    """Compute aggregate KPIs from a simulation vehicle list."""
    served = [v for v in vehicles if v.get("path_found")]
    n_total = len(vehicles)
    n_served = len(served)

    m = compute_sw_time_only(
        vehicles,
        served_only=True,
        delay_units="slots",
        slot_seconds=60,
        fee_per="per_slot",
        include_entry_delay=True,
        include_arrival_delay=True,
        travel_time_units="slots",
    )
    social_welfare = m["social_welfare"]

    avg_travel_time = (
        (sum(v.get("travel_time", 0.0) for v in served) / n_served) if n_served else float("nan")
    )
    total_revenue = sum(v.get("paid_fee", 0.0) for v in served)
    service_rate = (n_served / n_total) if n_total else 0.0

    return {
        "social_welfare": social_welfare,
        "avg_travel_time": avg_travel_time,
        "total_revenue": total_revenue,
        "service_rate": service_rate,
    }


def collect_edge_metrics(edge_data_dict, strategy_name, run_idx):
    """Aggregate per-edge pricing/capacity metrics into a DataFrame."""
    records = []
    for edge, data in edge_data_dict.items():
        initial = data.price_history[0] if data.price_history else 0.0
        final = data.price_history[-1] if data.price_history else 0.0
        peak_price = max(data.price_history) if data.price_history else final

        records.append(
            {
                "run": run_idx,
                "strategy": strategy_name,
                "edge": edge,
                "capacity": data.capacity,
                "demand": data.demand,
                "request_count": data.request_count,
                "alloc_count": data.alloc_count,
                "utilization": (data.alloc_count / data.capacity) if data.capacity else 0.0,
                "initial_price": initial,
                "final_price": final,
                "price_increase": final - initial,
                "peak_price": peak_price,
                "time_of_peak": data.price_history.index(peak_price) if data.price_history else 0,
                "last_used_price": data.last_used_price(),
            }
        )
    return pd.DataFrame(records)


def analyze_edges(edges_df):
    """Print strategy-level summary statistics for edge utilization and prices."""
    edge_summary = edges_df.groupby("strategy").agg(
        {
            "utilization": ["mean", "std"],
            "price_increase": ["mean", "std"],
            "peak_price": ["mean", "std"],
            "demand": "sum",
            "alloc_count": "sum",
        }
    )
    print(edge_summary)


def vehicles_to_df(vehicles, run_idx, strat_name):
    """Convert vehicle dicts into a flat per-vehicle DataFrame for reporting/export."""
    rows = []
    for i, v in enumerate(vehicles):
        served = bool(v.get("path_found"))
        entry_t = v.get("entry_time")
        exit_t = v.get("exit_time")
        travel = v.get("travel_time")

        reserve = v.get("reserved_price", np.nan)
        paid = v.get("paid_fee", 0.0)
        surplus = (reserve - paid) if (served and pd.notna(reserve)) else np.nan

        desired_entry = v.get("desired_entry")
        desired_arrival = v.get("desired_arrival")

        entry_delay = (
            (entry_t - desired_entry)
            if (entry_t is not None and desired_entry is not None)
            else np.nan
        )
        arrival_delay = (
            (exit_t - desired_arrival)
            if (exit_t is not None and desired_arrival is not None)
            else np.nan
        )

        # Path length (hops) if present
        raw_path = v.get("allocated_path") or v.get("path")
        path_len = len(raw_path) if isinstance(raw_path, list) else np.nan

        rows.append(
            {
                "run": run_idx,
                "strategy": strat_name,
                "served": served,
                "alpha": v.get("alpha"),
                "reserve": reserve,
                "paid_fee": paid,
                "surplus": surplus,
                "entry_time": entry_t,
                "exit_time": exit_t,
                "travel_time": travel,
                "entry_delay": entry_delay,
                "arrival_delay": arrival_delay,
                "path_len": path_len,
                "path": raw_path,
                "real_cost": v.get("real_cost", np.nan),
                "dijkstra_cost": v.get("dijkstra_cost", np.nan),
                "source": v.get("source"),
                "destination": v.get("destination"),
                "entry_fee": v.get("entry_fee"),
                "lateness_fee": v.get("lateness_fee"),
                "reject_reason": v.get("reject_reason"),
                "desired_entry": desired_entry,
                "desired_arrival": desired_arrival,
                "N": v.get("N"),
            }
        )
    return pd.DataFrame(rows)


def edge_timeslices_to_df(edge_data_dict, run_idx, strat_name):
    """Flatten time-expanded edge data into one row per time-edge for reporting/export."""
    rows = []
    for edge, d in edge_data_dict.items():
        (u, t) = edge[0]
        (v, t2) = edge[1]

        cap = getattr(d, "capacity", 0)
        alloc = getattr(d, "alloc_count", 0)
        req = getattr(d, "request_count", 0)
        price = getattr(d, "price", 0.0)
        unit0 = getattr(d, "unit_initial_price", 0.0)
        hist = getattr(d, "price_history", [price])

        peak = max(hist) if hist else price
        util = (alloc / cap) if cap else 0.0
        over = max(0, alloc - cap) if cap else 0

        rows.append(
            {
                "run": run_idx,
                "strategy": strat_name,
                "edge": edge,
                "u": u,
                "v": v,
                "t": t,
                "t2": t2,
                "capacity": cap,
                "alloc_count": alloc,
                "request_count": req,
                "util": util,
                "over_demand": over,
                "price": price,
                "unit_initial_price": unit0,
                "peak_price": peak,
            }
        )
    return pd.DataFrame(rows)


def run_summary(veh_df: pd.DataFrame, ts_df: pd.DataFrame, run_idx, strat_name):
    """Compute a compact per-run summary row (KPIs, percentiles, fairness, congestion)."""
    s = veh_df[veh_df["served"]]
    n_total = len(veh_df)
    n_served = len(s)
    svc_rate = (n_served / n_total) if n_total else 0.0

    # KPIs
    social_welfare = float(s["reserve"].sum())
    avg_travel_time = float(s["travel_time"].mean()) if n_served else float("nan")
    total_revenue = float(s["paid_fee"].sum())

    def pct(series, q):
        return float(series.quantile(q)) if not series.empty else float("nan")

    tt_p50 = pct(s["travel_time"], 0.5)
    tt_p90 = pct(s["travel_time"], 0.9)
    sur_p50 = pct(s["surplus"], 0.5)
    sur_p90 = pct(s["surplus"], 0.9)

    def gini(x: pd.Series) -> float:
        x = x.dropna().values
        if len(x) == 0:
            return float("nan")
        x = np.sort(x)
        n = len(x)
        cum = np.cumsum(x)
        return float((n + 1 - 2 * (cum.sum() / cum[-1])) / n) if cum[-1] != 0 else 0.0

    gini_tt = gini(s["travel_time"])
    gini_sur = gini(s["surplus"])

    # Congestion proxy from time-slice table
    ts = ts_df.copy()
    sat_rate = float((ts["util"] >= 1.0).mean()) if not ts.empty else 0.0
    over_demand_total = int(ts["over_demand"].sum()) if "over_demand" in ts else 0

    return pd.DataFrame(
        [
            {
                "run": run_idx,
                "strategy": strat_name,
                "social_welfare": social_welfare,
                "avg_travel_time": avg_travel_time,
                "total_revenue": total_revenue,
                "service_rate": svc_rate,
                "tt_p50": tt_p50,
                "tt_p90": tt_p90,
                "surplus_p50": sur_p50,
                "surplus_p90": sur_p90,
                "gini_travel_time": gini_tt,
                "gini_surplus": gini_sur,
                "time_slice_saturation_rate": sat_rate,
                "total_over_demand": over_demand_total,
                "n_total": n_total,
                "n_served": n_served,
            }
        ]
    )
