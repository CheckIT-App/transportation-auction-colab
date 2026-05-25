# ExpandedTimeSimulation/simulation_zefat/colab_utils.py
# ---------------------------------------------------------------------------
# Thin wrapper functions for the Google Colab notebook interface.
# All core simulation logic stays in the existing modules.
# ---------------------------------------------------------------------------

from __future__ import annotations

import json
import os
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

# ---------------------------------------------------------------------------
# Reject-reason decoder
# ---------------------------------------------------------------------------

_REJECT_LABELS: dict[int, str] = {
    0: "Accepted",
    1: "Infeasible path",
    2: "Exceeds reserve price",
}


def decode_reject_reason(code: Any) -> str:
    """Map a reject_reason integer to a human-readable string."""
    try:
        return _REJECT_LABELS.get(int(code), f"Unknown ({code})")
    except (TypeError, ValueError):
        return str(code)


# ---------------------------------------------------------------------------
# Widget config extractor
# ---------------------------------------------------------------------------


def load_config_from_widgets(w: dict) -> dict:
    """Extract .value from each ipywidget and return a plain config dict.

    Normalizes display values to internal format (e.g., "Only Entry" -> "entry").

    Usage::

        config = load_config_from_widgets(widgets)
    """
    config = {k: widget.value for k, widget in w.items()}
    
    # Normalize time_mode from display format to internal format
    if "time_mode" in config:
        mode_map = {
            "Only Entry": "entry",
            "Only Arrival": "arrival",
            "Both": "both",
        }
        config["time_mode"] = mode_map.get(config["time_mode"], "both")
    
    return config


# ---------------------------------------------------------------------------
# Network preview
# ---------------------------------------------------------------------------


def preview_network(config: dict) -> None:
    """Load the OSM graph and print a summary table + map plot.

    Loads from *graph_file* if it exists on disk, otherwise downloads from OSM.
    Does NOT build the time-expanded graph — this is a lightweight look at the
    physical road network only.
    """
    import osmnx as ox

    from ExpandedTimeSimulation.simulation_zefat.real_data import RealData

    graph_file = config.get("graph_file", "har_nof.gpickle")
    place_name = config.get("place_name", "Har Nof, Jerusalem, Israel")
    od_count = config.get("od_count_start", config.get("od_count", 200))
    slot_seconds = config.get("slot_seconds", 60)

    loader = RealData(place_name, time_slot_duration=slot_seconds, od_count=od_count)

    if os.path.exists(graph_file):
        print(f"Loading graph from: {graph_file}")
        loader.load_graph_from_file(graph_file)
        crs = loader.G.graph.get("crs") if loader.G is not None else None
        try:
            is_proj = ox.projection.is_projected(crs)
        except Exception:
            is_proj = False
        if not is_proj:
            print("  Projecting graph to meters ...")
            loader.G = ox.project_graph(loader.G)
    else:
        print(f"Graph file '{graph_file}' not found — downloading from OSM: '{place_name}'")
        print("  This may take a minute on the first run ...")
        try:
            loader.load_graph()
        except Exception as exc:
            raise RuntimeError(
                f"\n\nCould not download the road network for '{place_name}'.\n"
                f"Possible causes:\n"
                f"  - The place name is misspelled or not recognised by OpenStreetMap.\n"
                f"  - No internet connection in this Colab session.\n"
                f"  - OSM servers are temporarily unavailable.\n\n"
                f"Tip: Try a well-known city name like 'Tel Aviv, Israel' or upload a "
                f".gpickle file and set 'graph_file' to its path.\n\n"
                f"Original error: {exc}"
            ) from exc

    G = loader.G
    try:
        nodes, edges = ox.graph_to_gdfs(G)
        bbox = edges.total_bounds  # minx, miny, maxx, maxy
        bbox_str = f"x=[{bbox[0]:.0f}, {bbox[2]:.0f}]  y=[{bbox[1]:.0f}, {bbox[3]:.0f}] m"
    except Exception:
        bbox_str = "unavailable"

    print()
    print("-- Network Summary ------------------------------")
    print(f"  Nodes        : {G.number_of_nodes():,}")
    print(f"  Edges        : {G.number_of_edges():,}")
    print(f"  CRS          : {G.graph.get('crs', 'unknown')}")
    print(f"  Bounding box : {bbox_str}")
    print("-------------------------------------------------")
    print()

    fig, ax = ox.plot_graph(
        G,
        node_size=5,
        edge_linewidth=0.8,
        figsize=(8, 7),
        show=False,
        close=False,
    )
    ax.set_title(f"Road Network: {place_name}", fontsize=13)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Demand generation
# ---------------------------------------------------------------------------


def generate_demand_preview(config: dict) -> tuple[list[dict], pd.DataFrame]:
    """Generate a vehicle fleet and return ``(vehicles, preview_df)``.

    *preview_df* contains the first 10 rows with key columns for quick inspection.
    The returned *vehicles* list can be passed to :func:`plot_demand_distribution`.
    """
    import osmnx as ox

    from ExpandedTimeSimulation.simulation_zefat.experiments.vehicle_generation import (
        PeakSchedule,
        assign_peak_desired_time_by_mode,
        mixed_alpha_sampler,
    )
    from ExpandedTimeSimulation.simulation_zefat.real_data import RealData

    graph_file = config.get("graph_file", "har_nof.gpickle")
    place_name = config.get("place_name", "Har Nof, Jerusalem, Israel")
    od_count = config.get("od_count_start", config.get("od_count", 200))
    slot_seconds = config.get("slot_seconds", 60)
    peak_slot = config.get("peak_slot", 10)
    peak_sigma = config.get("peak_sigma_start", config.get("peak_sigma", 5.0))
    vehicle_T = config.get("vehicle_T", 100)
    base_seed = config.get("base_seed", 42)
    time_mode = config.get("time_mode", "both")
    arrival_percentage = config.get("arrival_percentage", 0.5)

    loader = RealData(place_name, time_slot_duration=slot_seconds, od_count=od_count)

    if os.path.exists(graph_file):
        loader.load_graph_from_file(graph_file)
        crs = loader.G.graph.get("crs") if loader.G is not None else None
        try:
            is_proj = ox.projection.is_projected(crs)
        except Exception:
            is_proj = False
        if not is_proj:
            loader.G = ox.project_graph(loader.G)
    else:
        try:
            loader.load_graph()
        except Exception as exc:
            raise RuntimeError(
                f"\n\nCould not download the road network for '{place_name}'.\n"
                f"Possible causes:\n"
                f"  - The place name is misspelled or not recognised by OpenStreetMap.\n"
                f"  - No internet connection in this Colab session.\n\n"
                f"Tip: Try a well-known city name like 'Tel Aviv, Israel' or upload a "
                f".gpickle file and set 'graph_file' to its path.\n\n"
                f"Original error: {exc}"
            ) from exc

    loader.enrich_graph()
    loader.seed = base_seed
    loader.generate_od_pairs()

    alpha_mix = mixed_alpha_sampler(
        [(0.4, (0.0, 0.3)), (0.4, (0.3, 0.7)), (0.2, (0.7, 1.0))]
    )
    vehicles = loader.generate_vehicles(alpha=alpha_mix)
    assign_peak_desired_time_by_mode(
        vehicles,
        schedule=PeakSchedule(
            peak_slot=peak_slot, sigma=peak_sigma, horizon_T=vehicle_T
        ),
        mode=time_mode,
        arrival_percentage=arrival_percentage,
    )

    cols = [
        "source", "destination", "alpha", "reserve",
        "desired_entry", "desired_arrival", "entry_fee", "lateness_fee",
    ]
    preview_df = (
        pd.DataFrame([{k: v[k] for k in cols if k in v} for v in vehicles[:10]])
        .round(3)
    )
    return vehicles, preview_df


def plot_demand_distribution(vehicles: list[dict]) -> None:
    """Histogram(s) for desired times and alpha distribution.
    
    Shows desired_entry, desired_arrival, or both depending on which fields are set.
    """
    desired_entries = [v.get("desired_entry") for v in vehicles]
    desired_arrivals = [v.get("desired_arrival") for v in vehicles]
    alphas = [v.get("alpha", 0.0) for v in vehicles]
    
    # Filter to only non-None values
    has_entries = any(e is not None for e in desired_entries)
    has_arrivals = any(a is not None for a in desired_arrivals)
    
    desired_entries_clean = [e for e in desired_entries if e is not None]
    desired_arrivals_clean = [a for a in desired_arrivals if a is not None]
    
    # Determine layout based on which fields exist
    if has_entries and has_arrivals:
        n_cols = 3
        n_time_plots = 2
    else:
        n_cols = 2
        n_time_plots = 1
    
    fig, axes = plt.subplots(1, n_cols, figsize=(5 + 4 * n_cols, 4))
    if n_cols == 1:
        axes = [axes]
    
    plot_idx = 0
    
    if has_entries:
        axes[plot_idx].hist(desired_entries_clean, bins=30, color="steelblue", 
                          edgecolor="white", linewidth=0.4)
        axes[plot_idx].set_xlabel("Desired Entry Time Slot", fontsize=12)
        axes[plot_idx].set_ylabel("Number of Vehicles", fontsize=12)
        axes[plot_idx].set_title("Desired Entry Time Distribution", fontsize=13)
        axes[plot_idx].grid(axis="y", alpha=0.3)
        plot_idx += 1
    
    if has_arrivals:
        axes[plot_idx].hist(desired_arrivals_clean, bins=30, color="seagreen", 
                          edgecolor="white", linewidth=0.4)
        axes[plot_idx].set_xlabel("Desired Arrival Time Slot", fontsize=12)
        axes[plot_idx].set_ylabel("Number of Vehicles", fontsize=12)
        axes[plot_idx].set_title("Desired Arrival Time Distribution", fontsize=13)
        axes[plot_idx].grid(axis="y", alpha=0.3)
        plot_idx += 1
    
    axes[plot_idx].hist(alphas, bins=30, color="darkorange", edgecolor="white", linewidth=0.4)
    axes[plot_idx].set_xlabel("Alpha  (price–time weight)", fontsize=12)
    axes[plot_idx].set_ylabel("Number of Vehicles", fontsize=12)
    axes[plot_idx].set_title("Alpha Distribution", fontsize=13)
    axes[plot_idx].grid(axis="y", alpha=0.3)

    plt.suptitle(
        f"Vehicle Fleet Preview  (n={len(vehicles):,})", fontsize=14, y=1.02
    )
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Simulation runner
# ---------------------------------------------------------------------------


def run_experiment(config: dict) -> str:
    """Run the batch experiment and return the path to the output Excel file.

    Supports parameter sweeps: set start != end for exactly one of the three
    sweep parameters (od_count, peak_sigma, capacity_factor).  The simulation
    runs once per value in the generated range and results are merged into a
    single Excel file with ``sweep_param`` / ``sweep_value`` columns added to
    every sheet.

    Diagnostic plots are suppressed here — use :func:`plot_results` after.
    """
    import numpy as np
    import tempfile

    from ExpandedTimeSimulation.simulation_zefat.experiments.batch_run import run_batch

    excel_file = config.get("excel_file", "results.xlsx")

    strategy_keys = config.get("strategy_keys")
    if strategy_keys is not None:
        strategy_keys = list(strategy_keys)

    # ── Shared non-sweep kwargs ──────────────────────────────────────────────
    shared = dict(
        num_runs=config.get("num_runs", 1),
        base_seed=config.get("base_seed", 2025),
        place_name=config.get("place_name", "Har Nof, Jerusalem, Israel"),
        graph_file=config.get("graph_file", "har_nof.gpickle"),
        max_time_slots=config.get("max_time_slots", 20),
        vmax=float(config.get("vmax", 100.0)),
        r=int(config.get("r", 30)),
        capacity_is_hourly=bool(config.get("capacity_is_hourly", True)),
        slot_seconds=int(config.get("slot_seconds", 60)),
        vehicle_T=int(config.get("vehicle_T", 100)),
        peak_slot=int(config.get("peak_slot", 10)),
        strategy_keys=strategy_keys or None,
        run_diagnostics_plots=False,
        smooth_tail_u0=float(config.get("smooth_tail_u0", 0.95)),
        time_mode=config.get("time_mode", "entry"),
        arrival_percentage=float(config.get("arrival_percentage", 0.5)),
        alpha_lo=float(config.get("alpha_lo", 0.0)),
        alpha_hi=float(config.get("alpha_hi", 1.0)),
        entry_fee_lo=float(config.get("entry_fee_lo", 1.0)),
        entry_fee_hi=float(config.get("entry_fee_hi", 5.0)),
        lateness_fee_lo=float(config.get("lateness_fee_lo", 1.0)),
        lateness_fee_hi=float(config.get("lateness_fee_hi", 5.0)),
    )

    # ── Build sweep ranges ───────────────────────────────────────────────────
    def _int_range(start_key, end_key, step_key, default):
        s = int(config.get(start_key, default))
        e = int(config.get(end_key, default))
        st = max(1, int(config.get(step_key, default)))
        if s == e:
            return [s]
        return list(range(s, e + 1, st)) if s < e else [s]

    def _float_range(start_key, end_key, step_key, default):
        s = float(config.get(start_key, default))
        e = float(config.get(end_key, default))
        st = float(config.get(step_key, 1.0))
        if abs(s - e) < 1e-9:
            return [round(s, 6)]
        if s > e:
            return [round(s, 6)]
        vals = np.arange(s, e + st * 0.5, st)
        return [round(float(v), 6) for v in vals]

    od_values = _int_range("od_count_start", "od_count_end", "od_count_step", 200)
    sigma_values = _float_range("peak_sigma_start", "peak_sigma_end", "peak_sigma_step", 5.0)
    # capacity factor stored as percentage in widgets → convert to fraction
    cap_pct_values = _float_range("cap_factor_start", "cap_factor_end", "cap_factor_step", 100.0)
    cap_values = [round(p / 100.0, 6) for p in cap_pct_values]

    active_sweeps = []
    if len(od_values) > 1:
        active_sweeps.append("od_count")
    if len(sigma_values) > 1:
        active_sweeps.append("peak_sigma")
    if len(cap_values) > 1:
        active_sweeps.append("capacity_factor")

    if len(active_sweeps) > 1:
        raise ValueError(
            f"Only one sweep parameter may have start ≠ end at a time. "
            f"Currently active: {', '.join(active_sweeps)}."
        )

    # ── Single run (no sweep) ────────────────────────────────────────────────
    if not active_sweeps:
        run_batch(
            excel_file=excel_file,
            od_count=od_values[0],
            peak_sigma=sigma_values[0],
            capacity_factor=cap_values[0],
            **shared,
        )
        return excel_file

    # ── Sweep loop ───────────────────────────────────────────────────────────
    sweep_param = active_sweeps[0]
    if sweep_param == "od_count":
        sweep_list = [(v, sigma_values[0], cap_values[0]) for v in od_values]
    elif sweep_param == "peak_sigma":
        sweep_list = [(od_values[0], v, cap_values[0]) for v in sigma_values]
    else:  # capacity_factor
        sweep_list = [(od_values[0], sigma_values[0], v) for v in cap_values]

    print(f"Sweep: {sweep_param} over {len(sweep_list)} values")

    temp_files: list[tuple[str, float]] = []
    for od, sigma, cap in sweep_list:
        raw_val = od if sweep_param == "od_count" else (sigma if sweep_param == "peak_sigma" else round(cap * 100, 2))
        print(f"  {sweep_param} = {raw_val}")
        tmp = tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False)
        tmp_path = tmp.name
        tmp.close()
        run_batch(
            excel_file=tmp_path,
            od_count=int(od),
            peak_sigma=float(sigma),
            capacity_factor=float(cap),
            **shared,
        )
        temp_files.append((tmp_path, raw_val))

    # ── Merge all temp Excel files into one ──────────────────────────────────
    from ExpandedTimeSimulation.simulation_zefat.constants import (
        SHEET_EDGE_METRICS,
        SHEET_EDGE_TIMESLICES,
        SHEET_RUN_SUMMARY,
        SHEET_VEHICLE_METRICS,
        SHEET_VEHICLES_TABLE,
    )
    sheet_names = [
        SHEET_VEHICLE_METRICS,
        SHEET_EDGE_METRICS,
        SHEET_VEHICLES_TABLE,
        SHEET_EDGE_TIMESLICES,
        SHEET_RUN_SUMMARY,
    ]

    merged: dict[str, list[pd.DataFrame]] = {s: [] for s in sheet_names}
    for tmp_path, sweep_val in temp_files:
        for sheet in sheet_names:
            try:
                df = pd.read_excel(tmp_path, sheet_name=sheet)
                df.insert(0, "sweep_param", sweep_param)
                df.insert(1, "sweep_value", sweep_val)
                merged[sheet].append(df)
            except Exception:
                pass
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    with pd.ExcelWriter(excel_file, engine="openpyxl") as writer:
        for sheet in sheet_names:
            frames = merged[sheet]
            if frames:
                pd.concat(frames, ignore_index=True).to_excel(writer, sheet_name=sheet, index=False)

    print(f"\nSweep complete. Results saved to: {excel_file}")
    return excel_file


# ---------------------------------------------------------------------------
# Custom X/Y plot
# ---------------------------------------------------------------------------

_X_LABELS: dict[str, str] = {
    "time_slot":         "Time Slot",
    "vehicles_sweep":    "Number of Vehicles",
    "sigma_sweep":       "Peak Sigma",
    "capacity_sweep":    "Capacity Factor (%)",
    "alpha":             "Alpha",
    "entry_delay_cost":  "Entry Delay Cost",
    "arrival_delay_cost":"Arrival Delay Cost",
}

_Y_LABELS: dict[str, str] = {
    "social_welfare":   "Social Welfare",
    "acceptance":       "% Acceptance",
    "avg_price":        "Avg Price per Route",
    "avg_entry_delay":  "Avg Entry Delay (slots)",
    "avg_arrival_delay":"Avg Arrival Delay (slots)",
    "speed":            "Speed (path-units / slot)",
    "travel_time":      "Avg Travel Time (slots)",
}

_SWEEP_PARAM_MAP: dict[str, str] = {
    "vehicles_sweep": "od_count",
    "sigma_sweep":    "peak_sigma",
    "capacity_sweep": "capacity_factor",
}


def plot_custom(
    excel_file: str,
    x_metric: str,
    y_metric: str,
    strategies: list[str] | None = None,
    n_bins: int = 10,
) -> None:
    """Plot any Y metric against any X metric, one line per strategy.

    x_metric choices: time_slot | vehicles_sweep | sigma_sweep | capacity_sweep |
                      alpha | entry_delay_cost | arrival_delay_cost
    y_metric choices: social_welfare | acceptance | avg_price |
                      avg_entry_delay | avg_arrival_delay | speed | travel_time
    """
    import numpy as np
    from ExpandedTimeSimulation.simulation_zefat.constants import (
        SHEET_VEHICLES_TABLE,
        COL_STRATEGY,
        COL_RUN,
    )

    try:
        vt = pd.read_excel(excel_file, sheet_name=SHEET_VEHICLES_TABLE)
    except Exception as exc:
        print(f"Could not read '{excel_file}': {exc}")
        return

    if vt.empty:
        print("vehicles_table sheet is empty — run the simulation first.")
        return

    # ── Ensure sweep columns exist ───────────────────────────────────────────
    if "sweep_param" not in vt.columns:
        vt["sweep_param"] = None
    if "sweep_value" not in vt.columns:
        vt["sweep_value"] = float("nan")

    # ── Derive computed columns ──────────────────────────────────────────────
    vt["entry_delay_cost"]    = vt.get("entry_fee",    0) * vt.get("entry_delay",    0)
    vt["arrival_delay_cost"]  = vt.get("lateness_fee", 0) * vt.get("arrival_delay",  0)
    travel_time_safe          = pd.to_numeric(vt.get("travel_time"), errors="coerce").replace(0, float("nan"))
    vt["speed"]               = pd.to_numeric(vt.get("path_len"), errors="coerce") / travel_time_safe

    # ── Filter strategies ────────────────────────────────────────────────────
    if strategies:
        vt = vt[vt[COL_STRATEGY].isin(strategies)]

    # ── Resolve X column & binning ───────────────────────────────────────────
    bin_x = False
    if x_metric in _SWEEP_PARAM_MAP:
        param_key = _SWEEP_PARAM_MAP[x_metric]
        if vt["sweep_param"].isna().all() or not (vt["sweep_param"] == param_key).any():
            print(
                f"Warning: no sweep data found for '{param_key}'. "
                "Run the simulation with this parameter as a sweep first."
            )
            return
        vt = vt[vt["sweep_param"] == param_key].copy()
        x_col = "sweep_value"
    elif x_metric == "time_slot":
        x_col = "entry_time"
    else:
        x_col = x_metric  # "alpha", "entry_delay_cost", "arrival_delay_cost"
        bin_x = True

    vt[x_col] = pd.to_numeric(vt[x_col], errors="coerce")
    vt = vt.dropna(subset=[x_col])

    if bin_x:
        try:
            vt["_x_bin"] = pd.cut(vt[x_col], bins=n_bins)
            vt["_x_mid"] = vt["_x_bin"].apply(lambda b: b.mid if pd.notna(b) else float("nan"))
        except Exception as exc:
            print(f"Binning failed: {exc}")
            return
        x_col = "_x_mid"

    # ── Y aggregation function ───────────────────────────────────────────────
    def _agg_y(grp: pd.DataFrame) -> float:
        served_mask = grp["served"].astype(bool) if "served" in grp.columns else pd.Series(True, index=grp.index)
        if y_metric == "social_welfare":
            return grp.loc[served_mask, "reserve"].sum() if "reserve" in grp.columns else float("nan")
        if y_metric == "acceptance":
            return served_mask.mean() * 100.0
        if y_metric == "avg_price":
            vals = grp.loc[served_mask, "paid_fee"] if "paid_fee" in grp.columns else pd.Series(dtype=float)
            return vals.mean() if not vals.empty else float("nan")
        if y_metric == "avg_entry_delay":
            return grp["entry_delay"].mean() if "entry_delay" in grp.columns else float("nan")
        if y_metric == "avg_arrival_delay":
            return grp["arrival_delay"].mean() if "arrival_delay" in grp.columns else float("nan")
        if y_metric == "speed":
            vals = grp.loc[served_mask, "speed"] if "speed" in grp.columns else pd.Series(dtype=float)
            return vals.mean() if not vals.empty else float("nan")
        if y_metric == "travel_time":
            vals = grp.loc[served_mask, "travel_time"] if "travel_time" in grp.columns else pd.Series(dtype=float)
            return vals.mean() if not vals.empty else float("nan")
        return float("nan")

    # ── Aggregate: per (x, strategy, run) then average runs ─────────────────
    run_col = COL_RUN if COL_RUN in vt.columns else "run"
    if run_col not in vt.columns:
        vt[run_col] = 1

    records = []
    for (x_val, strat, run_id), grp in vt.groupby([x_col, COL_STRATEGY, run_col], observed=True):
        records.append({"x": x_val, "strategy": strat, "run": run_id, "y": _agg_y(grp)})

    if not records:
        print("No data to plot after grouping.")
        return

    agg_df = pd.DataFrame(records)
    summary = (
        agg_df.groupby(["x", "strategy"], observed=True)["y"]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = ["x", "strategy", "y_mean", "y_std"]
    summary["y_std"] = summary["y_std"].fillna(0.0)

    n_runs = agg_df["run"].nunique()
    show_band = n_runs > 1

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, (strat, sdf) in enumerate(summary.groupby("strategy", observed=True)):
        sdf = sdf.sort_values("x")
        x_vals = sdf["x"].values
        y_vals = sdf["y_mean"].values
        y_std  = sdf["y_std"].values
        color  = colors[i % len(colors)]

        ax.plot(x_vals, y_vals, marker="o", label=strat, color=color, linewidth=1.8)
        if show_band:
            ax.fill_between(x_vals, y_vals - y_std, y_vals + y_std,
                            color=color, alpha=0.15)

    ax.set_xlabel(_X_LABELS.get(x_metric, x_metric), fontsize=12)
    ax.set_ylabel(_Y_LABELS.get(y_metric, y_metric), fontsize=12)
    ax.set_title(
        f"{_Y_LABELS.get(y_metric, y_metric)}  vs  {_X_LABELS.get(x_metric, x_metric)}",
        fontsize=13,
    )
    ax.legend(title="Strategy", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()

    return summary


# ---------------------------------------------------------------------------
# Results loading and summary
# ---------------------------------------------------------------------------


def summarize_results(excel_file: str) -> dict[str, pd.DataFrame]:
    """Read all result sheets from the Excel file. Returns a dict keyed by sheet name."""
    from ExpandedTimeSimulation.simulation_zefat.constants import (
        SHEET_EDGE_METRICS,
        SHEET_EDGE_TIMESLICES,
        SHEET_RUN_SUMMARY,
        SHEET_VEHICLE_METRICS,
        SHEET_VEHICLES_TABLE,
    )

    sheet_names = [
        SHEET_VEHICLE_METRICS,
        SHEET_EDGE_METRICS,
        SHEET_VEHICLES_TABLE,
        SHEET_EDGE_TIMESLICES,
        SHEET_RUN_SUMMARY,
    ]
    sheets: dict[str, pd.DataFrame] = {}
    for name in sheet_names:
        try:
            sheets[name] = pd.read_excel(excel_file, sheet_name=name)
        except Exception as exc:
            print(f"Warning: could not read sheet '{name}': {exc}")
            sheets[name] = pd.DataFrame()
    return sheets


def build_summary_tables(sheets: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Build per-metric pivot tables from the *run_summary* sheet.

    Returns a dict mapping a display label to a DataFrame with columns
    [Mean, Std, Min, Max] indexed by strategy name.
    """
    from ExpandedTimeSimulation.simulation_zefat.constants import (
        COL_AVG_TRAVEL_TIME,
        COL_SERVICE_RATE,
        COL_SOCIAL_WELFARE,
        COL_STRATEGY,
        COL_TOTAL_REVENUE,
        SHEET_RUN_SUMMARY,
    )

    df = sheets.get(SHEET_RUN_SUMMARY, pd.DataFrame())
    if df.empty:
        print("run_summary sheet is empty — nothing to summarise.")
        return {}

    metric_defs = [
        ("Social Welfare", COL_SOCIAL_WELFARE),
        ("Service Rate", COL_SERVICE_RATE),
        ("Avg Travel Time (slots)", COL_AVG_TRAVEL_TIME),
        ("Total Revenue", COL_TOTAL_REVENUE),
    ]

    tables: dict[str, pd.DataFrame] = {}
    for label, col in metric_defs:
        if col not in df.columns:
            continue
        pivot = (
            df.groupby(COL_STRATEGY)[col]
            .agg(["mean", "std", "min", "max"])
            .round(3)
        )
        pivot.columns = ["Mean", "Std Dev", "Min", "Max"]
        pivot.index.name = "Strategy"
        tables[label] = pivot

    return tables


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_results(excel_file: str, config: dict | None = None) -> None:
    """Run all diagnostic plots from the Excel results file.

    Wraps :func:`~plots.plots.plot_sim_diagnostics`.
    Pass *config* to forward ``vehicle_T`` (used to set the time-axis horizon).
    Individual plots that fail are skipped with a printed warning.
    """
    from ExpandedTimeSimulation.simulation_zefat.plots.plots import plot_sim_diagnostics

    vehicle_T = (config or {}).get("vehicle_T", None)
    try:
        plot_sim_diagnostics(excel_file, horizon_T=vehicle_T, fs=13)
    except (ValueError, KeyError) as exc:
        print(f"Warning: some plots could not be generated: {exc}")


def preview_network_map(config: dict):
    """Return an interactive folium map of the road network.

    Displays the physical road edges on an OpenStreetMap basemap.
    Call ``display(preview_network_map(config))`` in Colab to render it.
    """
    import osmnx as ox
    import pandas as pd
    try:
        import folium
    except ImportError:
        print("folium is not installed. Run:  !pip install -q folium")
        return None

    from ExpandedTimeSimulation.simulation_zefat.real_data import RealData

    # Request English name tags when downloading graph data from OSM
    if "name:en" not in ox.settings.useful_tags_way:
        ox.settings.useful_tags_way = list(ox.settings.useful_tags_way) + ["name:en"]

    graph_file = config.get("graph_file", "har_nof.gpickle")
    place_name = config.get("place_name", "Har Nof, Jerusalem, Israel")
    slot_seconds = config.get("slot_seconds", 60)

    loader = RealData(place_name, time_slot_duration=slot_seconds, od_count=1)

    if os.path.exists(graph_file):
        loader.load_graph_from_file(graph_file)
    else:
        try:
            loader.load_graph()
        except Exception as exc:
            print(f"Could not load graph for map: {exc}")
            return None

    G = loader.G

    # Get node positions in lat/lon (un-projected)
    try:
        G_latlon = ox.projection.project_graph(G, to_crs="EPSG:4326")
    except Exception:
        G_latlon = G  # already in lat/lon if projection fails

    nodes, edges = ox.graph_to_gdfs(G_latlon)
    center_lat = float(nodes.geometry.y.mean())
    center_lon = float(nodes.geometry.x.mean())

    fmap = folium.Map(location=[center_lat, center_lon], zoom_start=15,
                      tiles="OpenStreetMap")

    # Draw road edges with English name tooltips
    for _, row in edges.iterrows():
        coords = [(lat, lon) for lon, lat in row.geometry.coords]

        # Use English name only (name:en); never fall back to Hebrew default name
        en_name = row.get("name:en")
        road_name = str(en_name) if pd.notna(en_name) and str(en_name).strip() else ""

        folium.PolyLine(
            coords,
            color="#3388ff",
            weight=2.5,
            opacity=0.7,
            tooltip=road_name if road_name else None,
        ).add_to(fmap)

    folium.Marker(
        [center_lat, center_lon],
        popup=f"Network centre\n{place_name}",
        icon=folium.Icon(color="red", icon="info-sign"),
    ).add_to(fmap)

    return fmap


def print_vehicle_summary(vehicles: list[dict]) -> None:
    """Print a summary of vehicle allocation outcomes.

    Useful for debugging or inspection in Colab.
    """
    print("\nVehicle Allocation Summary (one line per vehicle):")
    for i, v in enumerate(vehicles, 1):
        served = "SERVED" if v.get("path_found") else "REJECTED"
        src = v.get("source", "?")
        dst = v.get("destination", "?")
        alpha = v.get("alpha", 0.0)
        reserve = v.get("reserve", 0.0)
        cost = v.get("paid_fee", 0.0)
        entry = v.get("entry_time")
        exit_t = v.get("exit_time")
        print(f"  {i:3d}: {src}→{dst} α={alpha:.2f} res={reserve:.2f} cost={cost:.2f} "
              f"entry={entry} exit={exit_t} {served}")


# ---------------------------------------------------------------------------


def export_results(
    excel_file: str,
    config: dict,
    output_dir: str = "experiment_output",
) -> None:
    """Save each Excel sheet as a CSV and write the config as JSON.

    Creates *output_dir* if it does not exist.
    The Excel file itself is already written by :func:`run_experiment`.
    """
    os.makedirs(output_dir, exist_ok=True)

    sheets = summarize_results(excel_file)
    for name, df in sheets.items():
        if not df.empty:
            out_path = os.path.join(output_dir, f"{name}.csv")
            df.to_csv(out_path, index=False)
            print(f"Saved: {out_path}")

    # Serialise config (convert tuple/set values from SelectMultiple to list)
    serializable: dict[str, Any] = {}
    for k, v in config.items():
        if hasattr(v, "__iter__") and not isinstance(v, str):
            serializable[k] = list(v)
        else:
            serializable[k] = v

    cfg_path = os.path.join(output_dir, "experiment_config.json")
    with open(cfg_path, "w", encoding="utf-8") as fh:
        json.dump(serializable, fh, indent=2)
    print(f"Saved: {cfg_path}")
