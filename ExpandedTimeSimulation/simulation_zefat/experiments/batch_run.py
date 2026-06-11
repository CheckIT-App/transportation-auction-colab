# ExpandedTimeSimulation/simulation_zefat/experiments/batch_run.py
# ---------------------------------------------------------------------------
# Batch utilities:
#   1) run_batch: run repetitions across multiple strategies, export full artifacts to Excel
#   2) run_batch_for_demand: build expected-demand baseline (capacity doesn't block routing)
# ---------------------------------------------------------------------------

from __future__ import annotations

import copy
import os
import random
from typing import Optional

import numpy as np
import pandas as pd
import osmnx as ox

from ExpandedTimeSimulation.simulation_zefat.auction_simulator import AuctionSimulator
from ExpandedTimeSimulation.simulation_zefat.network import TimeExpandedRoadNetwork
from ExpandedTimeSimulation.simulation_zefat.plots.plots import plot_sim_diagnostics
from ExpandedTimeSimulation.simulation_zefat.real_data import RealData
from ExpandedTimeSimulation.simulation_zefat.strategy_factory import make_strategy
from ExpandedTimeSimulation.simulation_zefat.analyzer import (
    collect_edge_metrics,
    compute_metrics,
    edge_timeslices_to_df,
    run_summary,
    vehicles_to_df,
)

from ExpandedTimeSimulation.simulation_zefat.constants import (
    # sheets
    SHEET_VEHICLE_METRICS,
    SHEET_EDGE_METRICS,
    SHEET_VEHICLES_TABLE,
    SHEET_EDGE_TIMESLICES,
    SHEET_RUN_SUMMARY,
    # common cols
    COL_RUN,
    COL_STRATEGY,
    COL_N,
    # strategy names
    STRAT_ZERO,
    STRAT_TRANSPORT_ADAPTED,
    STRAT_STATIC_MEDIAN,
    STRAT_ONLINE_COMPETITIVE,
    STRAT_SMOOTH_TAIL,
)

from ExpandedTimeSimulation.simulation_zefat.experiments.vehicle_generation import (
    mixed_alpha_sampler,
    assign_peak_desired_entry,
    assign_peak_desired_time_by_mode,
    PeakSchedule,
)

# ---------------------------------------------------------------------------
# Optional console diagnostics (keep here for now; you can move later if you want)
# ---------------------------------------------------------------------------


def analyze_edges(edges_df: pd.DataFrame) -> None:
    """Print summary stats per strategy for edge-level metrics."""
    if edges_df.empty:
        print("analyze_edges: edges_df is empty.")
        return

    cols = ["utilization", "price_increase", "peak_price", "demand", "alloc_count"]
    missing = [c for c in cols if c not in edges_df.columns]
    if missing:
        print(f"analyze_edges: missing columns: {missing}")
        return

    edge_summary = edges_df.groupby(COL_STRATEGY).agg(
        {
            "utilization": ["mean", "std"],
            "price_increase": ["mean", "std"],
            "peak_price": ["mean", "std"],
            "demand": "sum",
            "alloc_count": "sum",
        }
    )
    print(edge_summary)

    if "request_count" in edges_df.columns and "capacity" in edges_df.columns:
        print("\n=== Top 10% most utilized segments by alloc/capacity ratio (per strategy) ===")
        edges_df = edges_df.copy()
        edges_df["request_ratio"] = edges_df["alloc_count"] / edges_df["capacity"].replace(0, float("nan"))
        # Fixed N = 10% of the smallest strategy group — same count for all strategies
        n_fixed = int(min(len(grp) for _, grp in edges_df.groupby(COL_STRATEGY)) * 0.2)
        for strategy, grp in edges_df.groupby(COL_STRATEGY):
            top10pct = grp.nlargest(n_fixed, "request_ratio")
            over = top10pct[top10pct["request_count"] > top10pct["capacity"]]
            under = top10pct[top10pct["request_count"] <= top10pct["capacity"]]
            print(f"  {strategy} (n={len(top10pct)} segments):")
            for label, subset in [("requests > capacity", over), ("requests <= capacity", under)]:
                if subset.empty:
                    print(f"    [{label}] — none")
                else:
                    req = subset['request_count'].sum()
                    alloc = subset['alloc_count'].sum()
                    pct = alloc / req * 100 if req else 0
                    print(f"    [{label}] n={len(subset)}, mean utilization = {subset['utilization'].mean():.3f}, total requested = {req}, total allocated = {alloc} ({pct:.1f}%)")


def analyze_expected_demand(edges_df: pd.DataFrame) -> None:
    """
    Print expected-demand diagnostics aggregated at base-edge level.
    Expects edge to be ((u,t),(v,t2)) tuples.
    """
    if edges_df.empty:
        print("analyze_expected_demand: edges_df is empty.")
        return

    if "edge" not in edges_df.columns:
        print("analyze_expected_demand: missing 'edge' column.")
        return

    df = edges_df.copy()

    # Unpack ((u,t),(v,t2)) → u,v,t and per-slot utilization
    df["u"] = df["edge"].apply(lambda e: e[0][0])
    df["v"] = df["edge"].apply(lambda e: e[1][0])
    df["t"] = df["edge"].apply(lambda e: e[0][1])

    if "capacity" not in df.columns or "alloc_count" not in df.columns:
        print("analyze_expected_demand: missing 'capacity' and/or 'alloc_count'.")
        return

    df["util_slot"] = df["alloc_count"] / df["capacity"].replace(0, np.nan)

    # request_count vs requested_count
    if "request_count" in df.columns:
        req_col = "request_count"
    elif "requested_count" in df.columns:
        req_col = "requested_count"
    else:
        df["request_count"] = 0
        req_col = "request_count"

    df[req_col] = pd.to_numeric(df[req_col], errors="coerce").fillna(0)

    keys = [COL_RUN, COL_STRATEGY, "u", "v"]

    agg1 = df.groupby(keys, observed=False, as_index=False).agg(
        util_max=("util_slot", "max"),
        util_mean=("util_slot", "mean"),
        total_alloc=("alloc_count", "sum"),
        total_requested=(req_col, "sum"),
        total_capacity=("capacity", "sum"),
    )

    slots = (
        df.assign(_has_alloc=df["alloc_count"] > 0)
        .groupby(keys, observed=False)["_has_alloc"]
        .sum()
        .rename("slots_with_demand")
        .reset_index()
    )

    base = agg1.merge(slots, on=keys, how="left").fillna(
        {"slots_with_demand": 0, "total_requested": 0}
    )

    time_sat_rate = (df["util_slot"] >= 1.0).mean()
    print(f"time-slice saturation rate (util>=1): {time_sat_rate:.2%}")

    base_sat_rate = (base["util_max"] >= 1.0).mean()
    print(f"base-edge saturation rate (peak util>=1): {base_sat_rate:.2%}")

    print("\n=== Base-edge level (aggregated over time) ===")
    base_active = base[(base["total_alloc"] > 0) | (base["total_requested"] > 0)]
    print("base edges active:", len(base_active))
    if not base_active.empty:
        print(
            "mean util (time-slot mean) over active base-edges:",
            base_active["util_mean"].mean(),
        )
        print(
            "mean of peak util over active base-edges:",
            base_active["util_max"].mean(),
        )
        print(
            "base-edge peak util p90/p95:",
            base_active["util_max"].quantile([0.9, 0.95]).to_dict(),
        )

    top = base_active.sort_values("util_max", ascending=False).head(15)[
        ["u", "v", "util_max", "slots_with_demand", "total_alloc", "total_requested"]
    ]
    print("\nTop 15 base-edges by peak utilization:")
    print(top)


# ---------------------------------------------------------------------------
# Strategy selection
# ---------------------------------------------------------------------------


def _build_strategies(
    strategy_keys: Optional[list[str]] = None,
    smooth_tail_u0: float = 0.95,
) -> dict[str, object]:
    """
    Build strategies dict {strategy_name: instance}.
    Defaults to your 4 named strategies in constants.py (stable order).
    """
    if strategy_keys is None:
        strategy_keys = [
            STRAT_TRANSPORT_ADAPTED,
            STRAT_ONLINE_COMPETITIVE,
            STRAT_STATIC_MEDIAN,
            STRAT_ZERO,
            STRAT_SMOOTH_TAIL,
        ]
    from ExpandedTimeSimulation.simulation_zefat.strategies import SmoothTailPricingStrategy
    out: dict[str, object] = {}
    for k in strategy_keys:
        if str(k) == STRAT_SMOOTH_TAIL:
            out[str(k)] = SmoothTailPricingStrategy(u0=smooth_tail_u0)
        else:
            out[str(k)] = make_strategy(str(k))
    return out


# ---------------------------------------------------------------------------
# Batch runs
# ---------------------------------------------------------------------------


def run_batch(
    num_runs: int,
    *,
    excel_file: str = "all_runs.xlsx",
    base_seed: int = 2025,
    place_name: str = "Har Nof, Jerusalem, Israel",
    graph_file: str = "har_nof.gpickle",
    od_count: int = 1000,
    max_time_slots: int = 50,
    vmax: float = 100,
    r: int = 30,
    capacity_is_hourly: bool = True,
    slot_seconds: int = 60,
    vehicle_T: int = 100,
    peak_slot: int = 30,
    peak_sigma: float = 5.0,
    strategy_keys: Optional[list[str]] = None,
    run_diagnostics_plots: bool = True,
    diagnostics_strategy: Optional[str] = STRAT_TRANSPORT_ADAPTED,
    plots_only: bool = False,
    osm_xml_file: Optional[str] = None,
    demand_csv: Optional[str] = None,
    demand_hour: int = 16,
    buffer_dist: float = 0.0,
    distribute_demand: bool = False,
    distribute_od_count: int = 5000,
    smooth_tail_u0: float = 0.95,
    expected_demand_file: Optional[str] = None,
    demand_fraction: Optional[float] = None,
    time_mode: str = "entry",
    arrival_percentage: float = 0.5,
    capacity_factor: float = 1.0,
    alpha_lo: float = 0.0,
    alpha_hi: float = 1.0,
    entry_fee_lo: float = 1.0,
    entry_fee_hi: float = 5.0,
    lateness_fee_lo: float = 1.0,
    lateness_fee_hi: float = 5.0,
    reserve_lo: float = 1.0,
    reserve_hi: float = 100.0,
    bpr_alpha: float = 0.15,
    bpr_beta: float = 4.0,
) -> None:
    """
    Runs multiple repetitions.
    For each run: runs all strategies, exports:
      - vehicle_metrics
      - edge_metrics
      - vehicles_table
      - edge_timeslices
      - run_summary

    plots_only:
      - if True, skip simulation and only run diagnostics plots from `excel_file`
    """
    if plots_only:
        if not run_diagnostics_plots:
            print("plots_only=True and run_diagnostics_plots=False; nothing to do.")
            return
        if not os.path.exists(excel_file):
            raise FileNotFoundError(
                f"plots_only=True requires an existing Excel file, but not found: {excel_file}"
            )
        print(f"Plots-only mode: loading diagnostics from {excel_file}")
        plot_sim_diagnostics(
            excel_file,
            horizon_T=vehicle_T,
            fs=15,
        )
        return

    if num_runs < 1:
        raise ValueError("num_runs must be >= 1 when plots_only=False.")

    # 1) Load or build graph + projected-meter XY map once
    node_xy_file = os.path.splitext(graph_file)[0] + "_node_xy_meters.pkl"

    loader, node_xy_file = _load_or_build_graph_and_xy(
        place_name=place_name,
        graph_file=graph_file,
        node_xy_file=node_xy_file,
        time_slot_duration=slot_seconds,
        od_count=od_count,
        osm_xml_file=osm_xml_file,
        demand_csv=demand_csv,
        demand_hour=demand_hour,
        buffer_dist=buffer_dist,
        distribute_demand=distribute_demand,
        distribute_od_count=distribute_od_count,
    )
    # 2) Generate vehicles once (copied per strategy/run)
    vehicles = loader.generate_vehicles(
        alpha=(alpha_lo, alpha_hi),
        reserve_range=(reserve_lo, reserve_hi),
        entry_fee_range=(entry_fee_lo, entry_fee_hi),
        lateness_fee_range=(lateness_fee_lo, lateness_fee_hi),
    )
    assign_peak_desired_time_by_mode(
        vehicles,
        schedule=PeakSchedule(
            peak_slot=peak_slot, sigma=peak_sigma, horizon_T=vehicle_T
        ),
        mode=time_mode,
        arrival_percentage=arrival_percentage,
    )

    # 3) Choose strategies
    strategies = _build_strategies(strategy_keys, smooth_tail_u0=smooth_tail_u0)

    records: list[dict] = []
    all_edge_frames: list[pd.DataFrame] = []
    all_vehicle_rows: list[pd.DataFrame] = []
    all_timeslice_rows: list[pd.DataFrame] = []
    all_run_summaries: list[pd.DataFrame] = []

    last_net = None  # for diagnostics plots that want net.edge_data

    for run_idx in range(1, num_runs + 1):
        print(f"\n=== BATCH RUN {run_idx}/{num_runs} ===")

        # base edges reused for all strategies within this run
        loader.seed = base_seed + run_idx  # if your RealData uses it internally
        loader.generate_od_pairs()
        base_edges = loader.convert_to_base_edges(demand_fraction=demand_fraction)

        if capacity_factor != 1.0:
            base_edges = [
                (u, v, t, cap * capacity_factor, demand)
                for u, v, t, cap, demand in base_edges
            ]

        for strat_name, strat in strategies.items():
            print(f"  running strategy: {strat_name}")

            net = TimeExpandedRoadNetwork(
                base_edges,
                max_time_slots=max_time_slots,
                vmax=vmax,
                r=r,
                pricing_strategy=strat,
                for_demand=False,
                capacity_is_hourly=capacity_is_hourly,
                slot_seconds=slot_seconds,
                node_xy_file=node_xy_file,
                expected_demand_file=expected_demand_file,
                bake_expected_demand=expected_demand_file is not None,
            )
            last_net = net

            sim = AuctionSimulator(net, copy.deepcopy(vehicles))
            sim.run()
            print("    simulation finished")

            net.measure_realized_times(sim.vehicles, bpr_alpha=bpr_alpha, bpr_beta=bpr_beta)

            metrics = compute_metrics(sim.vehicles)
            records.append({COL_RUN: run_idx, COL_STRATEGY: strat_name, **metrics})

            vdf = vehicles_to_df(sim.vehicles, run_idx, strat_name)
            all_vehicle_rows.append(vdf)

            tsdf = edge_timeslices_to_df(net.edge_data, run_idx, strat_name)
            all_timeslice_rows.append(tsdf)

            df_edges = collect_edge_metrics(net.edge_data, strat_name, run_idx)
            all_edge_frames.append(df_edges)

            run_sum = run_summary(vdf, tsdf, run_idx, strat_name)
            all_run_summaries.append(run_sum)

    df_vehicle_metrics = pd.DataFrame(records)
    edges_df = (
        pd.concat(all_edge_frames, ignore_index=True)
        if all_edge_frames
        else pd.DataFrame()
    )
    df_v_table = (
        pd.concat(all_vehicle_rows, ignore_index=True)
        if all_vehicle_rows
        else pd.DataFrame()
    )
    df_ts_table = (
        pd.concat(all_timeslice_rows, ignore_index=True)
        if all_timeslice_rows
        else pd.DataFrame()
    )
    df_run_summ = (
        pd.concat(all_run_summaries, ignore_index=True)
        if all_run_summaries
        else pd.DataFrame()
    )

    # Optional console summaries
    if not edges_df.empty:
        analyze_edges(edges_df)

    # Export
    with pd.ExcelWriter(excel_file) as writer:
        df_vehicle_metrics.to_excel(
            writer, sheet_name=SHEET_VEHICLE_METRICS, index=False
        )
        edges_df.to_excel(writer, sheet_name=SHEET_EDGE_METRICS, index=False)
        df_v_table.to_excel(writer, sheet_name=SHEET_VEHICLES_TABLE, index=False)
        df_ts_table.to_excel(writer, sheet_name=SHEET_EDGE_TIMESLICES, index=False)
        df_run_summ.to_excel(writer, sheet_name=SHEET_RUN_SUMMARY, index=False)

    print(f"\nDone—all results saved to {excel_file}")

    if run_diagnostics_plots:
        if last_net is None:
            print("Diagnostics skipped: last_net is None.")
            return
        plot_sim_diagnostics(
            excel_file,
            horizon_T=vehicle_T,
            fs=15,
        )


def run_batch_for_demand(
    num_runs: int,
    *,
    excel_file: str = "demand.xlsx",
    base_seed: int = 2025,
    place_name: str = "Har Nof, Jerusalem, Israel",
    graph_file: str = "har_nof.gpickle",
    od_count: int = 4000,
    max_time_slots: int = 100,
    vmax: float = 100,
    r: int = 30,
    slot_seconds: int = 60,
    vehicle_T: int = 100,
    peak_slot: int = 30,
    peak_sigma: float = 5.0,
    osm_xml_file: Optional[str] = None,
    demand_csv: Optional[str] = None,
    demand_hour: int = 16,
    buffer_dist: float = 0.0,
    distribute_demand: bool = False,
    distribute_od_count: int = 5000,
) -> None:
    """
    Expected-demand baseline:
      - for_demand=True so routing is NOT blocked by capacity
      - uses Zero strategy (from registry) under STRAT_ZERO key
      - exports only vehicle_metrics + edge_metrics (like your previous version)
    """
    node_xy_file = os.path.splitext(graph_file)[0] + "_node_xy_meters.pkl"

    loader, node_xy_file = _load_or_build_graph_and_xy(
        place_name=place_name,
        graph_file=graph_file,
        node_xy_file=node_xy_file,
        time_slot_duration=slot_seconds,
        od_count=od_count,
        osm_xml_file=osm_xml_file,
        demand_csv=demand_csv,
        demand_hour=demand_hour,
        buffer_dist=buffer_dist,
        distribute_demand=distribute_demand,
        distribute_od_count=distribute_od_count,
    )
    zero = make_strategy(STRAT_ZERO)

    records: list[dict] = []
    all_edge_frames: list[pd.DataFrame] = []

    for run_idx in range(1, num_runs + 1):
        print(f"\n=== EXPECTED-DEMAND RUN {run_idx}/{num_runs} ===")

        loader.seed = base_seed + run_idx
        loader.generate_od_pairs()
        base_edges = loader.convert_to_base_edges()

        net = TimeExpandedRoadNetwork(
            base_edges,
            max_time_slots=max_time_slots,
            vmax=vmax,
            r=r,
            pricing_strategy=zero,
            for_demand=True,
            slot_seconds=slot_seconds,
            node_xy_file=node_xy_file,  # NEW
        )

        vehicles = loader.generate_vehicles()
        assign_peak_desired_entry(
            vehicles,
            schedule=PeakSchedule(
                peak_slot=peak_slot, sigma=peak_sigma, horizon_T=vehicle_T
            ),
            write_arrival=True,
        )
        sim = AuctionSimulator(net, copy.deepcopy(vehicles))
        sim.run()
        print("    simulation finished")

        metrics = compute_metrics(sim.vehicles)
        records.append({COL_RUN: run_idx, COL_STRATEGY: STRAT_ZERO, **metrics})

        df_edges = collect_edge_metrics(net.edge_data, STRAT_ZERO, run_idx)
        all_edge_frames.append(df_edges)

    df_vehicle_metrics = pd.DataFrame(records)
    edges_df = (
        pd.concat(all_edge_frames, ignore_index=True)
        if all_edge_frames
        else pd.DataFrame()
    )

    if not edges_df.empty:
        analyze_edges(edges_df)

    with pd.ExcelWriter(excel_file) as writer:
        df_vehicle_metrics.to_excel(
            writer, sheet_name=SHEET_VEHICLE_METRICS, index=False
        )
        edges_df.to_excel(writer, sheet_name=SHEET_EDGE_METRICS, index=False)

    print(f"\nDone—expected-demand baseline built; results saved to {excel_file}")


def _load_or_build_graph_and_xy(
    *,
    place_name: str,
    graph_file: str,
    node_xy_file: str,
    network_type: str = "drive",
    time_slot_duration: int = 60,
    od_count: int = 1000,
    osm_xml_file: Optional[str] = None,
    demand_csv: Optional[str] = None,
    demand_hour: int = 16,
    buffer_dist: float = 0.0,
    distribute_demand: bool = False,
    distribute_od_count: int = 5000,
) -> tuple[RealData, str]:
    """
    Returns:
      loader: RealData with loader.G loaded (projected to meters + enriched)
      node_xy_file: path to {node_id: (x,y)} in meters

    Behavior:
      - If graph_file exists: load it; if not projected -> project; then enrich; then save.
      - If graph_file missing and osm_xml_file provided: load from OSM XML, enrich, save.
      - If graph_file missing and no osm_xml_file: download from OSM (with buffer_dist), enrich, save.
      - If demand_csv provided: use enrich_graph_from_csv() instead of enrich_graph().
      - If distribute_demand: call distribute_internal_demand() after CSV enrichment.
      - If node_xy_file exists: keep it; else extract+save.
    """
    loader = RealData(
        place_name,
        network_type=network_type,
        time_slot_duration=time_slot_duration,
        od_count=od_count,
    )

    # ---- 1) Graph ----
    if os.path.exists(graph_file):
        loader.load_graph_from_file(graph_file)

        # Ensure projected CRS (meters). If CRS missing or not projected -> project now.
        crs = loader.G.graph.get("crs") if loader.G is not None else None
        try:
            is_proj = ox.projection.is_projected(crs)
        except Exception:
            is_proj = False

        if not is_proj:
            loader.G = ox.project_graph(loader.G)

    elif osm_xml_file is not None and os.path.exists(osm_xml_file):
        print(f"[graph] Loading from OSM XML: {osm_xml_file}")
        loader.load_graph_from_osm_xml(osm_xml_file)
    else:
        # Download + project to meters (optionally with buffer around place boundary)
        loader.load_graph(buffer_dist=buffer_dist)

    # Enrich edges with capacity, speed, travel_time (and optionally real demand)
    if demand_csv is not None and os.path.exists(demand_csv):
        print(f"[graph] Enriching from demand CSV: {demand_csv} (hour={demand_hour})")
        loader.enrich_graph_from_csv(demand_csv, hour=demand_hour)
        if distribute_demand:
            print(f"[graph] Distributing internal demand (od_count={distribute_od_count})")
            loader.distribute_internal_demand()
    else:
        loader.enrich_graph()

    # Persist so next run skips loading/downloading
    loader.save_graph(graph_file)

    # ---- 2) XY map ----
    if not os.path.exists(node_xy_file):
        loader.save_node_xy_map(node_xy_file)

    return loader, node_xy_file


# ---------------------------------------------------------------------------
# TNTP batch run
# ---------------------------------------------------------------------------


def run_batch_tntp(
    num_runs: int,
    *,
    tntp_dir: str,
    network_name: str,
    excel_file: str = "tntp_results.xlsx",
    base_seed: int = 2025,
    od_count: int = 500,
    max_time_slots: int = 50,
    vmax: float = 100,
    r: int = 30,
    slot_seconds: int = 60,
    free_flow_time_unit: str = "minutes",
    demand_od_sample: int = 3000,
    vehicle_T: int = 100,
    peak_slot: int = 30,
    peak_sigma: float = 5.0,
    strategy_keys: Optional[list[str]] = None,
    run_diagnostics_plots: bool = True,
    smooth_tail_u0: float = 0.95,
    bpr_alpha: float = 0.15,
    bpr_beta: float = 4.0,
) -> None:
    """Run a batch experiment on a TNTP benchmark network.

    Args:
        tntp_dir:            Directory containing the TNTP files.
        network_name:        File prefix (e.g. "SiouxFalls").
        excel_file:          Output Excel workbook path.
        od_count:            Vehicles per run.
        demand_od_sample:    OD pairs routed to estimate per-edge demand.
        free_flow_time_unit: "minutes" (Sioux Falls, Anaheim) or "hours".
        other args:          Same semantics as run_batch().
    """
    from ExpandedTimeSimulation.simulation_zefat.tntp_data import TNTPData

    # ── 1. Load TNTP network (once) ─────────────────────────────────────────
    td = TNTPData.from_directory(
        tntp_dir, network_name,
        time_slot_seconds=slot_seconds,
        free_flow_time_unit=free_flow_time_unit,
    )

    node_xy_file = os.path.join(tntp_dir, f"{network_name}_node_xy.pkl")
    node_xy_file = td.save_node_xy_map(node_xy_file) or None   # None if no coords

    base_edges = td.get_base_edges(demand_od_sample=demand_od_sample)

    # ── 2. Strategies ────────────────────────────────────────────────────────
    strategies = _build_strategies(strategy_keys, smooth_tail_u0=smooth_tail_u0)

    records: list[dict] = []
    all_edge_frames: list[pd.DataFrame] = []
    all_vehicle_rows: list[pd.DataFrame] = []
    all_timeslice_rows: list[pd.DataFrame] = []
    all_run_summaries: list[pd.DataFrame] = []

    for run_idx in range(1, num_runs + 1):
        print(f"\n=== TNTP BATCH RUN {run_idx}/{num_runs} ({network_name}) ===")
        random.seed(base_seed + run_idx)

        vehicles = td.get_vehicles(
            od_count=od_count,
            peak_slot=peak_slot,
            peak_sigma=peak_sigma,
            vehicle_T=vehicle_T,
        )

        for strat_name, strat in strategies.items():
            print(f"  running strategy: {strat_name}")

            net = TimeExpandedRoadNetwork(
                base_edges,
                max_time_slots=max_time_slots,
                vmax=vmax,
                r=r,
                pricing_strategy=strat,
                for_demand=False,
                capacity_is_hourly=True,
                slot_seconds=slot_seconds,
                node_xy_file=node_xy_file,
            )

            sim = AuctionSimulator(net, copy.deepcopy(vehicles))
            sim.run()

            net.measure_realized_times(sim.vehicles, bpr_alpha=bpr_alpha, bpr_beta=bpr_beta)

            metrics = compute_metrics(sim.vehicles)
            records.append({COL_RUN: run_idx, COL_STRATEGY: strat_name, **metrics})

            all_vehicle_rows.append(vehicles_to_df(sim.vehicles, run_idx, strat_name))
            all_timeslice_rows.append(edge_timeslices_to_df(net.edge_data, run_idx, strat_name))
            all_edge_frames.append(collect_edge_metrics(net.edge_data, strat_name, run_idx))
            all_run_summaries.append(run_summary(
                all_vehicle_rows[-1], all_timeslice_rows[-1], run_idx, strat_name
            ))

    df_vehicle_metrics = pd.DataFrame(records)
    edges_df    = pd.concat(all_edge_frames,    ignore_index=True) if all_edge_frames    else pd.DataFrame()
    df_v_table  = pd.concat(all_vehicle_rows,   ignore_index=True) if all_vehicle_rows   else pd.DataFrame()
    df_ts_table = pd.concat(all_timeslice_rows, ignore_index=True) if all_timeslice_rows else pd.DataFrame()
    df_run_summ = pd.concat(all_run_summaries,  ignore_index=True) if all_run_summaries  else pd.DataFrame()

    if not edges_df.empty:
        analyze_edges(edges_df)

    with pd.ExcelWriter(excel_file) as writer:
        df_vehicle_metrics.to_excel(writer, sheet_name=SHEET_VEHICLE_METRICS, index=False)
        edges_df.to_excel(   writer, sheet_name=SHEET_EDGE_METRICS,    index=False)
        df_v_table.to_excel( writer, sheet_name=SHEET_VEHICLES_TABLE,  index=False)
        df_ts_table.to_excel(writer, sheet_name=SHEET_EDGE_TIMESLICES, index=False)
        df_run_summ.to_excel(writer, sheet_name=SHEET_RUN_SUMMARY,     index=False)

    print(f"\nDone — results saved to {excel_file}")

    if run_diagnostics_plots:
        plot_sim_diagnostics(excel_file, horizon_T=vehicle_T, fs=15)
