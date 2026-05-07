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

    Usage::

        config = load_config_from_widgets(widgets)
    """
    return {k: widget.value for k, widget in w.items()}


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
    od_count = config.get("od_count", 200)
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
        assign_peak_desired_entry,
        mixed_alpha_sampler,
    )
    from ExpandedTimeSimulation.simulation_zefat.real_data import RealData

    graph_file = config.get("graph_file", "har_nof.gpickle")
    place_name = config.get("place_name", "Har Nof, Jerusalem, Israel")
    od_count = config.get("od_count", 200)
    slot_seconds = config.get("slot_seconds", 60)
    peak_slot = config.get("peak_slot", 10)
    peak_sigma = config.get("peak_sigma", 5.0)
    vehicle_T = config.get("vehicle_T", 100)
    base_seed = config.get("base_seed", 42)

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
    assign_peak_desired_entry(
        vehicles,
        schedule=PeakSchedule(
            peak_slot=peak_slot, sigma=peak_sigma, horizon_T=vehicle_T
        ),
        write_arrival=True,
    )

    cols = [
        "source", "destination", "alpha", "reserve",
        "desired_entry", "entry_fee", "lateness_fee",
    ]
    preview_df = (
        pd.DataFrame([{k: v[k] for k in cols if k in v} for v in vehicles[:10]])
        .round(3)
    )
    return vehicles, preview_df


def plot_demand_distribution(vehicles: list[dict]) -> None:
    """Two-panel histogram: desired entry time distribution and alpha distribution."""
    desired_entries = [v.get("desired_entry", 0) for v in vehicles]
    alphas = [v.get("alpha", 0.0) for v in vehicles]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(desired_entries, bins=30, color="steelblue", edgecolor="white", linewidth=0.4)
    axes[0].set_xlabel("Desired Entry Time Slot", fontsize=12)
    axes[0].set_ylabel("Number of Vehicles", fontsize=12)
    axes[0].set_title("Desired Entry Time Distribution", fontsize=13)
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].hist(alphas, bins=30, color="darkorange", edgecolor="white", linewidth=0.4)
    axes[1].set_xlabel("Alpha  (price–time weight)", fontsize=12)
    axes[1].set_ylabel("Number of Vehicles", fontsize=12)
    axes[1].set_title("Alpha Distribution", fontsize=13)
    axes[1].grid(axis="y", alpha=0.3)

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

    Thin wrapper around :func:`~experiments.batch_run.run_batch`.
    Diagnostic plots are suppressed here — use :func:`plot_results` after.
    """
    from ExpandedTimeSimulation.simulation_zefat.experiments.batch_run import run_batch

    excel_file = config.get("excel_file", "results.xlsx")

    strategy_keys = config.get("strategy_keys")
    if strategy_keys is not None:
        strategy_keys = list(strategy_keys)  # ipywidgets SelectMultiple returns tuple

    run_batch(
        num_runs=config.get("num_runs", 1),
        excel_file=excel_file,
        base_seed=config.get("base_seed", 2025),
        place_name=config.get("place_name", "Har Nof, Jerusalem, Israel"),
        graph_file=config.get("graph_file", "har_nof.gpickle"),
        od_count=config.get("od_count", 200),
        max_time_slots=config.get("max_time_slots", 20),
        vmax=float(config.get("vmax", 100.0)),
        r=int(config.get("r", 30)),
        capacity_is_hourly=bool(config.get("capacity_is_hourly", True)),
        slot_seconds=int(config.get("slot_seconds", 60)),
        vehicle_T=int(config.get("vehicle_T", 100)),
        peak_slot=int(config.get("peak_slot", 10)),
        peak_sigma=float(config.get("peak_sigma", 5.0)),
        strategy_keys=strategy_keys or None,
        run_diagnostics_plots=False,  # handled separately via plot_results()
        smooth_tail_u0=float(config.get("smooth_tail_u0", 0.95)),
    )
    return excel_file


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
    try:
        import folium
    except ImportError:
        print("folium is not installed. Run:  !pip install -q folium")
        return None

    from ExpandedTimeSimulation.simulation_zefat.real_data import RealData

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

    # Draw road edges
    for _, row in edges.iterrows():
        coords = [(lat, lon) for lon, lat in row.geometry.coords]
        folium.PolyLine(
            coords,
            color="#3388ff",
            weight=2.5,
            opacity=0.7,
        ).add_to(fmap)

    folium.Marker(
        [center_lat, center_lon],
        popup=f"Network centre\n{place_name}",
        icon=folium.Icon(color="red", icon="info-sign"),
    ).add_to(fmap)

    return fmap


# ---------------------------------------------------------------------------
# Export
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
