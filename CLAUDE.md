# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Purpose

Academic research project simulating and comparing online auction mechanisms for allocating road capacity in a time-expanded road network. Five dynamic pricing strategies are implemented and benchmarked against each other.

## Running the Project

### Interactive (Google Colab notebook)
Open `Transportation_Auction_Simulation.ipynb` and run cells B–F sequentially for a full experiment. Sections G–I provide analysis and visualization.

### Command-Line (C:\Transportation_vmax_per_segment)
```powershell
# Batch run all strategies, export to Excel
python -m ExpandedTimeSimulation.simulation_zefat.experiments.cli batch --runs 3 --excel results.xlsx

# Expected-demand baseline (capacity doesn't block routing)
python -m ExpandedTimeSimulation.simulation_zefat.experiments.cli demand --runs 5 --excel demand.xlsx

# TNTP benchmark network
python -m ExpandedTimeSimulation.simulation_zefat.experiments.cli tntp --dir tntp_networks --network SiouxFalls

# Toy-network entry point
python ExpandedTimeSimulation/main.py

# Full Zefat simulation with parameter sweep
python ExpandedTimeSimulation/simulation_zefat/main.py
```

### Dependencies
```powershell
pip install networkx osmnx pandas numpy matplotlib folium openpyxl
```

## Architecture

architecture:
- `c:\Users\חוה\Documents\transportation-auction-colab` — Jupyter/Colab notebook version (this repo)
- 
### Core Data Flow

```
RealData (OSM → base_edges)
  → TimeExpandedRoadNetwork (builds time-expanded graph, cached to cache/)
    → AuctionSimulator (routes vehicles, updates prices per strategy)
      → Analyzer (converts results to DataFrames)
        → Excel export (5 sheets) + Plots
```

### Key Modules

**`network.py` — `TimeExpandedRoadNetwork`**
Nodes are `(physical_node, time_slot)` pairs. Builds a directed graph where each edge carries an `EdgeData` object. Caches built graphs to `cache/expanded_net_<hash>.pkl`.

**`edge_data.py` — `EdgeData`**
Per-edge mutable state: price, capacity, demand, alloc_count, request_count, price_history. Core methods: `unit_price()`, `is_saturated()`, `update_price()`.

**`strategies.py` — Pricing strategies**
Abstract `PricingStrategy` base; five implementations:
- `DynamicPricingStrategy` — Transport-adapted, exponential with vmax scaling + travel-time component
- `AlternativePricingStrategy` (Online Competitive) — BG-style exponential with global `r`, `s_max`
- `ZeroPricingStrategy` — Baseline, no tolls
- `MedianPricingStrategy` — Price frozen at half-capacity utilization
- `SmoothTailPricingStrategy` — Exponential on [0, u0], cubic Hermite transition at saturation

**`auction_simulator.py` — `AuctionSimulator`**
Processes vehicles sequentially. For each vehicle: adds virtual source/target nodes, finds shortest path (configurable solver), allocates if cost ≤ reserve price, then calls the active strategy's price update.

**`real_data.py` — `RealData`**
Downloads OSM graphs via osmnx, enriches edges with capacity/speed/travel-time, generates synthetic vehicle fleets with random OD pairs and timing preferences.

**`analyzer.py`**
`compute_metrics()`, `collect_edge_metrics()`, `run_summary()` — converts simulation state to DataFrames. Excel output has sheets: `vehicle_metrics`, `edge_metrics`, `vehicles_table`, `edge_timeslices`, `run_summary`.

**`experiments/batch_run.py` + `cli.py`**
`run_batch()` / `run_batch_for_demand()` / `run_batch_tntp()` execute multi-strategy, multi-repetition sweeps and export all metrics to Excel. Strategy instances are created via a factory dict.

### Vehicle Model

Each vehicle is a dict with: `source`, `destination`, `alpha` (price–time weight), `reserve` (max WTP), `desired_entry`, `desired_arrival`, `entry_fee`, `lateness_fee`, `mode` ("entry"/"arrival"). Post-allocation fields: `path_found`, `real_cost`, `paid_fee`, `travel_time`, `entry_delay`, `arrival_delay`, `reject_reason`.

### Edge Cost Formula

```
cost(edge) = unit_price(edge) + alpha × travel_time(edge)
```

Virtual source/target edges contribute only delay penalties (`entry_fee` or `lateness_fee` × slot offset).

## Key Configuration Parameters

| Parameter | Role |
|---|---|
| `max_time_slots` (T) | Network time horizon |
| `vmax` | Price ceiling / urgency scaling |
| `r` | Online Competitive base rate |
| `od_count` | Number of OD pairs |
| `slot_seconds` | Time slot duration (affects capacity scaling) |
| `path_solver` | `dijkstra`, `bidirectional_dijkstra`, `astar_euclidean`, `astar_fflb`, `astar_fflb_delay`, `astar_reverse_arrival` |
| `smooth_tail_u0` | Transition occupancy for Smooth Tail strategy |
| `capacity_is_hourly` | Whether to scale edge capacity to per-slot |
