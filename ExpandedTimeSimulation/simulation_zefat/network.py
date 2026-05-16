import ast
import pickle
import hashlib
import json
import math
import os
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict
from pprint import pprint
import random
import copy
import pandas as pd
from pathlib import Path

from ExpandedTimeSimulation.simulation_zefat.strategies import ZeroPricingStrategy
from .utils import edge_key
from .edge_data import EdgeData
import os, json, pickle, hashlib, ast


SCHEMA_VERSION = 6  # bump if EdgeData / cache layout changes

class TimeExpandedRoadNetwork:
    def __init__(self,
                 base_edges,
                 max_time_slots,
                 vmax,
                 r,
                 pricing_strategy,
                 for_demand=False,
                 capacity_is_hourly=True,
                 # NEW — bake once and persist:
                 expected_demand_file=None,        # Excel/CSV/Parquet path
                 expected_sheet="edge_metrics",    # for Excel
                 expected_col="alloc_count",       # or "requested_count"
                 expected_agg="mean",               # "sum" | "mean" | "max"
                 expected_run=None,                # filter run
                 expected_strategy=None,           # filter strategy
                 bake_expected_demand=False,       # set True ONCE to bake & save
                 slot_seconds=60,  
                 node_xy_file=None,     # NEW
                 node_xy_map=None       # NEW# correct per-slot capacity scaling
                 ):
        self.base_edges        = base_edges
        self.max_time_slots    = max_time_slots
        self.vmax              = vmax
        self.r                 = r
        self.pricing_strategy  = pricing_strategy
        self.for_demand        = for_demand
        self.capacity_is_hourly = capacity_is_hourly
        self.slot_seconds      = slot_seconds
        self.node_xy_map       = node_xy_map
        self.node_xy_file      = node_xy_file
        
        # cache fingerprint — include schema + core params + edge content
        edges_hash = hashlib.sha1(json.dumps(
            [list(e) for e in base_edges], sort_keys=True
        ).encode()).hexdigest()[:12]
        fingerprint = hashlib.sha1(json.dumps({
            "schema": SCHEMA_VERSION,
            "edges_hash": edges_hash,
            "T": max_time_slots,
            "vmax": vmax,
            "r": r
        }, sort_keys=True).encode()).hexdigest()[:8]
          
        cache_dir  = "cache"
        cache_file = f"expanded_net_{fingerprint}.pkl"
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, cache_file)
        
        if for_demand:
            self.graph     = nx.DiGraph()
            self.edge_data = {}
            self.build_time_expanded_graph(self.base_edges)
            self._save_cache(cache_path)
        else:  
            if os.path.exists(cache_path):
                with open(cache_path, "rb") as f:
                    loaded = pickle.load(f)
                if isinstance(loaded, tuple) and len(loaded) == 2:
                    self.graph, self.edge_data = loaded
                elif isinstance(loaded, nx.DiGraph):
                    self.graph = loaded
                    self.edge_data = {}
                    self.build_time_expanded_graph(self.base_edges)
                elif isinstance(loaded, dict):
                    self.graph     = loaded.get("graph")
                    self.edge_data = loaded.get("edge_data", {})
                else:
                    raise ValueError(f"Unexpected cache format: {type(loaded)}")
            else:
                self.graph     = nx.DiGraph()
                self.edge_data = {}
                self.build_time_expanded_graph(self.base_edges)
                self._save_cache(cache_path)
        # Optional one-time bake of expected demand (and re-save)
        if bake_expected_demand and expected_demand_file:
            demap = self._load_expected_demand_map(
                expected_demand_file, expected_sheet, expected_col,
                expected_agg, expected_run, expected_strategy
            )
            applied = self._apply_expected_demand(demap)
            print(f"[expected-demand] applied to {applied} time-edges; saving cache…")
            #self._save_cache(cache_path)
        self.node_xy_map = node_xy_map
        if self.node_xy_map is None and node_xy_file is not None:
            with open(node_xy_file, "rb") as f:
                self.node_xy_map = pickle.load(f)
    def _save_cache(self, cache_path):
        with open(cache_path, "wb") as f:
            pickle.dump((self.graph, self.edge_data), f)

    def _load_expected_demand_map(self, path, sheet, col, agg, run, strategy):
        # Load DataFrame from Excel/CSV/Parquet
        ext = os.path.splitext(path)[1].lower()
        if ext in [".xlsx", ".xls"]:
            df = pd.read_excel(path, sheet_name=sheet)
        elif ext == ".csv":
            df = pd.read_csv(path)
        elif ext in [".parquet", ".pq"]:
            df = pd.read_parquet(path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")

        # Optional filters
        if run is not None and "run" in df.columns:
            df = df[df["run"] == run]
        if strategy is not None and "strategy" in df.columns:
            df = df[df["strategy"] == strategy]

        if "edge" not in df.columns:
            raise ValueError("expected-demand file must contain 'edge' column")
        if col not in df.columns:
            raise ValueError(f"expected-demand file must contain '{col}' column")

        # Parse edge literal "((u,t),(v,t2))"
        def parse_edge(x):
            if isinstance(x, tuple): return x
            try:
                return ast.literal_eval(x)
            except Exception:
                return None

        df = df.assign(edge_parsed=df["edge"].apply(parse_edge)).dropna(subset=["edge_parsed"])
        gb = df.groupby("edge_parsed")[col]
        if   agg == "sum":  s = gb.sum()
        elif agg == "mean": s = gb.mean()
        elif agg == "max":  s = gb.max()
        else: raise ValueError("expected_agg must be one of: 'sum','mean','max'")
        return {edge: float(val) for edge, val in s.items()}

    def _apply_expected_demand(self, demap):
        hit = 0
        for edge, val in demap.items():
            if edge in self.edge_data:
                self.edge_data[edge].demand = val
                hit += 1
        return hit

    def build_time_expanded_graph(self, base_edges):
        for u, v, time_to_travel, capacity_in, demand in base_edges:
            # correct per-slot capacity scaling
            cap = capacity_in
            if self.capacity_is_hourly:
                cap = max(2, int(round(capacity_in * self.slot_seconds / 3600.0)))
            else:
                cap = max(2, int(capacity_in))

            # physical endpoints' (x,y) = (lon,lat)
            if self.node_xy_map is not None:
                ux, uy = self.node_xy_map.get(u, (None, None))
                vx, vy = self.node_xy_map.get(v, (None, None))
            else:
                ux = uy = vx = vy = None

            for t in range(self.max_time_slots):
                t_arrival = t + time_to_travel
                if t_arrival > self.max_time_slots:
                    continue

                source_node = (u, t)
                target_node = (v, t_arrival)

                # Ensure nodes exist with x/y attributes for A*
                self.graph.add_node(source_node, x=ux, y=uy)
                self.graph.add_node(target_node, x=vx, y=vy)

                edge = (source_node, target_node)
                self.graph.add_edge(*edge)
                self.edge_data[edge] = EdgeData(time_to_travel, cap, demand)
            
    def plot_time_expanded_network(self, include_virtual=False):
        plt.figure(figsize=(14, 8))
        G = self.graph
        pos = {}

        base_nodes = sorted({node[0] for node in G.nodes()})
        y_levels = {name: -i for i, name in enumerate(base_nodes)}

        nodes_to_draw = []
        edges_to_draw = []
        for u, v in G.edges():
            if not include_virtual and ("VIRTUAL" in u[0] or "VIRTUAL" in v[0]):
                continue
            nodes_to_draw.extend([u, v])
            edges_to_draw.append((u, v))
        nodes_to_draw = list(set(nodes_to_draw))

        for node in nodes_to_draw:
            name, t = node
            pos[node] = (t, y_levels.get(name, 0))

        H = G.subgraph(nodes_to_draw).copy()
        labels = {node: f"{node[0]}@{node[1]}" for node in H.nodes()}

        nx.draw(H, pos, with_labels=True, labels=labels, node_size=800, node_color='lightgray', font_size=8)
        plt.title("Time-Expanded Network (Aligned and Complete)")
        plt.show()

    def get_cost(self, edge, alpha, vehicle=None,time_only=False):
        # Define the virtual entry/exit nodes
        vs = ("VIRTUAL_SOURCE", edge[0][1])
        vt = ("VIRTUAL_TARGET", edge[1][1])
        # If this is a virtual edge, cost is purely delay‐based
        if edge[0] == vs or edge[1] == vt:
            # if isinstance(self.pricing_strategy, ZeroPricingStrategy):
            #     return 0.0
            cost = 0
            if vehicle:
                if edge[0] == vs and vehicle.get("mode") == "enter":
                    fee_rate = vehicle.get("entry_fee", 1.0)
                    entry_delay = abs(edge[1][1] - vehicle["desired_entry"])
                    cost += fee_rate * entry_delay
                elif edge[1] == vt and vehicle.get("mode") == "arrive":
                    fee_rate = vehicle.get("lateness_fee", 1.0)
                    desired_arrival = vehicle.get("desired_arrival", edge[0][1])
                    arrival_delay = abs(edge[0][1] - desired_arrival)
                    cost += fee_rate * arrival_delay
            return cost

        # Otherwise, retrieve the real edge data
        data = self.edge_data[edge]
        if data.is_saturated(): 
        #  if data.is_saturated() and not self.for_demand: #to mark
            return float('inf')

        if data.price == 0:
            self.pricing_strategy.init_price(self, edge)
        if time_only: #smart reserve only
            return data.travel_time
        # Blend price and travel time
        return data.unit_price() +  alpha * data.travel_time

    def allocate_path(self, path, vehicle, s_max):
        """
        Try to allocate `path` to `vehicle`. Records:
        - reserved_price, path_found, entry/exit/travel_time, paid_fee
        - request_count per edge (true demand)
        """
        vs = ("VIRTUAL_SOURCE", path[0][1])
        vt = ("VIRTUAL_TARGET", path[-1][1])
        allocated_edge_list = [(path[i], path[i+1]) for i in range(len(path) - 1)]

        # --- NEW: record true demand pressure on each real edge ---
        for u, v in allocated_edge_list:
            if (u == vs) or (v == vt):
                continue
            self.edge_data[(u, v)].request_count += 1

        # 1) Compute total toll cost (skip virtual edges)
        real_cost = 0.0
        for u, v in allocated_edge_list:
            if (u == vs) or (v == vt):
                continue
            data = self.edge_data[(u, v)]
            # allow traversal beyond capacity in expected-demand runs
            if data.alloc_count >= data.capacity: 
            #  if data.alloc_count >= data.capacity and not self.for_demand: #to mark
                real_cost = float('inf')
                break
            real_cost += data.unit_price()

        # 2) Decide whether to allocate
        #    Use real_cost as a safe fallback if vehicle has no 'reserve'
        reserve = vehicle.get("reserve", real_cost)
        dijkstra_cost = vehicle.get("dijkstra_cost", real_cost)
        success = (dijkstra_cost != float('inf') 
                    and dijkstra_cost <= reserve
                    and real_cost <= reserve
                   )
        if not success:
            if real_cost == float('inf'):
                vehicle["reject_reason"] = 1
            elif dijkstra_cost > reserve or real_cost > reserve:
                vehicle["reject_reason"] = 2
        else:
            vehicle["reject_reason"] = 0
        # 3) Record per-vehicle metrics (use first/last REAL nodes for times)
        real_nodes = [n for n in path if n != vs and n != vt]
        if real_nodes:
            entry_time = real_nodes[0][1]
            exit_time  = real_nodes[-1][1]
        else:
            entry_time = path[0][1]
            exit_time  = path[-1][1]

        vehicle["reserved_price"] = reserve
        vehicle["path_found"]     = success
        vehicle["entry_time"]     = entry_time
        vehicle["exit_time"]      = exit_time
        vehicle["travel_time"]    = (exit_time - entry_time) if success else None
        vehicle["paid_fee"]       = real_cost if success else 0.0
        
        # --- SAFE DELAY COMPUTATION ---
        et = vehicle.get("entry_time")
        xt = vehicle.get("exit_time")
        de = vehicle.get("desired_entry")
        da = vehicle.get("desired_arrival")

        # if desired_* is None, treat delay as 0 for that side
        vehicle["entry_delay"] = (
            abs(et - de) if (et is not None and de is not None) else 0
        )
        vehicle["arrival_delay"] = (
            abs(xt - da) if (xt is not None and da is not None) else 0
        )
        # 4) If allocation succeeded, update counters (no price moves under Null strategy)
        if success:
            for u, v in allocated_edge_list:
                if (u == vs) or (v == vt):
                    continue
                data = self.edge_data[(u, v)]
                data.record_price()     # will just append 0 under Null strategy
                data.alloc_count += 1
                if not self.for_demand:  # skip price updates in expected-demand mode
                    data.update_price(self.pricing_strategy, self, (u, v), vehicle, s_max)
                    # print("expected demand",data.demand, data.capacity)  # Debugging output@
        return success, real_cost

    def plot_base_network(self):
        G = nx.DiGraph()
        # Use base_edges for static network
        for u, v, time, capacity, demand in self.base_edges:
            G.add_edge(u, v, label=f"T:{time}, C:{capacity}, D:{demand}")

        pos = nx.spring_layout(G, seed=42)
        edge_labels = nx.get_edge_attributes(G, 'label')

        plt.figure(figsize=(10, 6))
        nx.draw(G, pos, with_labels=True, node_size=1000, node_color='lightblue', font_size=10)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
        plt.title("Base Road Network (Non-Time-Expanded)")
        plt.tight_layout()
        plt.show()

    def plot_base_network2(self):
        G = nx.DiGraph()
        for u, v in self.graph.edges():
            G.add_edge(u, v)

        base_nodes = sorted({node[0] for node in G.nodes()})
        y_levels = {name: -i for i, name in enumerate(base_nodes)}
        # fallback to spring layout
        pos = nx.spring_layout(G, seed=42)

        plt.figure(figsize=(10, 6))
        nx.draw(G, pos, with_labels=True, node_size=1000, node_color='lightblue', font_size=10)
        plt.title("Base Road Network (Non-Time-Expanded)")
        plt.tight_layout()
        plt.show()

 

    def calculate_path_cost(self, path, alpha, vehicle=None):
        total_cost = 0.0
        for u, v in zip(path, path[1:]):
            edge = (u, v)
            c = self.get_cost(edge, alpha, vehicle)
            if c == float('inf'):
                return float('inf')
            total_cost += c
        return total_cost

    def print_vehicle_info(self, vehicles):
        # print("\nVehicle Allocation Summary (one line per vehicle):")
        # print("# | Src->Dst | mode | alpha | entry->arrival | real_cost | reserve | status | path")
        for i, v in enumerate(vehicles, 1):
            if v.get("allocated_path"):
                filtered = [
                n for n in v["allocated_path"]
                if n[0] not in ("VIRTUAL_SOURCE", "VIRTUAL_TARGET")
            ]
                path_str = "->".join(f"{n[0]}@{n[1]}" for n in filtered)
            else:
                path_str = "None"

            rc = v.get("real_cost")
            rc_str = "∞" if rc == float("inf") else f"{rc:.2f}" if rc is not None else "N/A"
            status = "ACCEPT" if rc not in (None, float("inf")) and rc <= v.get("reserve", 0) else "REJECT"

            # print(f"{i:2d} | {v['source']}->{v['destination']} | {v['mode']} | "
            #       f"{v['alpha']:.2f} | {v['entry_time']}->{v['exit_time']} | "
            #       f"{v['real_cost']:.5f} | {v['reserve']:.2f} | {status:^7} | {path_str}")

    def print_edge_price_summary(self):
        print("\nFinal Edge Prices (total, per unit, and capacity):")
        for edge, data in self.edge_data.items():
            unit_price = data.unit_price()
            used_price = data.last_used_price()
            used_unit = used_price / data.capacity if data.capacity else 0
            tj = data.travel_time
            ttotal = self.max_time_slots
            vmax_j = self.vmax * ((min(data.capacity, data.demand) / data.capacity) if data.capacity else 0) * (tj / ttotal)
            print(f"Edge {edge}: Total={data.price:.2f}, Per Unit={unit_price:.2f}, Capacity={data.capacity}, Demand={data.demand}, actual_demand={data.alloc_count} Time={tj} Last Used={used_price:.2f}, Used Per Unit={used_unit:.2f}, vmax_j={vmax_j:.2f}")

    def plot_edge_price_history(self):
        plt.figure(figsize=(12, 6))
        for edge, data in self.edge_data.items():
            times = list(range(len(data.price_history)))
            unit_history = [(p / data.capacity) if data.capacity else 0 for p in data.price_history]
            label = f"{edge[0][0]}{edge[0][1]}->{edge[1][0]}{edge[1][1]}"
            plt.plot(times, unit_history, label=label)
        plt.xlabel("Round")
        plt.ylabel("Unit Price")
        plt.title("Segment Price History per Unit Capacity")
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.tight_layout()
        plt.show()
