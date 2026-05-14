import math
import os
import pickle
import re
import random
from collections import defaultdict
from typing import Callable, Optional, Tuple, Union

import networkx as nx
import osmnx as ox


class RealData:
    """Loads and processes OSM road networks for time-expanded network simulations."""

    AlphaSpec = Union[float, Tuple[float, float], Callable[[], float]]

    def __init__(
        self,
        place_name: str,
        network_type: str = "drive",
        time_slot_duration: float = 1.0,
        od_count: int = 100,
    ):
        """Configure OSM source and sampling parameters."""
        self.place_name = place_name
        self.network_type = network_type
        self.time_slot_duration = time_slot_duration
        self.od_count = od_count

        self.G: nx.MultiDiGraph | None = None
        self.od_pairs: list[tuple] = []
        self.vehicles: list[dict] = []

    def load_graph(self, buffer_dist: float = 0.0) -> nx.MultiDiGraph:
        """Download the OSM graph for the specified place and network type.

        Args:
            buffer_dist: extra meters to expand beyond the place polygon boundary.
                         0 = strict boundary (default). Use e.g. 600 to include
                         surrounding access roads.
        """
        if buffer_dist > 0:
            self.G = ox.graph_from_place(
                self.place_name,
                network_type=self.network_type,
                buffer_dist=buffer_dist,
            )
        else:
            self.G = ox.graph_from_place(self.place_name, network_type=self.network_type)
        self.G = ox.project_graph(self.G)
        return self.G

    @staticmethod
    def extract_lanes(lanes_raw) -> int:
        """Parse a lanes attribute into an integer lane count (fallback=1)."""
        if isinstance(lanes_raw, list):
            try:
                return int(lanes_raw[0])
            except Exception:
                return 1

        if isinstance(lanes_raw, str):
            try:
                return int(lanes_raw)
            except Exception:
                return 1

        if isinstance(lanes_raw, (int, float)):
            return int(lanes_raw)

        return 1

    @staticmethod
    def estimate_capacity(road_type: str, lanes_raw) -> int:
        """Estimate edge capacity using road type defaults scaled by lane count."""
        lanes = RealData.extract_lanes(lanes_raw)
        base = {
            "motorway": 2000,
            "trunk": 1800,
            "primary": 1500,
            "secondary": 1200,
            "tertiary": 800,
            "residential": 600,
            "unclassified": 500,
            "service": 400,
        }
        return base.get(road_type, 600) * lanes

    @staticmethod
    def extract_speed(speed_raw, road_type: str) -> float:
        """Parse a maxspeed attribute into km/h using reasonable road-type defaults."""
        defaults = {
            "motorway": 100,
            "trunk": 80,
            "primary": 60,
            "secondary": 50,
            "tertiary": 40,
            "residential": 30,
            "unclassified": 30,
            "service": 20,
        }

        raw = speed_raw[0] if isinstance(speed_raw, list) else speed_raw

        if isinstance(raw, str):
            # Keep digits only (handles strings like "50", "50 km/h", etc.)
            nums = "".join(c for c in raw if c.isdigit())
            return float(nums) if nums else defaults.get(road_type, 30)

        if isinstance(raw, (int, float)):
            return float(raw)

        return defaults.get(road_type, 30)

    @staticmethod
    def compute_travel_time(length_m: float, speed_kph: float) -> float:
        """Compute travel time in seconds from length (meters) and speed (km/h)."""
        mps = speed_kph * 1000 / 3600
        return length_m / mps if mps > 0 else float("inf")

    def enrich_graph(self) -> None:
        """Add 'capacity', 'speed_kph', and 'travel_time' attributes to each OSM edge."""
        if self.G is None:
            raise RuntimeError("Graph not loaded. Call load_graph() first.")

        for _, _, _, data in self.G.edges(keys=True, data=True):
            rtype = data.get("highway", "residential")
            dtype = rtype[0] if isinstance(rtype, list) else rtype

            data["capacity"] = RealData.estimate_capacity(dtype, data.get("lanes", 1))
            speed = RealData.extract_speed(data.get("maxspeed"), dtype)

            data["speed_kph"] = speed
            data["travel_time"] = RealData.compute_travel_time(data.get("length", 0.0), speed)

    def generate_od_pairs(self) -> list[tuple]:
        """Randomly sample origin-destination node pairs from the loaded graph."""
        if self.G is None:
            raise RuntimeError("Graph not loaded. Call load_graph() first.")

        rng = random.Random(getattr(self, "seed", None))
        nodes = list(self.G.nodes)
        self.od_pairs = []

        for _ in range(self.od_count):
            o = rng.choice(nodes)
            d = rng.choice(nodes)
            while d == o:
                d = rng.choice(nodes)
            self.od_pairs.append((o, d))

        return self.od_pairs

    def convert_to_base_edges(self) -> list[tuple]:
        """Convert the enriched OSM graph and sampled OD demand into base_edges tuples.

        If edges already have a 'demand' attribute (set by enrich_graph_from_csv),
        that value is used directly and OD-routing is skipped.
        Otherwise demand is estimated by routing the sampled OD pairs.
        """
        if self.G is None:
            raise RuntimeError("Graph not loaded. Call load_graph() first.")

        # Check whether CSV-based demand was pre-set on every edge
        has_csv_demand = all(
            "demand" in data
            for _, _, _, data in self.G.edges(keys=True, data=True)
        )

        routing_demand: dict[tuple, int] = {}
        if not has_csv_demand:
            # Count demand by routing each OD through the graph (free-flow travel_time)
            routing_demand = defaultdict(int)
            for o, d in self.od_pairs:
                try:
                    path = nx.shortest_path(
                        self.G,
                        o,
                        d,
                        weight=lambda _, __, data: data.get("travel_time", 0.0),
                    )
                    for u, v in zip(path, path[1:]):
                        routing_demand[(u, v)] += 1
                except nx.NetworkXNoPath:
                    continue

        base_edges: list[tuple] = []
        for u, v, _, data in self.G.edges(keys=True, data=True):
            secs = data.get("travel_time", 0.0)
            slots = max(1, math.ceil(secs / self.time_slot_duration))
            cap = data["capacity"]
            dem = data.get("demand", routing_demand.get((u, v), cap))
            base_edges.append((u, v, slots, cap, dem))

        return base_edges

    def sample_alpha(self, a: AlphaSpec) -> float:
        """Sample an alpha value from a constant, range, or callable and clamp to [0, 2]."""
        if callable(a):
            x = a()
        elif isinstance(a, (tuple, list)) and len(a) == 2:
            lo, hi = a
            x = random.uniform(lo, hi)
        else:
            x = float(a)

        return max(0.0, min(2.0, x))

    def generate_vehicles(
        self,
        entry_fee: float = 5.0,
        reserve_range: tuple[float, float] = (1.0, 30.0),
        alpha: AlphaSpec = (0.2, 0.9),
        desired_entry: Optional[int] = 0,
        entry_fee_range: tuple[float, float] = (1.0, 5.0),
        lateness_fee_range: tuple[float, float] = (1.0, 5.0),
    ) -> list[dict]:
        """Generate vehicle dicts from OD pairs with sampled reserve and alpha values."""
        if not self.od_pairs:
            self.generate_od_pairs()

        vehicles: list[dict] = []
        for o, d in self.od_pairs:
            reserve = random.uniform(*reserve_range)
            a = self.sample_alpha(alpha)

            vehicles.append(
                {
                    "source": o,
                    "destination": d,
                    "desired_entry": 0 if desired_entry is None else desired_entry,
                    "desired_arrival": 0 if desired_entry is None else desired_entry,
                    "mode": "arrive",
                    "entry_fee": random.uniform(*entry_fee_range),
                    "lateness_fee": random.uniform(*lateness_fee_range),
                    "reserve": reserve,
                    "alpha": a,
                    "N": self.od_count,
                    # Metrics placeholders (filled later, e.g., by allocate_path)
                    "reserved_price": None,
                    "path_found": False,
                    "entry_time": None,
                    "exit_time": None,
                    "travel_time": None,
                    "paid_fee": 0.0,
                    "entry_delay": 0,
                    "arrival_delay": 0,
                    "reject_reason": 0,
                }
            )

        self.vehicles = vehicles
        return vehicles

    def plot(self, **kwargs) -> None:
        """Quick matplotlib plot of the loaded OSM graph."""
        if self.G is None:
            raise RuntimeError("Graph not loaded. Call load_graph() first.")
        ox.plot_graph(self.G, **kwargs)

    def save_graph(self, filepath: str) -> None:
        """Serialize the loaded/enriched graph to disk to avoid re-downloading."""
        if self.G is None:
            raise RuntimeError("Graph not loaded. Call load_graph()/load_graph_from_file() first.")
        with open(filepath, "wb") as f:
            pickle.dump(self.G, f)

    def load_graph_from_file(self, filepath: str) -> nx.MultiDiGraph:
        """Load a previously saved graph from disk."""
        with open(filepath, "rb") as f:
            self.G = pickle.load(f)
        return self.G

    def load_graph_from_osm_xml(self, osm_filepath: str) -> nx.MultiDiGraph:
        """Load and project an OSM graph from a local .osm XML file."""
        self.G = ox.graph_from_xml(osm_filepath, simplify=True)
        self.G = ox.project_graph(self.G)
        return self.G

    def enrich_graph_from_csv(self, csv_path: str, hour: int = 8) -> None:
        """
        Set capacity and demand on each graph edge from a SUMO-derived hourly CSV.

        CSV columns: osm_road_id (e.g. '1088477771#0' or '-1088477771#0'), hour,
                     capacity (vehicles/hour), expected_demand (vehicles/hour).

        SUMO splits each OSM way into segments (#0, #1, ...) and uses negative IDs
        for the reverse direction of two-way roads. Aggregation rules per direction:
          - capacity = MIN across segments (bottleneck limits the whole road)
          - demand   = MAX across segments (same vehicles pass through every segment)

        osmnx edges carry a boolean 'reversed' attribute (True when the edge runs
        opposite to the OSM way direction). We use it to match each edge to the
        correct SUMO direction (positive = forward, negative = reverse).

        Falls back to estimate_capacity() for edges not found in the CSV.
        Also sets speed_kph and travel_time (same as enrich_graph()).
        """
        import pandas as pd

        if self.G is None:
            raise RuntimeError("Graph not loaded. Call load_graph() first.")

        df = pd.read_csv(csv_path)
        df_hour = df[df["hour"] == hour]

        # Build per-direction lookups keyed by abs osmid string.
        # For each (way, direction), aggregate across all segments:
        #   capacity = MIN  — the bottleneck segment limits the whole road
        #   demand   = MAX  — same vehicles traverse all segments; peak segment
        #                     reflects true load (demand > capacity is valid and
        #                     intentional: it signals a saturated road that should
        #                     receive higher pricing pressure in the simulation)
        #   fwd_*: positive SUMO IDs  (forward direction of OSM way)
        #   rev_*: negative SUMO IDs  (reverse direction of OSM way)
        fwd_cap: dict[str, float] = {}
        fwd_dem: dict[str, float] = {}
        rev_cap: dict[str, float] = {}
        rev_dem: dict[str, float] = {}

        for _, row in df_hour.iterrows():
            rid = str(row["osm_road_id"])
            m = re.match(r"^(-?)(\d+)(?:#\d+)?$", rid)
            if not m:
                continue
            is_reverse = m.group(1) == "-"
            key = m.group(2)
            cap_val = float(row["capacity"])
            dem_val = float(row["expected_demand"])

            if is_reverse:
                rev_cap[key] = min(rev_cap.get(key, cap_val), cap_val)
                rev_dem[key] = max(rev_dem.get(key, 0.0),      dem_val)
            else:
                fwd_cap[key] = min(fwd_cap.get(key, cap_val), cap_val)
                fwd_dem[key] = max(fwd_dem.get(key, 0.0),      dem_val)

        matched = 0
        for _, _, _, data in self.G.edges(keys=True, data=True):
            rtype = data.get("highway", "residential")
            dtype = rtype[0] if isinstance(rtype, list) else rtype

            osmid = data.get("osmid")
            osmids = osmid if isinstance(osmid, list) else [osmid]
            # osmnx sets reversed=True when the edge runs opposite to the OSM way
            is_reversed = bool(data.get("reversed", False))

            found = False
            for oid in osmids:
                key = str(abs(int(oid)))
                if is_reversed:
                    if key in rev_cap:
                        data["capacity"] = rev_cap[key]
                        data["demand"]   = rev_dem[key]
                        found = True
                    elif key in fwd_cap:          # fallback: no reverse data, use forward
                        data["capacity"] = fwd_cap[key]
                        data["demand"]   = fwd_dem[key]
                        found = True
                else:
                    if key in fwd_cap:
                        data["capacity"] = fwd_cap[key]
                        data["demand"]   = fwd_dem[key]
                        found = True
                    elif key in rev_cap:          # fallback: no forward data, use reverse
                        data["capacity"] = rev_cap[key]
                        data["demand"]   = rev_dem[key]
                        found = True
                if found:
                    matched += 1
                    data["_has_csv_data"] = True
                    break

            if not found:
                data["_has_csv_data"] = False
                data["capacity"] = RealData.estimate_capacity(dtype, data.get("lanes", 1))
                data["demand"]   = data["capacity"]

            speed = RealData.extract_speed(data.get("maxspeed"), dtype)
            data["speed_kph"] = speed
            data["travel_time"] = RealData.compute_travel_time(data.get("length", 0.0), speed)

        print(f"[enrich_graph_from_csv] Matched {matched}/{self.G.number_of_edges()} edges (hour={hour}).")

        # Scale capacity of unmatched edges to match the real-world scale observed
        # in the CSV.  estimate_capacity() returns theoretical maximums; CSV values
        # reflect actual measured capacity for this specific neighbourhood.
        # Derive a per-road-type scale factor from the matched CSV edges.
        type_csv_cap: dict[str, list] = defaultdict(list)
        for _, _, _, data in self.G.edges(keys=True, data=True):
            if not data.get("_has_csv_data"):
                continue
            rtype = data.get("highway", "residential")
            dtype = rtype[0] if isinstance(rtype, list) else rtype
            theoretical = RealData.estimate_capacity(dtype, data.get("lanes", 1))
            if theoretical > 0:
                type_csv_cap[dtype].append(data["capacity"] / theoretical)

        # Fallback: if no matched edges for a type, use the overall mean scale factor
        overall_factors = [f for factors in type_csv_cap.values() for f in factors]
        overall_scale = sum(overall_factors) / len(overall_factors) if overall_factors else 1.0

        for _, _, _, data in self.G.edges(keys=True, data=True):
            if data.get("_has_csv_data"):
                continue
            rtype = data.get("highway", "residential")
            dtype = rtype[0] if isinstance(rtype, list) else rtype
            factors = type_csv_cap.get(dtype)
            scale = sum(factors) / len(factors) if factors else overall_scale
            data["capacity"] = max(data["capacity"] * scale, 1.0)

    def distribute_internal_demand(self) -> None:
        """Assign demand to internal edges by routing boundary flows through the network.

        Assumption: the neighbourhood has no through-traffic.
          - Vehicles entering via a boundary edge → random internal destination.
          - Vehicles leaving  via a boundary edge → random internal origin.

        The number of trips generated per boundary node is proportional to the
        total CSV demand on its incident edges (vehicles/hour).  Each routed trip
        adds 1 to the flow counter of every internal edge it traverses; the final
        counter is the demand (vehicles/hour) for that edge.

        Capacity is NOT modified — it stays as set by enrich_graph_from_csv()
        (estimate_capacity fallback for internal edges), so demand/capacity ratios
        will vary across edges and reflect real congestion potential.

        Must be called AFTER enrich_graph_from_csv().
        """
        if self.G is None:
            raise RuntimeError("Graph not loaded.")

        # ── 1. Classify nodes
        csv_nodes: set = set()
        for u, v, data in self.G.edges(data=True):
            if data.get("_has_csv_data", False):
                csv_nodes.add(u)
                csv_nodes.add(v)

        internal_nodes = list(set(self.G.nodes()) - csv_nodes)

        if len(internal_nodes) < 1:
            print("[distribute_internal_demand] No internal nodes; skipping.")
            return

        print(f"[distribute_internal_demand] Internal nodes: {len(internal_nodes)}, "
              f"boundary nodes: {len(csv_nodes)}")

        # ── 2. Collect CSV edges as (source_node, demand) for inbound trips and
        #       (dest_node, demand) for outbound trips.
        #       Each directed CSV edge (u→v, demand D) represents D vehicles/hour
        #       travelling on that road.  Since there is no through-traffic:
        #         • the vehicle entered the neighbourhood via this edge  →  origin=u
        #         • the vehicle will leave the neighbourhood via this edge →  dest=v
        #       We generate exactly D inbound trips (u → random internal) and
        #       D outbound trips (random internal → v), counting each vehicle once.
        csv_edge_flows: list[tuple] = []   # (u, v, demand)
        for u, v, data in self.G.edges(data=True):
            if data.get("_has_csv_data", False):
                csv_edge_flows.append((u, v, data["demand"]))

        total_trips = sum(d for _, _, d in csv_edge_flows)
        print(f"[distribute_internal_demand] CSV boundary edges: {len(csv_edge_flows)}, "
              f"total flow: {total_trips:.0f} veh/hr")

        # ── 3. Route trips: per-edge, not per-node, to avoid double-counting
        routing_count: dict[tuple, float] = defaultdict(float)
        routes_found = 0

        for b_src, b_dst, flow in csv_edge_flows:
            trips = max(1, int(round(flow)))

            # inbound: vehicle arrived via (b_src→b_dst) → heading to internal dest
            for _ in range(trips):
                dest = random.choice(internal_nodes)
                try:
                    path = nx.shortest_path(
                        self.G, b_dst, dest,
                        weight=lambda _u, _v, d: d.get("travel_time", 1.0),
                    )
                    for pu, pv in zip(path, path[1:]):
                        routing_count[(pu, pv)] += 1
                    routes_found += 1
                except nx.NetworkXNoPath:
                    pass

            # outbound: vehicle leaving via (b_src→b_dst) → came from internal origin
            for _ in range(trips):
                orig = random.choice(internal_nodes)
                try:
                    path = nx.shortest_path(
                        self.G, orig, b_src,
                        weight=lambda _u, _v, d: d.get("travel_time", 1.0),
                    )
                    for pu, pv in zip(path, path[1:]):
                        routing_count[(pu, pv)] += 1
                    routes_found += 1
                except nx.NetworkXNoPath:
                    pass

        print(f"[distribute_internal_demand] Routed {routes_found} trips.")

        # ── 4. Write demand onto non-CSV edges (capacity unchanged)
        assigned = 0
        for u, v, data in self.G.edges(data=True):
            if data.get("_has_csv_data", False):
                continue
            data["demand"] = routing_count.get((u, v), 0.0)
            assigned += 1

        print(f"[distribute_internal_demand] Demand assigned to {assigned} internal edges.")

    def inspect_graph(self, sample_edges: int = 5, sample_nodes: int = 5) -> None:
        """Print a summary of the loaded graph plus a small sample of nodes/edges."""
        if self.G is None:
            raise RuntimeError("Graph not loaded. Call load_graph() first.")

        print(f"Graph Summary for {self.place_name} ({self.network_type}):")
        print(f"  Number of nodes: {self.G.number_of_nodes()}")
        print(f"  Number of edges: {self.G.number_of_edges()}")

        print(f"Sample {sample_nodes} nodes:")
        for node in list(self.G.nodes)[:sample_nodes]:
            print(f"  Node: {node}")

        print(f"Sample {sample_edges} edges with attributes:")
        count = 0
        for u, v, k, data in self.G.edges(keys=True, data=True):
            print(f"  Edge: ({u} -> {v}, key={k})")
            processed = {kk: data[kk] for kk in ["capacity", "speed_kph", "travel_time"] if kk in data}
            raw = {kk: data.get(kk) for kk in ["name", "highway", "lanes", "maxspeed", "length"]}
            attrs = {**raw, **processed}
            print(f"    Attributes: {attrs}")

            count += 1
            if count >= sample_edges:
                break
    def get_node_xy_map(self) -> dict:
        """
        Return {node_id: (x, y)} in METERS.
        After ox.project_graph(), OSMnx nodes have x/y in projected CRS units (meters).
        """
        self.ensure_node_xy()
        xy_map = {}
        for n, data in self.G.nodes(data=True):
            x = float(data["x"])  # meters
            y = float(data["y"])  # meters
            xy_map[n] = (x, y)
        return xy_map

    def ensure_node_xy(self) -> None:
        """Ensure OSM nodes have x/y attributes (x=lon, y=lat)."""
        if self.G is None:
            raise RuntimeError("Graph not loaded. Call load_graph() first.")

        missing = []
        for n, data in self.G.nodes(data=True):
            if "x" not in data or "y" not in data:
                missing.append(n)
                if len(missing) >= 5:
                    break

        if missing:
            raise ValueError(f"Missing x/y on some nodes, e.g.: {missing[:5]}")

    def save_node_xy_map(self, filepath: str) -> str:
        """Persist normalized node (x,y) map to disk for later use."""
        if self.G is None:
            raise RuntimeError("Graph not loaded. Call load_graph() first.")
        xy_map = self.get_node_xy_map()
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(xy_map, f)
        return filepath

        
        
