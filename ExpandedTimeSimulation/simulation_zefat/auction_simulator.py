import math
import os
import pickle
import networkx as nx
import osmnx as ox

from ExpandedTimeSimulation.simulation_zefat.strategies import DynamicPricingStrategy, ZeroPricingStrategy, AlternativePricingStrategy, MedianPricingStrategy


def precompute_fflb(network, fflb_file: str, *, nodes=None) -> dict:
    """
    Pre-compute free-flow lower-bound shortest-path maps for all physical nodes and
    save to *fflb_file*.  Returns the cache dict so the caller can use it immediately.

    Keys: fflb["by_src"][n] = {node: travel_time}  (forward Dijkstra from n)
          fflb["by_dst"][n] = {node: travel_time}  (reverse Dijkstra from n)

    Pass nodes=<iterable> to limit computation to a subset (e.g. only OD nodes).
    """
    phys_fwd = nx.DiGraph()
    phys_rev = nx.DiGraph()
    for edge, data in network.edge_data.items():
        u, v = edge[0][0], edge[1][0]
        w = float(data.travel_time)
        if phys_fwd.get_edge_data(u, v, default={}).get("weight", float("inf")) > w:
            phys_fwd.add_edge(u, v, weight=w)
        if phys_rev.get_edge_data(v, u, default={}).get("weight", float("inf")) > w:
            phys_rev.add_edge(v, u, weight=w)

    compute_nodes = list(nodes) if nodes is not None else list(phys_fwd.nodes())
    by_src, by_dst = {}, {}
    for n in compute_nodes:
        if n in phys_fwd:
            by_src[n] = nx.single_source_dijkstra_path_length(phys_fwd, n, weight="weight")
        if n in phys_rev:
            by_dst[n] = nx.single_source_dijkstra_path_length(phys_rev, n, weight="weight")

    cache = {"by_src": by_src, "by_dst": by_dst}
    os.makedirs(os.path.dirname(os.path.abspath(fflb_file)), exist_ok=True)
    with open(fflb_file, "wb") as f:
        pickle.dump(cache, f)
    return cache


def load_fflb(fflb_file: str) -> dict | None:
    """Load a precomputed FFLB cache. Returns None if the file does not exist."""
    if not os.path.exists(fflb_file):
        return None
    with open(fflb_file, "rb") as f:
        return pickle.load(f)


class AuctionSimulator:
    """Runs an online auction simulation by finding and allocating a path per vehicle."""

    def __init__(self, network, vehicles, path_solver="astar_reverse_arrival",
                 fflb_cache: dict | None = None):
        """Initialize simulator state and prepare a reusable time-expanded graph skeleton."""
        self.strategies = {
            "Transport-Adapted": DynamicPricingStrategy(),
            # "Online": AlternativePricingStrategy(),
            # "Zero Pricing": ZeroPricingStrategy(),
            # "Static Median": MedianPricingStrategy(),
        }
        
        self.network = network
        self.vehicles = vehicles
        self.s_max = 1
        self.accepted = []
        self.path_solver = path_solver

        # Build once: time-expanded graph (no per-vehicle virtual edges yet)
        self.G = self.network.graph.copy()

        # Virtual source/target are kept in the graph; edges are added/removed per vehicle
        self.vs = ("VIRTUAL_SOURCE", 0)
        self.vt = ("VIRTUAL_TARGET", 0)
        self.G.add_node(self.vs)
        self.G.add_node(self.vt)
        # self.G = ox.project_graph(self.G)
        self.vmax_mps = 40.0  # For heuristic scaling in A* (meters per second)
        pos_xy = {}
        for node, data in self.G.nodes(data=True):
            # node is (phys, t)
            if not isinstance(node, tuple) or len(node) != 2:
                continue
            phys, _t = node
            if isinstance(phys, str) and phys.startswith("VIRTUAL"):
                continue

            x = data.get("x")  # lon
            y = data.get("y")  # lat
            if x is None or y is None:
                continue

            if phys not in pos_xy:
                pos_xy[phys] = (float(x), float(y))

        self.pos = self.network.node_xy_map or pos_xy

        # Physical reverse graph (dst -> all nodes shortest path tree) for FFLB A* heuristic.
        self._physical_rev = nx.DiGraph()
        self._physical_fwd = nx.DiGraph()
        for edge, data in self.network.edge_data.items():
            u = edge[0][0]
            v = edge[1][0]
            w = float(data.travel_time)
            prev = self._physical_rev.get_edge_data(v, u, default={}).get("weight")
            if prev is None or w < prev:
                self._physical_rev.add_edge(v, u, weight=w)
            prev_fwd = self._physical_fwd.get_edge_data(u, v, default={}).get("weight")
            if prev_fwd is None or w < prev_fwd:
                self._physical_fwd.add_edge(u, v, weight=w)
        if fflb_cache is not None:
            self._fflb_cache_by_dst = fflb_cache.get("by_dst", {})
            self._fflb_cache_by_src = fflb_cache.get("by_src", {})
        else:
            self._fflb_cache_by_dst = {}
            self._fflb_cache_by_src = {}

    def run(self):
        """Process vehicles sequentially: find a path, allocate it, and record outcomes."""
        for idx, v in enumerate(self.vehicles):
            # print(f"Vehicle {idx}")

            path = self.find_path_for_vehicle(v)
            # print(f"{self.path_solver} path found")

            if path:
                self.s_max = max(self.s_max, len(path))

                dijkstra_cost = self.network.calculate_path_cost(path, v["alpha"], v)
                # print("dijkstra cost calculated", dijkstra_cost)
                v["dijkstra_cost"] = dijkstra_cost

                _, real_cost = self.network.allocate_path(path, v, self.s_max)

                v["allocated_path"] = path
                v["real_cost"] = real_cost

                if (
                    dijkstra_cost != float("inf")
                    and dijkstra_cost <= v["reserve"]
                    and real_cost <= v["reserve"]
                ):
                    self.accepted.append(v)
                    # print("path allocated","v[reserve]:", v["reserve"], "real_cost:", real_cost, "dijkstra_cost:", dijkstra_cost)
                    
            else:
                dijkstra_cost = float("inf")
                real_cost = float("inf")
                v["dijkstra_cost"] = dijkstra_cost
                v["real_cost"] = float("inf")
                # print("path rejected", "v[reserve]:", v["reserve"], "real_cost:", real_cost, "dijkstra_cost:", dijkstra_cost)

            # print(f"Vehicle {v['source']} -> {v['destination']} ")

        self.network.print_vehicle_info(self.vehicles)
        self.report_objectives()

    def find_path_for_vehicle(self, vehicle):
        """Find a feasible (time-consistent) path by adding temporary virtual entry/exit edges."""
        alpha = vehicle["alpha"]
        src = vehicle["source"]
        dst = vehicle["destination"]
        T = self.network.max_time_slots

        entry_edges = []
        exit_edges = []

        # Add temporary virtual edges: vs -> (src,t) and (dst,t) -> vt
        for t in range(T):
            src_node = (src, t)
            if self.G.has_node(src_node):
                e = (self.vs, src_node)
                entry_edges.append(e)
                self.G.add_edge(*e)

            dst_node = (dst, t)
            if self.G.has_node(dst_node):
                e = (dst_node, self.vt)
                exit_edges.append(e)
                self.G.add_edge(*e)

        try:
            src_phys = vehicle["source"]
            dst_phys = vehicle["destination"]

            # Give virtual nodes the coordinates of THIS vehicle's physical endpoints
            if src_phys in self.pos:
                self.pos["VIRTUAL_SOURCE"] = self.pos[src_phys]
            if dst_phys in self.pos:
                self.pos["VIRTUAL_TARGET"] = self.pos[dst_phys]

            weight = lambda u, v, d: self.network.get_cost((u, v), vehicle["alpha"], vehicle)

            if self.path_solver == "dijkstra":
                path = nx.dijkstra_path(self.G, self.vs, self.vt, weight=weight)
            elif self.path_solver == "bidirectional_dijkstra":
                _, path = nx.bidirectional_dijkstra(self.G, self.vs, self.vt, weight=weight)
            elif self.path_solver == "astar_euclidean":
                path = nx.astar_path(
                    self.G,
                    self.vs,
                    self.vt,
                    heuristic=lambda u, v: self.astar_heuristic(u, v, self.pos, self.vmax_mps, vehicle["alpha"]),
                    weight=weight,
                )
            elif self.path_solver == "astar_fflb":
                self._ensure_fflb_for_destination(dst_phys)
                path = nx.astar_path(
                    self.G,
                    self.vs,
                    self.vt,
                    heuristic=lambda u, v: self.astar_fflb_heuristic(u, src_phys, dst_phys, vehicle["alpha"]),
                    weight=weight,
                )
            elif self.path_solver == "astar_fflb_delay":
                self._ensure_fflb_for_destination(dst_phys)
                h_cache = {}
                path = nx.astar_path(
                    self.G,
                    self.vs,
                    self.vt,
                    heuristic=lambda u, v: self.astar_fflb_delay_heuristic(
                        u, src_phys, dst_phys, vehicle, h_cache
                    ),
                    weight=weight,
                )
            elif self.path_solver == "astar_reverse_arrival":
                if vehicle.get("mode") == "arrival":
                    # Arrival mode: reverse search is usually more effective.
                    self._ensure_fflb_from_source(src_phys)
                    Gr = self.G.reverse(copy=False)
                    h_cache = {}
                    rev_path = nx.astar_path(
                        Gr,
                        self.vt,
                        self.vs,
                        heuristic=lambda u, v: self.astar_reverse_arrival_heuristic(
                            u, src_phys, dst_phys, vehicle["alpha"], h_cache
                        ),
                        weight=lambda u, v, d: self.network.get_cost((v, u), vehicle["alpha"], vehicle),
                    )
                    path = list(reversed(rev_path))
                else:
                    # Enter mode: use the regular forward graph heuristic.
                    self._ensure_fflb_for_destination(dst_phys)
                    h_cache = {}
                    path = nx.astar_path(
                        self.G,
                        self.vs,
                        self.vt,
                        heuristic=lambda u, v: self.astar_fflb_delay_heuristic(
                            u, src_phys, dst_phys, vehicle, h_cache
                        ),
                        weight=weight,
                    )
            elif self.path_solver == "reverse_dijkstra":
                if vehicle.get("mode") == "arrival":
                    # Arrival mode: reverse the graph and run Dijkstra from vt -> vs.
                    Gr = self.G.reverse(copy=False)
                    rev_path = nx.dijkstra_path(
                        Gr,
                        self.vt,
                        self.vs,
                        weight=lambda u, v, d: self.network.get_cost((v, u), vehicle["alpha"], vehicle),
                    )
                    path = list(reversed(rev_path))
                else:
                    # Entry mode: plain forward Dijkstra.
                    path = nx.dijkstra_path(self.G, self.vs, self.vt, weight=weight)
            else:
                raise ValueError(
                    "Unknown path_solver '{}'. Use one of: dijkstra, bidirectional_dijkstra, "
                    "astar_euclidean, astar_fflb, astar_fflb_delay, astar_reverse_arrival, reverse_dijkstra".format(self.path_solver)
                )
            return path
        except nx.NetworkXNoPath:
            return None
        finally:
            # Always clean up temporary virtual edges
            for u, v in entry_edges + exit_edges:
                if self.G.has_edge(u, v):
                    self.G.remove_edge(u, v)

    def _ensure_fflb_for_destination(self, dst_phys):
        """Cache free-flow lower-bound travel-time map to a given physical destination."""
        if dst_phys in self._fflb_cache_by_dst:
            return
        if dst_phys not in self._physical_rev:
            self._fflb_cache_by_dst[dst_phys] = {}
            return
        self._fflb_cache_by_dst[dst_phys] = nx.single_source_dijkstra_path_length(
            self._physical_rev, dst_phys, weight="weight"
        )

    def _ensure_fflb_from_source(self, src_phys):
        """Cache free-flow lower-bound travel-time map from a given physical source."""
        if src_phys in self._fflb_cache_by_src:
            return
        if src_phys not in self._physical_fwd:
            self._fflb_cache_by_src[src_phys] = {}
            return
        self._fflb_cache_by_src[src_phys] = nx.single_source_dijkstra_path_length(
            self._physical_fwd, src_phys, weight="weight"
        )

    def astar_fallback_heuristic(self, u, src_phys, dst_phys, alpha):
        """
        A* heuristic from free-flow lower bound (FFLB):
        h((node,t)) = alpha * min_travel_time_slots(node -> dst) .
        This is admissible because all ignored terms are non-negative.
        """
        if u == self.vt:
            return 0.0
        if u == self.vs:
            dist = self._fflb_cache_by_dst.get(dst_phys, {}).get(src_phys, float("inf"))
            return 0.0 if dist == float("inf") else alpha * dist

        u_phys = u[0] if isinstance(u, tuple) else u
        dist = self._fflb_cache_by_dst.get(dst_phys, {}).get(u_phys, float("inf"))
        return 0.0 if dist == float("inf") else alpha * dist

    def astar_fflb_delay_heuristic(self, u, src_phys, dst_phys, vehicle, cache):
        """
        Tighter admissible lower bound:
          alpha * LB(travel_time to destination)
          + LB(arrival delay penalty from earliest possible arrival).
        """
        if u in cache:
            return cache[u]

        alpha = float(vehicle.get("alpha", 1.0))
        mode = vehicle.get("mode")
        desired_arrival = vehicle.get("desired_arrival")
        lateness_fee = float(vehicle.get("lateness_fee", 0.0))

        if u == self.vt:
            h = 0.0
        elif u == self.vs:
            dist = self._fflb_cache_by_dst.get(dst_phys, {}).get(src_phys, float("inf"))
            if dist == float("inf"):
                h = 0.0
            else:
                # Lower bound from source cannot assume entry-delay lower bound > 0.
                h = alpha * dist
        else:
            u_phys = u[0] if isinstance(u, tuple) else u
            t_now = u[1] if isinstance(u, tuple) and len(u) == 2 else 0

            dist = self._fflb_cache_by_dst.get(dst_phys, {}).get(u_phys, float("inf"))
            if dist == float("inf"):
                h = 0.0
            else:
                h = alpha * dist
                if mode == "arrival" and desired_arrival is not None and lateness_fee > 0:
                    earliest_arrival = t_now + dist
                    # Since time only moves forward, if earliest arrival is already late,
                    # this lateness is unavoidable and can be added as a lower bound.
                    unavoidable_late = max(0.0, earliest_arrival - float(desired_arrival))
                    h += lateness_fee * unavoidable_late

        cache[u] = h
        return h

    def astar_reverse_arrival_heuristic(self, u, src_phys, dst_phys, alpha, cache):
        """
        Heuristic for reversed-graph A* (searching vt -> vs).
        Lower bound is the free-flow time from source to current physical node.
        """
        if u in cache:
            return cache[u]

        if u == self.vs:
            h = 0.0
        elif u == self.vt:
            dist = self._fflb_cache_by_src.get(src_phys, {}).get(dst_phys, float("inf"))
            h = 0.0 if dist == float("inf") else alpha * dist
        else:
            u_phys = u[0] if isinstance(u, tuple) else u
            dist = self._fflb_cache_by_src.get(src_phys, {}).get(u_phys, float("inf"))
            h = 0.0 if dist == float("inf") else alpha * dist

        cache[u] = h
        return h

    def report_objectives(self):
        """Print primal/dual objectives using accepted vehicles and last-used edge prices."""
        dual_sum = sum(v["reserve"] for v in self.accepted)

        primal_sum = dual_sum + sum(
            data.price_history[-2] if len(data.price_history) > 1 else 0.0
            for data in self.network.edge_data.values()
        )

        ratio = primal_sum / dual_sum if dual_sum > 0 else float("inf")
        print(f"Dual Objective (utility of accepted vehicles): {dual_sum:.2f}")
        print(f"Primal Objective (dual + total edge prices): {primal_sum:.2f}")
        print(f"Ratio (Primal/Dual): {ratio:.2f}")

    def astar_heuristic(self, u, v, pos_xy, vmax_mps, alpha):
        u_phys = u[0] if isinstance(u, tuple) else u
        v_phys = v[0] if isinstance(v, tuple) else v

        # Virtual SOURCE has no spatial position as a regular graph node.
        if isinstance(u_phys, str) and u_phys.startswith("VIRTUAL"):
            return 0.0

        if u_phys not in pos_xy or v_phys not in pos_xy:
            return 0.0

        x1, y1 = pos_xy[u_phys]
        x2, y2 = pos_xy[v_phys]
        dx, dy = x2 - x1, y2 - y1
        dist_m = (dx*dx + dy*dy) ** 0.5

        return alpha * (dist_m / vmax_mps)
