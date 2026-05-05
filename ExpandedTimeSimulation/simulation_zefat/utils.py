import random

import networkx as nx


def edge_key(edge):
    """Return a stable key for an edge based on its endpoint node IDs."""
    return (edge[0][0], edge[1][0])


def smart_reserve(
    network,
    vehicle,
    tau_time=4.0,          # ₪ per time-slot
    k_fee=1.5,             # fee weight
    safety_margin=(0.10, 0.30),
    rng=None,
):
    """Compute a reserve value R_i based on toll/time tradeoff plus a safety margin and fees."""
    if rng is None:
        rng = random

    alpha = vehicle["alpha"]
    fee = vehicle.get("entry_fee", vehicle.get("lateness_fee", 0.0))
    src = vehicle["source"]
    dst = vehicle["destination"]

    G = network.graph
    vs = ("VIRTUAL_SOURCE", 0)
    vt = ("VIRTUAL_TARGET", 0)

    # Ensure the virtual nodes exist once
    if not G.has_node(vs):
        G.add_node(vs)
    if not G.has_node(vt):
        G.add_node(vt)

    # Hook up temporary virtual edges and remember them for cleanup
    added = []
    for t in range(network.max_time_slots):
        e_in = (vs, (src, t))
        G.add_edge(*e_in)
        added.append(e_in)

        node = (dst, t)
        if G.has_node(node):
            e_out = (node, vt)
            G.add_edge(*e_out)
            added.append(e_out)

    try:
        # Cheapest-toll path (alpha=1)
        price_w = lambda u, v, d: network.get_cost((u, v), alpha=1.0, time_only=True)
        try:
            p_path = nx.dijkstra_path(G, vs, vt, weight=price_w)
            C_toll = sum(
                network.edge_data[(u, v)].unit_price()
                for u, v in zip(p_path, p_path[1:])
                if u != vs and v != vt
            )
        except nx.NetworkXNoPath:
            C_toll = 0.0

        # Fastest-time path (alpha=0)
        time_w = lambda u, v, d: network.get_cost((u, v), alpha=0.0)
        try:
            t_path = nx.dijkstra_path(G, vs, vt, weight=time_w)
            T_fast = sum(
                network.edge_data[(u, v)].travel_time
                for u, v in zip(t_path, t_path[1:])
                if u != vs and v != vt
            )
        except nx.NetworkXNoPath:
            T_fast = network.max_time_slots

        # Combine into reserve with a random safety margin
        rho = rng.uniform(*safety_margin)
        return (1 + rho) * (alpha * C_toll + (1 - alpha) * tau_time * T_fast) + k_fee * fee

    finally:
        # Always clean up the temporary virtual edges
        for u, v in added:
            if G.has_edge(u, v):
                G.remove_edge(u, v)
