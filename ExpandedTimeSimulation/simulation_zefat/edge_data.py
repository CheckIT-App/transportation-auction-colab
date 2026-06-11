class EdgeData:
    """Holds all mutable, edge-specific state used during the simulation."""

    def __init__(self, time, capacity, demand):
        """Initialize edge attributes and runtime tracking variables."""
        self.travel_time = time
        self.capacity = capacity
        self.demand = demand

        # Pricing state
        self.price = 0.0
        self.unit_initial_price = 0.0
        self.price_history = [0.0]

        # Allocation / demand tracking
        self.alloc_count = 0
        self.request_count = 0  # tracks how many vehicles requested this edge

    def unit_price(self):
        """Return the current per-unit price of the edge."""
        return (self.price / self.capacity) + self.unit_initial_price

    def is_saturated(self):
        """Check whether the edge has reached its capacity."""
        return self.alloc_count >= self.capacity

    def actual_travel_time(self, bpr_alpha=0.15, bpr_beta=4.0):
        """Realized (congested) travel time via the BPR volume-delay function.

        t_actual = t_free * (1 + a * (alloc/capacity)^b)

        `travel_time` stays the free-flow/planned value the auction routes on;
        this returns the slower time actually experienced once the edge fills.
        Computed from the final `alloc_count`, so it is a post-hoc measurement
        and does not feed back into allocation. At zero load it equals free-flow.
        """
        if not self.capacity:
            return self.travel_time
        vc = self.alloc_count / self.capacity
        return self.travel_time * (1.0 + bpr_alpha * (vc ** bpr_beta))

    def update_price(self, strategy, network, edge, vehicle, s_max):
        """Delegate price update to the active pricing strategy."""
        strategy.update_price(network, edge, vehicle, s_max)

    def record_price(self):
        """Append the current price to the price history."""
        self.price_history.append(self.price)

    def last_used_price(self):
        """Return the previous price value (before the most recent update)."""
        return self.price_history[-2] if len(self.price_history) > 1 else 0.0
