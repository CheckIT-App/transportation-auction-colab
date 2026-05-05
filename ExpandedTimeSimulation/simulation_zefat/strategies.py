import math


# =====================
# Pricing Strategy Classes
# =====================
class PricingStrategy:
    """Base class for edge pricing strategies used by the network pricing mechanism."""

    def update_price(self, network, edge, vehicle, s_max):
        """Update the price of a single edge after an allocation attempt."""
        raise NotImplementedError()

    def init_price(self, network, edge):
        """Initialize the edge price to its unit initial price."""
        data = network.edge_data[edge]
        data.price = data.unit_initial_price


class DynamicPricingStrategy(PricingStrategy):
    """Exponential dynamic pricing with edge-specific vmax scaling and travel-time component."""

    def update_price(self, network, edge, vehicle, s_max):
        """Apply the exponential update rule for the given edge."""
        data = network.edge_data[edge]

        bj = data.capacity
        xij = data.price
        tj = data.travel_time

        # vmax is scaled by demand/capacity (clipped by capacity)
        vmax_j = network.vmax * (min(data.demand, bj) / bj)

        # In the current formulation, rj equals bj.
        rj = bj

        # Growth-rate constant derived from the theoretical bound.
        ci = math.log(1 + vmax_j) / (1 - 1 / rj)

        # With yis = 1 (allocating one unit), the exponent is scaled by 1/bj.
        exp_term = (1 / bj) * ci

        # Price update: multiplicative growth + additive travel-time component.
        data.price = xij * math.exp(exp_term) + (bj * tj / network.max_time_slots) * (
            math.exp(exp_term) - 1
        )


class AlternativePricingStrategy(PricingStrategy):
    """Alternative exponential pricing that uses a global r and scales by s_max."""

    def update_price(self, network, edge, vehicle, s_max):
        """Apply the alternative exponential update rule for the given edge."""
        data = network.edge_data[edge]

        bj = data.capacity
        xij = data.price

        # Global growth-rate constant.
        c = math.log(1 + s_max * network.vmax) / (1 - 1 / network.r)

        # With aj=yis=1, the exponent is scaled by 1/bj.
        exp_term = (1 / bj) * c

        # Price update: multiplicative growth + additive scaling term.
        data.price = xij * math.exp(exp_term) + (bj / s_max) * (math.exp(exp_term) - 1)


class ZeroPricingStrategy(PricingStrategy):
    """Keeps all edge prices fixed (effectively a zero / constant-pricing baseline)."""

    def update_price(self, network, edge, vehicle, s_max):
        """No-op: price remains unchanged."""
        return


class SmoothTailPricingStrategy(PricingStrategy):
    """Smooth-tail pricing: exponential on [0, u0], cubic Hermite on (u0, 1].

    On [0, u0] uses the same demand-aware exponential rule as DynamicPricingStrategy.
    On (u0, 1] transitions to a cubic Hermite curve that reaches V* = vmax + 1 at u = 1,
    guaranteeing that saturated edges block further allocations (dual feasibility).
    """

    def __init__(self, u0: float = 0.95):
        self.u0 = u0

    def _p_orig(self, u, beta_j, lambda_j):
        return beta_j * (math.exp(lambda_j * u) - 1)

    def _hermite_tail(self, u, beta_j, lambda_j, V_star, h):
        u0 = self.u0
        p0 = self._p_orig(u0, beta_j, lambda_j)
        m0 = beta_j * lambda_j * math.exp(lambda_j * u0)
        D = V_star - p0
        m1 = 0.5 * D / h
        s = (u - u0) / h
        s2, s3 = s * s, s * s * s
        return (
            (2 * s3 - 3 * s2 + 1) * p0
            + (s3 - 2 * s2 + s) * h * m0
            + (-2 * s3 + 3 * s2) * V_star
            + (s3 - s2) * h * m1
        )

    def update_price(self, network, edge, vehicle, s_max):
        """Apply smooth-tail update: exponential below u0, Hermite above u0."""
        data = network.edge_data[edge]
        bj = data.capacity
        tj = data.travel_time
        T = network.max_time_slots

        vmax_j = network.vmax * (min(data.demand, bj) / bj)
        beta_j = tj / T
        lambda_j = math.log(1 + vmax_j) / (1 - 1 / bj) if bj > 1 else math.log(1 + vmax_j)

        # alloc_count is already incremented before update_price is called
        u_new = data.alloc_count / bj
        V_star = network.vmax + 1
        h = 1 - self.u0

        if u_new <= self.u0:
            p = self._p_orig(u_new, beta_j, lambda_j)
        else:
            p = self._hermite_tail(u_new, beta_j, lambda_j, V_star, h)

        data.price = bj * p


class MedianPricingStrategy(PricingStrategy):
    """Sets each edge's price to the price after half-capacity iterations (cached per edge)."""

    def __init__(self):
        """Initialize the per-edge median-price cache."""
        self._median_price: dict[tuple, float] = {}

    def update_price(self, network, edge, vehicle, s_max):
        """No-op: median price is constant after initialization."""
        return

    def init_price(self, network, edge):
        """Compute and set the cached median price for this edge (once)."""
        data = network.edge_data[edge]

        # Compute once per edge, then reuse.
        if edge not in self._median_price:
            bj = data.capacity
            iterations = bj // 2  # integer half-capacity
            tj = data.travel_time

            temp_price = data.unit_initial_price
            for _ in range(iterations):
                xij = temp_price
                vmax_j = network.vmax * (min(data.demand, bj) / bj)

                # Same growth-rate constant as DynamicPricingStrategy.
                ci = math.log(1 + vmax_j) / (1 - 1 / bj)
                exp_term = (1 / bj) * ci

                # Simulate the dynamic update for "half capacity" steps.
                temp_price = xij * math.exp(exp_term) + (bj * tj / network.max_time_slots) * (
                    math.exp(exp_term) - 1
                )

            self._median_price[edge] = temp_price

        data.price = self._median_price[edge]
