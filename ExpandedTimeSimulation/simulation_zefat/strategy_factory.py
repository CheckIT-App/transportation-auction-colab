from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

from .constants import (
    STRAT_ZERO,
    STRAT_TRANSPORT_ADAPTED,
    STRAT_STATIC_MEDIAN,
    STRAT_ONLINE_COMPETITIVE,
    STRAT_SMOOTH_TAIL,
)

StrategyFactory = Callable[[], object]


@dataclass(frozen=True)
class StrategySpec:
    """Metadata + factory for a pricing strategy."""
    key: str
    factory: StrategyFactory
    description: str = ""


# -----------------------------------------------------------------------------
# Lazy factories (avoid circular imports at module import time)
# -----------------------------------------------------------------------------
def _make_dynamic():
    from .strategies import DynamicPricingStrategy
    return DynamicPricingStrategy()


def _make_alternative():
    from .strategies import AlternativePricingStrategy
    return AlternativePricingStrategy()


def _make_zero():
    from .strategies import ZeroPricingStrategy
    return ZeroPricingStrategy()


def _make_median():
    from .strategies import MedianPricingStrategy
    return MedianPricingStrategy()


def _make_smooth_tail():
    from .strategies import SmoothTailPricingStrategy
    return SmoothTailPricingStrategy()


# -----------------------------------------------------------------------------
# Registry
# -----------------------------------------------------------------------------
_STRATEGY_REGISTRY: Dict[str, StrategySpec] = {
    STRAT_TRANSPORT_ADAPTED: StrategySpec(
        key=STRAT_TRANSPORT_ADAPTED,
        factory=_make_dynamic,
        description="Exponential dynamic pricing (edge-specific vmax + travel-time term)",
    ),
    STRAT_ONLINE_COMPETITIVE: StrategySpec(
        key=STRAT_ONLINE_COMPETITIVE,
        factory=_make_alternative,
        description="Alternative exponential pricing (global r, scaled by s_max)",
    ),
    STRAT_ZERO: StrategySpec(
        key=STRAT_ZERO,
        factory=_make_zero,
        description="Zero/constant pricing baseline",
    ),
    STRAT_STATIC_MEDIAN: StrategySpec(
        key=STRAT_STATIC_MEDIAN,
        factory=_make_median,
        description="Median (half-capacity) cached price per edge",
    ),
    STRAT_SMOOTH_TAIL: StrategySpec(
        key=STRAT_SMOOTH_TAIL,
        factory=_make_smooth_tail,
        description="Smooth-tail pricing: exponential on [0, u0], cubic Hermite on (u0, 1]",
    ),
}


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
def list_strategy_keys() -> list[str]:
    """Return all registered strategy keys (stable ordering)."""
    return sorted(_STRATEGY_REGISTRY.keys())


def get_strategy_spec(key: str) -> StrategySpec:
    """Return the StrategySpec for a given key."""
    k = str(key)
    if k not in _STRATEGY_REGISTRY:
        raise KeyError(f"Unknown strategy key: {k}. Known: {list_strategy_keys()}")
    return _STRATEGY_REGISTRY[k]


def make_strategy(key: str) -> object:
    """Create a new strategy instance by key."""
    return get_strategy_spec(key).factory()


def register_strategy(key: str, factory: StrategyFactory, description: str = "") -> None:
    """Register/override a strategy factory (useful for experiments)."""
    k = str(key)
    _STRATEGY_REGISTRY[k] = StrategySpec(key=k, factory=factory, description=description)


def ensure_registered(key: str) -> bool:
    """Return True if key exists in registry."""
    return str(key) in _STRATEGY_REGISTRY