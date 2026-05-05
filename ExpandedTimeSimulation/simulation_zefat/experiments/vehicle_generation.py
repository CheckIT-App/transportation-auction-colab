# ExpandedTimeSimulation/simulation_zefat/experiments/vehicle_generation.py
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Iterable, Union

from ExpandedTimeSimulation.simulation_zefat.constants import (
    COL_ALPHA,
    COL_DESIRED_ARRIVAL,
    COL_DESIRED_ENTRY,
)

# -----------------------------------------------------------------------------
# Types
# -----------------------------------------------------------------------------
AlphaSpec = Union[float, tuple[float, float], Callable[[], float]]


@dataclass(frozen=True)
class PeakSchedule:
    """Gaussian peak for desired_entry/desired_arrival (in discrete time slots)."""
    peak_slot: int = 30
    sigma: float = 5.0
    horizon_T: int = 100


# -----------------------------------------------------------------------------
# Alpha sampling
# -----------------------------------------------------------------------------
def clamp(x: float, lo: float, hi: float) -> float:
    """Clamp x into [lo, hi]."""
    return max(lo, min(hi, x))


def sample_alpha(spec: AlphaSpec, *, lo: float = 0.0, hi: float = 2.0) -> float:
    """
    Sample alpha from:
      - float: fixed value
      - (lo, hi): uniform draw
      - callable: user-provided sampler
    Then clamp into [lo, hi].
    """
    if callable(spec):
        return clamp(float(spec()), lo, hi)
    if isinstance(spec, (tuple, list)) and len(spec) == 2:
        a, b = float(spec[0]), float(spec[1])
        return clamp(random.uniform(min(a, b), max(a, b)), lo, hi)
    return clamp(float(spec), lo, hi)


def mixed_alpha_sampler(
    mix: list[tuple[float, AlphaSpec]],
    *,
    clamp_lo: float = 0.0,
    clamp_hi: float = 2.0,
) -> Callable[[], float]:
    """
    Build a sampler for alpha from a mixture distribution.

    mix = [(weight, alpha_spec), ...]
    alpha_spec can be:
      - float
      - (lo, hi) tuple
      - callable () -> float

    Returns a callable that samples alpha in [clamp_lo, clamp_hi].
    """
    if not mix:
        raise ValueError("mix is empty")

    weights = [float(w) for w, _ in mix]
    specs = [spec for _, spec in mix]
    tot = sum(weights)
    if tot <= 0:
        raise ValueError("sum of mixture weights must be > 0")

    probs = [w / tot for w in weights]

    def pick() -> float:
        r = random.random()
        acc = 0.0
        for p, spec in zip(probs, specs):
            acc += p
            if r <= acc:
                return sample_alpha(spec, lo=clamp_lo, hi=clamp_hi)
        # numeric edge-case
        return sample_alpha(specs[-1], lo=clamp_lo, hi=clamp_hi)

    return pick


# -----------------------------------------------------------------------------
# Desired schedule assignment
# -----------------------------------------------------------------------------
def sample_peak_time(peak_slot: int, sigma: float, T: int) -> int:
    """Sample an integer slot from N(peak_slot, sigma), clamped to [0, T-1]."""
    if T <= 0:
        return 0
    t = int(round(random.gauss(peak_slot, sigma)))
    return int(clamp(t, 0, T - 1))


def assign_peak_desired_entry(
    vehicles: list[dict],
    *,
    schedule: PeakSchedule = PeakSchedule(),
    write_arrival: bool = True,
) -> None:
    """
    Mutates vehicles in-place:
      vehicles[i][COL_DESIRED_ENTRY]   = sampled slot
      vehicles[i][COL_DESIRED_ARRIVAL] = same slot (optional)
    """
    for v in vehicles:
        t = sample_peak_time(schedule.peak_slot, schedule.sigma, schedule.horizon_T)
        v[COL_DESIRED_ENTRY] = t
        if write_arrival:
            v[COL_DESIRED_ARRIVAL] = t


def ensure_alpha_field(
    vehicles: list[dict],
    *,
    alpha_sampler: Callable[[], float] | None = None,
    default_alpha: float = 1.0,
) -> None:
    """
    Ensures each vehicle has an alpha value:
      - if alpha_sampler is provided -> overwrite/assign
      - else if missing -> assign default_alpha
    """
    for v in vehicles:
        if alpha_sampler is not None:
            v[COL_ALPHA] = float(alpha_sampler())
        elif COL_ALPHA not in v:
            v[COL_ALPHA] = float(default_alpha)


# -----------------------------------------------------------------------------
# Convenience helper for RealData
# -----------------------------------------------------------------------------
def generate_vehicles_with_peak(
    loader,
    *,
    alpha: AlphaSpec | Callable[[], float] | None = None,
    schedule: PeakSchedule = PeakSchedule(),
    clamp_alpha_to: tuple[float, float] = (0.0, 2.0),
) -> list[dict]:
    """
    Convenience wrapper around your RealData.generate_vehicles(...):
      - builds vehicles
      - assigns alpha (if provided)
      - assigns peak desired_entry/desired_arrival
    """
    # 1) Generate
    vehicles: list[dict]
    if callable(alpha):
        vehicles = loader.generate_vehicles(alpha=alpha)
    elif alpha is None:
        vehicles = loader.generate_vehicles()
    else:
        # allow float/tuple to be used as a spec; we convert it to a sampler
        sampler = lambda: sample_alpha(alpha, lo=clamp_alpha_to[0], hi=clamp_alpha_to[1])
        vehicles = loader.generate_vehicles(alpha=sampler)

    # 2) Ensure alpha exists (some generators may not set it consistently)
    if alpha is None:
        ensure_alpha_field(vehicles, alpha_sampler=None, default_alpha=1.0)

    # 3) Peak schedule
    assign_peak_desired_entry(vehicles, schedule=schedule, write_arrival=True)
    return vehicles
