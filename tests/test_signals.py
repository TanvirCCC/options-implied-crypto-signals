"""Regression test for the audit-found D4 == D1 bug."""
import numpy as np
import pandas as pd

from src.signals import compute_d1, compute_d4


def _synthetic_dvol(n=500, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    close = pd.Series(100 + np.cumsum(rng.normal(0, 1, n)), index=idx)
    high = close + rng.uniform(0, 2, n)
    low = close - rng.uniform(0, 2, n)
    change = np.log(close).diff().fillna(0)
    return close, high, low, change


def test_d4_is_strict_subset_of_d1():
    """Every D4 event timestamp must also be a D1 event (D4 = D1 ∩ range_spike)."""
    close, high, low, change = _synthetic_dvol()
    d1 = compute_d1(change, cooldown=0)
    d4 = compute_d4(close, high, low, change, range_mult=1.5, cooldown=0)
    d4_times = set(d4[d4 != 0].index)
    d1_times = set(d1[d1 != 0].index)
    assert d4_times.issubset(d1_times), \
        f"D4 has {len(d4_times - d1_times)} events that are not in D1"


def test_d4_is_not_identical_to_d1():
    """With a non-trivial range threshold D4 must filter at least one event."""
    close, high, low, change = _synthetic_dvol()
    d1 = compute_d1(change, cooldown=0)
    d4 = compute_d4(close, high, low, change, range_mult=2.5, cooldown=0)
    assert (d1 != 0).sum() > (d4 != 0).sum(), \
        "D4 with range_mult=2.5 should drop some D1 events"
