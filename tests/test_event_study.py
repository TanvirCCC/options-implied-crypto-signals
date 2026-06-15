"""Regression test for the audit-found alpha-ignored bug in event study bands."""
import numpy as np
import pandas as pd

from src.event_study import _car_with_bands


def test_alpha_controls_band_width():
    """Smaller alpha must produce wider confidence bands. Pre-fix this was hardcoded."""
    rng = np.random.default_rng(0)
    windows = pd.DataFrame(rng.normal(0, 0.01, size=(200, 50)))

    car_95 = _car_with_bands(windows, alpha=0.05)
    car_99 = _car_with_bands(windows, alpha=0.01)

    width_95 = (car_95["upper"] - car_95["lower"]).iloc[-1]
    width_99 = (car_99["upper"] - car_99["lower"]).iloc[-1]
    assert width_99 > width_95, \
        f"alpha=0.01 should give wider band than alpha=0.05; got {width_99:.4f} vs {width_95:.4f}"
