import numpy as np
import pandas as pd
from scipy.stats import norm


def build_event_windows(signal: pd.Series, returns: pd.Series,
                         pre: int = 60, post: int = 120) -> dict:
    """
    Extract return windows of [-pre, +post] bars around each signal event.
    Returns separate matrices for bullish (+1) and bearish (-1) events.
    """
    lags = list(range(-pre, post + 1))
    bull, bear = [], []

    for ts, direction in signal[signal != 0].items():
        pos = returns.index.searchsorted(ts)
        if pos < pre or pos + post >= len(returns):
            continue
        window = returns.iloc[pos - pre: pos + post + 1].values
        if len(window) != pre + post + 1:
            continue
        (bull if direction == 1 else bear).append(window)

    return {
        "bullish": pd.DataFrame(bull, columns=lags),
        "bearish": pd.DataFrame(bear, columns=lags),
        "lags": lags
    }


def _car_with_bands(windows: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    mean_r = windows.mean(axis=0)
    se = windows.std(axis=0) / np.sqrt(len(windows))
    z = norm.ppf(1 - alpha / 2)
    return pd.DataFrame({
        "car": mean_r.cumsum(),
        "lower": (mean_r - z * se).cumsum(),
        "upper": (mean_r + z * se).cumsum()
    })


def aggregate_car(events: dict, alpha: float = 0.05) -> pd.DataFrame:
    """
    Cumulative abnormal return for both directions.
    Bearish windows are sign-flipped so CAR represents directional edge.
    """
    frames = {}
    if not events["bullish"].empty:
        bull = _car_with_bands(events["bullish"], alpha)
        bull.columns = [f"bull_{c}" for c in bull.columns]
        frames.update(bull)
    if not events["bearish"].empty:
        bear = _car_with_bands(-events["bearish"], alpha)
        bear.columns = [f"bear_{c}" for c in bear.columns]
        frames.update(bear)

    return pd.DataFrame(frames, index=events["lags"])


def peak_post_car(car_df: pd.DataFrame, col: str = "bull_car") -> float:
    """Peak CAR in the post-event window (lag >= 0)."""
    post = car_df[car_df.index >= 0]
    return float(post[col].max()) if col in post.columns else float("nan")
