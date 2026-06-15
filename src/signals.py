import numpy as np
import pandas as pd


def _cooldown(sig: pd.Series, cooldown: int) -> pd.Series:
    """Zero out signals within cooldown bars of a prior signal."""
    out = sig.copy()
    last = -(cooldown + 1)
    for i, (_, v) in enumerate(sig.items()):
        if v != 0:
            if i - last <= cooldown:
                out.iloc[i] = 0
            else:
                last = i
    return out


def compute_d1(dvol_change: pd.Series, window: int = 30, z_thresh: float = 2.0,
               min_abs: float = 0.02, cooldown: int = 24) -> pd.Series:
    """
    D1: DVOL log-change z-score spike.
    Up signal = IV spike up (fear/uncertainty entering the market).
    Down signal = IV collapse (complacency / relief).
    """
    mu = dvol_change.rolling(window).mean()
    sigma = dvol_change.rolling(window).std().replace(0, np.nan)
    z = (dvol_change - mu) / sigma

    sig = pd.Series(0, index=dvol_change.index, dtype=int)
    sig[(z > z_thresh) & (dvol_change > min_abs)] = 1
    sig[(z < -z_thresh) & (dvol_change < -min_abs)] = -1
    return _cooldown(sig, cooldown)


def compute_d2(dvol_change: pd.Series, min_abs: float = 0.05,
               cooldown: int = 24) -> pd.Series:
    """D2: Absolute DVOL log-change exceeds threshold in a single bar."""
    sig = pd.Series(0, index=dvol_change.index, dtype=int)
    sig[dvol_change >= min_abs] = 1
    sig[dvol_change <= -min_abs] = -1
    return _cooldown(sig, cooldown)


def compute_d3(iv_premium: pd.Series, window: int = 30, z_thresh: float = 2.0,
               cooldown: int = 24) -> pd.Series:
    """
    D3: IV risk premium spike (DVOL - realized vol).
    A sudden jump means the options market expects a larger move than history
    would predict — a classic informed trading signature.
    """
    mu = iv_premium.rolling(window).mean()
    sigma = iv_premium.rolling(window).std().replace(0, np.nan)
    z = (iv_premium - mu) / sigma

    sig = pd.Series(0, index=iv_premium.index, dtype=int)
    sig[z > z_thresh] = 1
    sig[z < -z_thresh] = -1
    return _cooldown(sig, cooldown)


def compute_d4(dvol_close: pd.Series, dvol_high: pd.Series, dvol_low: pd.Series,
               dvol_change: pd.Series, window: int = 30, z_thresh: float = 2.0,
               min_abs: float = 0.02, range_mult: float = 1.5,
               cooldown: int = 24) -> pd.Series:
    """
    D4: D1 conditions plus elevated intrabar DVOL range (range z-score >= range_mult).
    Wide high-low range = panic or large directional IV move within the hour.
    Intended as a stricter subset of D1; selectivity depends on range_mult.
    With range_mult=1.5 only a small fraction of D1 events are filtered out; use
    range_mult>=2.0 if a clearly more selective subset is wanted.
    """
    mu = dvol_change.rolling(window).mean()
    sigma = dvol_change.rolling(window).std().replace(0, np.nan)
    z = (dvol_change - mu) / sigma

    bar_range = dvol_high - dvol_low
    range_z = (bar_range - bar_range.rolling(window).mean()) / bar_range.rolling(window).std().replace(0, np.nan)
    range_spike = range_z >= range_mult  # range_mult now acts as a z-score threshold

    sig = pd.Series(0, index=dvol_change.index, dtype=int)
    sig[(z > z_thresh) & (dvol_change > min_abs) & range_spike] = 1
    sig[(z < -z_thresh) & (dvol_change < -min_abs) & range_spike] = -1
    return _cooldown(sig, cooldown)


def compute_all(feat: pd.DataFrame, cfg: dict = None) -> pd.DataFrame:
    """
    Compute D1–D4 from IV feature DataFrame (output of deribit_fetch.compute_iv_features).
    feat must contain: dvol_change, dvol_z, dvol_close, dvol_high (optional), dvol_low (optional),
                       iv_premium (optional for D3).
    """
    cfg = cfg or {}
    w = cfg.get("rolling_window", 30)
    cd = cfg.get("cooldown_bars", 24)
    d1c = cfg.get("d1", {})
    d2c = cfg.get("d2", {})
    d3c = cfg.get("d3", {})
    d4c = cfg.get("d4", {})

    dc = feat["dvol_change"]

    d3 = (compute_d3(feat["iv_premium"], window=w,
                     z_thresh=d3c.get("z_score_threshold", 2.0), cooldown=cd)
          if "iv_premium" in feat.columns
          else pd.Series(0, index=feat.index, dtype=int))

    has_range = "dvol_high" in feat.columns and "dvol_low" in feat.columns
    d4 = (compute_d4(feat["dvol_close"], feat["dvol_high"], feat["dvol_low"], dc,
                     window=w, z_thresh=d4c.get("z_score_threshold", 2.0),
                     min_abs=d4c.get("min_abs_change", 0.02),
                     range_mult=d4c.get("range_multiplier", 1.5), cooldown=cd)
          if has_range
          else pd.Series(0, index=feat.index, dtype=int))

    return pd.DataFrame({
        "D1": compute_d1(dc, window=w, z_thresh=d1c.get("z_score_threshold", 2.0),
                         min_abs=d1c.get("min_abs_change", 0.02), cooldown=cd),
        "D2": compute_d2(dc, min_abs=d2c.get("min_abs_change", 0.05), cooldown=cd),
        "D3": d3,
        "D4": d4
    })


def signal_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = {}
    for col in df.columns:
        s = df[col]
        rows[col] = {
            "bullish (+1)": int((s == 1).sum()),
            "bearish (-1)": int((s == -1).sum()),
            "total": int((s != 0).sum())
        }
    return pd.DataFrame(rows).T
