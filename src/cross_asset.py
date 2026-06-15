import numpy as np
import pandas as pd
from scipy import stats
from .lead_lag import cross_correlation


def propagation_table(btc_signal: pd.Series,
                       asset_returns: dict[str, pd.Series],
                       max_lag: int = 60) -> pd.DataFrame:
    """
    Core novel contribution: test whether BTC Polymarket probability jumps
    predict returns across the broader crypto universe.

    For each asset, measures:
    - Peak lead correlation (signal fires before price moves)
    - The lag at which that peak occurs
    - Whether the lead correlation exceeds contemporaneous correlation
    """
    rows = []
    for asset, rets in asset_returns.items():
        ccf = cross_correlation(btc_signal, rets, max_lag=max_lag)
        lead_ccf = ccf[ccf.index < 0]
        contemp = ccf.loc[0] if 0 in ccf.index else pd.Series(dtype=float)

        if lead_ccf.empty:
            continue

        best_lag = int(lead_ccf["correlation"].abs().idxmax())
        best_corr = float(lead_ccf.loc[best_lag, "correlation"])
        best_p = float(lead_ccf.loc[best_lag, "p_value"])
        contemp_corr = float(contemp.get("correlation", np.nan))

        rows.append({
            "asset": asset,
            "peak_lead_lag_min": abs(best_lag),
            "peak_lead_correlation": best_corr,
            "p_value": best_p,
            "contemporaneous_corr": contemp_corr,
            "lead_beats_contemp": abs(best_corr) > abs(contemp_corr)
        })

    df = pd.DataFrame(rows).set_index("asset")
    return df.sort_values("peak_lead_correlation", ascending=False, key=abs)


def decay_profile(btc_signal: pd.Series, returns: pd.Series,
                   lags: list[int] = [1, 5, 10, 15, 30, 60]) -> pd.DataFrame:
    """
    How fast does the predictive signal decay?
    Correlation at each specific lead lag from signal to returns.
    Uses positional slicing on common-index-aligned arrays.
    """
    common = btc_signal.index.intersection(returns.index)
    s_base = btc_signal.reindex(common).fillna(0).values
    r_base = returns.reindex(common).values

    rows = []
    for lag in lags:
        s_v = s_base[:-lag] if lag > 0 else s_base
        r_v = r_base[lag:] if lag > 0 else r_base
        mask = np.isfinite(s_v) & np.isfinite(r_v)
        s_c, r_c = s_v[mask], r_v[mask]
        if len(s_c) < 30:
            continue
        corr, p = stats.pearsonr(s_c, r_c)
        rows.append({"lag_min": lag, "correlation": corr, "p_value": p})
    return pd.DataFrame(rows).set_index("lag_min")


def signal_to_asset_heatmap(btc_signal: pd.Series,
                              asset_returns: dict[str, pd.Series],
                              lags: list[int] = [1, 5, 10, 15, 30, 60]
                              ) -> pd.DataFrame:
    """
    Correlation matrix: rows = assets, columns = lead lag.
    Useful for visualising cross-asset propagation speed.
    """
    rows = {}
    for asset, rets in asset_returns.items():
        profile = decay_profile(btc_signal, rets, lags=lags)
        rows[asset] = profile["correlation"]
    return pd.DataFrame(rows).T
