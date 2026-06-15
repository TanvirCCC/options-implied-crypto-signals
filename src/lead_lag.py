import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import grangercausalitytests


def cross_correlation(signal: pd.Series, returns: pd.Series,
                       max_lag: int = 60) -> pd.DataFrame:
    """
    Pearson correlation between signal and returns at lags -max_lag to +max_lag.
    Negative lag means signal leads returns (the effect we're looking for).

    Aligns to common index first, then uses positional slicing so each lag
    actually shifts the series rather than re-selecting the same timestamps.
    """
    # Align to common index, then work positionally
    common = signal.index.intersection(returns.index)
    s_base = signal.reindex(common).fillna(0).values
    r_base = returns.reindex(common).values

    rows = []
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            s_v, r_v = s_base[:lag], r_base[-lag:]
        elif lag > 0:
            s_v, r_v = s_base[lag:], r_base[:-lag]
        else:
            s_v, r_v = s_base, r_base

        mask = np.isfinite(s_v) & np.isfinite(r_v)
        s_c, r_c = s_v[mask], r_v[mask]
        if len(s_c) < 30:
            continue

        corr, p = stats.pearsonr(s_c, r_c)
        rows.append({"lag": lag, "correlation": corr, "p_value": p})

    return pd.DataFrame(rows).set_index("lag")


def granger_test(signal: pd.Series, returns: pd.Series,
                  max_lags: int = 30, alpha: float = 0.05) -> pd.DataFrame:
    """
    Test whether signal Granger-causes returns.
    Applies Bonferroni correction across lag levels.
    """
    df = pd.DataFrame({"returns": returns, "signal": signal}).dropna()
    raw = grangercausalitytests(df[["returns", "signal"]],
                                 maxlag=max_lags, verbose=False)
    rows = []
    for lag, res in raw.items():
        f, p = res[0]["ssr_ftest"][0], res[0]["ssr_ftest"][1]
        rows.append({"lag": lag, "f_stat": f, "p_value": p})

    result = pd.DataFrame(rows).set_index("lag")
    result["p_bonferroni"] = (result["p_value"] * max_lags).clip(upper=1.0)
    result["significant"] = result["p_bonferroni"] < alpha
    return result


def placebo_time_shuffle(signal: pd.Series, returns: pd.Series,
                          n_iter: int = 500, max_lag: int = 60) -> dict:
    """
    Shuffle signal timestamps randomly and recompute peak |correlation|.
    Empirical p-value: fraction of shuffles exceeding the observed peak.
    """
    observed_peak = cross_correlation(signal, returns, max_lag)["correlation"].abs().max()
    vals = signal.values.copy()
    null_peaks = []

    for _ in range(n_iter):
        np.random.shuffle(vals)
        null_peaks.append(
            cross_correlation(pd.Series(vals, index=signal.index), returns, max_lag)
            ["correlation"].abs().max()
        )

    return {
        "observed_peak": observed_peak,
        "empirical_p": float(np.mean(np.array(null_peaks) >= observed_peak)),
        "null_peaks": null_peaks
    }


def placebo_random_jumps(signal: pd.Series, returns: pd.Series,
                          n_iter: int = 500, max_lag: int = 60) -> dict:
    """
    Replace real signals with random jumps at the same frequency.
    Tests whether structured timing drives correlation or just jump frequency.
    """
    observed_peak = cross_correlation(signal, returns, max_lag)["correlation"].abs().max()
    n_events = int((signal != 0).sum())
    null_peaks = []

    for _ in range(n_iter):
        fake = pd.Series(0, index=signal.index, dtype=float)
        idx = np.random.choice(len(signal), size=n_events, replace=False)
        fake.iloc[idx] = np.random.choice([-1, 1], size=n_events)
        null_peaks.append(
            cross_correlation(fake, returns, max_lag)["correlation"].abs().max()
        )

    return {
        "observed_peak": observed_peak,
        "empirical_p": float(np.mean(np.array(null_peaks) >= observed_peak)),
        "null_peaks": null_peaks
    }
