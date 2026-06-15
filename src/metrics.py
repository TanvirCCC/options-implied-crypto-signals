import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.special import ndtri


def trades_to_hourly_returns(trades: pd.DataFrame,
                              index: pd.DatetimeIndex | None = None) -> pd.Series:
    """
    Convert event-trade records into a continuous hourly strategy return series.
    Each trade's net_return is spread uniformly across its holding bars.
    Idle hours have zero return. Annualisation is then by sqrt(8760) on the series.

    For cross-strategy comparison pass `index` = full test-period hourly index so
    that idle time before the first trade and after the last is counted identically
    across strategies. Without `index`, the series spans only first-entry to last-exit.
    """
    if trades.empty:
        return pd.Series(dtype=float)
    if index is None:
        start = trades["entry_time"].min().floor("h")
        end = trades["exit_time"].max().ceil("h")
        index = pd.date_range(start, end, freq="h", tz=start.tz)

    r = pd.Series(0.0, index=index)
    for _, t in trades.iterrows():
        bars = max(int(t["holding_bars"]), 1)
        per_bar = (t["net_return"] * t["position_size"]) / bars
        window = pd.date_range(t["entry_time"], periods=bars, freq="h", tz=index.tz)
        window = window.intersection(index)
        r.loc[window] += per_bar
    return r


def sharpe(trades: pd.DataFrame, rf: float = 0.0) -> float:
    """Sharpe on the hourly calendar-time strategy return series, annualised by sqrt(8760)."""
    if len(trades) < 2:
        return np.nan
    hr = trades_to_hourly_returns(trades)
    if hr.std() == 0 or hr.empty:
        return np.nan
    return float(np.sqrt(8760) * (hr.mean() - rf) / hr.std())


def sortino(trades: pd.DataFrame, rf: float = 0.0) -> float:
    if len(trades) < 2:
        return np.nan
    hr = trades_to_hourly_returns(trades)
    dd = hr[hr < rf].std()
    if dd == 0 or np.isnan(dd):
        return np.nan
    return float(np.sqrt(8760) * (hr.mean() - rf) / dd)


def max_drawdown(trades: pd.DataFrame) -> float:
    if trades.empty:
        return np.nan
    hr = trades_to_hourly_returns(trades)
    if hr.empty:
        return np.nan
    equity = (1 + hr).cumprod()
    dd = (equity - equity.cummax()) / equity.cummax()
    return float(dd.min())


def annualized_return(trades: pd.DataFrame,
                       mins_per_year: int = 525_600) -> float:
    if trades.empty or len(trades) < 2:
        return np.nan
    total = (trades["net_return"] * trades["position_size"]).sum()
    span = (trades["exit_time"].max() - trades["entry_time"].min()).total_seconds() / 60
    return float(total * mins_per_year / span) if span > 0 else np.nan


def calmar(trades: pd.DataFrame) -> float:
    mdd = abs(max_drawdown(trades))
    ann = annualized_return(trades)
    return float(ann / mdd) if mdd > 0 else np.nan


def win_rate(trades: pd.DataFrame) -> float:
    return float((trades["net_return"] > 0).mean()) if not trades.empty else np.nan


def profit_factor(trades: pd.DataFrame) -> float:
    wins = trades[trades["net_return"] > 0]["net_return"].sum()
    losses = abs(trades[trades["net_return"] < 0]["net_return"].sum())
    return float(wins / losses) if losses > 0 else np.inf


def deflated_sharpe(sr: float, n_trials: int, n_obs: int,
                     skew: float = 0.0, excess_kurt: float = 0.0) -> float:
    """
    Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014).
    Adjusts observed Sharpe for selection bias across n_trials strategy variants.
    """
    gamma = 0.5772
    e_max = ((1 - gamma) * ndtri(1 - 1 / n_trials) +
             gamma * ndtri(1 - 1 / (n_trials * np.e)))
    sr_std = np.sqrt(
        (1 + (1 - skew * sr + (excess_kurt + 1) / 4 * sr ** 2)) / (n_obs - 1)
    )
    return float(norm.cdf((sr - e_max) / sr_std)) if sr_std > 0 else np.nan


def full_report(trades: pd.DataFrame, label: str = "",
                 n_strategy_variants: int = 40) -> dict:
    sr = sharpe(trades)
    n_obs = len(trades_to_hourly_returns(trades)) if not trades.empty else 2
    return {
        "label": label,
        "n_trades": len(trades),
        "win_rate": win_rate(trades),
        "profit_factor": profit_factor(trades),
        "sharpe": sr,
        "sortino": sortino(trades),
        "max_drawdown": max_drawdown(trades),
        "calmar": calmar(trades),
        "ann_return": annualized_return(trades),
        "deflated_sharpe": deflated_sharpe(sr or 0, n_strategy_variants, max(n_obs, 2)),
        "total_pnl": float((trades["net_return"] * trades["position_size"]).sum())
        if not trades.empty else 0.0
    }
