import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta


def run_backtest(signal: pd.Series, returns: pd.Series,
                  holding_period: int = 30,
                  position_size: float = 0.01,
                  commission: float = 0.00075,
                  slippage: float = 0.00025,
                  stop_loss_sigma: float = 1.5) -> pd.DataFrame:
    """
    Event-driven backtest. Enter at T+1 after signal fires, exit at T+H.
    Stop-loss exits early if cumulative loss exceeds stop_loss_sigma * pre-event vol.
    """
    cost = commission + slippage
    ret_vals = returns.values
    ret_idx = returns.index
    trades = []

    for ts, direction in signal[signal != 0].items():
        entry_pos = ret_idx.searchsorted(ts) + 1
        if entry_pos >= len(ret_vals):
            continue

        exit_pos = min(entry_pos + holding_period - 1, len(ret_vals) - 1)
        pre_vol = returns.iloc[max(0, entry_pos - 60): entry_pos].std()
        stop = stop_loss_sigma * pre_vol if pre_vol > 0 else np.inf

        cum = 0.0
        actual_exit = exit_pos
        for i in range(entry_pos, exit_pos + 1):
            cum += ret_vals[i] * direction
            if cum < -stop:
                actual_exit = i
                break

        trades.append({
            "entry_time": ret_idx[entry_pos],
            "exit_time": ret_idx[actual_exit],
            "direction": int(direction),
            "holding_bars": actual_exit - entry_pos + 1,
            "gross_return": cum,
            "net_return": cum - cost,
            "position_size": position_size
        })

    return pd.DataFrame(trades)


def run_all_holding_periods(signal: pd.Series, returns: pd.Series,
                              holding_periods: list[int] = [10, 30, 60, 240],
                              **kwargs) -> dict[int, pd.DataFrame]:
    return {h: run_backtest(signal, returns, holding_period=h, **kwargs)
            for h in holding_periods}


def chronological_split(trades: pd.DataFrame,
                          train_r: float = 0.60,
                          val_r: float = 0.20
                          ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(trades)
    t1 = int(n * train_r)
    t2 = int(n * (train_r + val_r))
    return trades.iloc[:t1], trades.iloc[t1:t2], trades.iloc[t2:]


def walk_forward(signal: pd.Series, returns: pd.Series,
                  holding_period: int = 30,
                  train_months: int = 6,
                  test_months: int = 2,
                  gap_days: int = 7,
                  **kwargs) -> list[dict]:
    """
    Roll train/test windows forward in time.
    Gap between train end and test start prevents look-ahead leakage.
    """
    start = returns.index.min().to_pydatetime()
    end = returns.index.max().to_pydatetime()
    window_start = start
    windows = []

    while True:
        train_end = window_start + relativedelta(months=train_months)
        test_start = train_end + pd.Timedelta(days=gap_days)
        test_end = test_start + relativedelta(months=test_months)
        if test_end.replace(tzinfo=None) > end.replace(tzinfo=None):
            break

        def _filter(s, a, b):
            ta = pd.Timestamp(a) if pd.Timestamp(a).tzinfo else pd.Timestamp(a, tz="UTC")
            tb = pd.Timestamp(b) if pd.Timestamp(b).tzinfo else pd.Timestamp(b, tz="UTC")
            return s[(s.index >= ta) & (s.index < tb)]

        train_trades = run_backtest(
            _filter(signal, window_start, train_end),
            _filter(returns, window_start, train_end),
            holding_period=holding_period, **kwargs
        )
        test_trades = run_backtest(
            _filter(signal, test_start, test_end),
            _filter(returns, test_start, test_end),
            holding_period=holding_period, **kwargs
        )

        windows.append({
            "window": len(windows),
            "train_start": window_start,
            "train_end": train_end,
            "test_start": test_start,
            "test_end": test_end,
            "train_trades": train_trades,
            "test_trades": test_trades
        })
        window_start += relativedelta(months=test_months)

    return windows
