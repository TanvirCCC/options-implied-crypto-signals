"""Regression tests for the audit-found backtest off-by-one bug."""
import numpy as np
import pandas as pd
import pytest

from src.backtest import run_backtest


@pytest.fixture
def constant_returns():
    """100 hourly bars of constant 1% returns, one bullish signal at hour 10."""
    idx = pd.date_range("2024-01-01", periods=100, freq="h", tz="UTC")
    returns = pd.Series([0.01] * 100, index=idx)
    signal = pd.Series(0, index=idx)
    signal.iloc[10] = 1
    return signal, returns


@pytest.mark.parametrize("h", [1, 2, 4, 8, 24])
def test_holding_period_sums_exactly_h_bars(constant_returns, h):
    """holding_period=H must sum exactly H return bars, no more, no less."""
    sig, ret = constant_returns
    trades = run_backtest(sig, ret, holding_period=h,
                          commission=0, slippage=0, stop_loss_sigma=1e9)
    assert len(trades) == 1
    assert trades.iloc[0]["holding_bars"] == h
    np.testing.assert_allclose(trades.iloc[0]["gross_return"], 0.01 * h, atol=1e-12)


def test_entry_is_at_t_plus_one(constant_returns):
    """Entry must be the bar AFTER the signal, never the same bar."""
    sig, ret = constant_returns
    trades = run_backtest(sig, ret, holding_period=1,
                          commission=0, slippage=0, stop_loss_sigma=1e9)
    signal_time = sig[sig != 0].index[0]
    assert trades.iloc[0]["entry_time"] > signal_time
