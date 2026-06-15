"""
Run this script once to download all data before opening the notebooks.

    python download_data.py

Downloads:
  1. Binance 1-hour OHLCV for 15 crypto pairs (data/crypto/)
  2. Deribit DVOL volatility index for BTC + ETH (data/deribit/)
"""

import yaml
from pathlib import Path
from src.binance_fetch import fetch_all
from src.deribit_fetch import fetch_all as fetch_deribit

cfg = yaml.safe_load(open("config.yaml"))

print("=" * 60)
print("Step 1: Downloading Binance 1-hour OHLCV data")
print("=" * 60)
fetch_all(
    symbols=cfg["binance"]["pairs"],
    start=cfg["binance"]["start_date"],
    end=cfg["binance"]["end_date"]
)

print("\n" + "=" * 60)
print("Step 2: Downloading Deribit DVOL (BTC + ETH)")
print("=" * 60)
fetch_deribit(
    currencies=cfg["deribit"]["currencies"],
    start=cfg["deribit"]["start_date"],
    end=cfg["deribit"]["end_date"]
)

print("\nDone. Open the notebooks in order: 01 → 02 → 03 → 04 → 05 → 06")
