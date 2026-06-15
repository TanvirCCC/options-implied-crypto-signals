import time
import requests
import pandas as pd
import numpy as np
from pathlib import Path

API = "https://www.deribit.com/api/v2/public"
DATA_DIR = Path(__file__).parent.parent / "data" / "deribit"
CHUNK = 1000  # max bars per API call


def fetch_dvol(currency: str, start: str, end: str,
               resolution: int = 3600) -> pd.DataFrame:
    """
    Fetch Deribit Volatility Index (DVOL) for BTC or ETH.
    resolution = seconds per bar (3600 = hourly).
    Returns OHLCV DataFrame indexed by UTC timestamp.
    """
    start_ms = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    end_ms = int(pd.Timestamp(end, tz="UTC").timestamp() * 1000)
    chunk_ms = resolution * 1000 * CHUNK

    frames = []
    cursor = start_ms

    while cursor < end_ms:
        chunk_end = min(cursor + chunk_ms, end_ms)
        resp = requests.get(
            f"{API}/get_volatility_index_data",
            params={"currency": currency.upper(), "resolution": resolution,
                    "start_timestamp": cursor, "end_timestamp": chunk_end},
            timeout=30
        )
        if resp.status_code != 200:
            break
        bars = resp.json().get("result", {}).get("data", [])
        if bars:
            frames.append(pd.DataFrame(bars, columns=["ts_ms", "open", "high", "low", "close"]))
        cursor = chunk_end
        time.sleep(0.2)

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames).drop_duplicates("ts_ms").sort_values("ts_ms")
    df.index = pd.to_datetime(df["ts_ms"], unit="ms", utc=True)
    df.index.name = "timestamp"
    return df[["open", "high", "low", "close"]].astype(float)


def fetch_realized_vol(currency: str) -> pd.DataFrame:
    """
    Fetch daily historical (realized) volatility from Deribit.
    Used to compute the IV risk premium: DVOL - realized vol.
    """
    resp = requests.get(f"{API}/get_historical_volatility",
                        params={"currency": currency.upper()}, timeout=30)
    data = resp.json().get("result", [])
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data, columns=["ts_ms", "realized_vol"])
    df.index = pd.to_datetime(df["ts_ms"], unit="ms", utc=True)
    df.index.name = "timestamp"
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df[["realized_vol"]].astype(float)


def fetch_all(currencies: list[str], start: str, end: str) -> None:
    """Download DVOL and realized vol for all currencies, save as parquet."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for ccy in currencies:
        dvol_path = DATA_DIR / f"{ccy.upper()}_dvol_1h.parquet"
        rvol_path = DATA_DIR / f"{ccy.upper()}_realized_vol.parquet"

        if not dvol_path.exists():
            print(f"Fetching {ccy.upper()} DVOL...")
            df = fetch_dvol(ccy, start, end)
            if not df.empty:
                df.to_parquet(dvol_path)
                print(f"  {len(df):,} hourly bars → {dvol_path.name}")
        else:
            print(f"{ccy.upper()} DVOL: already exists, skipping")

        if not rvol_path.exists():
            print(f"Fetching {ccy.upper()} realized vol...")
            df = fetch_realized_vol(ccy)
            if not df.empty:
                df.to_parquet(rvol_path)
                print(f"  {len(df):,} daily bars → {rvol_path.name}")
        else:
            print(f"{ccy.upper()} realized vol: already exists, skipping")


def load_dvol(currency: str) -> pd.DataFrame:
    path = DATA_DIR / f"{currency.upper()}_dvol_1h.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{currency} DVOL not found. Run fetch_all() first.")
    return pd.read_parquet(path)


def load_realized_vol(currency: str) -> pd.DataFrame:
    path = DATA_DIR / f"{currency.upper()}_realized_vol.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{currency} realized vol not found. Run fetch_all() first.")
    return pd.read_parquet(path)


def compute_iv_features(dvol: pd.DataFrame,
                         rvol: pd.DataFrame = None) -> pd.DataFrame:
    """
    Derive signal features from DVOL data:
    - dvol_close:  raw DVOL level
    - dvol_change: 1-period log change in DVOL
    - dvol_z:      rolling z-score of DVOL change (30-bar window)
    - iv_premium:  DVOL minus realized vol (requires rvol)
    """
    feat = pd.DataFrame(index=dvol.index)
    feat["dvol_close"] = dvol["close"]
    feat["dvol_change"] = np.log(dvol["close"]).diff()
    feat["dvol_z"] = (
        feat["dvol_change"] - feat["dvol_change"].rolling(30).mean()
    ) / feat["dvol_change"].rolling(30).std()

    if rvol is not None:
        # Forward-fill daily realized vol to hourly index
        rvol_hourly = rvol["realized_vol"].reindex(feat.index, method="ffill")
        feat["iv_premium"] = feat["dvol_close"] - rvol_hourly
        feat["iv_premium_z"] = (
            feat["iv_premium"] - feat["iv_premium"].rolling(30).mean()
        ) / feat["iv_premium"].rolling(30).std()

    return feat
