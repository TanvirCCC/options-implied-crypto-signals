import io
import zipfile
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime
from dateutil.relativedelta import relativedelta
from tqdm import tqdm

BASE_URL = "https://data.binance.vision/data/spot/monthly/klines"
DATA_DIR = Path(__file__).parent.parent / "data" / "crypto"
INTERVAL = "1h"


def _download_month(symbol: str, year: int, month: int,
                     interval: str = INTERVAL) -> pd.DataFrame | None:
    fname = f"{symbol}-{interval}-{year}-{month:02d}.zip"
    url = f"{BASE_URL}/{symbol}/{interval}/{fname}"
    r = requests.get(url, timeout=60)
    if r.status_code != 200:
        return None
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        with z.open(z.namelist()[0]) as f:
            df = pd.read_csv(
                f, header=None, usecols=[0, 1, 2, 3, 4, 5],
                names=["open_time", "open", "high", "low", "close", "volume"]
            )
    # Binance changed to microsecond timestamps in 2025; detect and normalise
    ts = pd.to_numeric(df["open_time"], errors="coerce")
    df = df[ts.notna() & (ts > 0)].copy()
    ts = ts[ts.notna() & (ts > 0)]
    # Microsecond values are ~1e15; millisecond values are ~1e12
    unit = "us" if ts.iloc[0] > 1e13 else "ms"
    df["open_time"] = pd.to_datetime(ts.astype("int64"), unit=unit, utc=True)
    return df.set_index("open_time").astype(float)


def fetch_symbol(symbol: str, start: str, end: str) -> pd.DataFrame:
    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.strptime(end, "%Y-%m-%d")
    current = start_dt.replace(day=1)
    frames = []
    while current <= end_dt:
        df = _download_month(symbol, current.year, current.month)
        if df is not None:
            frames.append(df)
        current += relativedelta(months=1)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames).sort_index().drop_duplicates()


def fetch_all(symbols: list[str], start: str, end: str) -> None:
    """Download all symbols from Binance public archive and save as parquet."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for symbol in symbols:
        out = DATA_DIR / f"{symbol}_1h.parquet"
        if out.exists():
            print(f"{symbol}: already exists, skipping")
            continue
        print(f"Fetching {symbol}...")
        df = fetch_symbol(symbol, start, end)
        if not df.empty:
            df.to_parquet(out)
            print(f"  {len(df):,} bars saved → {out.name}")
        else:
            print(f"  No data found for {symbol}")


def load_symbol(symbol: str) -> pd.DataFrame:
    path = DATA_DIR / f"{symbol}_1h.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{symbol} not found. Run fetch_all() first.")
    return pd.read_parquet(path)


def compute_returns(df: pd.DataFrame) -> pd.Series:
    return df["close"].pct_change().rename("returns")
