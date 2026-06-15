import time
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone

GAMMA_URL = "https://gamma-api.polymarket.com"
CLOB_URL = "https://clob.polymarket.com"
DATA_DIR = Path(__file__).parent.parent / "data" / "polymarket"

CRYPTO_KEYWORDS = [
    "bitcoin", "btc", "ethereum", "eth", "solana", "sol",
    "crypto", "above", "below", "price", "end of"
]


def fetch_crypto_markets(min_volume: float = 5000,
                          closed: bool = True) -> pd.DataFrame:
    """Fetch all crypto-related prediction markets from Polymarket Gamma API."""
    markets = []
    offset = 0
    limit = 100

    while True:
        resp = requests.get(
            f"{GAMMA_URL}/markets",
            params={"limit": limit, "offset": offset,
                    "closed": str(closed).lower(), "tag_slug": "crypto"},
            timeout=30
        )
        if resp.status_code in (422, 400):
            break
        resp.raise_for_status()
        data = resp.json()
        if not data:
            break
        markets.extend(data)
        if len(data) < limit:
            break
        offset += limit
        time.sleep(0.2)

    if not markets:
        return pd.DataFrame()

    df = pd.DataFrame(markets)

    # Filter by crypto keyword in question text
    mask = df["question"].str.lower().str.contains(
        "|".join(CRYPTO_KEYWORDS), na=False
    )
    df = df[mask].copy()

    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
        df = df[df["volume"] >= min_volume]

    return df.reset_index(drop=True)


SEVEN_DAYS = 7 * 24 * 3600


def fetch_market_prices(token_id: str, start_ts: int, end_ts: int,
                         fidelity: int = 1) -> pd.DataFrame:
    """
    Fetch probability history for a single market token.
    Automatically chunks requests into ≤7-day windows (API limit).
    fidelity = bar size in minutes.
    """
    frames = []
    chunk_start = start_ts

    while chunk_start < end_ts:
        chunk_end = min(chunk_start + SEVEN_DAYS, end_ts)
        resp = requests.get(
            f"{CLOB_URL}/prices-history",
            params={"market": token_id, "startTs": chunk_start,
                    "endTs": chunk_end, "fidelity": fidelity},
            timeout=60
        )
        if resp.status_code == 200:
            history = resp.json().get("history", [])
            if history:
                frames.append(pd.DataFrame(history))
        chunk_start = chunk_end
        time.sleep(0.15)

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames)
    df["t"] = pd.to_datetime(df["t"], unit="s", utc=True)
    df = df.rename(columns={"t": "timestamp", "p": "prob"})
    df = df.set_index("timestamp")[["prob"]].sort_index().drop_duplicates()
    return df.astype(float)


def fetch_and_save_all(markets_df: pd.DataFrame, start: str, end: str,
                        min_bars: int = 500) -> None:
    """Download price history for all markets and save as parquet files."""
    import json as _json
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    start_ts = int(datetime.strptime(start, "%Y-%m-%d")
                   .replace(tzinfo=timezone.utc).timestamp())
    end_ts = int(datetime.strptime(end, "%Y-%m-%d")
                 .replace(tzinfo=timezone.utc).timestamp())

    saved = 0
    for _, row in markets_df.iterrows():
        slug = str(row.get("slug", row.get("id", "")))[:80].replace("/", "-")
        out = DATA_DIR / f"{slug}.parquet"
        if out.exists():
            continue

        # Use YES token ID (index 0 of clobTokenIds)
        raw_tokens = row.get("clobTokenIds", "[]")
        try:
            tokens = _json.loads(raw_tokens) if isinstance(raw_tokens, str) else raw_tokens
            token_id = tokens[0] if tokens else None
        except Exception:
            token_id = None

        if not token_id:
            continue

        df = fetch_market_prices(token_id, start_ts, end_ts)
        if df.empty or len(df) < min_bars:
            continue

        df.to_parquet(out)
        saved += 1
        print(f"  [{saved}] {slug[:60]}: {len(df):,} bars")

    print(f"\nSaved {saved} markets to {DATA_DIR}")


def load_all_markets() -> dict[str, pd.DataFrame]:
    """Load all saved Polymarket probability series from disk."""
    markets = {}
    for path in sorted(DATA_DIR.glob("*.parquet")):
        df = pd.read_parquet(path)
        if not df.empty:
            markets[path.stem] = df
    return markets


def align_to_common_index(markets: dict[str, pd.DataFrame],
                           freq: str = "1min") -> pd.DataFrame:
    """
    Forward-fill all sparse probability series onto a common UTC grid.
    Polymarket only records bars when trades occur — ffill handles gaps.
    """
    if not markets:
        return pd.DataFrame()
    series = {name: df["prob"].resample(freq).last().ffill()
              for name, df in markets.items()}
    aligned = pd.DataFrame(series).ffill().dropna(how="all")
    return aligned
