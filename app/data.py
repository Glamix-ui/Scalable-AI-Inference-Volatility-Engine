"""Feature engineering shared by training and serving (so they can never drift apart)."""
import hashlib

import numpy as np
import pandas as pd

WINDOW = 30        # days of history fed to the model
N_FEATURES = 5
MIN_ROWS = 60      # minimum history needed to build one window
ANN = np.sqrt(252)


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    c = df["Close"].astype(float)
    r = np.log(c).diff()
    return pd.DataFrame({
        "ret": r,
        "abs_ret": r.abs(),
        "range": (df["High"] - df["Low"]) / c,
        "vol5": r.rolling(5).std(),
        "vol20": r.rolling(20).std(),
    })


def latest_window(df: pd.DataFrame) -> np.ndarray:
    f = make_features(df).dropna()
    if len(f) < WINDOW:
        raise ValueError(f"Need at least {MIN_ROWS} daily rows of data.")
    return f.tail(WINDOW).to_numpy(np.float32)


def baseline_vol(df: pd.DataFrame) -> float:
    """Naive benchmark: trailing 20-day realized volatility, annualized."""
    r = np.log(df["Close"].astype(float)).diff()
    return float(r.tail(20).std() * ANN)


def rolling_vol_pct(df: pd.DataFrame) -> pd.Series:
    r = np.log(df["Close"].astype(float)).diff()
    return r.rolling(20).std() * ANN * 100


def df_from_closes(closes: list[float]) -> pd.DataFrame:
    c = np.asarray(closes, dtype=float)
    if len(c) < MIN_ROWS or not np.all(np.isfinite(c)) or np.any(c <= 0):
        raise ValueError(f"Provide at least {MIN_ROWS} positive daily closes (oldest first).")
    return pd.DataFrame({"Open": c, "High": c, "Low": c, "Close": c, "Volume": 0})


def synthetic(ticker: str, n: int = 260) -> pd.DataFrame:
    """Deterministic fallback so the demo never breaks if Yahoo is unreachable."""
    rng = np.random.default_rng(int(hashlib.md5(ticker.encode()).hexdigest(), 16) % 2**32)
    sig = (0.008 + 0.02 * rng.random()) * (1 + 0.6 * np.abs(np.sin(np.linspace(0, 12, n))))
    close = 100 * np.exp(np.cumsum(rng.normal(0, sig)))
    open_ = np.r_[close[0], close[:-1]] * (1 + rng.normal(0, 0.002, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.004, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.004, n)))
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n)
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close,
                         "Volume": rng.integers(1_000_000, 5_000_000, n)}, index=idx)


def history(ticker: str, rng: str = "1y") -> pd.DataFrame:
    """Daily OHLCV from Yahoo's public chart endpoint (pure-Python httpx: no compiled dependencies)."""
    try:
        import httpx
        r = httpx.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}",
                      params={"range": rng, "interval": "1d"},
                      headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
        r.raise_for_status()
        res = r.json()["chart"]["result"][0]
        q = res["indicators"]["quote"][0]
        df = pd.DataFrame({"Open": q["open"], "High": q["high"], "Low": q["low"],
                           "Close": q["close"], "Volume": q["volume"]},
                          index=pd.to_datetime(res["timestamp"], unit="s").normalize())
        return df.dropna()
    except Exception as first_error:
        try:                                   # optional fallback if the user installed yfinance
            import yfinance as yf
            df = yf.Ticker(ticker).history(period=rng, interval="1d")
            return df[["Open", "High", "Low", "Close", "Volume"]]
        except Exception:
            raise RuntimeError(f"Could not download {ticker}: {first_error}")


def fetch_ohlcv(ticker: str) -> tuple[pd.DataFrame, str]:
    """Blocking call: run it with asyncio.to_thread. Returns (dataframe, 'live'|'synthetic')."""
    try:
        df = history(ticker, "1y")
        if len(df) >= MIN_ROWS:
            return df, "live"
    except Exception:
        pass
    return synthetic(ticker), "synthetic"
