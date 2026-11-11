import asyncio
import json
import re
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import data
from .batcher import MicroBatcher
from .config import settings
from .metrics import Hub, Metrics
from .model import Predictor

ROOT = Path(__file__).resolve().parent.parent
metrics, hub = Metrics(), Hub()
state: dict = {}
_cache: dict = {}
_locks: dict[str, asyncio.Lock] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    torch.set_num_threads(settings.torch_threads)
    predictor = Predictor(settings.model_path, settings.allow_untrained)
    batcher = MicroBatcher(predictor, metrics, hub, settings.max_batch, settings.max_wait_ms)
    batcher.start()
    state.update(predictor=predictor, batcher=batcher)

    async def stats_loop():
        while True:
            await asyncio.sleep(0.5)
            hub.publish({"type": "stats", "batching": batcher.enabled, **metrics.snapshot(batcher.depth)})

    ticker = asyncio.create_task(stats_loop())
    yield
    ticker.cancel()
    await batcher.stop()


app = FastAPI(title="Volatility Inference Engine", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")


class PredictIn(BaseModel):
    ticker: str = "AAPL"
    closes: list[float] | None = None      # optional custom data: daily closes, oldest -> newest


class ConfigIn(BaseModel):
    batching: bool | None = None
    max_batch: int | None = None
    max_wait_ms: float | None = None


def _ticker(raw: str) -> str:
    t = raw.strip().upper()
    if not re.fullmatch(r"[A-Z0-9.\-^=]{1,10}", t):
        raise HTTPException(422, "Invalid ticker symbol.")
    return t


async def get_market(ticker: str):
    hit = _cache.get(ticker)
    if hit and time.time() - hit[0] < settings.cache_ttl:
        return hit[1], hit[2]
    async with _locks.setdefault(ticker, asyncio.Lock()):    # one fetch per ticker during bursts
        hit = _cache.get(ticker)
        if hit and time.time() - hit[0] < settings.cache_ttl:
            return hit[1], hit[2]
        df, source = await asyncio.to_thread(data.fetch_ohlcv, ticker)
        _cache[ticker] = (time.time(), df, source)
        return df, source


def risk_level(vol: float) -> str:
    return "Low" if vol < 0.20 else "Moderate" if vol < 0.40 else "High" if vol < 0.60 else "Extreme"


@app.get("/")
async def index():
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/health")
async def health():
    return {"status": "ok", "model": state["predictor"].info, "batching": state["batcher"].enabled}


@app.get("/api/candles/{ticker}")
async def candles(ticker: str):
    df, source = await get_market(_ticker(ticker))
    df = df.tail(150)
    days = df.index.strftime("%Y-%m-%d")
    vol = data.rolling_vol_pct(df)
    return {
        "source": source,
        "candles": [{"time": d, "open": o, "high": h, "low": l, "close": c}
                    for d, o, h, l, c in zip(days, df.Open, df.High, df.Low, df.Close)],
        "vol": [{"time": d, "value": float(v)} for d, v in zip(days, vol) if np.isfinite(v)],
    }


@app.post("/api/predict")
async def predict(body: PredictIn):
    ticker = _ticker(body.ticker)
    try:
        if body.closes:
            df, source = data.df_from_closes(body.closes), "custom"
        else:
            df, source = await get_market(ticker)
        x = data.latest_window(df)
    except ValueError as e:
        raise HTTPException(422, str(e))
    t0 = time.perf_counter()
    vol, batch_size = await state["batcher"].submit(x)
    return {
        "ticker": ticker, "source": source,
        "predicted_vol": round(vol, 4),                       # annualized, next 5 days
        "baseline_vol": round(data.baseline_vol(df), 4),      # trailing 20-day benchmark
        "var_95_1d": round(1.645 * vol / data.ANN, 4),        # 1-day 95% parametric VaR (fraction of position)
        "risk": risk_level(vol),
        "batch_size": batch_size,
        "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
    }


@app.get("/api/metrics")
async def get_metrics():
    return {"batching": state["batcher"].enabled, **metrics.snapshot(state["batcher"].depth)}


@app.post("/api/config")
async def set_config(c: ConfigIn):
    b = state["batcher"]
    if c.batching is not None:
        b.enabled = c.batching
    if c.max_batch:
        b.max_batch = max(1, min(c.max_batch, 256))
    if c.max_wait_ms is not None:
        b.max_wait_ms = max(0.0, min(c.max_wait_ms, 100.0))
    metrics.reset()
    return {"batching": b.enabled, "max_batch": b.max_batch, "max_wait_ms": b.max_wait_ms}


@app.get("/api/events")
async def events(request: Request):
    """Server-Sent Events stream (works on App Runner / Render / ECS, unlike WebSockets on some)."""
    q = hub.subscribe()

    async def gen():
        try:
            while True:
                try:
                    msg = await asyncio.wait_for(q.get(), 15)
                except asyncio.TimeoutError:
                    yield ": ping\n\n"
                    continue
                yield f"data: {json.dumps(msg)}\n\n"
        finally:
            hub.unsubscribe(q)

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
