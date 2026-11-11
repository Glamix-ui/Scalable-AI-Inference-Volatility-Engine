"""Benchmark batching ON vs OFF. Server must be running.  python scripts/loadtest.py --url http://localhost:8000"""
import argparse
import asyncio
import time

import httpx
import numpy as np


async def run(url, n, conc, batching):
    limits = httpx.Limits(max_connections=conc)
    async with httpx.AsyncClient(base_url=url, timeout=60, limits=limits) as c:
        await c.post("/api/config", json={"batching": batching})
        await c.post("/api/predict", json={"ticker": "AAPL"})        # warm the market-data cache
        sem, lat = asyncio.Semaphore(conc), []

        async def one():
            async with sem:
                t = time.perf_counter()
                (await c.post("/api/predict", json={"ticker": "AAPL"})).raise_for_status()
                lat.append((time.perf_counter() - t) * 1000)

        t0 = time.perf_counter()
        await asyncio.gather(*(one() for _ in range(n)))
        dur = time.perf_counter() - t0
    m = np.array(lat)
    return f"| {'on' if batching else 'off'} | {n / dur:7.1f} | {np.percentile(m, 50):7.1f} | {np.percentile(m, 99):7.1f} |"


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://localhost:8000")
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--concurrency", type=int, default=64)
    a = p.parse_args()
    print("| batching | req/s | P50 ms | P99 ms |\n|---|---|---|---|")
    for mode in (False, True):
        print(await run(a.url, a.n, a.concurrency, mode))


asyncio.run(main())
