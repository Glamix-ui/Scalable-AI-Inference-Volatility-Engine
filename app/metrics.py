import asyncio
import time
from collections import deque

import numpy as np


class Metrics:
    """Rolling in-memory latency / throughput stats."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.lat = deque(maxlen=2000)       # end-to-end ms (queue wait + inference)
        self.stamps = deque(maxlen=2000)
        self.batch_sizes = deque(maxlen=500)
        self.infer_ms = deque(maxlen=500)
        self.total = 0

    def record_request(self, ms: float):
        self.lat.append(ms)
        self.stamps.append(time.time())
        self.total += 1

    def record_batch(self, size: int, infer_ms: float):
        self.batch_sizes.append(size)
        self.infer_ms.append(infer_ms)

    def snapshot(self, depth: int = 0) -> dict:
        lat = np.fromiter(self.lat, float) if self.lat else np.array([0.0])
        now = time.time()
        recent = sum(1 for t in self.stamps if now - t <= 5)
        avg = lambda d: float(np.mean(d)) if d else 0.0
        return {
            "p50": round(float(np.percentile(lat, 50)), 2),
            "p99": round(float(np.percentile(lat, 99)), 2),
            "mean": round(float(lat.mean()), 2),
            "rps": round(recent / 5, 1),
            "avg_batch": round(avg(self.batch_sizes), 1),
            "avg_infer_ms": round(avg(self.infer_ms), 2),
            "queue_depth": depth,
            "total": self.total,
        }


class Hub:
    """Fan-out of live events (enqueue / batch / stats) to every connected browser."""

    def __init__(self):
        self.subs: set[asyncio.Queue] = set()

    def subscribe(self) -> asyncio.Queue:
        q = asyncio.Queue(maxsize=2000)
        self.subs.add(q)
        return q

    def unsubscribe(self, q):
        self.subs.discard(q)

    def publish(self, msg: dict):
        for q in list(self.subs):
            try:
                q.put_nowait(msg)
            except asyncio.QueueFull:
                pass   # slow client: drop events instead of blocking the engine
