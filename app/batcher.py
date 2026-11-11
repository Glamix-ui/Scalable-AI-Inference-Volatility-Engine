"""Async micro-batching: many concurrent requests -> one batched forward pass."""
import asyncio
import time
from dataclasses import dataclass

import numpy as np


@dataclass
class _Item:
    id: int
    x: np.ndarray
    fut: asyncio.Future
    t0: float


class MicroBatcher:
    def __init__(self, predict_fn, metrics, hub, max_batch=32, max_wait_ms=8.0, enabled=True):
        self.predict_fn, self.metrics, self.hub = predict_fn, metrics, hub
        self.max_batch, self.max_wait_ms, self.enabled = max_batch, max_wait_ms, enabled
        self.queue: asyncio.Queue = asyncio.Queue()
        self._seq = 0
        self._batches = 0
        self._task: asyncio.Task | None = None

    def start(self):
        self._task = asyncio.create_task(self._run())

    async def stop(self):
        if self._task:
            self._task.cancel()

    @property
    def depth(self) -> int:
        return self.queue.qsize()

    async def submit(self, x: np.ndarray) -> tuple[float, int]:
        """Queue one request; resolves to (prediction, size_of_batch_it_rode_in)."""
        self._seq += 1
        item = _Item(self._seq, x, asyncio.get_running_loop().create_future(), time.perf_counter())
        self.queue.put_nowait(item)
        self.hub.publish({"type": "enqueue", "id": item.id})
        result = await item.fut
        self.metrics.record_request((time.perf_counter() - item.t0) * 1000)
        return result

    async def _collect(self) -> list[_Item]:
        batch = [await self.queue.get()]
        if not self.enabled:                       # batching off => batch size is always 1
            return batch
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.max_wait_ms / 1000
        while len(batch) < self.max_batch:
            remaining = deadline - loop.time()
            if remaining <= 0:
                break
            try:
                async with asyncio.timeout(remaining):
                    batch.append(await self.queue.get())
            except TimeoutError:
                break
        return batch

    async def _run(self):
        while True:
            batch = await self._collect()
            t = time.perf_counter()
            try:
                out = await asyncio.to_thread(self.predict_fn, np.stack([i.x for i in batch]))
            except Exception as exc:               # fail the batch, keep the worker alive
                for i in batch:
                    if not i.fut.done():
                        i.fut.set_exception(exc)
                continue
            infer_ms = (time.perf_counter() - t) * 1000
            self._batches += 1
            self.metrics.record_batch(len(batch), infer_ms)
            self.hub.publish({"type": "batch", "n": self._batches, "ids": [i.id for i in batch],
                              "size": len(batch), "infer_ms": round(infer_ms, 2)})
            for i, v in zip(batch, out):
                if not i.fut.done():
                    i.fut.set_result((float(v), len(batch)))
