import asyncio

import numpy as np

from app import data
from app.batcher import MicroBatcher
from app.metrics import Hub, Metrics


def test_requests_are_grouped_and_answered_correctly():
    async def go():
        sizes = []

        def fake_model(batch):
            sizes.append(len(batch))
            return batch.sum(axis=(1, 2))

        b = MicroBatcher(fake_model, Metrics(), Hub(), max_batch=16, max_wait_ms=20)
        b.start()
        xs = [np.full((30, 5), i, dtype=np.float32) for i in range(50)]
        res = await asyncio.gather(*(b.submit(x) for x in xs))
        await b.stop()
        return sizes, res

    sizes, res = asyncio.run(go())
    assert [round(r[0]) for r in res] == [i * 150 for i in range(50)]   # each caller got ITS answer
    assert max(sizes) > 1 and len(sizes) < 50                            # batching actually happened
    assert max(sizes) <= 16


def test_batching_disabled_means_batch_of_one():
    async def go():
        sizes = []
        b = MicroBatcher(lambda x: (sizes.append(len(x)), x.sum(axis=(1, 2)))[1], Metrics(), Hub(), enabled=False)
        b.start()
        await asyncio.gather(*(b.submit(np.zeros((30, 5), np.float32)) for _ in range(10)))
        await b.stop()
        return sizes

    assert set(asyncio.run(go())) == {1}


def test_feature_window_shape():
    df = data.synthetic("AAPL")
    assert data.latest_window(df).shape == (data.WINDOW, data.N_FEATURES)
