"""
Generic batch processor — runs any NLP pipeline function over a list of texts
in fixed-size batches using run_in_executor to avoid blocking the event loop.
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable

from app.config import get_settings


async def process_text_column(
    texts: list[str],
    pipeline_fn: Callable[[list[str]], list[Any]],
    batch_size: int | None = None,
) -> list[Any]:
    """
    Process *texts* through *pipeline_fn* in batches of *batch_size*.

    Uses ``asyncio.get_event_loop().run_in_executor`` to offload the
    CPU-bound NLP work to a thread-pool, keeping the FastAPI event loop
    responsive.

    Parameters
    ----------
    texts : list[str]
        Input texts to process.
    pipeline_fn : callable
        A function that accepts ``list[str]`` and returns ``list[Any]``.
    batch_size : int, optional
        Override the default ``NLP_BATCH_SIZE`` from settings.

    Returns
    -------
    list[Any]
        Flat list of results aligned 1-to-1 with *texts*.
    """
    if batch_size is None:
        batch_size = get_settings().NLP_BATCH_SIZE

    if not texts:
        return []

    loop = asyncio.get_event_loop()
    results: list[Any] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        batch_results = await loop.run_in_executor(None, pipeline_fn, batch)
        results.extend(batch_results)

    return results
