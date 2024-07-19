import asyncio
import functools
import logging
from typing import AsyncIterator, Iterator, Optional, TypeVar, Union, Tuple, Any

import lamini

from lamini.generation.base_generation_queue import BaseGenerationQueue
from lamini.generation.process_generation_batch import process_generation_batch
from lamini.generation.token_optimizer import TokenOptimizer

T = TypeVar("T")

logger = logging.getLogger(__name__)

global_inference_queue = None


def get_global_inference_queue(api_key, api_url, config):
    global global_inference_queue
    if global_inference_queue is None or global_inference_queue.client.closed:
        global_inference_queue = GenerationQueue(api_key, api_url, config=config)
    return global_inference_queue


class AppendableAsyncGenerator:
    """Holds a async generator and a list, and prefer return elements in the list

    Always first returns elements from the list.
    This allows different priorities between items.
    E.g., failed item can be appended, so that they would be retried first.
    """

    def __init__(self, generator):
        self.generator = generator
        self.appendlist = []

    def __aiter__(self):
        return self

    async def __anext__(self):
        if len(self.appendlist):
            return self.appendlist.pop()
        next_val = await self.generator.__anext__()
        return next_val

    def append(self, item):
        self.appendlist.append(item)


class GenerationQueue(BaseGenerationQueue):
    async def submit(
        self,
        request: dict,
        token_optimizer: Optional[TokenOptimizer] = None,
    ):
        batches = self.form_batches(
            request,
            self.client,
            self.api_key,
            self.api_prefix,
            token_optimizer,
        )
        batches = AppendableAsyncGenerator(batches)
        wrapped = return_args_and_exceptions(process_generation_batch)
        async_iterator = map_unordered(wrapped, batches, limit=self.get_max_workers())

        async for result in async_iterator:
            if isinstance(result[1], Exception):
                if (
                    result[0]["batch"]["prompt"][0].error is not None
                    and len(result[0]["batch"]["prompt"][0].error)
                    < self.get_retry_limit()
                ):
                    logger.debug(
                        f"Retrying up to {self.get_retry_limit()}, prompt: {result[0]}"
                    )
                    batches.append(result[0])
                    # Retried prompt batch is not yielded.
                    # They will eventually be yielded if 1) succeed within retry limit
                    # or fail even after retry.
                    continue

            for elem in result[0]["batch"]["prompt"]:
                yield elem

    def combine_results(self, results):
        results = dict(sorted(results.items()))
        combined_results = []
        for _, result_future in results.items():
            assert isinstance(result_future, list)
            combined_results.extend(result_future)
        return combined_results

    async def form_batches(
        self,
        request,
        client,
        key,
        api_prefix,
        token_optimizer: Optional[TokenOptimizer],
    ):
        batch_size_func = self.reservation_api.get_dynamic_max_batch_size
        async for prompt in next_n_w_step_func(request["prompt"], batch_size_func):
            batch = request.copy()
            batch["prompt"] = prompt
            if token_optimizer is not None and "max_new_tokens" in batch:
                batch["max_tokens"] = (
                    token_optimizer.calculate_heuristic_max_tokens_from_prompt(
                        [p.prompt for p in batch["prompt"]], batch["max_new_tokens"]
                    )
                )
                self.reservation_api.max_tokens = batch["max_tokens"]
            yield {
                "api_prefix": api_prefix,
                "key": key,
                "batch": batch,
                "client": client,
            }


async def next_n_w_step_func(iterator: Union[AsyncIterator, Iterator], step_func):
    if isinstance(iterator, AsyncIterator):
        async for x in async_chunks(iterator, step_func):
            yield x
    elif isinstance(iterator, Iterator):
        for x in chunks(iterator, step_func):
            yield x
    else:
        raise TypeError("iterator must be an iterator or an async iterator")


async def map_unordered(func, iterable, *, limit):
    try:
        aws = map(func, iterable)
    except TypeError:
        aws = (func(x) async for x in iterable)
    async for task in limit_concurrency(aws, limit):
        yield await task


async def limit_concurrency(aws, limit):
    """Yields up-to a number of limit items from aws."""
    try:
        aws = aiter(aws)
        is_async = True
    except TypeError:
        aws = iter(aws)
        is_async = False

    aws_ended = False
    pending = set()

    while pending or not aws_ended:
        while len(pending) < limit and not aws_ended:
            try:
                aw = await anext(aws) if is_async else next(aws)
            except StopAsyncIteration if is_async else StopIteration:
                aws_ended = True
            else:
                pending.add(asyncio.ensure_future(aw))

        if not pending:
            return

        done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
        while done:
            yield done.pop()


def return_args_and_exceptions(func) -> Tuple[Any, Any]:
    return functools.partial(_return_args_and_exceptions, func)


async def _return_args_and_exceptions(func, *args) -> Tuple[Any, Any]:
    try:
        return *args, await func(*args)
    except Exception as e:
        return *args, e


async def arange(start, stop=None, step=1):
    if stop:
        range_ = range(start, stop, step)
    else:
        range_ = range(start)
    for i in range_:
        yield i
        await asyncio.sleep(0)


def chunks(
    iterator: Iterator[T],
    size_fn,
) -> Iterator[list[T]]:
    """Yield successive n-sized chunks from lst."""
    finished = False

    while not finished:
        results: list[T] = []
        size = size_fn()

        for _ in range(size):
            try:
                result = None
                while result is None:
                    result = next(iterator)
            except StopIteration:
                finished = True
            else:
                results.append(result)

        if results:
            yield results


async def async_chunks(
    async_iterator: AsyncIterator[T],
    size_fn,
) -> AsyncIterator[list[T]]:
    """Generate chunks from an asynchronous sequence.

    Chunks are lists consists of original ``T`` elements.
    The chunk can't contain more than ``size`` elements.
    The last chunk might contain less than ``size`` elements,
    but can't be empty.
    """
    finished = False

    while not finished:
        results: list[T] = []
        size = size_fn()

        for _ in range(size):
            try:
                result = None
                while result is None:
                    result = await anext(async_iterator)
            except StopAsyncIteration:
                finished = True
            else:
                results.append(result)

        if results:
            yield results
