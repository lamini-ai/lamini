import asyncio
import functools
import logging

import aiohttp
from lamini.api.utils.base_async_inference_queue import BaseAsyncInferenceQueue
from lamini.api.utils.process_batch import process_batch
from lamini.api.utils.reservations import create_reservation_api

logger = logging.getLogger(__name__)


class AsyncInferenceQueue(BaseAsyncInferenceQueue):
    async def submit(self, request, local_cache_file, callback=None):
        # Break the request into batches
        results = {}
        exceptions = []
        local_cache = None
        if local_cache_file:
            local_cache = self.read_local_cache(local_cache_file)
        self.reservation_api = create_reservation_api(
            self.api_key, self.api_url, self.config
        )
        self.reservation_api.initialize_reservation(
            len(request["prompt"]), request["model_name"], request["max_tokens"]
        )
        self.reservation_api.pause_for_reservation_start()
        connector = aiohttp.TCPConnector(limit=self.get_max_workers())
        async with aiohttp.ClientSession(connector=connector) as client:
            batches = self.form_batches(
                request,
                client,
                self.api_key,
                self.api_prefix,
                local_cache_file,
                local_cache,
                callback,
            )
            self.reservation_polling_task = asyncio.create_task(
                self.reservation_api.kickoff_reservation_polling(client)
            )
            wrapped = return_args_and_exceptions(process_batch)
            async for result in map_unordered(
                wrapped, batches, limit=self.get_max_workers()
            ):
                if isinstance(result[1], Exception):
                    exceptions.append(result[1])
                else:
                    results[result[0]["index"]] = result[1]
        self.reservation_api.is_working = False
        if self.reservation_polling_task is not None:
            self.reservation_polling_task.cancel()
        if self.reservation_api.polling_task is not None:
            self.reservation_api.polling_task.cancel()
        if len(exceptions) > 0:
            print(
                f"Encountered {len(exceptions)} errors during run. Raising first as an exception."
            )
            raise exceptions[0]
        # Combine the results and return them
        return self.combine_results(results)

    def combine_results(self, results):
        results = dict(sorted(results.items()))
        combined_results = []
        for _, result_future in results.items():
            logger.info(f"inference result_future: {result_future}")
            assert isinstance(result_future, list)
            combined_results.extend(result_future)
        return combined_results

    async def form_batches(
        self, request, client, key, api_prefix, local_cache_file, local_cache, callback
    ):
        batch_size = self.get_batch_size()
        assert isinstance(request["prompt"], list)
        async for i in arange(0, len(request["prompt"]), batch_size):
            batch = request.copy()
            end = min(i + batch_size, len(request["prompt"]))
            batch["prompt"] = request["prompt"][i:end]
            yield {
                "api_prefix": api_prefix,
                "key": key,
                "batch": batch,
                "client": client,
                "local_cache_file": local_cache_file,
                "local_cache": local_cache,
                "index": i,
                "callback": callback,
            }


async def map_unordered(func, iterable, *, limit):
    try:
        aws = map(func, iterable)
    except TypeError:
        aws = (func(x) async for x in iterable)
    async for task in limit_concurrency(aws, limit):
        yield await task


async def limit_concurrency(aws, limit):
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


def return_args_and_exceptions(func):
    return functools.partial(_return_args_and_exceptions, func)


async def _return_args_and_exceptions(func, *args):
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
