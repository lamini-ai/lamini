import asyncio
import concurrent
import functools
import json
import logging
import os

import aiohttp
import lamini
from lamini.api.lamini_config import get_config, get_configured_key, get_configured_url
from lamini.api.rest_requests import make_async_web_request

logger = logging.getLogger(__name__)


class AsyncInferenceQueue:
    def __init__(self, api_key, api_url, config):
        self.config = get_config(config)
        self.api_key = api_key or lamini.api_key or get_configured_key(self.config)
        self.api_url = api_url or lamini.api_url or get_configured_url(self.config)
        self.api_prefix = self.api_url + "/v1/"

    async def submit(self, request, local_cache_file, callback=None):
        # Break the request into batches
        results = {}
        exceptions = []
        local_cache = None
        if local_cache_file:
            local_cache = self.read_local_cache(local_cache_file)
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

            wrapped = return_args_and_exceptions(process_batch)
            async for result in map_unordered(
                wrapped, batches, limit=self.get_max_workers()
            ):
                if isinstance(result[1], Exception):
                    exceptions.append(result[1])
                else:
                    results[result[0]["index"]] = result[1]
        if local_cache_file:
            if len(exceptions) > 0:
                raise exceptions[0]
        # Combine the results and return them
        return self.combine_results(results)

    def read_local_cache(self, local_cache_file):
        if not os.path.exists(local_cache_file):
            return {}

        with open(local_cache_file, "r") as file:
            content = file.read()

        content = content.strip()
        if content.strip() == "":
            return {}

        if content[-1] != ",":
            raise Exception(f"The last char in {local_cache_file} should be ','")

        content = "{" + content[:-1] + "}"
        cache = json.loads(content)

        if not isinstance(cache, dict):
            raise Exception(f"{local_cache_file} cannot be loaded as dict")

        return cache

    def combine_results(self, results):
        results = dict(sorted(results.items()))
        combined_results = []
        for index, result_future in results.items():
            if isinstance(result_future, concurrent.futures._base.Future):
                result = result_future.result()
            else:
                result = result_future
            logger.info(f"inference result: {result}")
            if isinstance(result, list):
                combined_results.extend(result)
            else:
                combined_results.append(result)

        return combined_results

    def get_max_workers(self):
        return lamini.max_workers

    async def form_batches(
        self, request, client, key, api_prefix, local_cache_file, local_cache, callback
    ):
        batch_size = self.get_batch_size()

        if isinstance(request["prompt"], str):
            yield {
                "api_prefix": api_prefix,
                "key": key,
                "batch": request,
                "client": client,
                "local_cache_file": local_cache_file,
                "local_cache": local_cache,
                "index": 0,
                "callback": callback,
            }
        else:
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

    def get_batch_size(self):
        return lamini.batch_size


async def process_batch(args):
    client = args["client"]
    key = args["key"]
    api_prefix = args["api_prefix"]
    batch = args["batch"]
    local_cache_file = args["local_cache_file"]
    local_cache = args["local_cache"]
    callback = args["callback"]
    logger.debug(f"Sending batch")
    url = api_prefix + "completions"
    batch_k = str(batch)
    if local_cache and batch_k in local_cache:
        return local_cache[batch_k]
    result = await make_async_web_request(client, key, url, "post", batch)
    logger.debug(f"Received batch response")
    if local_cache_file and result:
        append_local_cache(local_cache_file, batch_k, result)
    if callback:
        callback(batch, result)
    return result


def append_local_cache(local_cache_file, batch, res):
    batch_k = json.dumps(str(batch))
    batch_v = json.dumps(res)
    cache_line = f"{batch_k}: {batch_v},\n\n"

    with open(local_cache_file, "a") as file:
        file.write(cache_line)


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
