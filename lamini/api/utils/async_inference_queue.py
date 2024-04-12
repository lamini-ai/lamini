import asyncio
import logging
from typing import Optional

import aiohttp
import lamini
from lamini.api.utils.base_async_inference_queue import BaseAsyncInferenceQueue
from lamini.api.utils.process_batch import process_batch
from lamini.api.utils.reservations import create_reservation_api
from lamini.generation.token_optimizer import TokenOptimizer

logger = logging.getLogger(__name__)


class AsyncInferenceQueue(BaseAsyncInferenceQueue):
    async def submit(
        self,
        request,
        local_cache_file,
        callback=None,
        metadata=None,
        token_optimizer: Optional[TokenOptimizer] = None,
    ):
        # Break the request into batches
        results = []
        exceptions = []
        local_cache = None
        if local_cache_file:
            local_cache = self.read_local_cache(local_cache_file)
        loop = asyncio.get_running_loop()
        self.reservation_api = create_reservation_api(
            self.api_key, self.api_url, self.config
        )
        if token_optimizer is not None and "max_new_tokens" in request:
            request["max_tokens"] = (
                token_optimizer.calculate_heuristic_max_tokens_from_prompt(
                    request["prompt"], request["max_new_tokens"]
                )
            )
            logger.debug(f"Adjusted max_tokens to: {request['max_tokens']}")
        self.reservation_api.initialize_reservation(
            len(request["prompt"]),
            request["model_name"],
            self.get_batch_size(),
            request["max_tokens"],
        )
        self.reservation_api.pause_for_reservation_start()
        connector = aiohttp.TCPConnector(limit=self.get_max_workers(), loop=loop)
        async with aiohttp.ClientSession(connector=connector, loop=loop) as client:
            batches = self.form_batches(
                request,
                client,
                self.api_key,
                self.api_prefix,
                local_cache_file,
                local_cache,
                callback,
                metadata,
            )
            self.reservation_polling_task = loop.create_task(
                self.reservation_api.kickoff_reservation_polling(client)
            )
            semaphore = asyncio.Semaphore(lamini.max_workers)
            tasks = [
                loop.create_task(wrapper(semaphore, process_batch(batch)))
                for batch in batches
            ]
            mixed_results = await asyncio.gather(*tasks)
            for result in mixed_results:
                if isinstance(result, Exception):
                    exceptions.append(result)
                else:
                    results.append(result)
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
        combined_results = []
        for result in results:
            logger.info(f"inference result: {result}")
            assert isinstance(result, list)
            combined_results.extend(result)
        return combined_results

    def form_batches(
        self,
        request,
        client,
        key,
        api_prefix,
        local_cache_file,
        local_cache,
        callback,
        metadata,
    ):
        batch_size = self.get_batch_size()
        assert isinstance(request["prompt"], list)
        for i in range(0, len(request["prompt"]), batch_size):
            batch = request.copy()
            end = min(i + batch_size, len(request["prompt"]))
            batch["prompt"] = request["prompt"][i:end]
            metadata_batch = None
            if metadata is not None:
                metadata_batch = metadata[i:end]
            yield {
                "api_prefix": api_prefix,
                "key": key,
                "batch": batch,
                "client": client,
                "local_cache_file": local_cache_file,
                "local_cache": local_cache,
                "index": i,
                "callback": callback,
                "metadata": metadata_batch,
            }


async def wrapper(semaphore, aw):
    async with semaphore:
        return await aw
