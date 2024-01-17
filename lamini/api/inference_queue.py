import concurrent
import json
import os
import logging
import lamini
from concurrent.futures import ThreadPoolExecutor
from lamini.api.lamini_config import get_config, get_configured_key, get_configured_url
from lamini.api.rest_requests import make_web_request
from lamini.error.error import RateLimitError

logger = logging.getLogger(__name__)

thread_pool = None


class InferenceQueue:
    def __init__(self, api_key, api_url, config):
        self.config = get_config(config)
        self.api_key = api_key or lamini.api_key or get_configured_key(self.config)
        self.api_url = api_url or lamini.api_url or get_configured_url(self.config)
        self.api_prefix = self.api_url + "/v1/"
        self.thread_pool = self.create_thread_pool()

    def submit(self, request, local_cache_file):
        # Break the request into batches
        batches = self.form_batches(request)

        results = []
        exceptions = []

        logger.info(
            f"Launching {len(batches)} batches onto the thread pool of size {self.get_max_workers()}"
        )

        if local_cache_file:
            local_cache = self.read_local_cache(local_cache_file)

        for batch in batches:
            # Submit each batch to the thread pool

            batch_k = str(batch)
            if local_cache_file and batch_k in local_cache:
                results.append(local_cache[batch_k])
            else:
                results.append(
                    self.thread_pool.submit(
                        process_batch, self.api_key, self.api_prefix, batch
                    )
                )

        # Wait for all the results to come back
        for i, result in enumerate(results):
            if local_cache_file:
                if not isinstance(result, concurrent.futures._base.Future):
                    continue

                r = None
                try:
                    r = result.result()
                except Exception as e:
                    exceptions.append(e)

                if r:
                    self.append_local_cache(local_cache_file, batches[i], r)
            else:
                result.result()

        if local_cache_file:
            if len(exceptions) > 0:
                raise exceptions[0]

        # Combine the results and return them
        return self.combine_results(results)

    def read_local_cache(self, local_cache_file):
        if not os.path.exists(local_cache_file):
            return {}
        
        with open(local_cache_file, 'r') as file:
            content = file.read()

        content = content.strip()
        if content.strip() == '':
            return {}

        if content[-1] != ',':
            raise Exception(f"The last char in {local_cache_file} should be ','")

        content = '{' + content[:-1] + '}'
        cache = json.loads(content)

        if not isinstance(cache, dict):
            raise Exception(f"{local_cache_file} cannot be loaded as dict")

        return cache
    
    def append_local_cache(self, local_cache_file, batch, res):
        batch_k = json.dumps(str(batch))
        batch_v = json.dumps(res)
        cache_line = f"{batch_k}: {batch_v},\n\n"

        with open(local_cache_file, 'a') as file:
            file.write(cache_line)

    def combine_results(self, results):
        combined_results = []
        for result_future in results:
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

    def create_thread_pool(self):
        global thread_pool
        if thread_pool is None:
            thread_pool = ThreadPoolExecutor(max_workers=self.get_max_workers())

        return thread_pool

    def get_max_workers(self):
        return lamini.max_workers

    def form_batches(self, request):
        batch_size = self.get_batch_size()
        batches = []

        if isinstance(request["prompt"], str):
            batches.append(request)
        else:
            for i in range(0, len(request["prompt"]), batch_size):
                batch = request.copy()
                end = min(i + batch_size, len(request["prompt"]))
                batch["prompt"] = request["prompt"][i:end]
                batches.append(batch)

        if len(batches) > self.get_max_batch_count():
            raise RateLimitError(
                f"Too many requests, {len(request['prompt'])} >"
                f" {self.get_max_batch_count() * self.get_batch_size()} (max)",
            )

        return batches

    def get_batch_size(self):
        return lamini.batch_size

    def get_max_batch_count(self):
        return 512


def process_batch(key, api_prefix, batch):
    url = api_prefix + "completions"
    result = make_web_request(key, url, "post", batch)
    return result
