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

    def submit(self, request):
        # Break the request into batches
        batches = self.form_batches(request)

        results = []

        logger.info(
            f"Launching {len(batches)} batches onto the thread pool of size {self.get_max_workers()}"
        )

        for batch in batches:
            # Submit each batch to the thread pool
            results.append(
                self.thread_pool.submit(
                    process_batch, self.api_key, self.api_prefix, batch
                )
            )

        # Wait for all the results to come back
        for result in results:
            result.result()

        # Combine the results and return them
        return self.combine_results(results)

    def combine_results(self, results):
        combined_results = []
        for result_future in results:
            result = result_future.result()
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
        # url = self.api_prefix + "system/gpu_count"
        gpu_count = 12  # make_web_request(self.key, url, "get", {})["gpu_count"]

        return gpu_count

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
                "RateLimitError",
            )

        return batches

    def get_batch_size(self):
        return 10

    def get_max_batch_count(self):
        return 512


def process_batch(key, api_prefix, batch):
    url = api_prefix + "completions"
    result = make_web_request(key, url, "post", batch)
    return result
