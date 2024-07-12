import asyncio
import logging
from typing import Optional, List, Dict, Any, Callable, Generator

import aiohttp
import lamini
from lamini.api.utils.base_async_inference_queue import BaseAsyncInferenceQueue
from lamini.api.utils.process_batch import process_batch
from lamini.api.utils.reservations import create_reservation_api
from lamini.generation.token_optimizer import TokenOptimizer

logger = logging.getLogger(__name__)


class AsyncInferenceQueue(BaseAsyncInferenceQueue):
    """ 
    Child class to handle AsyncInferenceQueue functions for python <= 3.9. 
    AsyncInferenceQueue will handle the overhead of async calls of the 
    web requests. This class will handle the splitting and combination of
    the provided prompts within the 'request' parameter when calling submit.
    
    BaseAsyncInferenceQueue Inherited Parameters

    api_key: str
        Lamini platform API key, if not provided the key stored 
        within ~.lamini/configure.yaml will be used. If either
        don't exist then an error is raised.

    api_url: Optional[str]
        Lamini platform api url, only needed if a different url is needed outside of the
        defined ones here: https://github.com/lamini-ai/lamini-platform/blob/main/sdk/lamini/api/lamini_config.py#L68
            i.e. localhost, staging.lamini.ai, or api.lamini.ai
            Additionally, LLAMA_ENVIRONMENT can be set as an environment variable
            that will be grabbed for the url before any of the above defaults
        
    config: dict
        Dictionary that is handled from the following script:
            https://github.com/lamini-ai/lamini-platform/blob/main/sdk/lamini/api/lamini_config.py
        Configurations currently hold the following keys and data as a yaml format:
            local:
                url: <url>
            staging:
                url: <url>
            production:
                url: <url>
            
            local:
                key: <auth-key>
            staging:
                key: <auth-key>
            production:
                key:
                    <auth-key  

    """

    async def submit(
        self,
        request: Dict[str, Any],
        local_cache_file: str,
        callback: Optional[Callable] = None,
        metadata: Optional[Dict[str, Any]] = None,
        token_optimizer: Optional[TokenOptimizer] = None,
    ) -> List[Any]:
        """Handling of the logic around breaking a request into batches based
        on the size of the prompts given within the request and the size of the
        number of workers. Returned List is a combination of the results for each
        batch.

        Parameters
        ----------    
        request: Dict[str, Any]
            Data to be sent within a request
        
        local_cache_file: str
            Path to local cache file

        callback: Optional[Callable] = None
            Function to post process requests after successful requests have been made
        
        metadata: Optional[Dict[str, Any]] = None
            Data passed into the callback function if provided.

        token_optimizer: Optional[TokenOptimizer] = None
            Object to handle finding the optimal number of max tokens given the
            provided prompts within 'request'

        Returns
        -------
        List[Any]
            Combined results from the call to self.combine_results
        """

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

    def combine_results(self, results: List[List[Any]]) -> List[Any]:
        """Build a single list from the provided nested lists within results

        Parameters
        ----------    
        results: List[List[Any]]
            Nested list holding results from batch calls

        Returns
        -------
        combined_results: List[Any]
            Combined list of the contents of results
        """

        combined_results = []
        for result in results:
            logger.info(f"inference result: {result}")
            assert isinstance(result, list)
            combined_results.extend(result)
        return combined_results

    def form_batches(
        self,
        request: Dict[str, Any],
        client: aiohttp.ClientSession,
        key: str,
        api_prefix: str,
        local_cache_file: str,
        local_cache: Dict[str, Any],
        callback: Callable,
        metadata: Dict[str, Any],
    ) -> Generator[Dict[str, Any], Any, Any]:
        """Split the provided request into batches of size self.get_batch_size()

        Parameters
        ----------    
        request: Dict[str, Any]
            Request data for the web request

        client: aiohttp.ClientSession
            Interface for the http requests

        key: str
            API key

        api_prefix: str
            API url prefix

        local_cache_file: str
            Path of local cache file

        local_cache: Dict[str, Any]
            Contents of the last read in local cache

        callback: Callable
            Function for post processing of the results of the request

        metadata: Dict[str, Any]
            Data to be passed into the callback function

        Yields
        -------
        Dict[str, Any]
            New request with reduced request size to the batch size
        """

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


async def wrapper(semaphore: asyncio.Semaphore, aw: Any) -> Any:
    """Semaphore handler for waiting tasks within asyncio.running_loop

    Parameters
    ----------    
    semaphore: asyncio.Semaphore
        Async signal handler within asyncio

    aw: Any
        Code to be processed when the semaphore is open

    Returns
    -------
    Any
        Results of the provided aw code
    """
    async with semaphore:
        return await aw
