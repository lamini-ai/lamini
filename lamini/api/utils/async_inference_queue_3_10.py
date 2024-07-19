import asyncio
import functools
import logging
from typing import (
    Any,
    AsyncGenerator,
    AsyncIterator,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
)

import aiohttp
from lamini.api.utils.base_async_inference_queue import BaseAsyncInferenceQueue
from lamini.api.utils.process_batch import process_batch
from lamini.api.utils.reservations import create_reservation_api
from lamini.generation.token_optimizer import TokenOptimizer

logger = logging.getLogger(__name__)


class AsyncInferenceQueue(BaseAsyncInferenceQueue):
    """
    Child class to handle AsyncInferenceQueue functions for python >= 3.10.
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
        results = {}
        exceptions = []
        local_cache = None
        if local_cache_file:
            local_cache = self.read_local_cache(local_cache_file)
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
                metadata,
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

    def combine_results(self, results: Dict[str, List[Any]]) -> List[Any]:
        """Build a single list from the provided nested lists within results

        Parameters
        ----------
        results: Dict[str, List[Any]]
            Dictionary holding results from batch calls with request results
            within the values

        Returns
        -------
        combined_results: List[Any]
            Combined list of the contents of results
        """

        results = dict(sorted(results.items()))
        combined_results = []
        for _, result_future in results.items():
            logger.info(f"inference result_future: {result_future}")
            assert isinstance(result_future, list)
            combined_results.extend(result_future)
        return combined_results

    async def form_batches(
        self,
        request: Dict[str, Any],
        client: aiohttp.ClientSession,
        key: str,
        api_prefix: str,
        local_cache_file: str,
        local_cache: Dict[str, Any],
        callback: Callable,
        metadata: Dict[str, Any],
    ) -> AsyncGenerator:
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

        assert isinstance(request["prompt"], list)
        batch_size_func = self.reservation_api.get_dynamic_max_batch_size
        async for i, batch_size in arange_w_step_func(
            0, len(request["prompt"]), batch_size_func
        ):
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


async def map_unordered(
    func: Callable, iterable: Union[AsyncIterator, Iterator], *, limit: int
) -> AsyncGenerator:
    """Map function 'func' over the provided iterable, limit the
    number of concurrent calls to the provided limit.

    Parameters
    ----------
    func: Callable
        Function for which to map over the iterable

    iterable: Union[AsyncIterator, Iterator]
        Data structure to run func onto

    limit: int
        Limit to the number of concurrent calls

    Yields
    -------
    Returned result of the finished task
    """

    try:
        aws = map(func, iterable)
    except TypeError:
        aws = (func(x) async for x in iterable)
    async for task in limit_concurrency(aws, limit):
        yield await task


async def limit_concurrency(
    aws: Union[AsyncIterator, Iterator], limit: int
) -> Union[AsyncGenerator, None]:
    """Limit the number of calls for the provided Iterator 'aws'

    Parameters
    ----------
    aws: Union[AsyncIterator, Iterator]
        Task to be run

    limit: int
        Limit to the number of concurrent calls

    Yields
    ------
    Finished job

    Returns
    -------
    None if no pending jobs
    """

    try:
        aws = aiter(aws)
        is_async = True
    except TypeError:
        aws = iter(aws)
        is_async = False

    aws_ended = False
    pending = set()

    # TODO: there is a bug here, see
    # test_limit_concurrency_with_append_more_worker_than_items()
    # in sdk/test/lamini/generation/test_generation_queue_3_10.py.
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


def return_args_and_exceptions(func: Callable) -> Any:
    """Partial function wrapper for given returned args and exceptions

    Parameters
    ----------
    func: Callable
        Function to be called with the partial args and exceptions

    Returns
    -------
    partial object
    """

    return functools.partial(_return_args_and_exceptions, func)


async def _return_args_and_exceptions(func: Callable, *args) -> Tuple[Any, Any]:
    """Wrapper to handle exceptions to the provided func if
    not all arguments are provided

    Parameters
    ----------
    func: Callable
        Function to be called

    args:
        Function arguments

    Returns
    -------
    Tuple[Any, Any]
        Returns the decomposed args along with either the result of
        the provided function, or the exception from the function
    """

    try:
        return *args, await func(*args)
    except Exception as e:
        return *args, e


async def arange(
    start: int, stop: Optional[int] = None, step: Optional[int] = 1
) -> AsyncGenerator:
    """Async arange implementation

    Parameters
    ----------
    start: int
        Beginning value of the range

    stop: Optional[int] = None
        Ending value of the range

    step: Optional[int] = 1
        Step size of the range

    Yields
    ------
    Iterated next value of the range
    """

    if stop:
        range_ = range(start, stop, step)
    else:
        range_ = range(start)
    for i in range_:
        yield i
        await asyncio.sleep(0)


async def arange_w_step_func(
    start: int, stop: int, step_func: Callable
) -> AsyncGenerator:
    """Async arange implementation with complex step

    Parameters
    ----------
    start: int
        Beginning value of the range

    stop: int
        Ending value of the range

    step_func: Callable
        Function to handle the step size of the range

    Yields
    ------
    Iterated next value of the range
    """

    i = start
    while i < stop:
        batch_size = step_func()
        yield (i, batch_size)
        i += batch_size
        await asyncio.sleep(0)
