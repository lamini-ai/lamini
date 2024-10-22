import asyncio
import time
from typing import Any, Dict, List, Optional, Union

import aiohttp

import lamini
from lamini.api.lamini_config import get_config, get_configured_key, get_configured_url
from lamini.api.rest_requests import make_async_web_request, make_web_request


class BatchStreamingCompletionObject:

    def __init__(
        self,
        api_url: str,
        api_key: str,
        id: str,
        polling_interval: int,
        max_errors: int = 5,
    ):
        self.api_url = api_url + f"/{id}/result"
        self.api_key = api_key
        self.done_streaming = False
        self.polling_interval = polling_interval
        self.current_result = None
        self.error_count = 0
        self.max_errors = max_errors
        self.current_index = 0
        self.available_results = 0

    def __iter__(self) -> object:
        """Iteration definition

        Parameters
        ----------
        None

        Returns
        -------
        Reference to self
        """

        return self

    def __next__(self) -> str:
        """Iterator next step definition

        Parameters
        ----------
        None

        Returns
        -------
        str
            Streamed next result
        """

        return self.next()

    def next(self):
        """Retrieve the next iteration of the response stream

        Parameters
        ----------
        None

        Returns
        -------
        self.current_result: str
            Streamed result from the web request
        """
        if self.done_streaming:
            raise StopIteration()

        self.wait_for_results()
        result = {
            "output": self.current_result["outputs"][self.current_index],
            "finish_reason": self.current_result["finish_reason"][self.current_index],
        }
        self.current_index += 1
        if self.current_index >= len(self.current_result["finish_reason"]):
            self.done_streaming = True
        return result

    def wait_for_results(self):
        # Poll for results until more work is available
        while self.available_results <= self.current_index:
            time.sleep(self.polling_interval)
            try:
                self.current_result = make_web_request(
                    self.api_key,
                    self.api_url,
                    "get",
                )
                if self.current_result == {}:
                    continue
                else:
                    self.available_results = len(
                        self.current_result["finish_reason"]
                    ) - self.current_result["finish_reason"].count(None)

            except Exception as e:
                self.error_count += 1
                if self.error_count > self.max_errors:
                    raise e


class AsyncBatchStreamingCompletionObject:

    def __init__(
        self,
        api_url: str,
        api_key: str,
        id: str,
        polling_interval: int,
        max_errors: int = 5,
    ):
        self.api_url = api_url + f"/{id}/result"
        self.api_key = api_key
        self.done_streaming = False
        self.polling_interval = polling_interval
        self.current_result = None
        self.error_count = 0
        self.max_errors = max_errors
        self.current_index = 0
        self.available_results = 0

    def __aiter__(self) -> object:
        """Asychronous iteration definition

        Parameters
        ----------
        None

        Returns
        -------
        Reference to this instance of AsyncBatchStreamingCompletionObject
        """

        return self

    async def __anext__(self):
        """Asynchronous next definition

        Parameters
        ----------
        None

        Returns
        -------
        str
            Current streaming result from the web request
        """

        return await self.next()

    async def next(self):
        """Retrieve the next iteration of the response stream

        Parameters
        ----------
        None

        Returns
        -------
        self.current_result: str
            Streamed result from the web request
        """
        if self.done_streaming:
            raise StopAsyncIteration()

        await self.wait_for_results()
        # print(self.current_result)
        result = {
            "output": self.current_result["outputs"][self.current_index],
            "finish_reason": self.current_result["finish_reason"][self.current_index],
        }
        self.current_index += 1
        if self.current_index >= len(self.current_result["finish_reason"]):
            self.done_streaming = True
        return result

    async def wait_for_results(self):
        # Poll for results until more work is available
        while self.available_results <= self.current_index:
            await asyncio.sleep(self.polling_interval)
            try:
                async with aiohttp.ClientSession() as client:
                    self.current_result = await make_async_web_request(
                        client,
                        self.api_key,
                        self.api_url,
                        "get",
                    )
                    # print(self.current_result)
                if self.current_result == {}:
                    continue
                else:
                    self.available_results = len(
                        self.current_result["finish_reason"]
                    ) - self.current_result["finish_reason"].count(None)

            except Exception as e:
                self.error_count += 1
                if self.error_count > self.max_errors:
                    raise e


class BatchCompletions:
    """Handler for formatting and POST request for the batch submission API"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
    ) -> None:
        self.config = get_config()
        self.api_key = api_key or lamini.api_key or get_configured_key(self.config)
        self.api_url = api_url or lamini.api_url or get_configured_url(self.config)
        self.api_prefix = self.api_url + "/v1/"

    def submit(
        self,
        prompt: Union[str, List[str]],
        model_name: str,
        output_type: Optional[dict] = None,
        max_tokens: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:

        req_data = self.make_llm_req_map(
            prompt=prompt,
            model_name=model_name,
            output_type=output_type,
            max_tokens=max_tokens,
            max_new_tokens=max_new_tokens,
        )
        resp = make_web_request(
            self.api_key,
            self.api_prefix + "batch_completions",
            "post",
            req_data,
        )
        return resp

    async def async_submit(
        self,
        prompt: Union[str, List[str]],
        model_name: str,
        output_type: Optional[dict] = None,
        max_tokens: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:

        req_data = self.make_llm_req_map(
            prompt=prompt,
            model_name=model_name,
            output_type=output_type,
            max_tokens=max_tokens,
            max_new_tokens=max_new_tokens,
        )
        async with aiohttp.ClientSession() as client:
            resp = await make_async_web_request(
                client,
                self.api_key,
                self.api_prefix + "batch_completions",
                "post",
                req_data,
            )
        return resp

    def streaming_generate(
        self,
        prompt: Union[str, List[str]],
        model_name: str,
        output_type: Optional[dict] = None,
        max_tokens: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
        polling_interval: Optional[float] = 1,
    ) -> Dict[str, Any]:

        resp = self.submit(
            prompt=prompt,
            model_name=model_name,
            output_type=output_type,
            max_tokens=max_tokens,
            max_new_tokens=max_new_tokens,
        )
        return BatchStreamingCompletionObject(
            api_key=self.api_key,
            api_url=self.api_prefix + f"batch_completions",
            polling_interval=polling_interval,
            id=resp["id"],
        )

    async def async_streaming_generate(
        self,
        prompt: Union[str, List[str]],
        model_name: str,
        output_type: Optional[dict] = None,
        max_tokens: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
        polling_interval: Optional[float] = 1,
    ) -> Dict[str, Any]:

        resp = await self.async_submit(
            prompt=prompt,
            model_name=model_name,
            output_type=output_type,
            max_tokens=max_tokens,
            max_new_tokens=max_new_tokens,
        )
        return AsyncBatchStreamingCompletionObject(
            api_key=self.api_key,
            api_url=self.api_prefix + f"batch_completions",
            polling_interval=polling_interval,
            id=resp["id"],
        )

    async def async_check_result(
        self,
        id: str,
    ) -> Dict[str, Any]:
        """Check for the result of a batch request with the appropriate batch id."""
        async with aiohttp.ClientSession() as client:
            resp = await make_async_web_request(
                client,
                self.api_key,
                self.api_prefix + f"batch_completions/{id}/result",
                "get",
            )
        return resp

    def check_result(
        self,
        id: str,
    ) -> Dict[str, Any]:
        """Check for the result of a batch request with the appropriate batch id."""
        resp = make_web_request(
            self.api_key,
            self.api_prefix + f"batch_completions/{id}/result",
            "get",
        )
        return resp

    def make_llm_req_map(
        self,
        model_name: str,
        prompt: Union[str, List[str]],
        output_type: Optional[dict] = None,
        max_tokens: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:

        req_data = {}
        req_data["model_name"] = model_name
        req_data["prompt"] = prompt
        req_data["output_type"] = output_type
        req_data["max_tokens"] = max_tokens
        if max_new_tokens is not None:
            req_data["max_new_tokens"] = max_new_tokens
        return req_data
