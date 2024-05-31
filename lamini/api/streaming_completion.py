import asyncio
import time
from typing import List, Optional, Union

import aiohttp
import lamini
from lamini.api.lamini_config import get_config, get_configured_key, get_configured_url
from lamini.api.rest_requests import make_async_web_request, make_web_request


class StreamingCompletionObject:
    def __init__(
        self, request_params, api_url, api_key, polling_interval, max_errors=0
    ):
        self.request_params = request_params
        self.api_url = api_url
        self.api_key = api_key
        self.done_streaming = False
        self.server = None
        self.polling_interval = polling_interval
        self.current_result = None
        self.error_count = 0
        self.max_errors = max_errors

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if self.done_streaming:
            raise StopIteration()
        time.sleep(self.polling_interval)
        if self.server is not None:
            self.request_params["server"] = self.server
        try:
            resp = make_web_request(
                self.api_key,
                self.api_url,
                "post",
                self.request_params,
            )

            self.server = resp["server"]
            if resp["status"][0]:
                self.done_streaming = True
            self.current_result = resp["data"][0]
        except Exception as e:
            self.error_count += 1
            if self.error_count > self.max_errors:
                raise e
        return self.current_result


class AsyncStreamingCompletionObject:
    def __init__(
        self, request_params, api_url, api_key, polling_interval, max_errors=5
    ):
        self.request_params = request_params
        self.api_url = api_url
        self.api_key = api_key
        self.done_streaming = False
        self.server = None
        self.polling_interval = polling_interval
        self.current_result = None
        self.error_count = 0
        self.max_errors = max_errors

    def __aiter__(self):
        return self

    async def __anext__(self):
        return await self.next()

    async def next(self):
        if self.done_streaming:
            raise StopAsyncIteration()
        await asyncio.sleep(self.polling_interval)
        if self.server is not None:
            self.request_params["server"] = self.server
        try:
            async with aiohttp.ClientSession() as client:
                resp = await make_async_web_request(
                    client,
                    self.api_key,
                    self.api_url,
                    "post",
                    self.request_params,
                )
            self.server = resp["server"]
            if resp["status"][0]:
                self.done_streaming = True
            self.current_result = resp["data"][0]
        except Exception as e:
            self.error_count += 1
            if self.error_count > self.max_errors:
                raise e
        return self.current_result


class StreamingCompletion:
    def __init__(
        self,
        api_key: str = None,
        api_url: str = None,
        config={},
    ):
        self.config = get_config(config)
        self.api_key = api_key or lamini.api_key or get_configured_key(self.config)
        self.api_url = api_url or lamini.api_url or get_configured_url(self.config)
        self.api_prefix = self.api_url + "/v1/"
        self.streaming_completions_url = self.api_prefix + "streaming_completions"

    def submit(
        self,
        prompt: Union[str, List[str]],
        model_name: Optional[str] = None,
        output_type: Optional[dict] = None,
        max_tokens: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
    ):
        req_data = self.make_llm_req_map(
            prompt=prompt,
            model_name=model_name,
            output_type=output_type,
            max_tokens=max_tokens,
            max_new_tokens=max_new_tokens,
            server=None,
        )
        resp = make_web_request(
            self.api_key, self.streaming_completions_url, "post", req_data
        )
        return {
            "url": self.streaming_completions_url,
            "params": {**req_data, "server": resp["server"]},
        }

    async def async_submit(
        self,
        prompt: Union[str, List[str]],
        model_name: Optional[str] = None,
        output_type: Optional[dict] = None,
        max_tokens: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
    ):
        req_data = self.make_llm_req_map(
            prompt=prompt,
            model_name=model_name,
            output_type=output_type,
            max_tokens=max_tokens,
            max_new_tokens=max_new_tokens,
            server=None,
        )
        async with aiohttp.ClientSession() as client:
            resp = await make_async_web_request(
                client, self.api_key, self.streaming_completions_url, "post", req_data
            )
        return {
            "url": self.streaming_completions_url,
            "params": {**req_data, "server": resp["server"]},
        }

    def create(
        self,
        prompt: Union[str, List[str]],
        model_name: Optional[str] = None,
        output_type: Optional[dict] = None,
        max_tokens: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
        polling_interval: Optional[float] = 1,
    ):
        self.done_streaming = False
        self.server = None
        self.prompt = prompt
        self.model_name = model_name
        self.output_type = output_type
        self.max_tokens = max_tokens
        self.max_new_tokens = max_new_tokens
        req_data = self.make_llm_req_map(
            prompt=prompt,
            model_name=model_name,
            output_type=output_type,
            max_tokens=max_tokens,
            max_new_tokens=max_new_tokens,
            server=None,
        )
        return StreamingCompletionObject(
            req_data,
            api_key=self.api_key,
            api_url=self.streaming_completions_url,
            polling_interval=polling_interval,
        )

    def async_create(
        self,
        prompt: Union[str, List[str]],
        model_name: Optional[str] = None,
        output_type: Optional[dict] = None,
        max_tokens: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
        polling_interval: Optional[float] = 1,
    ):
        self.done_streaming = False
        self.server = None
        self.prompt = prompt
        self.model_name = model_name
        self.output_type = output_type
        self.max_tokens = max_tokens
        self.max_new_tokens = max_new_tokens
        req_data = self.make_llm_req_map(
            prompt=prompt,
            model_name=model_name,
            output_type=output_type,
            max_tokens=max_tokens,
            max_new_tokens=max_new_tokens,
            server=None,
        )
        return AsyncStreamingCompletionObject(
            req_data,
            api_key=self.api_key,
            api_url=self.streaming_completions_url,
            polling_interval=polling_interval,
        )

    def make_llm_req_map(
        self, model_name, prompt, output_type, max_tokens, max_new_tokens, server
    ):
        req_data = {}
        req_data["model_name"] = model_name
        req_data["prompt"] = prompt
        req_data["output_type"] = output_type
        req_data["max_tokens"] = max_tokens
        if max_new_tokens is not None:
            req_data["max_new_tokens"] = max_new_tokens
        if server is not None:
            req_data["server"] = server
        return req_data
