import asyncio
import time
from typing import Any, Dict, List, Optional, Union

import aiohttp
import lamini
from lamini.api.lamini_config import get_config, get_configured_key, get_configured_url
from lamini.api.rest_requests import make_async_web_request, make_web_request


class StreamingCompletionObject:
    """Handler for streaming API endpoint on the Lamini Platform

    Parameters
    ----------
    api_key: Optional[str]
        Lamini platform API key, if not provided the key stored
        within ~.lamini/configure.yaml will be used. If either
        don't exist then an error is raised.

    api_url: Optional[str]
        Lamini platform api url, only needed if a different url is needed outside of the
        defined ones here: https://github.com/lamini-ai/lamini-platform/blob/main/sdk/lamini/api/lamini_config.py#L68
            i.e. localhost, staging.lamini.ai, or api.lamini.ai
            Additionally, LLAMA_ENVIRONMENT can be set as an environment variable
            that will be grabbed for the url before any of the above defaults

    polling_interval: int
        Interval to wait before polling again

    max_errors: int = 0
        Number of errors before raising an exception
    """

    def __init__(
        self,
        api_url: str,
        api_key: str,
        id: str,
        polling_interval: int,
        max_errors: int = 0,
    ):
        self.api_url = api_url + f"/{id}/result"
        self.api_key = api_key
        self.done_streaming = False
        self.polling_interval = polling_interval
        self.current_result = None
        self.error_count = 0
        self.max_errors = max_errors

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

    def next(self) -> str:
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
        time.sleep(self.polling_interval)
        try:
            resp = make_web_request(
                self.api_key,
                self.api_url,
                "get",
            )
            if len(resp) == 0:
                self.current_result = None
                return self.current_result

            if all(r is not None for r in resp["finish_reason"]):
                self.done_streaming = True
            self.current_result = resp
        except Exception as e:
            self.error_count += 1
            if self.error_count > self.max_errors:
                raise e
        return self.current_result


class AsyncStreamingCompletionObject:
    """Handler for asynchronous streaming API endpoint on the Lamini Platform

    Parameters
    ----------
    api_key: Optional[str]
        Lamini platform API key, if not provided the key stored
        within ~.lamini/configure.yaml will be used. If either
        don't exist then an error is raised.

    api_url: Optional[str]
        Lamini platform api url, only needed if a different url is needed outside of the
        defined ones here: https://github.com/lamini-ai/lamini-platform/blob/main/sdk/lamini/api/lamini_config.py#L68
            i.e. localhost, staging.lamini.ai, or api.lamini.ai
            Additionally, LLAMA_ENVIRONMENT can be set as an environment variable
            that will be grabbed for the url before any of the above defaults

    polling_interval: int
        Interval to wait before polling again

    max_errors: int = 5
        Number of errors before raising an exception
    """

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

    def __aiter__(self) -> object:
        """Asychronous iteration definition

        Parameters
        ----------
        None

        Returns
        -------
        Reference to this instance of AsyncStreamingCompletionObject
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
        await asyncio.sleep(self.polling_interval)
        try:
            async with aiohttp.ClientSession() as client:
                resp = await make_async_web_request(
                    client,
                    self.api_key,
                    self.api_url,
                    "get",
                )
            if len(resp) == 0:
                self.current_result = None
                return self.current_result
            if all(r is not None for r in resp["finish_reason"]):
                self.done_streaming = True
            self.current_result = resp
        except Exception as e:
            self.error_count += 1
            if self.error_count > self.max_errors:
                raise e
        return self.current_result


class StreamingCompletion:
    """Handler for streaming completions API endpoint on the Lamini Platform

    Parameters
    ----------
    api_key: Optional[str]
        Lamini platform API key, if not provided the key stored
        within ~.lamini/configure.yaml will be used. If either
        don't exist then an error is raised.

    api_url: Optional[str]
        Lamini platform api url, only needed if a different url is needed outside of the
        defined ones here: https://github.com/lamini-ai/lamini-platform/blob/main/sdk/lamini/api/lamini_config.py#L68
            i.e. localhost, staging.lamini.ai, or api.lamini.ai
            Additionally, LLAMA_ENVIRONMENT can be set as an environment variable
            that will be grabbed for the url before any of the above defaults
    """

    def __init__(
        self,
        api_key: str = None,
        api_url: str = None,
    ):
        self.config = get_config()
        self.api_key = api_key or lamini.api_key or get_configured_key(self.config)
        self.api_url = api_url or lamini.api_url or get_configured_url(self.config)
        self.api_prefix = self.api_url + "/v3/"
        self.streaming_completions_url = self.api_prefix + "streaming_completions"

    def submit(
        self,
        prompt: Union[str, List[str]],
        model_name: str,
        max_new_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Conduct a web request to the streaming completions api endpoint with the
        provided prompt to the model_name if provided.
        max_new_tokens are related to the total amount of tokens
        the model can use and generate.

        Parameters
        ----------
        prompt: Union[str, List[str]]
            Prompt to send to LLM

        model_name: Optional[str] = None
            Which model to use from hugging face

        max_new_tokens: Optional[int] = None
            Max number of new tokens from the model's generation

        Returns
        -------
        Dict[str, Any]
            Returned response from the web request
        """

        req_data = self.make_llm_req_map(
            prompt=prompt,
            model_name=model_name,
            max_new_tokens=max_new_tokens,
        )
        resp = make_web_request(
            self.api_key, self.streaming_completions_url, "post", req_data
        )
        return resp

    async def async_submit(
        self,
        prompt: Union[str, List[str]],
        model_name: str,
        max_new_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Asynchronously send a web request to the streaming completions api endpoint with the
        provided prompt to the model_name if provided.
        max_new_tokens are related to the total amount of tokens
        the model can use and generate.

        Parameters
        ----------
        prompt: Union[str, List[str]]
            Prompt to send to LLM

        model_name: Optional[str] = None
            Which model to use from hugging face

        max_new_tokens: Optional[int] = None
            Max number of new tokens from the model's generation

        Returns
        -------
        Dict[str, Any]
            Returned response from the web request
        """

        req_data = self.make_llm_req_map(
            prompt=prompt,
            model_name=model_name,
            max_new_tokens=max_new_tokens,
        )
        async with aiohttp.ClientSession() as client:
            resp = await make_async_web_request(
                client, self.api_key, self.streaming_completions_url, "post", req_data
            )
        return resp

    def create(
        self,
        prompt: Union[str, List[str]],
        model_name: str,
        max_new_tokens: Optional[int] = None,
        polling_interval: Optional[float] = 1,
    ) -> object:
        """Instantiate a new StreamingCompletionObject

        Parameters
        ----------
        prompt: Union[str, List[str]]
            Prompt to send to LLM

        model_name: Optional[str] = None
            Which model to use from hugging face

        max_new_tokens: Optional[int] = None
            Max number of new tokens from the model's generation

        polling_interval: Optional[float] = 1
            Interval to wait before polling again

        Returns
        -------
        StreamingCompletionObject
            Newly instantiated object
        """

        req_data = self.submit(
            prompt=prompt, model_name=model_name, max_new_tokens=max_new_tokens
        )
        return StreamingCompletionObject(
            api_key=self.api_key,
            api_url=self.streaming_completions_url,
            id=req_data["id"],
            polling_interval=polling_interval,
        )

    async def async_create(
        self,
        prompt: Union[str, List[str]],
        model_name: str,
        max_new_tokens: Optional[int] = None,
        polling_interval: Optional[float] = 1,
    ) -> object:
        """Instantiate a new AsyncStreamingCompletionObject

        Parameters
        ----------
        prompt: Union[str, List[str]]
            Prompt to send to LLM

        model_name: Optional[str] = None
            Which model to use from hugging face

        max_new_tokens: Optional[int] = None
            Max number of new tokens from the model's generation

        polling_interval: Optional[float] = 1
            Interval to wait before polling again

        Returns
        -------
        AsyncStreamingCompletionObject
            Newly instantiated object
        """

        req_data = await self.async_submit(
            prompt=prompt, model_name=model_name, max_new_tokens=max_new_tokens
        )
        return AsyncStreamingCompletionObject(
            api_key=self.api_key,
            api_url=self.streaming_completions_url,
            polling_interval=polling_interval,
            id=req_data["id"],
        )

    def make_llm_req_map(
        self,
        model_name: Optional[str],
        prompt: Union[str, List[str]],
        max_new_tokens: Optional[int],
    ) -> Dict[str, Any]:
        """Make a web request to the Lamini Platform

        Parameters
        ----------
        model_name: Optional[str]
            Which model to use from hugging face

        prompt: Union[str, List[str]]
            Prompt to send to LLM


        max_new_tokens: Optional[int] = None
            Max number of new tokens from the model's generation

        Returns
        -------
        req_data: Dict[str, Any]
            Response from the web request
        """

        req_data = {}
        req_data["model_name"] = model_name
        req_data["prompt"] = prompt
        if max_new_tokens is not None:
            req_data["max_new_tokens"] = max_new_tokens
        return req_data
