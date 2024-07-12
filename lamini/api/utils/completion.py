from typing import List, Optional, Union

import aiohttp
import lamini
from lamini.api.lamini_config import get_config, get_configured_key, get_configured_url
from lamini.api.rest_requests import make_async_web_request, make_web_request


class Completion:
    def __init__(self, api_key, api_url, config):
        self.config = get_config(config)
        self.api_key = api_key or lamini.api_key or get_configured_key(self.config)
        self.api_url = api_url or lamini.api_url or get_configured_url(self.config)
        self.api_prefix = self.api_url + "/v1/"
        self.model_config = self.config.get("model_config", None)

    def generate(
        self,
        prompt: Union[str, List[str]],
        model_name: str,
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
        )
        resp = make_web_request(
            self.api_key, self.api_prefix + "completions", "post", req_data
        )
        return resp

    async def async_generate(self, params, client: aiohttp.ClientSession = None):
        if client is not None:
            assert isinstance(client, aiohttp.ClientSession)
            resp = await make_async_web_request(
                client,
                self.api_key,
                self.api_prefix + "streaming_completions",
                "post",
                params,
            )
            return resp

        async with aiohttp.ClientSession() as client:
            resp = await make_async_web_request(
                client, self.api_key, self.api_prefix + "completions", "post", params
            )
            return resp

    def make_llm_req_map(
        self,
        model_name: str,
        prompt: Union[str, List[str]],
        output_type: Optional[dict] = None,
        max_tokens: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
    ):
        """Returns a dict of parameters for calling the remote LLM inference API.

        NOTE: Copied from lamini.py.

        TODO: Create a helper function that accepts all values and returns a dict. And replace callers
        of self.make_llm_req_map() with the calling of the free function.
        """
        req_data = {}
        req_data["model_name"] = model_name
        # TODO: prompt should be named prompt to signal it's a batch.
        req_data["prompt"] = prompt
        req_data["output_type"] = output_type
        req_data["max_tokens"] = max_tokens
        if max_new_tokens is not None:
            req_data["max_new_tokens"] = max_new_tokens
        if self.model_config:
            req_data["model_config"] = self.model_config.as_dict()
        return req_data
