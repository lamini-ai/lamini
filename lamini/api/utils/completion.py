from typing import Any, Dict, List, Optional, Union

import aiohttp
import lamini
from lamini.api.lamini_config import get_config, get_configured_key, get_configured_url
from lamini.api.rest_requests import make_async_web_request, make_web_request


class Completion:
    """Handler for formatting and POST request for the completions
    and streaming_completions API endpoints.


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

    def __init__(self, api_key, api_url) -> None:
        """
        Configuration dictionary for platform metadata provided by the following function:
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
                        <auth-key>
        """
        self.config = get_config()

        self.api_key = api_key or lamini.api_key or get_configured_key(self.config)
        self.api_url = api_url or lamini.api_url or get_configured_url(self.config)
        self.api_prefix = self.api_url + "/v1/"

    def generate(
        self,
        prompt: Union[str, List[str]],
        model_name: str,
        output_type: Optional[dict] = None,
        max_tokens: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Handles construction of the POST request headers and body, then
        a web request is made with the response returned.

        Parameters
        ----------
        prompt: Union[str, List[str]]:
            Input prompt for the LLM

        model_name: str
            LLM model name from HuggingFace

        output_type: Optional[dict] = None
            Json format for the LLM output

        max_tokens: Optional[int] = None
            Upper limit in total tokens

        max_new_tokens: Optional[int] = None
            Upper limit for newly generated tokens

        Returns
        -------
        resp: Dict[str, Any]
            Json data returned from POST request
        """

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

    async def async_generate(
        self, params: Dict[str, Any], client: aiohttp.ClientSession = None
    ) -> Dict[str, Any]:
        """

        Parameters
        ----------
        params: Dict[str, Any]
            POST Request input parameters

        client: aiohttp.ClientSession = None
            ClientSession handler

        Returns
        -------
        resp: Dict[str, Any]
            Json data returned from POST request
        """

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
        cache_id: Optional[str] = None,
        output_type: Optional[dict] = None,
        max_tokens: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Returns a dict of parameters for calling the remote LLM inference API.

        NOTE: Copied from lamini.py.

        TODO: Create a helper function that accepts all values and returns a dict. And replace callers
        of self.make_llm_req_map() with the calling of the free function.

        Parameters
        ----------
        model_name: str
            LLM model name from HuggingFace

        prompt: Union[str, List[str]]:
            Input prompt for the LLM

        output_type: Optional[dict] = None
            Json format for the LLM output

        max_tokens: Optional[int] = None
            Upper limit in total tokens

        max_new_tokens: Optional[int] = None
            Upper limit for newly generated tokens

        Returns
        -------
        req_data: Dict[str, Any]
            Constructed dictionary with parameters provided into the correctly
            specified keys for a REST request.
        """

        req_data = {}
        req_data["model_name"] = model_name
        # TODO: prompt should be named prompt to signal it's a batch.
        if isinstance(prompt, list) and len(prompt) > 20:
            print(
                "For large inference batches, we strongly recommend using a Generation Pipeline to streamline your process: https://github.com/lamini-ai/lamini-examples/blob/main/05_data_pipeline/"
            )
        req_data["prompt"] = prompt
        req_data["output_type"] = output_type
        req_data["max_tokens"] = max_tokens
        if max_new_tokens is not None:
            req_data["max_new_tokens"] = max_new_tokens
        if cache_id is not None:
            req_data["cache_id"] = cache_id
        return req_data
