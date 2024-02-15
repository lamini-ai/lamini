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

    def generate(self, params):
        resp = make_web_request(
            self.api_key, self.api_prefix + "completions", "post", params
        )
        return resp

    async def async_generate(self, params):
        async with aiohttp.ClientSession() as client:
            resp = await make_async_web_request(
                client, self.api_key, self.api_prefix + "completions", "post", params
            )
            return resp
