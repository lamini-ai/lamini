from typing import Any, Dict, List, Optional, Union

import aiohttp
import lamini
from lamini.api.lamini_config import get_config, get_configured_key, get_configured_url
from lamini.api.rest_requests import make_async_web_request, make_web_request
from lamini.api.utils.completion import Completion

class SQLCompletion(Completion):
    def __init__(self, api_key, api_url) -> None:
        self.config = get_config()

        self.api_key = api_key or lamini.api_key or get_configured_key(self.config)
        self.api_url = api_url or lamini.api_url or get_configured_url(self.config)
        self.api_prefix = self.api_url + "/v1alpha/"

    def generate(
        self,
        prompt: Union[str, List[str]],
        cache_id: str,
        model_name: str,
        max_tokens: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        req_data = self.make_llm_req_map(
            prompt=prompt,
            cache_id=cache_id,
            model_name=model_name,
            max_tokens=max_tokens,
            max_new_tokens=max_new_tokens,
        )
        resp = make_web_request(
            self.api_key, self.api_prefix + "sql", "post", req_data
        )
        return resp

    async def async_generate(
        self, params: Dict[str, Any], client: aiohttp.ClientSession = None
    ) -> Dict[str, Any]:
        raise Exception("SQL streaming not implemented")
