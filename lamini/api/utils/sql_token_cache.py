from typing import Any, Dict, List, Optional, Union

import aiohttp
import lamini
from lamini.api.lamini_config import get_config, get_configured_key, get_configured_url
from lamini.api.rest_requests import make_async_web_request, make_web_request


class SQLTokenCache:
    def __init__(self, api_key, api_url) -> None:
        self.config = get_config()

        self.api_key = api_key or lamini.api_key or get_configured_key(self.config)
        self.api_url = api_url or lamini.api_url or get_configured_url(self.config)
        self.api_prefix = self.api_url + "/v1alpha/"

    def add_token_cache(self, base_model_name, col_vals=None):
        req_data = self.make_req_map(
            base_model_name,
            col_vals=col_vals,
        )
        resp = make_web_request(
            self.api_key, self.api_prefix + "add_sql_token_cache", "post", req_data
        )

        return resp

    def delete_token_cache(self, cache_id):
        resp = make_web_request(
            self.api_key, self.api_prefix + "sql_token_cache/" + cache_id, "delete"
        )

        return resp

    def make_req_map(
        self,
        base_model_name: str,
        col_vals: Optional[dict] = None,
    ) -> Dict[str, Any]:
        req_data = {}
        req_data["base_model_name"] = base_model_name

        if col_vals is not None:
            req_data["col_vals"] = col_vals

        return req_data
    
