import re

import lamini
from lamini import __version__
from lamini.api.lamini_config import get_config, get_configured_key, get_configured_url
from lamini.api.rest_requests import make_web_request


class APIVersion:
    """Handler for embedding requests to the Lamini Platform


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
        self.api_prefix = self.api_url + "/v1/"
        self.server_version = None
        self.client_version = __version__
        self.api_endpoints = None
        self.features_to_api_versions = {}
        try:
            self.get_versions()
        except:
            pass

    def __path_to_version_number(self, path: str) -> int:
        # Regular expression to match the version number in the format /v{number}
        match = re.search(r"/v(\d+)/", path)

        # If a version is found, return it as an integer, otherwise return None
        return int(match.group(1)) if match else None

    def _populate_features_to_api_versions(self):
        for endpoint in self.api_endpoints:
            if endpoint["name"] not in self.features_to_api_versions:
                self.features_to_api_versions[endpoint["name"]] = set()
            version = self.__path_to_version_number(endpoint["path"])
            if version is not None:
                self.features_to_api_versions[endpoint["name"]].add(version)

    def get_versions(self) -> dict:
        """Request to Lamini platform for an embedding encoding of the provided
        prompt
        """
        resp = make_web_request(self.api_key, self.api_prefix + "version", "get")
        self.api_endpoints = resp["api"]
        self.server_version = resp["server"]
        self.client_version = resp["client"]

        self._populate_features_to_api_versions()
        return self.features_to_api_versions
