from typing import Optional, Dict, Any
import json
import logging
import os

import lamini
from lamini.api.lamini_config import get_config, get_configured_key, get_configured_url

logger = logging.getLogger(__name__)


class BaseAsyncInferenceQueue:
    """ 
    Parent class to handle the basic functionality of an InferenceQueue. Such
    functions are local cache loading and retrieval of lamini work and batch
    properties.
            
    Parameters
    ----------        
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
        
    config: Optional[Dict[str, Any]]
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

    def __init__(
            self, 
            api_key: Optional[str], 
            api_url: Optional[str], 
            config: Optional[Dict[str, Any]]
    ) -> None:
        self.config = get_config(config)
        self.api_key = api_key or lamini.api_key or get_configured_key(self.config)
        self.api_url = api_url or lamini.api_url or get_configured_url(self.config)
        self.api_prefix = self.api_url + "/v1/"
        self.reservation_api = None
        self.reservation_polling_task = None

    def read_local_cache(self, local_cache_file: str) -> Dict[str, Any]:
        """ Read in the local cache and format as a dictionary. An empty
        dictionary is returned if the cache is empty. Exceptions are raised
        for incorrect formatting and failed loading of the cache into a 
        dictionary.

        Parameters
        ----------
        local_cache_file: str
            Path of local cache file
        
        Raises
        ------
            Exception: 
                Last char of the local cache file is not ','
            Exception:
                Data structure returned from json.loads is not a dictionary
            
        Returns
        -------
        cache: Dict[str, Any]
            Cache dictionary, can be empty is the cache is empty
        """

        if not os.path.exists(local_cache_file):
            return {}

        with open(local_cache_file, "r") as file:
            content = file.read()

        content = content.strip()
        if content.strip() == "":
            return {}

        if content[-1] != ",":
            raise Exception(f"The last char in {local_cache_file} should be ','")

        content = "{" + content[:-1] + "}"
        cache = json.loads(content)

        if not isinstance(cache, dict):
            raise Exception(f"{local_cache_file} cannot be loaded as dict")

        return cache

    def get_max_workers(self) -> int:
        """ Return the Lamini API max number of workers

        Parameters
        ----------
        None
            
        Returns
        -------
        int
            lamini.max_workers
        """
        return lamini.max_workers

    def get_batch_size(self) -> int:
        """ Return the Lamini API batch size

        Parameters
        ----------
        None
            
        Returns
        -------
        int
            lamini.batch_size
        """

        return int(lamini.batch_size)

    def __del__(self) -> None:
        """ Handle cancelling reservation_polling_task if one is present
        upon deletion of this object.

        Parameters
        ----------
        None
            
        Returns
        -------
        None
        """

        if self.reservation_polling_task is not None:
            self.reservation_polling_task.cancel()
