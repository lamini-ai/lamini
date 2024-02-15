import json
import logging
import os

import lamini
from lamini.api.lamini_config import get_config, get_configured_key, get_configured_url
from lamini.api.utils.reservations import create_reservation_api

logger = logging.getLogger(__name__)


class BaseAsyncInferenceQueue:
    def __init__(self, api_key, api_url, config):
        self.config = get_config(config)
        self.api_key = api_key or lamini.api_key or get_configured_key(self.config)
        self.api_url = api_url or lamini.api_url or get_configured_url(self.config)
        self.api_prefix = self.api_url + "/v1/"
        self.reservation_api = None
        self.reservation_polling_task = None

    def read_local_cache(self, local_cache_file):
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

    def get_max_workers(self):
        return lamini.max_workers

    def get_batch_size(self):
        return lamini.batch_size

    def __del__(self):
        if self.reservation_polling_task is not None:
            self.reservation_polling_task.cancel()
