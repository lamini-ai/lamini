from typing import List, Union
import requests
import os
from llama.program.util.config import get_config, edit_config
import llama
import lamini
import numpy as np
from llama.error.error import (
    APIError,
    AuthenticationError,
    ModelNameError,
    RateLimitError,
    UnavailableResourceError,
    ServerTimeoutError,
    UserError,
)


def query_run_embedding(prompt: Union[str, List[str]], config={}):
    params = {"prompt": prompt}
    edit_config(config)
    url = get_configured_url()
    resp = make_web_request("post", url + "/v1/inference/embedding", None, params)
    embeddings = resp["embedding"]

    if isinstance(prompt, str):
        return np.reshape(embeddings, (1, -1))
    return [np.reshape(embedding, (1, -1)) for embedding in embeddings]


def make_web_request(http_method, url, api_key, json):
    configured_key = get_configured_key()

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key or lamini.api_key or configured_key}",
    }
    if http_method == "post":
        resp = requests.post(url=url, headers=headers, json=json)
    elif http_method == "get":
        resp = requests.get(url=url, headers=headers)
    else:
        raise Exception("http_method must be 'post' or 'get'")
    try:
        resp.raise_for_status()
    except requests.exceptions.HTTPError as e:
        if resp.status_code == 404:
            try:
                json_response = resp.json()
            except Exception:
                json_response = {}
            raise ModelNameError(json_response.get("detail", "ModelNameError"))
        if resp.status_code == 429:
            try:
                json_response = resp.json()
            except Exception:
                json_response = {}
            raise RateLimitError(json_response.get("detail", "RateLimitError"))
        if resp.status_code == 401:
            try:
                json_response = resp.json()
            except Exception:
                json_response = {}
            raise AuthenticationError(lamini.MISSING_API_KEY_MESSAGE)
        if resp.status_code == 400:
            try:
                json_response = resp.json()
            except Exception:
                json_response = {}
            raise UserError(json_response.get("detail", "UserError"))
        if resp.status_code == 503:
            try:
                json_response = resp.json()
            except Exception:
                json_response = {}
            raise UnavailableResourceError(
                json_response.get("detail", "UnavailableResourceError")
            )
        if resp.status_code == 504:
            try:
                json_response = resp.json()
            except Exception:
                json_response = {}
            raise ServerTimeoutError(json_response.get("detail", "ServerTimeoutError"))
        if resp.status_code != 200:
            try:
                description = resp.json()
            except BaseException:
                description = resp.status_code
            finally:
                if description == {"detail": ""}:
                    raise APIError("500 Internal Server Error")
                raise APIError(f"API error {description}")

    return resp.json()


def get_configured_url():
    cfg = get_config()
    environment = os.environ.get("LLAMA_ENVIRONMENT")
    if environment == "LOCAL":
        url = cfg.get("local.url", "http://localhost:5001")
    elif environment == "STAGING":
        url = cfg.get("staging.url", "https://api.staging.powerml.co")
    else:
        url = cfg.get("production.url", "https://api.lamini.ai")
    return url


def get_configured_key():
    cfg = get_config()
    environment = os.environ.get("LLAMA_ENVIRONMENT")
    if environment == "LOCAL":
        key = cfg.get("local.key", None)
    elif environment == "STAGING":
        key = cfg.get("staging.key", None)
    else:
        key = cfg.get("production.key", None)
    return key


def get_model_config():
    cfg = get_config()
    return cfg.get("model_config", None)


def get_ui_url():
    cfg = get_config()
    environment = os.environ.get("LLAMA_ENVIRONMENT")
    if environment == "LOCAL":
        url = cfg.get("local.url", "http://localhost:5001")
    elif environment == "STAGING":
        url = cfg.get("staging.url", "https://staging.powerml.co")
    else:
        if cfg.get("production.key", "") == "test_token":
            url = cfg.get("production.url", "http://localhost:5001")
        else:
            url = "https://app.lamini.ai"
    return url
