import asyncio
import logging

import aiohttp
import lamini
import requests
from lamini.api.lamini_config import get_config, get_configured_key, get_configured_url
from lamini.error.error import (
    APIError,
    APIUnprocessableContentError,
    AuthenticationError,
    ModelNotFound,
    RateLimitError,
    UnavailableResourceError,
    UserError,
)

logger = logging.getLogger(__name__)

warn_once = False


def get_version(key, url, config):
    api_key = key or get_configured_key(config)
    api_url = url or get_configured_url(config)
    return make_web_request(api_key, api_url + "/v1/version", "get", None)


def check_version(resp):
    global warn_once
    if not warn_once:
        if resp.headers is not None and "X-Warning" in resp.headers:
            warn_once = True
            print(resp.headers["X-Warning"])


async def make_async_web_request(client, key, url, http_method, json=None):
    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + key,
        }
    except:
        raise AuthenticationError("Missing API Key")
    try:
        version = importlib.metadata.version("lamini")
        headers["Lamini-Version"] = version
    except:
        pass
    assert http_method == "post" or http_method == "get"
    logger.debug(f"Making {http_method} request to {url} with payload {json}")
    try:
        if http_method == "post":
            async with client.post(
                url,
                headers=headers,
                json=json,
            ) as resp:
                check_version(resp)
                if resp.status == 200:
                    json_response = await resp.json()
                    logger.debug("api response: " + str(json_response))
                else:
                    await handle_error(resp)
        elif http_method == "get":
            async with client.get(url, headers=headers) as resp:
                check_version(resp)
                if resp.status == 200:
                    json_response = await resp.json()
                else:
                    handle_error(resp)
    except asyncio.TimeoutError:
        raise APIError(
            "Request Timeout: The server did not respond in time.",
        )

    return json_response


async def handle_error(resp: aiohttp.ClientResponse):
    if resp.status == 594:
        try:
            json_response = await resp.json()
        except Exception:
            json_response = {}
        raise ModelNotFound(json_response.get("detail", "ModelNotFound"))
    if resp.status == 429:
        try:
            json_response = await resp.json()
        except Exception:
            json_response = {}
        raise RateLimitError(json_response.get("detail", "RateLimitError"))
    if resp.status == 401:
        try:
            json_response = await resp.json()
        except Exception:
            json_response = {}
        raise AuthenticationError(json_response.get("detail", "AuthenticationError"))
    if resp.status == 400:
        try:
            json_response = await resp.json()
        except Exception:
            json_response = {}
        raise UserError(json_response.get("detail", "UserError"))
    if resp.status == 422:
        try:
            json_response = await resp.json()
        except Exception:
            json_response = {}
        raise APIUnprocessableContentError(
            "The API has returned a 422 Error. This typically happens when the python package is outdated. Please consider updating the lamini python package version with `pip install --upgrade --force-reinstall lamini`"
        )
    if resp.status == 503:
        try:
            json_response = await resp.json()
        except Exception:
            json_response = {}
        raise UnavailableResourceError(
            json_response.get("detail", "UnavailableResourceError")
        )
    if resp.status != 200:
        try:
            description = await resp.json()
        except BaseException:
            description = resp.status
        finally:
            if description == {"detail": ""}:
                raise APIError("500 Internal Server Error")
            raise APIError(f"API error {description}")


def make_web_request(key, url, http_method, json=None):
    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + key,
        }
    except:
        raise AuthenticationError("Missing API Key")
    try:
        version = importlib.metadata.version("lamini")
        headers["Lamini-Version"] = version
    except:
        pass
    if http_method == "post":
        resp = requests.post(url=url, headers=headers, json=json)
    elif http_method == "get":
        resp = requests.get(url=url, headers=headers)
    else:
        raise Exception("http_method must be 'post' or 'get'")
    try:
        check_version(resp)
        resp.raise_for_status()
    except requests.exceptions.HTTPError as e:
        print("status code:", resp.status_code, url)
        if resp.status_code == 594:
            try:
                json_response = resp.json()
            except Exception:
                json_response = {}
            raise ModelNotFound(json_response.get("detail", "ModelNameError"))
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
            raise AuthenticationError(
                json_response.get("detail", "AuthenticationError")
            )
        if resp.status_code == 400:
            try:
                json_response = resp.json()
            except Exception:
                json_response = {}
            raise UserError(json_response.get("detail", "UserError"))
        if resp.status_code == 422:
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
