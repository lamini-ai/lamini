import asyncio

import aiohttp
import lamini
import requests
from lamini.error.error import (
    APIError,
    APIUnprocessableContentError,
    AuthenticationError,
    ModelNotFound,
    RateLimitError,
    UnavailableResourceError,
    UserError,
)


def retry_once(func):
    async def wrapped(*args, **kwargs):
        try:
            result = await func(*args, **kwargs)
        except Exception as e:
            if lamini.retry:
                result = await func(*args, **kwargs)
            else:
                raise e
        return result

    return wrapped


@retry_once
async def make_async_web_request(client, key, url, http_method, json=None):
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + key,
    }
    assert http_method == "post" or http_method == "get"

    try:
        if http_method == "post":
            async with client.post(
                url,
                headers=headers,
                json=json,
            ) as resp:
                if resp.status == 200:
                    json_response = await resp.json()
                else:
                    await handle_error(resp)
        elif http_method == "get":
            async with client.get(url, headers=headers) as resp:
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
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + key,
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
        print("status code:", resp.status_code)
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
