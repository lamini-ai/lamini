import asyncio
import importlib.metadata
import logging
from typing import Any, Dict, Optional

import aiohttp
import requests
from lamini.api.lamini_config import get_configured_key, get_configured_url
from lamini.error.error import (
    APIError,
    APIUnprocessableContentError,
    AuthenticationError,
    DownloadingModelError,
    DuplicateResourceError,
    ModelNotFoundError,
    RateLimitError,
    RequestTimeoutError,
    UnavailableResourceError,
    ProjectNotFoundError,
    UserError,
)

logger = logging.getLogger(__name__)

warn_once = False


def check_version(resp: Dict[str, Any]) -> None:
    """If the flag of warn_once is not set then print the X-warning
    from the post request response and set the flag to true.

    Parameters
    ----------
    resp: Dict[str, Any]
        Request response dictionary

    Returns
    -------
    None
    """

    global warn_once
    if not warn_once:
        if resp.headers is not None and "X-Warning" in resp.headers:
            warn_once = True
            print(resp.headers["X-Warning"])


def get_version(
    key: Optional[str], url: Optional[str], config: Optional[Dict[str, Any]]
) -> str:
    """Getter for the Lamini Platform version
    Parameters
    ----------
    key: Optional[str]
        Lamini platform API key, if not provided the key stored
        within ~.lamini/configure.yaml will be used. If either
        don't exist then an error is raised.
    url: Optional[str]
        Lamini platform api url, only needed if a different url is needed outside of the
        defined ones here: https://github.com/lamini-ai/lamini-platform/blob/main/sdk/lamini/api/lamini_config.py#L68
            i.e. localhost, staging.lamini.ai, or api.lamini.ai
            Additionally, LLAMA_ENVIRONMENT can be set as an environment variable
            that will be grabbed for the url before any of the above defaults
    config: Dict[str, Any]
        Configuration storing the key and url
    Returns
    -------
    str
        Version of the Lamini Platform
    """
    api_key = key or get_configured_key(config)
    api_url = url or get_configured_url(config)
    return make_web_request(api_key, api_url + "/v1/version", "get", None)


async def make_async_web_request(
    client: requests.Session,
    key: str,
    url: str,
    http_method: str,
    json: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Send asycn request to the Lamini Platform

    Parameters
    ----------
    client: requests.Session
        Session handler for web requests

    key: Optional[str]
        Lamini platform API key, if not provided the key stored
        within ~.lamini/configure.yaml will be used. If either
        don't exist then an error is raised.

    url: Optional[str]
        Lamini platform api url, only needed if a different url is needed outside of the
        defined ones here: https://github.com/lamini-ai/lamini-platform/blob/main/sdk/lamini/api/lamini_config.py#L68
            i.e. localhost, staging.lamini.ai, or api.lamini.ai
            Additionally, LLAMA_ENVIRONMENT can be set as an environment variable
            that will be grabbed for the url before any of the above defaults

    http_method: str
        Request type

    json: Optional[Dict[str, Any]]=None
        Data to send with request

    Raises
    ------
    AuthenticationError
        Raised if key is not valid or missing

    AssertionError
        http_method is not post or get

    asyncio.TimeoutError
        Timeout from server

    Returns
    -------
    json_response: Dict[str, Any]
        Response from the web request
    """

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
                    await handle_error(resp)
    except asyncio.TimeoutError:
        raise APIError(
            "Request Timeout: The server did not respond in time.",
        )

    return json_response


async def handle_error(resp: aiohttp.ClientResponse) -> None:
    """Given the response from a requests.Session, provide the proper
    readable output for the user.

    Parameters
    ----------
    resp: aiohttp.ClientResponse
        Response from the web request

    Raises
    ------
    UserError
        Raises from 400

    AuthenticationError
        Raises from 401

    APIUnprocessableContentError
        Raises from 422

    RateLimitError
        Raises from 429

    DuplicateResourceError
        Raises from 497

    JobNotFoundError
        Raises from 498

    ProjectNotFoundError
        Raises from 499

    UnavailableResourceError
        Raises from 503

    ModelNotFoundError
        Raises from 594

    APIError
        Raises from 200

    Returns
    -------
    None

    """

    if resp.status == 594:
        try:
            json_response = await resp.json()
        except Exception:
            json_response = {}
        raise ModelNotFoundError(json_response.get("detail", "ModelNotFound"))
    if resp.status == 499:
        try:
            json_response = await resp.json()
        except Exception:
            json_response = {}
        raise ProjectNotFoundError(json_response.get("detail", "ProjectNotFoundError"))
    if resp.status == 497:
        try:
            json_response = await resp.json()
        except Exception:
            json_response = {}
        raise DuplicateResourceError(
            json_response.get("detail", "DuplicateResourceError")
        )
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


def make_web_request(
    key: str, url: str, http_method: str, json: Optional[Dict[str, Any]] = None, stream: bool = False
) -> Dict[str, Any]:
    """Execute a web request

    Parameters
    ----------
    key: Optional[str]
        Lamini platform API key, if not provided the key stored
        within ~.lamini/configure.yaml will be used. If either
        don't exist then an error is raised.

    url: Optional[str]
        Lamini platform api url, only needed if a different url is needed outside of the
        defined ones here: https://github.com/lamini-ai/lamini-platform/blob/main/sdk/lamini/api/lamini_config.py#L68
            i.e. localhost, staging.lamini.ai, or api.lamini.ai
            Additionally, LLAMA_ENVIRONMENT can be set as an environment variable
            that will be grabbed for the url before any of the above defaults

    http_method: str
        Request type

    json: Optional[Dict[str, Any]]=None
        Data to send with request

    Raises
    ------
    AuthenticationError
        Raised from invalid or missing api key

    Exception
        http_method requested is not post or get

    HTTPError
        Raised from many possible reasons:
            if resp.status_code == 594:
                ModelNotFoundError
            if resp.status_code == 429:
                RateLimitError
            if resp.status_code == 401:
                AuthenticationError
            if resp.status_code == 400:
                UserError
            if resp.status_code == 422:
                UserError
            if resp.status_code == 503:
                UnavailableResourceError
            if resp.status_code == 513:
                DownloadingModelError
            if resp.status_code == 524:
                RequestTimeoutError
            if resp.status_code != 200:
                APIError

    Returns
    -------
    Dic[str, Any]
        Response from the request
    """

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
    elif http_method == "get" and stream:
        resp = requests.get(url=url, headers=headers, stream=True)
    elif http_method == "get":
        resp = requests.get(url=url, headers=headers)
    elif http_method == "delete":
        resp = requests.delete(url=url, headers=headers)
    else:
        raise Exception("http_method must be 'post' or 'get' or 'delete'")
    try:
        check_version(resp)
        resp.raise_for_status()
    except requests.exceptions.HTTPError as e:
        if resp.status_code == 594:
            try:
                json_response = resp.json()
            except Exception:
                json_response = {}
            raise ModelNotFoundError(json_response.get("detail", "ModelNameError"))
        if resp.status_code == 499:
            try:
                json_response = resp.json()
            except Exception:
                json_response = {}
            raise ProjectNotFoundError(
                json_response.get("detail", "ProjectNotFoundError")
            )
        if resp.status_code == 497:
            try:
                json_response = resp.json()
            except Exception:
                json_response = {}
            raise DuplicateResourceError(
                json_response.get("detail", "DuplicateResourceError")
            )
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
        if resp.status_code == 513:
            message = ""
            try:
                json_response = resp.json()
                message = json_response.get("detail")
                message = message.split("Downloading", 1)[1].join(["Downloding", ""])
            except Exception:
                json_response = {}
            raise DownloadingModelError(message)
        if resp.status_code == 524:
            try:
                json_response = resp.json()
            except Exception:
                json_response = {}
            raise RequestTimeoutError(
                json_response.get("detail", "RequestTimeoutError")
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

    if stream:
        return resp
    else:
        return resp.json()
