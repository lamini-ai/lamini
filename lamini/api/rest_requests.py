import requests

from lamini.error.error import (
    APIError,
    AuthenticationError,
    ModelNameError,
    RateLimitError,
    UnavailableResourceError,
    UserError,
)


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
