import os
from typing import Dict, Any

import config

global_config = None


def get_global_config(overrides: Dict[str, Any] = {}) -> config.Configuration:
    """Getter for the global lamini config, update config if overrides are provided

    Parameters
    ----------
    overrides: Dict[str, Any]={}
        Dictionary contents to override within the global_config

    Raises
    ------
    Assertion Exception
        Thrown if global_config has not been set

    Returns
    -------
    overriden_config: config.Configuration
        Copy of the global config with provided overrides
    """

    global global_config
    assert (
        global_config is not None
    ), "global_config must be set before calling get_config"

    overriden_config = global_config.copy()
    overriden_config.update(overrides)

    return overriden_config


def setup_config(dictionary: Dict[str, Any] = {}) -> config.Configuration:
    """Initialize the global config with the provided dictionary

    Parameters
    ----------
    dictionary: Dict[str, Any]
        Key/values to wrap into a config

    Returns
    -------
    global_config: config.Configuration
        Newly initalized config
    """

    global global_config
    global_config = get_config(dictionary)
    return global_config


def get_config(dictionary: Dict[str, Any] = {}) -> config.Configuration:
    """Construct a Configuration from the provided dictionary, along with
    the environment, lamini, and powerml configurations.

    Parameters
    ----------
    dictionary: Dict[str, Any]
        Key/values to wrap into a config

    Returns
    -------
    new_config: config.Configuration
        Newly construction config
    """

    new_config = config.ConfigurationSet(
        config.config_from_dict(dictionary),
        config.config_from_env(prefix="LAMINI", separator="__", lowercase_keys=True),
        home_lamini_config(),
        home_powerml_config(),
    )

    return new_config


def reset_config() -> None:
    """Reset the global config to None

    Parameters
    ----------
    None

    Returns
    -------
    None
    """

    global global_config
    global_config = None


def edit_config(dictionary: Dict[str, Any] = {}) -> config.Configuration:
    """Update the global_config with the provided dictionary

    Parameters
    ----------
    dictionary: Dict[str, Any]
        Key/values to update into global_config

    Returns
    -------
    global_config: config.Configuration
        Updated global_config
    """

    global global_config
    if global_config is None:
        global_config = setup_config(dictionary)
    else:
        global_config.update(config.config_from_dict(dictionary))
    return global_config


def home_lamini_config() -> config.Configuration:
    """Gather the local lamini configuration and wrap into a config

    Parameters
    ----------
    None

    Returns
    -------
    yaml_config: config.Configuration
        Home config key/values inside a config
    """

    home = os.path.expanduser("~")
    home_config_path = os.path.join(home, ".lamini/configure.yaml")
    if os.path.exists(home_config_path):
        yaml_config = config.config_from_yaml(home_config_path, read_from_file=True)
    else:
        yaml_config = config.config_from_dict({})
    return yaml_config


def home_powerml_config() -> config.Configuration:
    """Gather the local powerml configuration and wrap into a config

    Parameters
    ----------
    None

    Returns
    -------
    yaml_config: config.Configuration
        Home config key/values inside a config
    """

    home = os.path.expanduser("~")
    home_config_path = os.path.join(home, ".powerml/configure_llama.yaml")
    if os.path.exists(home_config_path):
        yaml_config = config.config_from_yaml(home_config_path, read_from_file=True)
    else:
        yaml_config = config.config_from_dict({})
    return yaml_config


def get_configured_url(config: config.Configuration) -> str:
    """Extract the Lamini platform url from the config

    Parameters
    ----------
    config: config.Configuration
        Config storing the url

    Returns
    -------
    url: str
        Extracted platform url
    """

    LAMINI_ENV_ENV_VAR_NAME = "LLAMA_ENVIRONMENT"
    LOCAL_URL_KEY = "local.url"
    LOCAL_URL_DEFAULT_VALUE = "http://localhost:5001"
    STAGING_URL_KEY = "staging.url"
    STAGING_URL_DEFAULT_VALUE = "https://staging.lamini.ai"
    PUBLIC_URL_DEFAULT_VALUE = "https://api.lamini.ai"
    PUBLIC_URL_KEY = "production.url"

    environment = os.environ.get(LAMINI_ENV_ENV_VAR_NAME)
    if environment == "LOCAL":
        url = config.get(LOCAL_URL_KEY, LOCAL_URL_DEFAULT_VALUE)
    elif environment == "STAGING":
        url = config.get(STAGING_URL_KEY, STAGING_URL_DEFAULT_VALUE)
    else:
        url = config.get(PUBLIC_URL_KEY, PUBLIC_URL_DEFAULT_VALUE)
    return url


def get_configured_key(config: config.Configuration) -> str:
    """Extract the Lamini platform key from the config

    Parameters
    ----------
    config: config.Configuration
        Config storing the key

    Returns
    -------
    key: str
        Extracted platform key
    """

    environment = os.environ.get("LLAMA_ENVIRONMENT")
    if environment == "LOCAL":
        key = config.get("local.key", None)
    elif environment == "STAGING":
        key = config.get("staging.key", None)
    else:
        key = config.get("production.key", None)
    return key
