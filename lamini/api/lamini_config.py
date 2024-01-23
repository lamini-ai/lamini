import os

import config

global_config = None


def get_global_config(overrides={}):
    global global_config
    assert (
        global_config is not None
    ), "global_config must be set before calling get_config"
    return global_config


def setup_config(dictionary={}):
    global global_config
    global_config = get_config(dictionary)
    return global_config


def get_config(dictionary={}):
    new_config = config.ConfigurationSet(
        config.config_from_dict(dictionary),
        config.config_from_env(prefix="LAMINI", separator="__", lowercase_keys=True),
        home_lamini_config(),
        home_powerml_config(),
    )

    return new_config


def reset_config():
    global global_config
    global_config = None


def edit_config(dictionary={}):
    global global_config
    if global_config is None:
        global_config = setup_config(dictionary)
    else:
        global_config.update(config.config_from_dict(dictionary))
    return global_config


def home_lamini_config():
    home = os.path.expanduser("~")
    home_config_path = os.path.join(home, ".lamini/configure.yaml")
    if os.path.exists(home_config_path):
        yaml_config = config.config_from_yaml(home_config_path, read_from_file=True)
    else:
        yaml_config = config.config_from_dict({})
    return yaml_config


def home_powerml_config():
    home = os.path.expanduser("~")
    home_config_path = os.path.join(home, ".powerml/configure_llama.yaml")
    if os.path.exists(home_config_path):
        yaml_config = config.config_from_yaml(home_config_path, read_from_file=True)
    else:
        yaml_config = config.config_from_dict({})
    return yaml_config


def get_configured_url(config):
    environment = os.environ.get("LLAMA_ENVIRONMENT")
    if environment == "LOCAL":
        url = config.get("local.url", "http://localhost:5001")
    elif environment == "STAGING":
        url = config.get("staging.url", "https://api.staging.powerml.co")
    else:
        url = config.get("production.url", "https://api.lamini.ai")
    return url


def get_configured_key(config):
    environment = os.environ.get("LLAMA_ENVIRONMENT")
    if environment == "LOCAL":
        key = config.get("local.key", None)
    elif environment == "STAGING":
        key = config.get("staging.key", None)
    else:
        key = config.get("production.key", None)
    return key
