import config
import os

global_config = None


def get_config():
    global global_config
    assert global_config is not None
    return global_config


def setup_config(dictionary={}):
    global global_config
    global_config = config.ConfigurationSet(
        config.config_from_dict(dictionary),
        config.config_from_env(prefix="POWERML", separator="__", lowercase_keys=True),
        home_lamini_config(),
        home_powerml_config(),
    )
    return global_config


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
