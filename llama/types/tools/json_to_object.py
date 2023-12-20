from pydantic import create_model, ValidationError, validator


def json_to_object(json, type):
    type.__init__(**json)


def json_to_dynamic_object(json):
    UserModel = create_model(
        "UserModel",
    )
