from llama.program.value import Value
from llama.types.base_specification import BaseSpecification

import json


def type_to_dict(type):
    if issubclass(type, BaseSpecification):
        return json.loads(type.schema_json())

    return str(type)


def value_to_dict(input_value):
    if isinstance(input_value, Value):
        # type Value is e.g. a return value of calling llm()
        return {
            "index": input_value._index,
            "type": type_to_dict(input_value._type),
        }

    return {
        "data": rewrite_dict_data(input_value),
        "type": rewrite_dict_type(input_value),
    }


def rewrite_dict_data(input_value):
    if isinstance(input_value, Value):
        raise ValueError("Value should not be passed to rewrite_dict_data")
    elif isinstance(input_value, BaseSpecification):
        input_value = input_value.dict()
        for key, value in input_value.items():
            input_value[key] = rewrite_dict_data(value)

    return input_value


def rewrite_dict_type(input_value):
    assert isinstance(input_value, BaseSpecification)
    return json.loads(type(input_value).schema_json())


def data_to_dict(data):
    if data is None:
        return None
    examples = []
    for example in data:
        if isinstance(example, list):
            # Related information in a list together
            examples_i = []
            for example_i in example:
                examples_i.append(value_to_dict(example_i))
            examples.append(examples_i)
        else:
            # Singleton type
            examples.append(value_to_dict(example))
    return examples
