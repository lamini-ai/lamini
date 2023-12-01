from typing import List, Optional

from llama.program.util.type_to_dict import value_to_dict
from llama.engine.lamini import Lamini
from llama.types.type import Type


class TypedLamini(Lamini):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        result = super().__call__(*args, **kwargs)
        if isinstance(result, list):
            if "output_type" in kwargs:
                return [kwargs["output_type"].parse_obj(r) for r in result]
            elif len(args) >= 2:
                return [args[1].parse_obj(r) for r in result]
        else:
            if "output_type" in kwargs:
                return kwargs["output_type"].parse_obj(result)
            elif len(args) >= 2:
                return args[1].parse_obj(result)

    def train_async(self, data: Optional[List] = None, *args, **kwargs):
        if data is not None:
            data = [
                [value_to_dict(d[0])["data"], value_to_dict(d[1])["data"]] for d in data
            ]
        return super().train_async(data, *args, **kwargs)

    def train(self, data: Optional[List] = None, **kwargs):
        return super().train(data, **kwargs)

    def same_type(self, t1, t2):
        return type(t1) == type(t2)

    def is_correct_type(self, t):
        return isinstance(t, Type)

    def make_save_data_req_map(self, data):
        req_data = super().make_save_data_req_map(data)
        req_data["data"] = [d.dict() for d in req_data["data"]]
        return req_data

    def make_save_data_pairs_req_map(self, data):
        req_data = {}
        req_data["id"] = self.id
        req_data["data"] = []
        type_err_msg = "data must be in the form [[input, output], [input, output], ...]. Each element in the data array must have the same type"

        if type(data) != list:
            raise TypeError(type_err_msg)

        for d in data:
            if len(d) != 2:
                raise TypeError(type_err_msg)

            input_data = d[0]
            output_data = d[1]

            if (
                not isinstance(input_data, Type)
                or not isinstance(output_data, Type)
                or not self.same_type(input_data, data[0][0])
                or not self.same_type(output_data, data[0][1])
            ):
                raise TypeError(type_err_msg)

            req_data["data"].append([input_data.dict(), output_data.dict()])

        return req_data

    def make_llm_req_map(
        self,
        id,
        model_name,
        input_value,
        output_type,
        prompt_template,
        stop_tokens,
        enable_peft,
        random,
        max_tokens,
        streaming,
    ):
        new_input_value = {}
        if isinstance(input_value, list):
            input = [v.dict() for v in input_value]
            new_input_value = input
        else:
            new_input_value = value_to_dict(input_value)["data"]

        new_out_type = {}
        for key, val in output_type.schema()["properties"].items():
            if val["type"] == "number":  # pydantic says float is number
                new_out_type[key] = "float"
            else:
                new_out_type[key] = val["type"]

        return super().make_llm_req_map(
            id,
            model_name,
            new_input_value,
            new_out_type,
            prompt_template,
            stop_tokens,
            enable_peft,
            random,
            max_tokens,
            streaming,
        )
