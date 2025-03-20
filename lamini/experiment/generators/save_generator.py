import pandas as pd

from lamini.experiment.utils import remove_non_ascii
from lamini.experiment.generators import BaseGenerator
from lamini.generation.base_prompt_object import PromptObject

class SaveGenerator(BaseGenerator):
    def __init__(self, save_path: str, save_keys: list[str] = None):
        self.save_path = save_path
        self.save_keys = save_keys
        self.name = "SaveGenerator"

    def __call__(self, prompt_obj: PromptObject):
        #print(prompt_obj)
        if self.save_keys:
            data = {
                key_: remove_non_ascii(prompt_obj.data[key_])
                for key_ in self.save_keys
            }
        else:
            data = {
                key_: remove_non_ascii(prompt_obj.data[key_])
                for key_ in prompt_obj.data  
            }
        #print(data)
        pd.DataFrame(
            [data]
        ).to_json(
            self.save_path,
            mode="a",
            orient="records",
            lines=True,
        )

        return prompt_obj