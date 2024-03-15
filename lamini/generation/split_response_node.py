import logging
from typing import AsyncIterator, Callable, Iterator, Optional, Union

from lamini.api.lamini_config import get_config
from lamini.generation.base_node_object import BaseGenerationNode
from lamini.generation.base_prompt_object import PromptObject

logger = logging.getLogger(__name__)


class SplitResponseNode(BaseGenerationNode):
    def __init__(self, prompt_lambda: Optional[Callable] = None):
        self.prompt_lambda = prompt_lambda

    def __call__(self, *args, **kwargs):
        return self.split(*args, **kwargs)

    async def split(
        self,
        prompt: Union[Iterator[PromptObject], AsyncIterator[PromptObject]],
    ):
        async for a in prompt:
            if a is None or a.response is None:
                continue
            new_prompt_objs = self.split_response(a)
            for new_prompt in new_prompt_objs:
                yield new_prompt

    def split_response(self, prompt_obj: PromptObject):
        if isinstance(prompt_obj.response, dict):
            for key, val in prompt_obj.response.items():
                new_prompt_obj = PromptObject("", val, prompt_obj.data)
                if self.prompt_lambda:
                    self.prompt_lambda(new_prompt_obj)
                yield new_prompt_obj
        else:
            if self.prompt_lambda:
                self.prompt_lambda(prompt_obj)
            return prompt_obj
