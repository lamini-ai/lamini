import logging
from typing import AsyncIterator, Callable, Iterator, Optional, Union

from lamini.api.lamini_config import get_config
from lamini.generation.base_node_object import BaseGenerationNode
from lamini.generation.base_prompt_object import PromptObject

logger = logging.getLogger(__name__)


class ModifyNode(BaseGenerationNode):
    def __init__(self, prompt_lambda: Optional[Callable] = None):
        self.prompt_lambda = prompt_lambda

    def __call__(self, *args, **kwargs):
        return self.modify(*args, **kwargs)

    async def modify(
        self,
        prompt: Union[Iterator[PromptObject], AsyncIterator[PromptObject]],
        prompt_lambda: Optional[Callable] = None,
    ):
        async for a in prompt:
            if a is None or a.response is None:
                continue
            if self.prompt_lambda:
                self.prompt_lambda(a)
            if prompt_lambda:
                prompt_lambda(a)
            yield a
