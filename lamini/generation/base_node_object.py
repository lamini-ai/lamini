import logging
from typing import AsyncIterator, Iterator, Optional, Union

from lamini.generation.base_prompt_object import PromptObject

logger = logging.getLogger(__name__)


class BaseGenerationNode:
    def __call__(self, *args, **kwargs):
        return self.generate(*args, **kwargs)

    def generate(
        self,
        prompt: Union[Iterator[PromptObject], AsyncIterator[PromptObject]],
        model_name: Optional[str] = None,
        output_type: Optional[dict] = None,
        max_tokens: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
    ):
        pass
