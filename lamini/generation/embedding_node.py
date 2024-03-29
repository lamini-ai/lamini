import logging
import sys
from typing import AsyncIterator, Iterator, Optional, Union

from lamini.api.lamini_config import get_config
from lamini.generation.base_prompt_object import PromptObject
from lamini.generation.generation_node import GenerationNode

logger = logging.getLogger(__name__)


class EmbeddingNode(GenerationNode):
    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
        config: dict = {},
        max_tokens: Optional[int] = None,
    ):
        super(EmbeddingNode, self).__init__(
            model_name=model_name, api_key=api_key, api_url=api_url, config=config
        )
        self.max_tokens = max_tokens

    def generate(
        self,
        prompt: Union[Iterator[PromptObject], AsyncIterator[PromptObject]],
        model_name: Optional[str] = None,
    ):
        assert isinstance(prompt, Iterator) or isinstance(prompt, AsyncIterator)
        req_data = self.make_llm_req_map(
            prompt=prompt,
            model_name=model_name or self.model_name,
        )
        return self.async_inference_queue.submit(req_data)

    def make_llm_req_map(
        self,
        model_name,
        prompt,
    ):
        req_data = {}
        req_data["model_name"] = model_name
        req_data["prompt"] = prompt
        req_data["type"] = "embedding"
        return req_data
