import logging
from typing import AsyncIterator, List, Optional

import lamini
from lamini.classify.lamini_classifier import LaminiClassifier
from lamini.generation.base_prompt_object import PromptObject
from lamini.generation.embedding_node import EmbeddingNode

logger = logging.getLogger(__name__)


class ClassifierNode(EmbeddingNode):
    def __init__(
        self,
        classifier: LaminiClassifier,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
        config: dict = {},
        max_tokens: Optional[int] = None,
    ):
        super(ClassifierNode, self).__init__(
            model_name=model_name, api_key=api_key, api_url=api_url, config=config
        )
        self.max_tokens = max_tokens
        self.classifier = classifier

    def __call__(self, prompt, *args, **kwargs):
        prompt = self.transform_prompt(prompt)
        results = self.generate(prompt, *args, **kwargs)
        results = self.classify_results(results)
        results = self.process_results(results)
        return results

    async def batch(self, examples) -> AsyncIterator[List[PromptObject]]:
        batch = []
        async for example in examples:
            batch.append(example)
            if len(batch) == lamini.batch_size:
                yield batch
                batch = []
        if len(batch) > 0:
            yield batch

    async def classify_results(self, results: AsyncIterator[PromptObject]):
        batches = self.batch(results)
        async for batch in batches:
            result = self.classifiy_batch(batch)
            for r in result:
                yield r

    def classifiy_batch(self, prompt_list: List[PromptObject]):
        try:
            probabilities = self.classifier.classify_from_embedding(
                [prompt.response for prompt in prompt_list]
            )
            for prompt, probability in zip(prompt_list, probabilities):
                prompt.data["predictions"] = probability
            return prompt_list
        except Exception as e:
            logger.error(e)
            return [None for _ in prompt_list]
