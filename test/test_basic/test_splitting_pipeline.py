import datetime
import random
import unittest
from typing import AsyncIterator, Iterator, Union
from unittest.mock import patch

import lamini
from lamini.generation.base_prompt_object import PromptObject
from lamini.generation.generation_node import GenerationNode
from lamini.generation.generation_pipeline import GenerationPipeline
from lamini.generation.split_response_node import SplitResponseNode

questions_type = {"question_1": "str", "question_2": "str", "question_3": "str"}


class SplittingResponsePipeline(GenerationPipeline):
    def __init__(self):
        super(SplittingResponsePipeline, self).__init__()

        self.question_generator = QuestionGenerator()

    def forward(self, x):
        x = self.question_generator(x)
        return x


class QuestionGenerator(GenerationNode):
    def __init__(self):
        super(QuestionGenerator, self).__init__(
            model_name="hf-internal-testing/tiny-random-gpt2"
        )

    def generate(
        self,
        prompt: Union[Iterator[PromptObject], AsyncIterator[PromptObject]],
        *args,
        **kwargs,
    ):
        results = super(QuestionGenerator, self).generate(
            prompt,
            output_type={
                "question_1": "string",
                "question_2": "string",
                "question_3": "string",
            },
            *args,
            **kwargs,
        )
        return results

    async def process_results(self, results):
        async for result in results:
            if (result is None) or (result.response is None):
                continue

            response = result.response
            questions = [
                response["question_1"],
                response["question_2"],
                response["question_3"],
            ]
            for question in questions:
                ans = PromptObject(prompt=question, data=result.data.copy())
                yield ans

    async def transform_prompt(self, prompts):
        for prompt in prompts:
            prompt.prompt = self.make_prompt(prompt)
            yield prompt

    def make_prompt(self, chunk):
        prompt = "<s>[INST] You are a customer shopping on Instacart. You are looking for a product to buy."
        prompt += f"Product ID: {chunk.data['product']['product_id']}\n"
        prompt += f"Product Name: {chunk.data['product']['product_name']}\n"
        prompt += f"Product Description: {chunk.data['descriptions']}\n"
        prompt += "Write three questions that would be a good fit for this product.\n"
        prompt += " [/INST]"
        return prompt


async def async_mock_side_effect(args):
    batch = args["batch"]
    for i, prompt_obj in enumerate(batch["prompt"]):
        if prompt_obj.response is None:
            prompt_obj.response = {
                "question_1": "string",
                "question_2": "string",
                "question_3": "string",
            }
        else:
            raise Exception("This should not be called")


class TestSplittingQAPipeline(unittest.IsolatedAsyncioTestCase):

    @patch("lamini.api.utils.reservations.make_web_request")
    @patch("lamini.api.utils.reservations.make_async_web_request")
    @patch("lamini.generation.generation_queue_3_10.process_generation_batch")
    async def test_pipeline_call_with_result(
        self,
        mock_process_generation_batch,
        mock_reservation_api_poll,
        mock_reservation_api_init,
    ):
        mock_process_generation_batch.side_effect = async_mock_side_effect
        mock_reservation_api_poll.return_value = {
            "reservation_id": 1,
            "capacity": 10,
            "start_time": str(datetime.datetime.now()),
            "end_time": str(datetime.datetime.now() + datetime.timedelta(seconds=60)),
            "capacity_remaining": 10,
        }
        mock_reservation_api_init.return_value = {
            "reservation_id": 1,
            "capacity": 10,
            "start_time": str(datetime.datetime.now()),
            "end_time": str(datetime.datetime.now() + datetime.timedelta(seconds=60)),
            "capacity_remaining": 10,
        }

        pipeline = SplittingResponsePipeline()

        prompts = list(
            PromptObject(
                f"{i}",
                data={
                    "product": {
                        "product_id": "9517",
                        "product_name": "Havarti Cheese",
                        "aisle_id": "21",
                        "department_id": "16",
                    },
                    "descriptions": "Havarti cheese is a type of semi-soft, creamy cheese that originates from the Caucasus region of Georgia.  It is made from the milk of goats or sheep, and is known for its rich, nutty flavor and smooth texture.  Unlike other cheeses, Havarti cheese is not aged for a long time, resulting in a fresher, more delicate taste",
                },
            )
            for i in range(10)
        )
        prompts_iterator = iter(prompts)
        results = await pipeline.call_with_result(prompts_iterator)
        print(results)
        assert len(results) == 30
