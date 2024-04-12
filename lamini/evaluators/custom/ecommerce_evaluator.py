import logging
from typing import Union, Iterator, AsyncIterator

import jsonlines
from lamini.generation.base_prompt_object import PromptObject
from lamini.generation.generation_node import GenerationNode
from lamini.generation.generation_pipeline import GenerationPipeline
from lamini.evaluators.utils.utils import save_results

logger = logging.getLogger(__name__)


class EcommerceEvaluator:
    def __init__(self, ds_path, model_type, model_name, max_examples):
        self.ds_path = ds_path
        self.model_type = model_type
        self.model_name = model_name
        self.max_examples = max_examples

    async def load_shopping_dataset(self):
        i = 0
        with jsonlines.open(self.ds_path) as reader:
            for line in reader:
                if i == self.max_examples:
                    break
                i += 1
                yield PromptObject(prompt="", data=line)

    async def evaluate_hallucination(self):
        dataset = self.load_shopping_dataset()
        answers = AnswerScorePipeline(answer_model=self.model_name).call(dataset)

        results = await save_results(
            answers, model_name=self.model_name, task_name="shopping"
        )
        try:
            mean_response_score = sum(
                [item["answer"]["score"] for item in results]
            ) / len(results)
            precision_score = sum(
                [int(item["is_exact_match"]) for item in results]
            ) / len(results)
            results = {
                "product_response_subjective_score": {
                    "product_response_subjective_score": mean_response_score
                },
                "product_id_precision_score": {
                    "product_id_precision_score": precision_score
                },
            }
        except ZeroDivisionError:
            raise ValueError("No results to evaluate")
        return results


class AnswerScorePipeline(GenerationPipeline):
    def __init__(
        self,
        answer_model="mistralai/Mistral-7B-Instruct-v0.2",
        score_model="mistralai/Mistral-7B-Instruct-v0.2",
    ):
        super(AnswerScorePipeline, self).__init__()

        self.answer_generator = AnswerGenerator(model_name=answer_model)
        self.score_generator = ScoreGenerator(model_name=score_model)

    def forward(self, x):
        ans = self.answer_generator(x)
        score = self.score_generator(ans)
        return score


class AnswerGenerator(GenerationNode):
    def __init__(self, model_name="mistralai/Mistral-7B-Instruct-v0.2"):
        super(AnswerGenerator, self).__init__(model_name, max_new_tokens=150)

    def generate(
        self,
        prompt: Union[Iterator[PromptObject], AsyncIterator[PromptObject]],
        *args,
        **kwargs,
    ):
        op_type = {
            "product_name": "str",
            "product_description": "str",
            "product_id": "int",
        }
        results = super(AnswerGenerator, self).generate(
            prompt, output_type=op_type, *args, **kwargs
        )
        return results

    async def process_results(self, results):
        async for result in results:
            logger.info(f"Generated answer")
            if result is None:
                continue
            yield result

    async def transform_prompt(self, prompts):
        async for prompt in prompts:
            prompt.data["expected_product_id"] = prompt.data["product_id"]
            prompt.data["expected_product_name"] = prompt.data["product_name"]

            # for clarity
            del prompt.data["product_id"]
            del prompt.data["product_name"]
            prompt.prompt = self.make_prompt(prompt)
            yield prompt

    def make_prompt(self, chunk):
        # TODO: this prompt template may not work with all models. Change it as needed.
        prompt = "<s>[INST] You are an expert shopper at Instacart.\n"
        prompt += "You are helping a customer find a product. "
        prompt += (
            "Include the product name, id, and detailed description in your answer. "
        )
        prompt += "A product id is a number between 0 and 100,000. "
        prompt += "The customer asks\n"
        prompt += chunk.data["question"]
        prompt += " [/INST]"
        return prompt


class ScoreGenerator(GenerationNode):
    def __init__(self, model_name="mistralai/Mistral-7B-Instruct-v0.2"):
        super(ScoreGenerator, self).__init__(model_name=model_name, max_new_tokens=150)

    def generate(
        self,
        prompt: Union[Iterator[PromptObject], AsyncIterator[PromptObject]],
        *args,
        **kwargs,
    ):
        op_type = {"explanation": "str", "score": "int"}
        results = super(ScoreGenerator, self).generate(
            prompt, output_type=op_type, *args, **kwargs
        )
        return results

    async def process_results(self, results):
        async for result in results:
            logger.info(f"Generated score.")
            if result is None:
                continue
            yield result

    def get_rubric(self):
        rubric = "Read this scoring rubric carefully and follow the instructions precisely:\n"
        rubric += (
            "A score of 5 means that model's id is the same as the gold answer's id.\n"
        )
        rubric += "A score of 4 means that the model's product name is the same or a paraphrase of the gold answer, but the id may be wrong.  For example, the product names 'Tuna' and 'Canned Tuna' are similar\n"
        rubric += "A score of 3 means that the model's description is similar as the gold answer's description, but the id and product name may be wrong.  For example, lemonade and iced tea are different products.\n"
        rubric += "A score of 2 means that the model's description is not similar to the gold answer, but the answer is plausible.\n"
        rubric += "A score of 1 means that the model's description is not similar to the gold answer, and the answer doesn't make sense.\n"

        rubric += "Here are three examples of how to score the model's response:\n"
        rubric += "gold answer == Product ID: 1234, Product Name: Tuna, Description: Canned Tuna, model response == Product ID: 1234, Product Name: Tuna, Description: Canned Tuna, score == 5\n"
        rubric += "gold answer == Product ID: 5678, Product Name: Tuna, Description: Canned Tuna, model response == Product ID: 1234, Product Name: Canned Tuna, Description: Tuna, score == 4\n"
        rubric += "gold answer == Product ID: 5678, Product Name: Tuna, Description: Canned Tuna, model response == Product ID: 1234, Product Name: Bubble Gum, Description: Delicious treat, score == 1\n"
        rubric += "Assign a 5 even if fields are missing, for example: gold answer == Product ID: 1234, model response == Product ID: 1234, score == 5\n"
        return rubric

    def is_exact_match(self, prompt):
        return prompt.response["product_id"] == prompt.data["expected_product_id"]

    async def transform_prompt(self, prompts):
        async for prompt in prompts:
            prompt.data["rubric"] = self.get_rubric()
            prompt.data["response"] = self.format_response(prompt)
            prompt.data["expected_response"] = self.get_expected_response(prompt)
            prompt.data["is_exact_match"] = self.is_exact_match(prompt)

            prompt.prompt = self.make_prompt(prompt)
            yield prompt

    def get_expected_response(self, chunk):
        expected_response = f"Product ID: {chunk.data['expected_product_id']}\n"

        if chunk.data["expected_product_name"] != "N/A":
            expected_response += (
                f"Product Name: {chunk.data['expected_product_name']}\n"
            )

        if chunk.data["description"] != "N/A":
            expected_response += f"Description: {chunk.data['description']}"

        return expected_response

    def format_response(self, chunk):
        response = chunk.response
        formatted_response = f"Product ID: {response['product_id']}\n"

        if response["product_name"] != "N/A":
            formatted_response += f"Product Name: {response['product_name']}\n"

        if response["product_description"] != "N/A":
            formatted_response += f"Description: {response['product_description']}"

        return formatted_response

    def make_prompt(self, chunk):
        # TODO: this prompt template may not work with all models. Change it as needed.
        prompt = "<s>[INST] A model is going to answer a question. Your job is to score the answer, comparing it to a golden reference. You are an expert scorer.\n"
        prompt += f"Rate the answer using a score from 1 (lowest match) to 5 (highest match).\n"
        prompt += chunk.data["rubric"]
        prompt += "Use the full range. Read the gold answer carefully. "
        prompt += "Explain your score in 2-3 short sentences not exceeding 100 words each, then assign a score. "
        prompt += 'Output your score as a JSON object in the format {"explanation" : str, "score" : int}\n'
        prompt += "Use single quotes within your explanation. End your explanation with a double quote.\n"
        prompt += "Prefer answers that are most similar to the gold answer, even if the gold answer refused to answer the question.\n\n"
        prompt += f"========== question =========\n{chunk.data['question']}\n\n"
        prompt += (
            f"========== gold answer =========\n{chunk.data['expected_response']}\n\n"
        )
        prompt += f"========== model answer =========\n{chunk.data['response']}\n\n"
        prompt += "=" * 40 + "\n\n"
        prompt += f"How would you score the model's answer compared to the gold answer (using the 1-5 scale defined above)?\n\n"
        prompt += " [/INST]"
        return prompt
