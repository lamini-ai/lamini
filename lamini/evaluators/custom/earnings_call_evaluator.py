import logging
from typing import Union, Iterator, AsyncIterator

import jsonlines
from lamini.generation.base_prompt_object import PromptObject
from lamini.generation.generation_node import GenerationNode
from lamini.generation.generation_pipeline import GenerationPipeline
from lamini.evaluators.utils.utils import save_results, format_results


logger = logging.getLogger(__name__)


class EarningsCallEvaluator:
    def __init__(self, ds_path, model_type, model_name, max_examples):
        self.ds_path = ds_path
        self.model_type = model_type
        self.model_name = model_name
        self.max_examples = max_examples

    async def load_earnings_dataset(self):
        i = 0
        with jsonlines.open(self.ds_path) as reader:
            for line in reader:
                if i == self.max_examples:
                    break
                i += 1
                yield PromptObject(prompt="", data=line)

    async def evaluate_hallucination(self):
        dataset = self.load_earnings_dataset()
        answers = EarningsPipeline(answer_model=self.model_name).call(dataset)
        results = await save_results(
            answers, model_name=self.model_name, task_name="earnings"
        )

        try:
            mean_response_score = sum(
                [item["answer"]["score"] for item in results]
            ) / len(results)
            precision_score = sum(
                [int(item["is_exact_match"]) for item in results]
            ) / len(results)
            results = {
                "earnings_response_subjective_score": {
                    "earnings_response_subjective_score": mean_response_score
                },
                "earnings_precision_score": {
                    "earnings_precision_score": precision_score
                },
            }
        except ZeroDivisionError:
            raise ValueError("No results to evaluate")
        return results


class EarningsPipeline(GenerationPipeline):
    def __init__(
        self,
        answer_model="mistralai/Mistral-7B-Instruct-v0.2",
        score_model="mistralai/Mistral-7B-Instruct-v0.2",
    ):
        super(EarningsPipeline, self).__init__()

        self.answer_generator = EarningsAnswerGenerator(model_name=answer_model)
        self.score_generator = EarningsScoreGenerator(model_name=score_model)

    def forward(self, x):
        ans = self.answer_generator(x)
        score = self.score_generator(ans)
        return score


class EarningsAnswerGenerator(GenerationNode):
    def __init__(self, model_name="mistralai/Mistral-7B-Instruct-v0.2"):
        super(EarningsAnswerGenerator, self).__init__(model_name)

    def generate(
        self,
        prompt: Union[Iterator[PromptObject], AsyncIterator[PromptObject]],
        *args,
        **kwargs,
    ):
        op_type = {
            "answer": "str",
            "value": "float",
            "units": "str",
        }
        results = super(EarningsAnswerGenerator, self).generate(
            prompt, output_type=op_type, *args, **kwargs
        )
        return results

    async def process_results(self, results):
        async for result in results:
            if result is None:
                continue
            yield result

    async def transform_prompt(self, prompts):
        async for prompt in prompts:
            # for clarity, delete the original value and units and call them "expected".
            prompt.data["expected_value"] = prompt.data["value"]
            prompt.data["expected_units"] = prompt.data["units"]
            prompt.data["expected_answer"] = prompt.data["answer"]

            del prompt.data["value"]
            del prompt.data["units"]
            del prompt.data["answer"]
            prompt.prompt = self.make_prompt(prompt)
            yield prompt

    def make_prompt(self, chunk):
        # TODO: this prompt template may not work with all models. Change it as needed.
        prompt = "<s>[INST] You are a financial analyst with extensive experience at Goldman Sachs.\n"
        prompt += "You are reading questions that you have heard from a client about a specific earnings call. "
        prompt += "The question asks about specific numbers mentioned in the call. "
        prompt += "The question is about this earnings call:\n"
        prompt += "====================\n"
        prompt += f"Date of the call: {chunk.data['date']}\n"
        prompt += f"Ticker: {chunk.data['ticker']}\n"
        prompt += f"Quarter: {chunk.data['q']}\n"
        prompt += "====================\n"
        prompt += "The client asks\n"
        prompt += chunk.data["question"]
        prompt += " [/INST]"
        return prompt


class EarningsScoreGenerator(GenerationNode):
    def __init__(self, model_name="mistralai/Mistral-7B-Instruct-v0.2"):
        super(EarningsScoreGenerator, self).__init__(model_name=model_name)

    def generate(
        self,
        prompt: Union[Iterator[PromptObject], AsyncIterator[PromptObject]],
        *args,
        **kwargs,
    ):
        op_type = {"explanation": "str", "score": "int"}
        results = super(EarningsScoreGenerator, self).generate(
            prompt, output_type=op_type, *args, **kwargs
        )
        return results

    async def process_results(self, results):
        async for result in results:
            logger.info(f"Generated score. {result}")
            logger.info(f"===")
            if result is None:
                continue
            yield result

    def get_rubric(self):
        rubric = "Read this scoring rubric carefully and follow the instructions precisely:\n"
        rubric += "A score of 5 means that model's value is the same as the gold answer's id.\n"
        rubric += "A score of 4 means that the model's answer is the same or a paraphrase of the gold answer, but the value may not be an exact match. For example, the values '1 million' and '1000.0 thousand' are different ways of describing the same value\n"
        rubric += "A score of 3 means that the model's answer is similar as the gold answer's description, but the value may be wrong. Both answers may indicate that revenue is increased but the gold says 12 percent and the model say 50 million USD.\n"
        rubric += "A score of 2 means that the model's answer is not similar to the gold answer, but the answer is plausible.\n"
        rubric += "A score of 1 means that the model's answer is not similar to the gold answer, and the answer doesn't make sense.\n"

        rubric += "Assign a 5 for a correct value even if other fields are missing.\n"
        return rubric

    def is_exact_match(self, prompt):
        if "units" not in prompt.response:
            return False

        if "value" not in prompt.response:
            return False

        return (
            prompt.response["units"] == prompt.data["expected_units"]
            and prompt.response["value"] == prompt.data["expected_value"]
        )

    async def transform_prompt(self, prompts):
        async for prompt in prompts:
            prompt.data["rubric"] = self.get_rubric()
            prompt.data["response"] = self.format_response(prompt)
            prompt.data["expected_response"] = self.get_expected_response(prompt)
            prompt.data["is_exact_match"] = self.is_exact_match(prompt)

            prompt.prompt = self.make_prompt(prompt)
            yield prompt

    def get_expected_response(self, chunk):
        expected_response = (
            f"Value: {chunk.data['expected_value']} {chunk.data['expected_units']}\n"
        )

        if "expected_answer" in chunk.data and chunk.data["expected_answer"] != "N/A":
            expected_response += f"Answer: {chunk.data['expected_answer']}\n"

        return expected_response

    def format_response(self, chunk):
        response = chunk.response
        if "units" not in response:
            return "Unknown units\n"

        if "value" not in response:
            return "Unknown value\n"

        formatted_response = f"Value: {response['value']} {response['units']}\n"

        if "answer" in response and response["answer"] != "N/A":
            formatted_response += f"Answer: {response['answer']}\n"

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
