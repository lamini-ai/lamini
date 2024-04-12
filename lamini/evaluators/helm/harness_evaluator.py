from lm_eval import tasks, evaluator, utils
from lm_eval.api.model import LM
from datetime import datetime
from tqdm import tqdm
import os
import jsonlines
from typing import List
from lamini.api.lamini import Lamini
from lamini.evaluators.helm.mmlu_evaluator import MMLUEvaluator
from lamini.evaluators.helm.truthfulqa_evaluator import TruthfulQAEvaluator
import logging

logger = logging.getLogger(__name__)


class HarnessEvaluator:
    def __init__(
        self,
        model,
        task_names=None,
        max_examples=2,
        batch_size=20,
        seed=42,
    ):
        if task_names is None:
            self.task_names = ["products", "earnings", "icd11"]
        else:
            self.task_names = task_names
        self.model = model
        self.max_examples = max_examples
        self.batch_size = batch_size
        self.seed = seed

    def get_harness_tasks(self, tasks):
        task_names = []
        alias_dict = {
            "mmlu": "mmlu_flan_n_shot_generative_global_facts",
            "truthfulqa": "truthfulqa_gen",
        }
        for t in tasks:
            if t in alias_dict:
                task_names.append(alias_dict[t])
            else:
                print(f"Invalid task: {t}. Skipping.")
        return task_names

    def evaluate(self, task_names):
        harness_tasks = self.get_harness_tasks(task_names)
        print(f"Selected Harness Tasks: {harness_tasks}")
        model_obj = BenchmarkModel(self.model)

        if len(harness_tasks) == 0:
            print("No harness tasks to evaluate. Skipping.")
            return {}
        harness_results = evaluator.simple_evaluate(
            model=model_obj,
            tasks=harness_tasks,
            batch_size=self.batch_size,
            num_fewshot=0,
            limit=self.max_examples,
            random_seed=self.seed,
        )
        res = self.format_harness_results(harness_results)
        return res

    def format_harness_results(self, harness_results):
        res = {
            "config": harness_results.get("config", {}),
            "results": {
                "mmlu_flan_n_shot_generative_global_facts": harness_results.get(
                    "results", {}
                ).get("mmlu_flan_n_shot_generative_global_facts", -1),
                "truthfulqa_gen": harness_results.get("results", {}).get(
                    "truthfulqa_gen", -1
                ),
            },
        }
        return res


class BenchmarkModel(LM):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.api = Lamini(model_name=model)

    def __call__(self, prompt, output_type):
        return self.api.generate(prompt=prompt, output_type=output_type)

    def get_helm_response(self, request):
        """
        Run helm evaluation tasks. You can add more tasks and write evaluators from here: https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks
        Make sure to add only 'generative' tasks.
        """

        if request.task_name == "truthfulqa_gen":
            question = request.doc["question"]
            obj = TruthfulQAEvaluator()
            prompt = obj.get_prompt(question)
            try:
                response = self.api.generate(prompt=prompt).strip()
                print("\n\n")
            except Exception as e:
                print("Error fetching response: ", e)
                # select random answer
                response = "\nA: none"

        else:
            obj = MMLUEvaluator()
            question = request.arguments[0]
            prompt = obj.get_prompt(question)
            try:
                op_type = {"explanation": "str", "answer": "str"}
                # TODO: replace with pipeline
                answer = self.api.generate(prompt=prompt, output_type=op_type)
                response = f"({answer['answer'].strip()})"
            except Exception as e:
                print("Error fetching response: ", e)
                # select random answer
                response = "(A)"

        print("\n\ntask_name: ", request.task_name)
        print("helm prompt: ", prompt)
        print("helm response: ", response)
        return response

    def generate_until(self, requests) -> List[str]:
        # This is called by helm tasks
        res = []
        directory = f"tmp/results/{self.model}"
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, f"helm-answers-{datetime.now()}.jsonl")
        print(f"Writing generation results to {path} inside the current directory.")

        try:
            with jsonlines.open(path, "w") as writer:
                for request in tqdm(requests):
                    write_dict = request.__dict__
                    response = self.get_helm_response(request)
                    write_dict["model_response"] = response
                    writer.write(write_dict)
                    res.append(response)
                    self.cache_hook.add_partial("generate_until", request, response)
            print("Written.")
            return res
        except Exception as e:
            print("Error generating response: ", e)
            return []

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError("No support for loglikelihood.")

    def loglikelihood(self, requests):
        raise NotImplementedError("No support for rolling likelihood.")
