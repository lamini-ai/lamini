import json
from lamini.api.lamini import Lamini
from typing import List, Optional
from lamini.evaluators.custom.custom_evaluator import CustomEvaluator
from lamini.evaluators.helm.harness_evaluator import HarnessEvaluator
import os
from datetime import datetime
import asyncio


class Benchmark:
    def __init__(
        self,
        model_name,
        api_key=None,
        api_url=None,
        config={},
    ):
        super().__init__()
        self.model_name = model_name
        self.api = Lamini(
            model_name=self.model_name,
            api_key=api_key,
            api_url=api_url,
            config=config,
        )

    def _get_task_names(self, tasks):
        HARNESS_TASKS = ["mmlu", "truthfulqa"]
        CUSTOM_TASKS = ["earnings", "products", "icd11"]

        if "all" in tasks:
            task_names = HARNESS_TASKS + CUSTOM_TASKS
        else:
            task_names = []
            for task in tasks:
                if task in CUSTOM_TASKS or task in HARNESS_TASKS:
                    task_names.append(task)
                else:
                    print(f"Invalid task: {task}. Skipping.")
        return task_names

    def get_benchmark_results(
        self, tasks: Optional[List] = ["all"], batch_size=20, limit=2, seed=42
    ):
        return asyncio.run(self._get_benchmark_results(tasks, batch_size, limit, seed))

    async def _get_benchmark_results(self, tasks, batch_size=20, limit=2, seed=42):
        task_names = self._get_task_names(tasks)
        custom_evaluator = CustomEvaluator(
            self.model_name, task_names, max_examples=limit
        )
        custom_results = await custom_evaluator.evaluate()

        harness_evaluator = HarnessEvaluator(
            self.model_name,
            task_names,
            max_examples=limit,
            batch_size=batch_size,
            seed=seed,
        )
        harness_results = harness_evaluator.evaluate(task_names)

        results_trimmed = self.prepare_and_save_results(custom_results, harness_results)
        return results_trimmed

    def prepare_and_save_results(self, custom_results, harness_results):
        combined_results = custom_results.get("results", {}) | harness_results.get(
            "results", {}
        )
        results_trimmed = {
            "config": harness_results.get("config", {}),
            "results": combined_results,
        }
        results_trimmed["config"]["model"] = self.model_name

        directory = f"tmp/results/{self.model_name}"
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, f"results-{datetime.now()}.json")
        print(f"Writing benchmark results to {path} inside the current directory.")
        with open(path, "w") as f:
            results_str = json.dumps(results_trimmed, indent=4)
            f.write(results_str)
        print("Write complete.")
        return results_trimmed
