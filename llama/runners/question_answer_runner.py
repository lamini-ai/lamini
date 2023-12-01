from typing import List, Union
from llama.engine.lamini import Lamini
import jsonlines
import pandas as pd
from llama.runners.runner import Runner
import os

class QuestionAnswerModel(Runner):
    """A class for running and training a question answering model"""

    def __init__(
            self,
            model_name: str = "EleutherAI/pythia-410m-deduped",
            task_name: str = "question_answer_runner_data",
            config={},
            enable_peft=False,
    ):
        self.model_name = model_name
        self.config = config
        self.llm = Lamini(
            id=task_name,
            model_name=self.model_name,
            prompt_template="{input:question}",
            config=self.config,
        )

        self.question_answer = []
        self.job_id = None
        self.enable_peft = enable_peft

        self.evaluation = None

    def get_answer(
            self,
            question: str,
    ) -> str:
        """Get answer to a single question"""
        question_object = {"question": question}
        answer_object = self.llm(
            input=question_object,
            output_type={"answer": "str"},
            model_name=self.model_name,
            enable_peft=self.enable_peft,
        )
        return answer_object["answer"]

    def get_answers(self, questions: List[str]) -> List[str]:
        """Get answers to a batch of questions"""
        print("Asking %d questions" % len(questions))
        question_objects = [{"question": q} for q in questions]
        answer_objects = self.llm(
            input=question_objects,
            output_type={"answer": "str"},
            model_name=self.model_name,
            enable_peft=self.enable_peft,
        )
        answers = [a["answer"] for a in answer_objects]
        return [{"question": q, "answer": a} for q, a in zip(questions, answers)]

    def call(self, inputs: Union[str, List[str]]):
        if type(inputs) == str:
            return self.get_answer(inputs)
        else:
            return self.get_answers(inputs)

    def load_question_answer(self, data, question_key="question", answer_key="answer"):
        """
        Load a list of json objects with question answer keys into the LLM
        Each object must have 'question' and 'answer' as keys.
        """
        try:
            question_answer_objects = [
                [{"question": d[question_key]}, {"answer": d[answer_key]}] for d in data
            ]
        except KeyError:
            raise ValueError("Each object must have 'question' and 'answer' as keys")
        self.question_answer.extend(question_answer_objects)

    def load_question_answer_from_jsonlines(
            self, file_path: str, question_key="question", answer_key="answer"
    ):
        """
        Load a jsonlines file with question answer keys into the LLM.
        Each line must be a json object with 'question' and 'answer' as keys.
        """
        data = []
        with open(file_path) as dataset_file:
            reader = jsonlines.Reader(dataset_file)
            data = list(reader)
        self.load_question_answer(data, question_key, answer_key)

    def load_question_answer_from_dataframe(
            self, df: pd.DataFrame, question_key="question", answer_key="answer"
    ):
        """
        Load a pandas dataframe with question answer keys into the LLM.
        Each row must have 'question' as a key.
        """
        try:
            for _, row in df.iterrows():
                self.question_answer.append(
                    [
                        {"question": row[question_key]},
                        {"answer": row[answer_key]},
                    ]
                )
        except KeyError:
            raise ValueError("Each object must have 'question' and 'answer' as keys")

    def load_question_answer_from_csv(
            self, file_path: str, question_key="question", answer_key="answer"
    ):
        """
        Load a csv file with question answer keys into the LLM.
        Each row must have 'question' and 'answer' as keys.
        """
        df = pd.read_csv(file_path)
        self.load_question_answer_from_dataframe(
            df, question_key=question_key, answer_key=answer_key
        )

    def upload_file(self,
                    file_path,
                    question_key="question",
                    answer_key="answer"):

        if os.path.getsize(file_path) > 1e+7:
            raise Exception("File size is too large, please upload file less than 10MB")

        # Convert file records to appropriate format before uploading file
        items = []
        if file_path.endswith(".jsonl") or file_path.endswith(".jsonlines"):
            with open(file_path) as dataset_file:
                reader = jsonlines.Reader(dataset_file)
                data = list(reader)

                for row in data:
                    item = [
                        {"question": row[question_key]},
                        {"answer": row[answer_key]}
                    ]
                    items.append(item)

        elif file_path.endswith(".csv"):
            df = pd.read_csv(file_path).fillna("")
            data_keys = df.columns
            if question_key not in data_keys:
                raise ValueError(
                    f"File must have question={question_key} and answer={answer_key} as columns. You "
                    "can pass in different question_key and answer_key."
                )

            try:
                for _, row in df.iterrows():
                    item = [
                        {"question": row[question_key]},
                        {"answer": row[answer_key]}
                    ]
                    items.append(item)
            except KeyError:
                raise ValueError("Each object must have 'question' and 'answer' as keys")

        else:
            raise Exception("Upload of only csv and jsonlines file supported at the moment.")

        self.llm.upload_data(items)

    def clear_data(self):
        """Clear the data from the LLM"""
        self.llm.delete_data()
        self.question_answer = []

    def train(
            self,
            verbose: bool = False,
            limit=500,
            is_public=False,
            **kwargs,
    ):
        """
        Train the LLM on added data. This function blocks until training is complete.
        """
        if len(self.question_answer) < 2 and not self.llm.upload_file_path:
            raise Exception(
                "Submit at least 2 question answer pairs to train to allow validation"
            )
        if limit is None:
            qa_pairs = self.question_answer
        elif len(self.question_answer) > limit:
            qa_pairs = self.question_answer[:limit]
        else:
            qa_pairs = self.question_answer

        if self.llm.upload_file_path:
            final_status = self.llm.train(
                **kwargs,
            )
        else:
            final_status = self.llm.train(
                qa_pairs,
                **kwargs,
            )
        try:
            self.model_name = final_status["model_name"]
            self.job_id = final_status["job_id"]
            self.llm.delete_data()
        except KeyError:
            raise Exception("Training failed")

        return final_status

    def evaluate(self) -> List:
        """Get evaluation results"""
        if self.job_id is None:
            raise Exception("Must train before getting results (no job id))")
        self.evaluation = self.llm.evaluate()
        return self.evaluation

    def get_eval_results(self) -> List:
        return self.evaluate()
