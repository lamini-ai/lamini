import unittest
from unittest.mock import patch

import lamini


class TestLaminiBackwardsCompatibility(unittest.TestCase):
    """Tests API stability of Lamini - think carefully about changing
    these tests once they are working. Breaking changes will require
    deprecating the corresponding python APIs.
    """

    @patch("lamini.api.utils.completion.Completion.generate")
    def test_lamini_generate(self, mock_generate):
        mock_generate.return_value = {"output": "Hi."}

        llm = lamini.Lamini(
            api_key="test_token",
            model_name="hf-internal-testing/tiny-random-gpt2",
        )
        res = llm.generate("How are you?", output_type={"Response": "str"})
        expected_res = {"output": "Hi."}
        print("Response", res)
        print("Expected Response", expected_res)

        self.assertEqual(res, expected_res)

    @patch("lamini.api.utils.completion.Completion.generate")
    def test_lamini_generate_with_output_type(self, mock_generate):
        mock_generate.return_value = {"Response": "Hi."}

        llm = lamini.Lamini(
            api_key="test_token",
            model_name="hf-internal-testing/tiny-random-gpt2",
        )
        res = llm.generate("How are you?", output_type={"Response": "str"})
        expected_res = {"Response": "Hi."}
        print("Response", res)
        print("Expected Response", expected_res)

        self.assertEqual(res, expected_res)

    @patch("lamini.api.train.Train.train")
    @patch("lamini.Lamini.upload_data")
    @patch("lamini.api.train.Train.get_existing_dataset")
    def test_lamini_train(self, mock_get_existing_dataset, mock_upload_data, mock_train):

        from importlib.metadata import version
        print(f"Lamini version: {version('lamini')}")

        mock_upload_data.return_value = "test_dataset_id"
        mock_get_existing_dataset.return_value = {"dataset_location": "test_dataset_id"}
        mock_train.return_value = {"job_id": "test_job_id"}

        llm = lamini.Lamini(
            api_key="test_token",
            model_name="hf-internal-testing/tiny-random-gpt2",
        )

        data = get_data()

        llm.train(data_or_dataset_id=data)



def get_data():
    data = [
        {
            "input": "Are there any step-by-step tutorials or walkthroughs available in the documentation?",
            "output": "Yes, there are step-by-step tutorials and walkthroughs available in the documentation section. Here\u2019s an example for using Lamini to get insights into any python SDK: https://lamini-ai.github.io/example/",
        },
        {
            "input": "Is the Lamini type system similar to a python type system?",
            "output": "Yes, the Lamini type system is built using Pydantic BaseModel.",
        },
        {
            "input": "Does Lamini have a limit on the number of API requests I can make?",
            "output": "Lamini provides each user with free tokens up front.",
        },
        {
            "input": "What does it mean to cancel a job using the `cancel_job()` function? Can we stop the machine from doing its task?",
            "output": "The `cancel_job()` function is used to stop a training job that is currently running.",
        },
        {
            "input": "Can Lamini automatically handle hyperparameter tuning during the customization process? How does it optimize the model for a specific use case?",
            "output": "Lamini is capable of automatically handling hyperparameter tuning during the model customization process. It employs an intelligent algorithm to explore the hyperparameter space and find the optimal combination of values. This is done through techniques such as heuristics, grid search, random search, Bayesian optimization, or genetic algorithms. Lamini efficiently utilizes computational resources to evaluate multiple model instances with different hyperparameter configurations. It incorporates techniques like cross-validation to prevent overfitting and ensure generalization. By automating hyperparameter tuning, Lamini streamlines the machine learning workflow and improves the chances of developing high-performing models for specific use cases.",
        },
        {
            "input": "Can you explain the CC-BY license mentioned? What does it allow me to do with the customized language model?",
            "output": "Lamini allows for commercial use of their LLM technology under a permissive Apache 2.0 license unless otherwise specified. You keep access and ownership of your own data, and we don't use your data to train models for anyone else but you. For more information, please reach out to Lamini directly.",
        },
        {
            "input": "Can Lamini be used on a regular computer, or do I need specialized hardware or software?",
            "output": "Lamini can be used on a regular computer without any specialized hardware or software by using the python client or REST APIs. It is designed to be easily accessible and user-friendly.  The LLM Engine itself requires GPU accelerated servers capable of running large language models.  Lamini hosts such machines that can be easily connected to, e.g. from a jupyter notebook or python program.  Lamini also offers enterprise deployments of the LLM Engine on your infrastructure.",
        },
        {
            "input": "Does Lamini require an internet connection to work, or can I use it offline?",
            "output": "Lamini requires an internet connection to work, as it is a cloud-based language model. However, you can use it offline by downloading and running a local instance of the Lamini API. To learn more about this deployment process, reach out to the Lamini team.",
        },
        {
            "input": "Can Lamini help me with tasks like translating text or answering questions, or is it focused on generating text?",
            "output": "Lamini is primarily focused on generating text, and it can be used for tasks like summarization and paraphrasing. Lamini can also be used to train a LLM for tasks like translation and question answering. You\u2019re talking to a model trained using Lamini right now!",
        },
        {
            "input": "What is Lamini? Is it like a robot or a computer program?",
            "output": "Lamini is a program for the execution of LLMs called a large language model engine. It is not a robot, but rather a tool for building and executing LLMs.",
        },
    ]
    return data
