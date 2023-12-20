from abc import ABCMeta, abstractmethod
from typing import List, Union
import pandas as pd


class BaseRunner(object, metaclass=ABCMeta):
    """
    A convenience class with some common methods for training and running LLMs
    """

    @abstractmethod
    def __init__(
        self,
        model_name: str,
        enable_peft: bool = False,
        config: dict = {},
        **kwargs,
    ):
        pass

    @abstractmethod
    def __call__(self, inputs: Union[str, List[str]]) -> Union[str, List[str]]:
        """Call the model runner on prompt"""
        pass

    @abstractmethod
    def evaluate_autocomplete(
        self, data: Union[str, List[str]]
    ) -> Union[str, List[str]]:
        pass

    @abstractmethod
    def load_data(self, data, verbose: bool = False):
        """
        Load arbitrary singleton types' values into the LLM
        """
        pass

    @abstractmethod
    def load_data_from_jsonlines(
        self, file_path: str, keys=None, verbose: bool = False
    ):
        """
        Load a jsonlines file with any keys into the LLM. Assume all independent strings.
        """

    @abstractmethod
    def load_data_from_dataframe(
        self, df: pd.DataFrame, columns=None, verbose: bool = False
    ):
        """
        Load a pandas dataframe into the LLM.
        """
        pass

    @abstractmethod
    def load_data_from_csv(self, file_path: str, columns=None, verbose: bool = False):
        """
        Load a csv file with input output keys into the LLM.
        Each row must have 'input' and 'output' as keys.
        """
        pass

    @abstractmethod
    def clear_data(self):
        """Clear the data from the LLM"""
        pass

    @abstractmethod
    def train(
        self, verbose: bool = False, finetune_args={}, limit=500, is_public=False
    ):
        """
        Train the LLM on added data. This function blocks until training is complete.
        """
        pass

    @abstractmethod
    def evaluate(self) -> List:
        """Get evaluation results"""
        pass
