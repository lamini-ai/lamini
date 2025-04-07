import logging

import pandas as pd
import logging

from lamini.experiment.utils import remove_non_ascii
from lamini.experiment.generators import BaseGenerator
from lamini.experiment.base_experiment_object import ExperimentObject

class SaveGenerator(BaseGenerator):
    """
    A generator class that saves experiment data to a JSON file.

    Parameters
    ----------
    save_path : str
        Path to the JSON file where data will be saved.
    save_keys : list[str], optional
        List of keys to save from the experiment object data.
        If None, all keys from the experiment object will be saved.

    Attributes
    ----------
    save_path : str
        Path to the JSON file where data will be saved.
    input : dict
        Dictionary mapping save keys to their expected type.
    output : dict
        Empty dictionary for output configuration.
    name : str
        Name of the generator.
    logger : logging.Logger
        Logger instance for this class.
    """

    def __init__(self, save_path: str, save_keys: list[str] = None):
        self.save_path = save_path
        self.input = {}
        if save_keys is not None:  # Only create key mapping if save_keys is provided
            self.input = {key_: "str" for key_ in save_keys}
        self.output = {}
        self.name = "SaveGenerator"
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.setLevel(logging.INFO)

    def __call__(self, exp_obj: ExperimentObject, debug=False):
        """
        Save experiment object data to a JSON file.

        Parameters
        ----------
        exp_obj : ExperimentObject
            The experiment object containing data to be saved.
        debug : bool, optional
            If True, sets logging level to DEBUG, by default False.

        Returns
        -------
        ExperimentObject
            The input experiment object with added SaveGenerator_output status.

        Notes
        -----
        The data is saved as a JSON Lines file, with each line containing a JSON object.
        Non-ASCII characters are removed from the data before saving.
        """
        if debug:
            self.logger.setLevel(logging.DEBUG)
        #print(prompt_obj)
        if self.input:
            data = {
                key_: remove_non_ascii(exp_obj.data[key_])
                for key_ in self.input
            }
        else:
            data = {
                key_: remove_non_ascii(exp_obj.data[key_])
                for key_ in exp_obj.data  
            }
        #print(data)
        try:
            pd.DataFrame(
                [data]
            ).to_json(
                self.save_path,
                mode="a",
                orient="records",
                lines=True,
            )
            exp_obj.data["SaveGenerator_output"] = "Success"
        except Exception as e:
            self.logger.error(f"Error saving data to {self.save_path} on {exp_obj}: {e}")
            exp_obj.data["SaveGenerator_output"] = "Error"
        
        return exp_obj