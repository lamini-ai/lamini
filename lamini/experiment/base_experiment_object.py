
from lamini.generation.base_prompt_object import PromptObject

class BaseExperimentObject(PromptObject):
    """
    A base class for experiment objects that extends PromptObject functionality.
    
    This class serves as a foundation for conducting experiments with prompt-based
    operations, providing a structured way to manage experimental data and responses
    in prompt engineering workflows.
    
    Parameters
    ----------
    data : dict, optional
        A dictionary containing additional data related to the experiment,
        by default {}
        
    Attributes
    ----------
    prompt : str
        The prompt text used in the experiment (inherited from PromptObject)
    response : Any
        The response generated from the prompt (inherited from PromptObject)
    data : dict
        Additional metadata and parameters associated with the experiment
    """
    
    def __init__(self, data: dict = None) -> None:
        """
        Initialize a new BaseExperimentObject instance.
        
        Parameters
        ----------
        data : dict, optional
            A dictionary containing additional data related to the experiment,
            by default {}
        """
        super().__init__(prompt="", response=None, data=data)
        
        
