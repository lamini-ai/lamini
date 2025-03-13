from lamini.experiment.validators import BaseValidator
from lamini.experiment.generators import BaseSQLGenerator

class BaseSQLValidator(BaseSQLGenerator, BaseValidator):
    """Base class for SQL validation that combines generator and validator capabilities."""

    def __init__(
        self,
        model,
        client=None,
        db_type=None,
        db_params=None,
        name="BaseSQLValidator",
        instruction=None,
        output_type=None,
        **kwargs,
    ):
        # Initialize BaseSQLGenerator first
        BaseSQLGenerator.__init__(
            self,
            model=model,
            client=client,
            db_type=db_type,
            db_params=db_params,
            name=name,
            instruction=instruction,
            output_type=output_type,
            **kwargs,
        )

        # Initialize BaseValidator
        BaseValidator.__init__(
            self,
            model=model,
            client=client,
            name=name,
            instruction=instruction,
            output_type=output_type,
            **kwargs,
        )

    def __call__(self, prompt_obj, debug=False):
        """Base validation method to be implemented by subclasses."""
        return BaseValidator.__call__(self, prompt_obj, debug)
    
