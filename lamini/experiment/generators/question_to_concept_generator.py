from lamini.experiment.generators.base_generator import BaseGenerator

class QuestionToConceptGenerator(BaseGenerator):
    """
    Takes a single question and returns a single concept that is relevant to the question.
    """

    def __init__(
        self,
        model,
        client=None,
        name=None,
        role=None,
        instruction=None,
        output_type=None,
        **kwargs,
    ):
        name = name or "QuestionToConceptGenerator"
        role = (
            role
            or "You are a helpful assistant that takes a single question and returns a single concept that is relevant to the question."
        )
        instruction = (
            instruction
            or """Given the question, return a single concept that is relevant to the question.
        Question:
        {question}

        Concept:"""
        )

        output_type = output_type or {"concept": "str"}

        super().__init__(
            client=client,
            model=model,
            name=name,
            role=role,
            instruction=instruction,
            output_type=output_type,
            **kwargs,
        )