from lamini.experiment.generators.base_generator import BaseGenerator

class ConceptToSQLInterpretationGenerator(BaseGenerator):
    """
    Takes a concept and returns the SQL interpretation of the concept.
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
        name = name or "ConceptToSQLInterpretationGenerator"
        role = (
            role
            or "You are a helpful assistant that takes a concept and returns the SQL interpretation of the concept, which is just the calculation of the concept in a SQL fragment."
        )
        instruction = (
            instruction
            or """Given the concept, return the SQL interpretation of the concept, which is just the calculation of the concept in a SQL fragment.
        Concept:
        {concept}

        SQL Interpretation:"""
        )

        output_type = output_type or {"sql": "str"}

        super().__init__(
            client=client,
            model=model,
            name=name,
            role=role,
            instruction=instruction,
            output_type=output_type,
            **kwargs,
        )