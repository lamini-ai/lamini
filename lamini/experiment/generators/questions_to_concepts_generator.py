from lamini.experiment.generators.base_generator import BaseGenerator
from lamini.generation.base_prompt_object import PromptObject
from copy import deepcopy

class QuestionsToConceptsGenerator(BaseGenerator):
    """Generator that extracts concepts from questions.

    Processes a list of questions and extracts key concepts, creating separate
    PromptObjects for each identified concept.

    Note:
        Expects response in format {"concepts_list": "concept1, concept2, concept3"}
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

        name = name or "QuestionsToConceptsGenerator"
        role = (
            role
            or "You are a helpful assistant that takes a list of common questions and returns a list of concepts that are relevant to the concepts."
        )
        instruction = (
            instruction
            or """Given the list of questions, return short concepts that are relevant to the questions, separated by commas. Do not include any other information in your response.
        Questions:
        {questions}

        Concepts:"""
        )

        output_type = output_type or {
            "concepts_list": "str"
        }  # This is the intermediate output type

        super().__init__(
            client=client,
            model=model,
            name=name,
            role=role,
            instruction=instruction,
            output_type=output_type,
            **kwargs,
        )

    def postprocess(self, result):
        """Process the LLM response into individual concept PromptObjects.

        Converts the comma-separated concepts string into a list of individual
        PromptObjects, each containing a single concept.

        Args:
            result (PromptObject): Contains response with concepts_list field

        Returns:
            List[PromptObject]: List of new PromptObjects, one per concept
        """
        # Turn the string result, formatted as {"concepts_list": "concept1, concept2, concept3"}, into a list of concept objects
        concepts_list_object = result.response["concepts_list"]

        concepts_list_object = concepts_list_object.replace(
            "[", ""
        )  # remove square brackets
        concepts_list_object = concepts_list_object.replace("]", "")

        concepts_list = concepts_list_object.split(",")  # split into list of concepts

        concepts_list = [
            concept.strip() for concept in concepts_list
        ]  # remove whitespace
        concepts_list = [
            concept for concept in concepts_list if concept
        ]  # remove empty strings

        # Create a list of concept PromptObjects, each with a concept field
        concepts = []
        for concept in concepts_list:
            # Deep copy the history stored in result prompt object to avoid shared references
            new_prompt_obj = PromptObject(
                prompt=deepcopy(result.prompt),
                data=deepcopy(result.data),
                response=deepcopy(result.response),
            )
            new_prompt_obj.data["concept"] = concept
            concepts.append(new_prompt_obj)

        return concepts