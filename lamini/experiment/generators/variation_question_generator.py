from lamini.experiment.generators import BaseGenerator

class VariationQuestionGenerator(BaseGenerator):
    """
    VariationQuestionGenerator class that inherits from BaseGenerator.

    Responsible for generating variations of an input SQL question considering guidelines for modification.

    Attributes:
    - model: The model used for generating question variations.
    - client: Optional client parameter for base class.
    - name: Optional name for the generator instance.
    - role: Role description for the generator.
    - instruction: Instructions for the model detailing how variations are generated.
    - output_type: Specifies the type of output expected from the model.
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
        """
        Initializes the VariationGenerator with necessary parameters.

        Args:
        - model: The model to be used for generating variations.
        - client (optional): A client instance if necessary.
        - name (optional): Name of the generator.
        - role (optional): Role description for the generator.
        - instruction (optional): Detailed instructions for the model.
        - output_type (optional): Expected output type from the model.
        - kwargs: Additional keyword arguments for BaseGenerator initialization.
        """

        # Default instruction for generating question variations.
        instruction = """
          Task: Given the following original question, SQL query, schema, and glossary, create 2 variations of the question by modifying certain aspects of the query.

            Input Format:

                Original Question: {question}
                Original SQL: {sql}
                Schema: {schema}
                Glossary: {glossary}

            Guidelines for Generating Variations:

                Time Periods or Date Ranges: Modify the time periods or date ranges referenced in the question to change the scope (e.g., last month, this quarter, specific dates).
                Add or Remove Conditions: Introduce new conditions or remove existing ones to slightly alter the question's focus (e.g., adding filters or eliminating some restrictions).
                Change Aggregation Requirements: Adjust how data is aggregated (e.g., switch from a count to a sum, or change the way averages are calculated).
                Adjust Grouping Criteria: Modify the grouping or partitioning logic, like grouping by a different attribute or adding/removing grouping columns.

            Ensure that:

                The variations maintain consistency with the original intent of the question.
                Terminology from the glossary is used where applicable.
                The variations can still be answered using the available schema.

            Please provide ONLY the variations (in a clear, concise format) without any explanation or markdown formatting.
        """

        # Define expected output types for the variations.
        output_type = {"q1": "string", "q2": "string"}

        # Define the input structure for the model.
        self.input = {
            "question": "str",
            "sql": "str",
            "schema": "str",
            "glossary": "str",
        }

        # Initialize the base class with parameters.
        super().__init__(
            client=client,
            model=model,
            name=name or "VariationQuestionGenerator",
            role=role
            or "You are an expert at creating meaningful variations of questions.",
            instruction=instruction,
            output_type=output_type,
            **kwargs,
        )

    def postprocess(self, prompt_obj):
        """
        Post-process the model's response into a list of question variations.

        Args:
        - prompt_obj: The prompt object containing the model's response.

        Returns:
        - The modified prompt object with structured variations added.
        """
        # Handle empty model responses.
        if not prompt_obj.response:
            print(
                f"Warning: Empty response from model for question: {prompt_obj.data.get('question', 'Unknown')}"
            )
            return prompt_obj

        try:
            # Initialize variations list.
            variations = []

            # Append variations if present in the response.
            if prompt_obj.response.get("q1"):
                variations.append({"question": prompt_obj.response["q1"]})
            if prompt_obj.response.get("q2"):
                variations.append({"question": prompt_obj.response["q2"]})

            # Update the prompt object with variations.
            prompt_obj.response = {"variations": variations}
            return prompt_obj

        except Exception as e:
            # Handle exceptions during processing.
            print(f"Error processing model response: {str(e)}")
            print(f"Original question: {prompt_obj.data.get('question', 'Unknown')}")
            print(f"Model response: {prompt_obj.response}")
            prompt_obj.response = {"variations": []}
            return prompt_obj