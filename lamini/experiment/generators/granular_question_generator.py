from lamini.experiment.generators import BaseGenerator

class GranularQuestionGenerator(BaseGenerator):
    """
    A class to generate new questions based on an original question and
    the corresponding SQL query, schema, and glossary. The new questions
    adjust the level of detail in the request, providing both high-level
    summaries and detailed breakdowns of the data.
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
        Initializes a GranularGenerator instance with the specified parameters.

        Parameters:
        - model: The model to use for generating questions.
        - client: Optional client configuration.
        - name: Optional name for the instance. Defaults to "GranularGenerator".
        - role: Optional role description. Defaults to a predefined role.
        - instruction: Optional instruction set. Defaults to a predefined instruction.
        - output_type: Optional output type description. Defaults to a predefined structure.
        """
        instruction = """
            Task: Given the following original question, SQL query, schema, and glossary, generate 2 new questions that adjust the level of detail in the request. One variation should ask for high-level summaries (aggregated data), while the other should request detailed, granular information (fine breakdowns).

            Input Format:

                Original Question: {question}
                Original SQL: {sql}
                Schema: {schema}
                Glossary: {glossary}

            Guidelines for Generating New Questions:

                High-Level Summary (Aggregated Data): Generate one question that asks for aggregated data, such as averages, totals, counts, or other forms of summary information. This should provide an overview of the data at a higher level, without diving into specific details.
                Detailed Breakdown (Granular Information): Generate another question that asks for more detailed, granular information. This should request specific data points or breakdowns at a lower level of granularity (e.g., individual records, detailed groupings, or time-specific data).
                Alignment with Schema and Glossary: Both variations should align with the provided schema and glossary, ensuring that the queries can be answered with the available tables and columns.
                Capture Different Perspectives: The new questions should provide different perspectives on the data—one with a summary view and the other with detailed breakdowns—while maintaining the overall context of the original question.
                SQL Query Validity: The SQL queries must be syntactically valid and appropriate for the schema provided.

            Please provide ONLY the new questions (in a clear, concise format) without any explanation or markdown formatting.
        """

        # Define the expected output type
        output_type = {"q1": "string", "q2": "string"}

        # Define the expected input structure
        self.input = {
            "question": "str",
            "sql": "str",
            "schema": "str",
            "glossary": "str",
        }

        # Initialize the superclass with the provided and default parameters
        super().__init__(
            client=client,
            model=model,
            name=name or "GranularQuestionGenerator",
            role=role
            or "You are an expert in making questions that focuses focus making granular questions.",
            instruction=instruction,
            output_type=output_type,
            **kwargs,
        )

    def postprocess(self, prompt_obj):
        """
        Process the model's response to extract the generated questions
        and handle any errors.

        Parameters:
        - prompt_obj: The object containing the model's response and input data.

        Returns:
        The updated prompt_obj with the processed response.
        """
        if not prompt_obj.response:
            print(
                f"Warning: Empty response from model for question: {prompt_obj.data.get('question', 'Unknown')}"
            )
            return prompt_obj

        try:
            grans = []
            if prompt_obj.response.get("q1"):
                grans.append({"question": prompt_obj.response["q1"]})
            if prompt_obj.response.get("q2"):
                grans.append({"question": prompt_obj.response["q2"]})

            prompt_obj.response = {"grans": grans}
            return prompt_obj

        except Exception as e:
            print(f"Error processing model response: {str(e)}")
            print(f"Original question: {prompt_obj.data.get('question', 'Unknown')}")
            print(f"Model response: {prompt_obj.response}")
            prompt_obj.response = {"grans": []}
            return prompt_obj