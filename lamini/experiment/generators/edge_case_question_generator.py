from lamini.experiment.generators import BaseGenerator

class EdgeCaseQuestionGenerator(BaseGenerator):
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
        Initializes an instance of EdgeCaseGenerator, a class that focuses on
        generating questions about unusual or extreme conditions such as outliers,
        rare events, or boundary cases.

        Args:
            model: The model to be used for generating questions.
            client: Optional; the client configuration.
            name: Optional; name of the generator.
            role: Optional; role description for the generator.
            instruction: Optional; the instruction set to guide question generation.
            output_type: Optional; the expected format of the output.
            **kwargs: Additional keyword arguments.
        """
        # Set the task instruction for generating edge case questions
        instruction = """
        Task: Given the following original question, SQL query, schema, and glossary, generate 2 new questions that focus on unusual or extreme conditions, such as outliers, rare events, or boundary cases. Modify filters or conditions to explore scenarios that might not occur frequently but are important for a comprehensive analysis.

        Input Format:

            Original Question: {question}
            Original SQL: {sql}
            Schema: {schema}
            Glossary: {glossary}

        Guidelines for Generating New Questions:

            Focus on Extreme or Unusual Conditions: Modify the question to explore outliers, rare events, or boundary cases. Think about what edge conditions might be interesting to analyze, such as data points that fall outside typical patterns or expectations.
            Modify Filters or Conditions: Adjust the filters, conditions, or aggregations in the SQL query to reflect these unusual or extreme scenarios. For example, using specific thresholds or conditions that focus on outliers or rare events (e.g., values above a certain threshold or extremely low frequencies).
            Maintain Original Question Intent: Ensure that the new questions still align with the original question's intent while exploring the extreme or unusual conditions.
            Answerability with Schema and Glossary: The new questions must be answerable using the provided schema and glossary. Use valid tables and columns as described in the schema to construct your queries.
            SQL Query Validity: The SQL queries generated must be syntactically valid and consistent with the schema.

        Please provide ONLY the new questions (in a clear, concise format) without any explanation or markdown formatting.
        """
        # Define the expected format for the output questions
        output_type = {"q1": "string", "q2": "string"}
        # Define the input format for the original data
        self.input = {
            "question": "str",
            "sql": "str",
            "schema": "str",
            "glossary": "str",
        }
        # Call the initializer of the BaseGenerator class
        super().__init__(
            client=client,
            model=model,
            name=name or "EdgeCaseQuestionGenerator",
            role=role
            or "You are an expert making questions that focuses on unusual or extreme conditions, such as outliers, rare events, or boundary cases.",
            instruction=instruction,
            output_type=output_type,
            **kwargs,
        )

    def postprocess(self, prompt_obj):
        """
        Processes the response from the model to convert it into a list of questions
        focusing on edge cases.

        Args:
            prompt_obj: The prompt object that includes the response from the model.

        Returns:
            The prompt object with the processed response added.
        """
        # Check if the response is empty and warn if it is
        if not prompt_obj.response:
            print(
                f"Warning: Empty response from model for question: {prompt_obj.data.get('question', 'Unknown')}"
            )
            return prompt_obj

        try:
            # Initialize an empty list of edges
            edges = []
            # Add questions to the edges list if they exist in the response
            if prompt_obj.response.get("q1"):
                edges.append({"question": prompt_obj.response["q1"]})
            if prompt_obj.response.get("q2"):
                edges.append({"question": prompt_obj.response["q2"]})

            # Set the processed edges in the prompt object response
            prompt_obj.response = {"edges": edges}
            return prompt_obj

        except Exception as e:
            # Handle exceptions by logging error details
            print(f"Error processing model response: {str(e)}")
            print(f"Original question: {prompt_obj.data.get('question', 'Unknown')}")
            print(f"Model response: {prompt_obj.response}")
            # Return an empty edges list in case of error
            prompt_obj.response = {"edges": []}
            return prompt_obj