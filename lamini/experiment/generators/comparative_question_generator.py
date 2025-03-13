from lamini.experiment.generators import BaseGenerator

class ComparativeQuestionGenerator(BaseGenerator):
    """Generator for creating comparative analysis questions.

    Transforms single questions into pairs of comparative questions that explore
    relationships between different groups, time periods, or conditions.

    Note:
        Generated questions maintain the original query's intent while adding
        comparative elements.
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
        # Instructions for generating comparative questions and SQL queries
        instruction = """
            Task: Given the following original question, SQL query, schema, and glossary, generate 2 new questions that involve comparisons between different groups, time periods, or conditions. The queries should reflect the original intent while incorporating comparative language such as "versus," "compared to," or "relative difference."

            Input Format:

                Original Question: {question}
                Original SQL: {sql}
                Schema: {schema}
                Glossary: {glossary}

            Guidelines for Generating New Questions:

                Incorporate Comparative Language: Modify the questions to include comparative phrases like "versus," "compared to," or "relative difference." These should frame the query as a comparison between two or more groups, time periods, or conditions.
                Adjust Grouping or Filtering: Modify the SQL query's grouping, filtering, or aggregation criteria to reflect the comparative nature of the question. Ensure the comparison is meaningful and logically consistent with the schema.
                Maintain Original Query Intent: While introducing a comparative element, ensure the new question still aligns with the original query's intent and logic.
                Answerability with Schema and Glossary: Ensure that the new questions can be answered using the available schema and glossary. The SQL queries should use valid tables and columns as per the schema.
                SQL Query Validity: The generated SQL queries must be syntactically correct and should work based on the schema provided.

            Please provide ONLY the new questions (in a clear, concise format) without any explanation or markdown formatting.
        """

        # Define the expected output structure
        output_type = {"q1": "string", "q2": "string"}

        # Define the expected input structure
        self.input = {
            "question": "str",
            "sql": "str",
            "schema": "str",
            "glossary": "str",
        }

        # Initialize the base generator with provided parameters
        super().__init__(
            client=client,
            model=model,
            name=name or "ComparativeQuestionGenerator",
            role=role
            or "You are an expert making questions that compare two groups, periods or conditions",
            instruction=instruction,
            output_type=output_type,
            **kwargs,
        )

    def postprocess(self, prompt_obj):
        """Process model response into comparative questions.

        Converts the model's response into a structured format of comparative
        questions with error handling.

        Args:
            prompt_obj (PromptObject): Contains model response with q1 and q2

        Returns:
            PromptObject: Updated with processed comparative questions in comps field

        Note:
            Handles empty responses and processing errors gracefully
        """
        if not prompt_obj.response:
            # Handle cases where the model returns an empty response
            print(
                f"Warning: Empty response from model for question: {prompt_obj.data.get('question', 'Unknown')}"
            )
            return prompt_obj

        try:
            comps = []
            # Add questions to the list if they exist in the model response
            if prompt_obj.response.get("q1"):
                comps.append({"question": prompt_obj.response["q1"]})
            if prompt_obj.response.get("q2"):
                comps.append({"question": prompt_obj.response["q2"]})

            # Update the response structure
            prompt_obj.response = {"comps": comps}
            return prompt_obj

        except Exception as e:
            # Handle and log any errors in processing the model response
            print(f"Error processing model response: {str(e)}")
            print(f"Original question: {prompt_obj.data.get('question', 'Unknown')}")
            print(f"Model response: {prompt_obj.response}")
            prompt_obj.response = {"comps": []}
            return prompt_obj