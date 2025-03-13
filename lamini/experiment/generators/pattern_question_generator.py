from lamini.experiment.generators import BaseGenerator

class PatternQuestionGenerator(BaseGenerator):
    """
    PatternGenerator extends the functionality of BaseGenerator to create new
    SQL questions based on an existing question, its SQL query, a schema, and a glossary.
    The generator ensures that the new questions follow a similar structure and can be
    answered using the provided schema, adhering to specific guidelines.
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
        Initializes the PatternGenerator with specific model parameters and instructions.

        :param model: The model used for generating patterns.
        :param client: Optional client instance.
        :param name: Optional name of the generator.
        :param role: Describes the role of the pattern generator.
        :param instruction: Instructions for the model.
        :param output_type: Expected output structure.
        :param kwargs: Additional keyword arguments.
        """
        instruction = """
            Task: Given the following original question, SQL query, schema, and glossary, generate 2 new questions that follow a similar structure and can be answered using the provided schema. The output SQL should be syntactically valid and correspond to the business context.

            Input Format:

                Original Question: {question}
                Original SQL: {sql}
                Schema: {schema}
                Glossary: {glossary}

            Guidelines for Generating New Questions:

                Contextual Relevance: Ensure that the new questions reflect a business context similar to the original question. The context may be different but should stay within the realm of the schema's purpose.
                SQL Query Validity: Ensure that the SQL queries generated are syntactically correct and align with the schema provided. Only valid queries should be generated.
                Complexity Level: The new questions should have a similar complexity level to the original question. Keep in mind the difficulty of the query when generating the new questions.
                Answerability with Schema: The generated questions should be answerable with the available schema, and you should only use the tables and columns described in the schema to generate your queries.

            Please provide ONLY the new questions (in a clear, concise format) without any explanation or markdown formatting."""

        output_type = {"q1": "string", "q2": "string"}

        self.input = {
            "question": "str",
            "sql": "str",
            "schema": "str",
            "glossary": "str",
        }

        super().__init__(
            client=client,
            model=model,
            name=name or "PatternQuestionGenerator",
            role=role
            or "You are an expert at identifying and replicating question patterns.",
            instruction=instruction,
            output_type=output_type,
            **kwargs,
        )

    def postprocess(self, prompt_obj):
        """
        Processes the model response to extract question variations.

        :param prompt_obj: The object containing the model's response and input data.
        :return: Modified prompt_obj with variations or an empty list if an error occurs.
        """
        if not prompt_obj.response:
            print(
                f"Warning: Empty response from model for question: {prompt_obj.data.get('question', 'Unknown')}"
            )
            return prompt_obj

        try:
            variations = []
            if prompt_obj.response.get("q1"):
                variations.append({"question": prompt_obj.response["q1"]})
            if prompt_obj.response.get("q2"):
                variations.append({"question": prompt_obj.response["q2"]})

            prompt_obj.response = {"variations": variations}
            return prompt_obj

        except Exception as e:
            print(f"Error processing model response: {str(e)}")
            print(f"Original question: {prompt_obj.data.get('question', 'Unknown')}")
            print(f"Model response: {prompt_obj.response}")
            prompt_obj.response = {"variations": []}
            return prompt_obj