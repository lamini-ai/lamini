from lamini.experiment.generators import BaseGenerator

class ParaphrasingQuestionGenerator(BaseGenerator):
    """
    ParaphrasingGenerator is responsible for generating paraphrases
    of given input questions while maintaining their original meaning
    and context. It utilizes a given model to perform paraphrasing tasks.
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
        Initializes the ParaphrasingGenerator with the provided parameters.

        :param model: The model used for generating paraphrases.
        :param client: (Optional) The client instance.
        :param name: (Optional) Name of the generator.
        :param role: (Optional) Role description for the generator.
        :param instruction: (Optional) Instruction for generating paraphrases.
        :param output_type: (Optional) The output type configuration.
        :param kwargs: (Optional) Additional keyword arguments.
        """
        instruction = """
            Task: Given the following original question, SQL query, schema, and glossary, generate 2 paraphrases that maintain the same intent but use different sentence structures, word choices, and phrasing. The new questions should preserve the original context and meaning.

            Input Format:

                Original Question: {question}
                Original SQL: {sql}
                Schema: {schema}
                Glossary: {glossary}

            Guidelines for Generating New Questions:

                Synonyms and Alternative Phrasing: Use synonyms and alternative ways of phrasing the question while maintaining the original meaning. This could involve changing sentence structure, swapping words with their synonyms, or rewording the question for variety.
                Preserve Context and Meaning: Ensure that the paraphrased questions preserve the original context and meaning, so the same SQL query can be used to answer them.
                Answerability with Schema and Glossary: The paraphrased questions should still be answerable using the schema and glossary provided. Ensure the new phrasing does not introduce ambiguity or misinterpretation that would make the question unanswerable with the schema.
                SQL Query Validity: The SQL queries associated with the paraphrased questions should remain syntactically valid and consistent with the provided schema.

            Please provide ONLY the new questions (in a clear, concise format) without any explanation or markdown formatting.       """

        # Define the expected output type
        output_type = {"q1": "string", "q2": "string"}

        # Define the expected input types
        self.input = {
            "question": "str",
            "sql": "str",
            "schema": "str",
            "glossary": "str",
        }

        # Initialize the BaseGenerator with specified parameters
        super().__init__(
            client=client,
            model=model,
            name=name or "ParaphrasingQuestionGenerator",
            role=role or "You are an expert at identifying and paraphrasing questions.",
            instruction=instruction,
            output_type=output_type,
            **kwargs,
        )

    def postprocess(self, prompt_obj):
        """
        Processes the model response to extract and format the paraphrased questions.

        :param prompt_obj: The prompt object containing the input data and model response.
        :return: The modified prompt object with processed paraphrases.
        """
        # Handle empty response scenario
        if not prompt_obj.response:
            print(
                f"Warning: Empty response from model for question: {prompt_obj.data.get('question', 'Unknown')}"
            )
            return prompt_obj

        try:
            paraphrases = []
            # Append paraphrased questions if available
            if prompt_obj.response.get("q1"):
                paraphrases.append({"question": prompt_obj.response["q1"]})
            if prompt_obj.response.get("q2"):
                paraphrases.append({"question": prompt_obj.response["q2"]})

            # Update the prompt object response with paraphrases
            prompt_obj.response = {"paraphrases": paraphrases}
            return prompt_obj

        except Exception as e:
            # Handle exceptions during processing
            print(f"Error processing model response: {str(e)}")
            print(f"Original question: {prompt_obj.data.get('question', 'Unknown')}")
            print(f"Model response: {prompt_obj.response}")
            prompt_obj.response = {"paraphrases": []}
            return prompt_obj