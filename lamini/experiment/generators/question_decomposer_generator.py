from lamini.experiment.generators import BaseGenerator
from lamini.generation.base_prompt_object import PromptObject

class QuestionDecomposerGenerator(BaseGenerator):
    """Generator for breaking down complex questions into sub-questions.

    Decomposes complex analytical questions into simpler, focused sub-questions
    that can be answered using the available schema and combined to address
    the original question.

    Note:
        Generates three sub-questions that maintain the original question's intent
        while simplifying the logical complexity.
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
        # Task instruction to guide the model on generating sub-questions.
        instruction = """
        Task: Given the following database schema, glossary, original complex question, and SQL query, break down the original question into 3 distinct sub-questions. These sub-questions should help answer the original question when combined.

        Input Format:

            Database Schema: {schema}
            Glossary Terms and Definitions: {glossary}
            Original Complex Question: {question}
            Original SQL Query: {sql}

        Guidelines for Sub-Questions:

            Self-contained and Specific: Each sub-question should be self-contained and focused on a specific part of the problem. Avoid overly broad or general questions.
            Consistency with Original Intent: The sub-questions should maintain the original question's intent and objectives. They should be logically connected and contribute to solving the original query.
            Use Glossary Terminology: Where applicable, use the glossary terms and definitions to ensure the language is consistent with the provided schema.
            Answerability with Schema: Ensure each sub-question can be answered using the available schema. Refer only to the tables and columns described in the schema when formulating the sub-questions.
            Avoid Complex Logic in Single Sub-Questions: Keep the logic of each sub-question simple. Do not combine multiple conditions or complex logic in a single sub-question.

        Please provide ONLY the sub-questions (in a clear, concise format) without any explanation or markdown formatting."""

        # Expected output types for sub-questions
        output_type = output_type or {"q1": "str", "q2": "str", "q3": "str"}

        # Input format requirements
        self.input = {
            "question": "str",
            "sql": "str",
            "schema": "str",
            "glossary": "str",
        }

        # Initialize the base generator with given parameters
        super().__init__(
            client=client,
            model=model,
            name=name or "QuestionDecomposerGenerator",
            role=role
            or "You are an expert at breaking down complex business questions into simpler ones.",
            instruction=instruction,
            output_type=output_type,
            **kwargs,
        )

    def postprocess(self, prompt_obj):
        """
        Processes the model's response to structure sub-questions 

        Args:
            prompt_obj: An instance of PromptObject containing the model's response.

        Returns:
            The original prompt_obj with its response 
        """
        # Handle case where response is empty
        if not prompt_obj.response:
            print(
                f"Warning: Empty response from model for question: {prompt_obj.data.get('question', 'Unknown')}"
            )
            prompt_obj.response = {"sub_questions": []}
            return prompt_obj

        try:
            # Extract sub-questions from the model's response
            sub_questions = []
            
            # Add each non-empty sub-question to the list
            if prompt_obj.response.get("q1"):
                sub_questions.append({"sub_question": prompt_obj.response["q1"]})
            if prompt_obj.response.get("q2"):
                sub_questions.append({"sub_question": prompt_obj.response["q2"]})
            if prompt_obj.response.get("q3"):
                sub_questions.append({"sub_question": prompt_obj.response["q3"]})

            prompt_obj.response = {"sub_questions": sub_questions}
            return prompt_obj

        except Exception as e:
            # Handle any exceptions during processing
            print(f"Error processing model response: {str(e)}")
            print(f"Original question: {prompt_obj.data.get('question', 'Unknown')}")
            print(f"Model response: {prompt_obj.response}")
            prompt_obj.response = {"sub_questions": []}
            return prompt_obj