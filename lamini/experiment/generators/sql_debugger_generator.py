from lamini.experiment.generators import BaseSQLGenerator
class SQLDebuggerGenerator(BaseSQLGenerator):
    def __init__(
        self,
        model,
        output_type=None,
        instruction=None,
        client=None,
        schema=None,
        db_type=None,
        db_params=None,
    ):
        instruction = (
            instruction
            or """SQL Query Debugger:
        You are provided with a user's question, SQL query, error information, schema, and glossary.
        
        Question: {sub_question}
        The query to fix is: {error_sql}
        Error Message: {error_message}
        Error Explanation: {error_explanation}
        
        Database Schema:
        {schema}
        
        Glossary:
        {glossary}
        
        Please provide ONLY the corrected SQL query without any explanation or markdown formatting.
        The query should reference only schema columns where applicable
        """

        )
        output_type = output_type or {
            "corrected_sql": "str"
        }
        
        super().__init__(
            client=client,
            model=model,
            schema=schema,
            db_type=db_type,
            db_params=db_params,
            name="SQLDebugger",
            role="You are a SQL debugging expert. Output only the corrected SQL query without any explanation.",
            instruction=instruction,
            output_type=output_type,
        )

    def __call__(self, prompt_obj, debug=False):
        """Process the debugging request with error handling"""
        required_fields = [
            "error_message",
            "error_explanation",
            "error_sql",
            "sub_question",
            "glossary"
        ]
        
        for field in required_fields:
            if field not in prompt_obj.data:
                if field in ["sub_question", "glossary"]:
                    prompt_obj.data[field] = "Not provided"
                else:
                    raise ValueError(f"Missing required field: {field}")

        prompt_obj.data["schema"] = self.schema
        return super().__call__(prompt_obj, debug)

    def postprocess(self, prompt_obj):
        """Process SQL query response"""
        if not prompt_obj.response:
            prompt_obj.response = {
                "corrected_sql": None
            }
            return prompt_obj

        if "corrected_sql" in prompt_obj.response:
            sql = prompt_obj.response["corrected_sql"].strip()
            if not sql.endswith(';'):
                sql += ';'
            prompt_obj.response["corrected_sql"] = sql

        return prompt_obj
