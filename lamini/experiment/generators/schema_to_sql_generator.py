from lamini.experiment.generators import BaseSQLGenerator

class SchemaToSQLGenerator(BaseSQLGenerator):
    """Generates SQL for questions using database schema and context."""
    
    def __init__(
        self,
        model,
        client=None,
        schema=None,
        db_type=None,
        db_params=None,
        name=None,
        role=None,
        instruction=None,
        output_type=None,
        **kwargs,
    ):
        """
        Initialize the SQL generator.
        
        Args:
            model: The model to use for SQL generation
            client: Optional API client
            schema: Database schema
            db_type: Database type (e.g., 'sqlite')
            db_params: Database connection parameters
            name: Generator name
            role: System role prompt
            instruction: Custom instruction prompt
            output_type: Expected output format
            **kwargs: Additional parameters
        """

        instruction = instruction or """
            Schema: {schema}
            Glossary: {glossary}
            Original Question: {original_question}
            Original SQL: {original_sql}
            Sub Question: {question}
            Write a complete SQL query and make sure that it references only schema columns where applicable
        """
        
        # Define expected output format
        output_type = output_type or {"sql_query": "str"}
        
        # Initialize with base SQL generator
        super().__init__(
            client=client,
            model=model,
            schema=schema,
            db_type=db_type,
            db_params=db_params,
            name=name or "SchemaToSQLGenerator",
            role=role or "You are a SQL expert who writes precise SQL queries.",
            instruction=instruction,
            output_type=output_type,
            **kwargs,
        )
    
    def preprocess(self, prompt_obj):
        """
        Prepare the prompt object by ensuring all required fields exist.
        
        Args:
            prompt_obj: The prompt object to process
            
        Returns:
            Processed prompt object
        """
        # Make sure all required template fields exist
        if "original_question" not in prompt_obj.data:
            # Use the main question as original if not provided
            prompt_obj.data["original_question"] = prompt_obj.data.get("question", "")
            
        if "original_sql" not in prompt_obj.data:
            # Use empty string if not provided
            prompt_obj.data["original_sql"] = ""
            
        if "schema" not in prompt_obj.data and hasattr(self, "schema"):
            prompt_obj.data["schema"] = self.schema
            
        if "glossary" not in prompt_obj.data:
            prompt_obj.data["glossary"] = ""
            
        # Make sure we have a question field
        if "question" not in prompt_obj.data and "sub_question" in prompt_obj.data:
            prompt_obj.data["question"] = prompt_obj.data["sub_question"]
        
        return prompt_obj
    
    def postprocess(self, prompt_obj):
        """
        Process the model's response.
        
        Args:
            prompt_obj: The prompt object with model response
            
        Returns:
            Processed prompt object
        """
        # Check if response exists and contains sql_query
        if not prompt_obj.response or "sql_query" not in prompt_obj.response:
            return prompt_obj
        
        # Clean and format the SQL query
        sql = prompt_obj.response["sql_query"].strip()
        if not sql.endswith(';'):
            sql += ';'
        
        prompt_obj.response["sql_query"] = sql
        prompt_obj.data["generated_sql"] = sql
        
        return prompt_obj
    