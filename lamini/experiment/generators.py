# ---- Lamini Copy Right ----
# © 2025 Lamini. All rights reserved.
# This software is licensed pursuant to the terms of the Lamini License Agreement.
# Unauthorized copying of this file, via any medium is strictly prohibited.

# ---- Default Generators ----
# These are default generator classes provided by Lamini for various purposes.
# Each generator is designed to transform inputs in specific ways using predefined roles and instructions.
# They facilitate operations such as concept extraction, SQL interpretation, question generation,
# and more.

from lamini.experiment.base_generator import BaseGenerator
from lamini.generation.base_prompt_object import PromptObject
from copy import deepcopy
import sqlite3


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


class QuestionToConceptGenerator(BaseGenerator):
    """
    Takes a single question and returns a single concept that is relevant to the question.
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
        name = name or "QuestionToConceptGenerator"
        role = (
            role
            or "You are a helpful assistant that takes a single question and returns a single concept that is relevant to the question."
        )
        instruction = (
            instruction
            or """Given the question, return a single concept that is relevant to the question.
        Question:
        {question}

        Concept:"""
        )

        output_type = output_type or {"concept": "str"}

        super().__init__(
            client=client,
            model=model,
            name=name,
            role=role,
            instruction=instruction,
            output_type=output_type,
            **kwargs,
        )


class ConceptToSQLInterpretationGenerator(BaseGenerator):
    """
    Takes a concept and returns the SQL interpretation of the concept.
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
        name = name or "ConceptToSQLInterpretationGenerator"
        role = (
            role
            or "You are a helpful assistant that takes a concept and returns the SQL interpretation of the concept, which is just the calculation of the concept in a SQL fragment."
        )
        instruction = (
            instruction
            or """Given the concept, return the SQL interpretation of the concept, which is just the calculation of the concept in a SQL fragment.
        Concept:
        {concept}

        SQL Interpretation:"""
        )

        output_type = output_type or {"sql": "str"}

        super().__init__(
            client=client,
            model=model,
            name=name,
            role=role,
            instruction=instruction,
            output_type=output_type,
            **kwargs,
        )


class BaseSQLGenerator(BaseGenerator):
    """Base class for SQL-related generators with database functionality.

    Provides common database connection and schema management capabilities for
    SQL-focused generators. Supports multiple database types with extensible
    connection handling.

    Attributes:
        SUPPORTED_DB_TYPES (dict): Mapping of database types to connection factories
        metadata_keys (list): Required schema metadata key
        conn: Active database connection
        schema (str): Database schema information
    """

    SUPPORTED_DB_TYPES = {
        "sqlite": lambda params: sqlite3.connect(
            params["database"] if isinstance(params, dict) else params
        )
    }
    metadata_keys = ["schema"]  # required before super init

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
        # Initialize conn before calling super().__init__ to ensure it exists
        self.conn = None

        super().__init__(
            client=client,
            model=model,
            name=name or "BaseSQLGenerator",
            role=role or "You are a SQL expert.",
            instruction=instruction or "Base SQL Generator",
            output_type=output_type,  # set by subclasses
            **kwargs,
        )

        # Initialize database connection if params provided
        if db_type and db_params:
            self._initialize_db(db_type, db_params)
            self.schema = self._get_schema_from_db() if not schema else schema

            # Add db_type and db_params to input, instead of schema
            self.input["db_type"] = db_type
            self.input["db_params"] = db_params
            self.input.pop("schema", None)
        elif schema:
            self.schema = schema
        else:
            raise ValueError(
                "Must provide schema string, or db_type and db_params to connect to a database and extract the schema"
            )

    @classmethod
    def add_db_support(cls, db_type, connection_factory):
        """Add support for a new database type.

        Args:
            db_type (str): Identifier for the database type
            connection_factory (Callable): Function to create database connections
        """
        cls.SUPPORTED_DB_TYPES[db_type] = connection_factory

    def _initialize_db(self, db_type, db_params):
        """Initialize connection to the specified database.

        Args:
            db_type (str): Type of database to connect to
            db_params (Union[str, dict]): Connection parameters

        Raises:
            ValueError: If db_type is not supported
            Exception: If connection fails
        """
        if db_type not in self.SUPPORTED_DB_TYPES:
            raise ValueError(
                f"Unsupported database type: {db_type}. "
                f"Supported types are: {list(self.SUPPORTED_DB_TYPES.keys())}"
            )

        try:
            connection_factory = self.SUPPORTED_DB_TYPES[db_type]
            self.conn = connection_factory(db_params)
        except Exception as e:
            print(f"Error initializing database: {e}")
            raise

    def _get_schema_from_db(self):
        """Extract schema information from connected database."""
        if not self.conn:
            raise RuntimeError("No database connection available")

        cur = self.conn.cursor()
        schema_string = ""

        # Get list of tables
        cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cur.fetchall()

        for table in tables:
            table_name = table[0]
            schema_string += f"Schema for table '{table_name}':\n"

            # Get table schema
            cur.execute(f"PRAGMA table_info({table_name});")
            columns = cur.fetchall()

            # Get sample row
            cur.execute(f"SELECT * FROM {table_name} LIMIT 1;")
            sample_row = cur.fetchone()
            sample_row = list(sample_row) if sample_row is not None else []

            # Format column information
            for index, column in enumerate(columns):
                column_name = column[1]
                column_type = column[2]
                sample_value = sample_row[index] if index < len(sample_row) else None
                schema_string += f"Column: {column_name} | Type: {column_type} | Sample Value: {sample_value}\n"

            schema_string += "\n"

        return schema_string

    def execute_query(self, query):
        """Execute a SQL query if database connection is available."""
        if not self.conn:
            raise RuntimeError(
                "No database connection available. Initialize with db_type and db_params to execute queries."
            )

        try:
            cur = self.conn.cursor()
            cur.execute(query)
            return cur.fetchall()
        except Exception as e:
            print(f"Error executing query: {e}")
            raise

    def __del__(self):
        """Cleanup database connection if it exists."""
        if self.conn:
            self.conn.close()


class SchemaToSQLGenerator(BaseSQLGenerator):
    def __init__(
        self,
        model,
        client=None,
        schema=None,
        db_type=None,
        db_params=None,
        schema_definitions=None,
        terms_in_questions=None,
        output_type=None,
        instruction=None,
        name="SchemaToSQLGenerator",
        role="You are a SQL expert with 15 years of experience writing complex SQL queries.",
        execute_sql=False,
    ):
        instruction = (
            instruction
            or """Consider the following database schema:
        {schema}"""
        )

        if schema_definitions:
            instruction += """
            Here are the definitions of terms used in the schema:
            {schema_definitions}
            """

        if terms_in_questions:
            instruction += """
            Here is a glossary of terms used in the questions:
            {terms_in_questions}
            """

        instruction += """
        Write a sqlite query to answer the following question: {question}
        
        Now, let's think step by step:
        1. Analyze the Question: Understand what information is being requested
        2. Map to Database Schema: Identify relevant tables and columns
        3. Construct SQL Query: Write an optimized query that answers the question
        4. Validate: Ensure the query is correct and efficient
        """

        output_type = output_type or {
            "schema_to_sql_thinking_steps": "str",
            "sql_query": "str",
        }

        super().__init__(
            client=client,
            model=model,
            schema=schema,
            db_type=db_type,
            db_params=db_params,
            name=name,
            role=role,
            instruction=instruction,
            output_type=output_type,
        )

        self.schema = schema
        self.schema_definitions = schema_definitions
        self.terms_in_questions = terms_in_questions

        self.execute_sql = execute_sql

    def __call__(self, prompt_obj, debug=False):
        # Add schema and optional components to prompt
        prompt_obj.data["schema"] = self.schema
        if self.schema_definitions:
            prompt_obj.data["schema_definitions"] = self.schema_definitions
        if self.terms_in_questions:
            prompt_obj.data["terms_in_questions"] = self.terms_in_questions
        result = super().__call__(prompt_obj, debug)

        if self.execute_sql:
            if hasattr(self, "conn"):
                query = result.response["sql_query"]
                try:
                    result.response["sql_execution"] = self.execute_query(query)
                    result.data["execution_status"] = "success"
                except Exception as e:
                    result.data["execution_status"] = "failed"

                    # For SQLDebugger, if it's the next generator in pipeline
                    result.data["error_sql"] = query
                    result.data["error_message"] = str(e)
            else:
                self.logger.warning(
                    "No database connection available. Cannot execute SQL query. Initialize with db_type and db_params to execute queries."
                )

        return result


class SQLDebuggerGenerator(BaseSQLGenerator):
    """Generator for debugging and fixing SQL queries.

    Analyzes SQL queries that produced errors and generates corrected versions
    based on the error message and database schema.

    Note:
        Includes specific rules for common SQL issues like case sensitivity,
        aliasing, and GROUP BY clauses.
    """

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
        You are provided with a user's SQL query and the following error message:
        Error: {error_message}

        Database Schema:
        {schema}

        Fix any issues in the SQL query using the provided schema. Apply these rules:
        1. String Comparisons: Use LOWER() for case-insensitive matches unless exact match needed
        2. Aliasing: Ensure calculated fields have appropriate aliases
        3. GROUP BY: Include all non-aggregated columns
        4. Calculations: Use parentheses to clarify order of operations
        5. WHERE Clauses: Verify conditions match column types
        6. Date Functions: Use appropriate date formatting
        7. CASE Statements: Handle all possible cases
        8. Performance: Avoid redundant operations

        The query to fix is:
        {error_sql}

        Think step by step. Then, provide the corrected SQL query.
        """
        )

        output_type = output_type or {
            "sql_debugger_thinking_steps": "str",
            "corrected_sql": "str",
        }

        super().__init__(
            client=client,
            model=model,
            schema=schema,
            db_type=db_type,
            db_params=db_params,
            name="SQLDebugger",
            role="You are a SQL debugging expert with 30 years of experience.",
            instruction=instruction,
            output_type=output_type,
        )

    def __call__(self, prompt_obj, debug=False):
        """Execute the SQL debugger on a prompt object.

        Adds schema information to the prompt object before processing.

        Args:
            prompt_obj (PromptObject): Contains error_message and error_sql
            debug (bool, optional): Enable debug logging. Defaults to False.

        Returns:
            PromptObject: Contains corrected SQL and debugging steps
        """
        prompt_obj.data["schema"] = self.schema
        return super().__call__(prompt_obj, debug)


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
        Processes the model's response to extract and return a list of sub-questions as PromptObjects.

        Args:
            prompt_obj: An instance of PromptObject containing the model's response.

        Returns:
            A list of PromptObject instances, each representing a sub-question.
        """

        # Handle case where response is empty
        if not prompt_obj.response:
            print(
                f"Warning: Empty response from model for question: {prompt_obj.data.get('question', 'Unknown')}"
            )
            return []

        try:
            # Extract sub-questions from the model's response
            questions = [
                prompt_obj.response.get("q1"),
                prompt_obj.response.get("q2"),
                prompt_obj.response.get("q3"),
            ]

            # Filter out any None values in the list of questions
            questions = [q for q in questions if q]

            # Check if any valid questions were extracted
            if not questions:
                print(
                    f"Warning: No valid questions extracted from model response for: {prompt_obj.data.get('question', 'Unknown')}"
                )
                return []

            result_prompts = []
            # Create PromptObject for each extracted sub-question
            for question in questions:
                new_prompt = PromptObject(prompt=question, data=prompt_obj.data.copy())
                new_prompt.data["sub_question"] = question
                result_prompts.append(new_prompt)

            return result_prompts

        except Exception as e:
            # Handle any exceptions during processing
            print(f"Error processing model response: {str(e)}")
            print(f"Original question: {prompt_obj.data.get('question', 'Unknown')}")
            print(f"Model response: {prompt_obj.response}")
            return []


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
