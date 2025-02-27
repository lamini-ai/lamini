from pydantic import BaseModel
from typing import Dict, List, Any
import pandas as pd
import numpy as np
import logging
import sqlite3

from lamini.experiment.base_validator import BaseValidator
from lamini.experiment.generators import BaseSQLGenerator


class FactualityValidator(BaseValidator):

    instruction = """
    You are a helpful assistant checking if the information is factual, or could've been hallucinated.
    """

    def __init__(
        self,
        model: str,
        instruction_metadata: List[str],
        name: str = "FactualityValidator",
        instruction: str = instruction,
        output_type: Dict = {"thinking_steps": "str", "is_factual": "bool"},
        is_valid_field: str = "is_factual",
    ):

        if instruction_metadata:
            # Append the metadata to the instruction as curly brace placeholders
            for metadata in instruction_metadata:
                instruction += f"\n\n{{{metadata}}}"

        super().__init__(
            name=name,
            model=model,
            instruction=instruction,
            output_type=output_type,
            is_valid_field=is_valid_field,
        )


class BaseSQLValidator(BaseSQLGenerator, BaseValidator):
    """Base class for SQL validation that combines generator and validator capabilities."""

    def __init__(
        self,
        model,
        client=None,
        db_type=None,
        db_params=None,
        name="BaseSQLValidator",
        instruction=None,
        output_type=None,
        **kwargs,
    ):
        # Initialize BaseSQLGenerator first
        BaseSQLGenerator.__init__(
            self,
            model=model,
            client=client,
            db_type=db_type,
            db_params=db_params,
            name=name,
            instruction=instruction,
            output_type=output_type,
            **kwargs,
        )

        # Initialize BaseValidator
        BaseValidator.__init__(
            self,
            model=model,
            client=client,
            name=name,
            instruction=instruction,
            output_type=output_type,
            **kwargs,
        )

    def __call__(self, prompt_obj, debug=False):
        """Base validation method to be implemented by subclasses."""
        return BaseValidator.__call__(self, prompt_obj, debug)


class SQLValidator(BaseSQLValidator):
    """Simple SQL validator that checks if a query can execute."""

    def __init__(
        self,
        model,
        db_type="sqlite",
        db_params=None,
        name="SQLValidator",
        sql_key="sql_query",  # key in prompt object data to get the SQL query from
        is_valid_field="is_valid",
        **kwargs,
    ):
        super().__init__(
            model=model,
            name=name,
            db_type=db_type,
            db_params=db_params,
            is_valid_field=is_valid_field,
            **kwargs,
        )

        self.sql_key = sql_key

        self.input[self.sql_key] = "str"

    def __call__(self, prompt_obj, debug=False):
        """
        Check if SQL query is valid by attempting to execute it.
        Note: This is a deterministic validator, so it does not call an LLM.
        It will look for the sql_key in the prompt object data.
        """
        if not prompt_obj.data.get(self.sql_key):
            prompt_obj.response = self.create_error_response("No SQL query provided")
            return prompt_obj

        query = prompt_obj.data[self.sql_key].strip()
        if not query:
            prompt_obj.response = self.create_error_response("Empty SQL query")
            return prompt_obj

        if not query.endswith(';'):
            query += ';'

        db_can_execute = False

        try:
            cursor = self.conn.cursor()
            cursor.execute(f"EXPLAIN QUERY PLAN {query}")
            prompt_obj.response = self.create_success_response()
            db_can_execute = True
        except Exception as e:
            prompt_obj.response = self.create_error_response(str(e))

        if not db_can_execute:
            new_query = self.fix_invalid_syntax(query)

            if new_query:
                if not new_query.endswith(';'):
                    new_query += ';'

                try:
                    cursor.execute(f"EXPLAIN QUERY PLAN {new_query}")
                    prompt_obj.data[self.sql_key] = new_query
                    prompt_obj.response = self.create_success_response()
                    db_can_execute = True
                except:
                    pass

        return prompt_obj

    def create_error_response(self, error_msg):
        """Create error response"""
        return {
            self.is_valid_field: False,
            "error": error_msg,
            "explanation": f"Query is invalid: {error_msg}"
        }

    def create_success_response(self):
        """Create success response"""
        return {
            self.is_valid_field: True,
            "error": None,
            "explanation": "Query is valid and can be executed"
        }

    def fix_invalid_syntax(self, query):
        i_codeblock = query.find('```')

        if i_codeblock != -1:
            query = query[i_codeblock:]

        queryl = query.lower()

        if not query.lower().startswith('with'):
            init_tokens = ['select', 'with']
            i_start = None

            for init_token in init_tokens:
                i_start = queryl.find(init_token)
                if i_start != -1:
                    break
                if not i_start:
                    return None

            query = query[i_start:]

        i_semicolon = query.find(';')

        if i_semicolon != -1:
            query = query[:i_semicolon + 1]

        return query

class SQLScoreValidator(BaseSQLValidator):
    """
    Validator that uses LLM to compare SQL query results with reference data.
    The sql_key refers to the generated SQL query, in PromptObject.data.
    The reference_df refers to the reference SQL query, in PromptObject.data.
    """

    def __init__(
        self,
        model,
        db_type="sqlite",
        db_params=None,
        name="SQLScoreValidator",
        instruction=None,
        output_type={"explanation": "str", "is_similar": "bool"},
        is_valid_field="is_similar",
        sql_key="sql_query",
        reference_df: pd.DataFrame = None,
        reference_sql_key=None,
        **kwargs,
    ):
        # Need either the reference_df or the reference_sql_key
        if reference_df is None and reference_sql_key is None:
            raise ValueError(
                "Either reference_df or reference_sql_key must be provided to compare against the generated SQL's results."
            )

        # Default instruction for comparison
        if instruction is None:
            instruction = """Compare the following two dataframes. They are similar if they are almost identical, or if they convey the same information about the given dataset.
            ========== Dataframe 1 =========
            {generated_df}
            ========== Dataframe 2 =========
            {reference_df}
            Can you tell me if these dataframes are similar or convey exactly similar information?
            Also write explanation in one or two lines about why you think the data frames are similar or otherwise.
            """

        super().__init__(
            model=model,
            db_type=db_type,
            db_params=db_params,
            name=name,
            instruction=instruction,
            output_type=output_type,
            is_valid_field=is_valid_field,
            **kwargs,
        )

        self.sql_key = sql_key
        self.reference_df = reference_df
        self.reference_sql_key = reference_sql_key

    def _execute_query(self, query):
        """Execute SQL query and return results as DataFrame."""
        if not self.conn:
            raise RuntimeError("No database connection available")

        try:
            return pd.read_sql_query(query, self.conn)
        except Exception as e:
            self.logger.error(f"Error executing query: {str(e)}")
            raise

    def __call__(self, prompt_obj, debug=False):
        """Compare query results with reference data using LLM."""
        if debug:
            self.logger.setLevel(logging.DEBUG)

        # Check if sql_key is in the data
        if self.sql_key not in prompt_obj.data:
            raise ValueError(
                f"{self.sql_key} not found in prompt object data. Set the right data key using the sql_key parameter."
            )
        if (
            self.reference_sql_key is not None
            and self.reference_sql_key not in prompt_obj.data
        ):
            raise ValueError(
                f"{self.reference_sql_key} not found in prompt object data. Set the right data key using the reference_sql_key parameter, or set it to None and use the reference_df parameter."
            )

        generated_query = prompt_obj.data[self.sql_key]

        if self.reference_df is not None:
            reference_df = self.reference_df
        else:
            reference_query = prompt_obj.data[self.reference_sql_key]
            try:
                reference_result = self._execute_query(reference_query)
                reference_df = pd.DataFrame(reference_result)
            except Exception as e:
                raise ValueError(
                    f"Error executing reference query: {str(e)}. Set a valid reference query using the reference_sql_key parameter."
                )

        try:
            # Execute query
            generated_df = self._execute_query(generated_query)

            # Update prompt object with results for comparison
            prompt_obj.data["generated_df"] = str(generated_df).lower()
            prompt_obj.data["reference_df"] = str(reference_df).lower()

            # The LLM comparison will happen in the parent class's processing
            # which uses the instruction template with the results
            return super().__call__(prompt_obj, debug)

        except Exception as e:
            prompt_obj.response = {"explanation": str(e), "similar": False}
            return prompt_obj
