import logging

import pandas as pd

from lamini.experiment.validators import BaseSQLValidator

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