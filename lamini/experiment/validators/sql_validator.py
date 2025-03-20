from sqlalchemy import create_engine, inspect

from lamini.experiment.validators import (
    BaseSQLValidator,
    sql_autofix,
)
    
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
        skip_autofixes=[],
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
        self.db_type = db_type
        self.skip_autofixes = skip_autofixes
        self.input[self.sql_key] = "str"
        self.col_table_map = self.get_db_col_info()

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

        valid_autofixes = set(['extract_sql', 'fix_statement_count', 'fix_column'])
        autofixes = valid_autofixes - set(self.skip_autofixes)

        if len(autofixes) + len(self.skip_autofixes) > len(valid_autofixes):
            raise Exception(f"Valid autofix options are {valid_autofixes}. You have used {self.skip_autofixes}")

        if not db_can_execute and len(autofixes) > 0:
            if 'extract_sql' in autofixes and query:
                query = sql_autofix.extract_sql_part(query, self.db_type)
            if 'fix_statement_count' in autofixes and query:
                if 'gpt4' in self.model and self.db_type == 'snowflake':
                    query = sql_autofix.fix_stmt_count(query, self.db_type)
            if 'fix_column' in autofixes and query:
                query = sql_autofix.fix_invalid_col(query, self.col_table_map, self.db_type)

            if query:
                if not query.endswith(';'):
                    query += ';'            

                try:
                    cursor.execute(f"EXPLAIN QUERY PLAN {query}")
                    prompt_obj.data[self.sql_key] = query
                    prompt_obj.response = self.create_success_response()
                    db_can_execute = True
                except:
                    pass

        return prompt_obj

    def get_db_col_info(self):
        if self.db_type == 'sqlite':
            engine = create_engine("sqlite://", creator=lambda: self.conn)
        else:
            raise Exception(f"DB type {self.db_type} not supported")

        inspector = inspect(engine)
        col_table_map = {}

        for table in inspector.get_table_names():
            columns = inspector.get_columns(table)
            table = table.upper()

            for c in columns:
                c = c['name'].upper()
                if c in col_table_map:
                    col_table_map[c].add(table)
                else:
                    col_table_map[c] = set()

        return col_table_map

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

