import sqlite3

from lamini.experiment.generators.base_generator import BaseGenerator

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