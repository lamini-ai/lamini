import os

from lamini.generation.base_prompt_object import PromptObject
from lamini.experiment.base_generator import BaseGenerator
from lamini.experiment.base_validator import BaseValidator
from lamini.experiment.base_agentic_pipeline import BaseAgenticPipeline
from lamini.experiment.base_memory_experiment import BaseMemoryExperiment
from lamini.experiment.memory_rag_experiment import MemoryRAGExperiment
from lamini.api.openai_client import BaseOpenAIClient

from lamini.experiment.generators import (
    QuestionToConceptGenerator,
    ConceptToSQLInterpretationGenerator,
    QuestionsToConceptsGenerator,
    SchemaToSQLGenerator,
    SQLDebuggerGenerator,
    ComparativeQuestionGenerator,
    GranularQuestionGenerator,
    ParaphrasingQuestionGenerator,
    PatternQuestionGenerator,
    QuestionDecomposerGenerator,
    VariationQuestionGenerator,
    EdgeCaseQuestionGenerator,
)

from lamini.experiment.validators import (
    FactualityValidator,
    SQLValidator,
    SQLScoreValidator,
)

from lamini.index.lamini_index import LaminiIndex

import sqlite3
import pandas as pd
from datetime import datetime, timedelta


def connect_to_db(db_path):
    conn = sqlite3.connect(db_path)
    return conn


def setup_test_db():
    """Create a simple test database."""
    if os.path.exists("test.db"):
        os.remove("test.db")

    conn = sqlite3.connect("test.db")
    cursor = conn.cursor()

    # Create a simple table
    cursor.execute(
        """
        CREATE TABLE sales (
            product TEXT,
            category TEXT,
            revenue INTEGER
        )
    """
    )

    # Add some data
    cursor.executemany(
        "INSERT INTO sales (product, category, revenue) VALUES (?, ?, ?)",
        [
            ("Laptop", "Electronics", 1000),
            ("Phone", "Electronics", 800),
            ("Headphones", "Electronics", 200),
            ("Chair", "Furniture", 300),
        ],
    )

    conn.commit()
    conn.close()
    return "test.db"


models = {
    "deepseek": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "llama": "meta-llama/Llama-3.1-8B-Instruct",
}


def make_llama_3_prompt(user, system=""):
    system_prompt = ""
    if system != "":
        system_prompt = (
            f"<|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|>"
        )
    return f"<|begin_of_text|>{system_prompt}<|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"


def make_prompt(user, system="", model="llama"):
    if "llama" in model:
        return make_llama_3_prompt(user, system)
    else:
        print("No prompt template for model", model)
        return system + user


def main():
    """Test the LLM SQL score validator."""

    models = {
        "deepseek": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "llama": "meta-llama/Llama-3.1-8B-Instruct",
    }

    # Setup
    db_path = setup_test_db()

    # Create reference data - total revenue by category
    reference_df = pd.DataFrame(
        {"category": ["Electronics", "Furniture"], "total_revenue": [2000, 300]}
    )

    # Initialize validator
    validator = SQLScoreValidator(
        model=models["llama"],
        db_type="sqlite",
        db_params={"database": db_path},
        sql_key="sql_query",
        reference_df=reference_df,
    )
    validator.instruction = make_prompt(
        user=validator.instruction, model=models["llama"]
    )

    # Test cases to test semantic similarity
    test_cases = [
        {
            "name": "Exact Match",
            "query": """
                SELECT category, SUM(revenue) as total_revenue 
                FROM sales 
                GROUP BY category
            """,
            "should_pass": True,
        },
        {
            "name": "Same Information, Different Format",
            "query": """
                SELECT category, 
                       COUNT(*) as num_products,
                       SUM(revenue) as total_revenue,
                       AVG(revenue) as avg_revenue
                FROM sales 
                GROUP BY category
            """,
            "should_pass": True,
        },
        {
            "name": "Different Information",
            "query": """
                SELECT product, revenue
                FROM sales
                ORDER BY revenue DESC
            """,
            "should_pass": False,
        },
    ]

    # Run tests
    print("\nRunning SQL score validation tests:")
    print("-" * 50)

    for test in test_cases:
        print(f"\nTest: {test['name']}")
        print(f"Query: {test['query']}")

        # Create prompt object with all required keys
        prompt_obj = PromptObject(
            prompt="Compare query results",
            data={
                "sql_query": test["query"],
                "reference_df": str(reference_df).lower(),
            },
        )

        try:
            result = validator(prompt_obj)
            is_similar = result.response["is_similar"]
            explanation = result.response["explanation"]

            print(f"Similar: {is_similar}")
            print(f"Explanation: {explanation}")

            if is_similar == test["should_pass"]:
                print("✓ Test passed")
            else:
                print("✗ Test failed")

        except Exception as e:
            print(f"Error running test: {str(e)}")
            import traceback

            traceback.print_exc()

    sql_validator = SQLValidator(
        model=models["llama"],
        db_type="sqlite",
        db_params={"database": db_path},
        sql_key="sql_query",
    )
    test_sql_validator(sql_validator)

    # Cleanup
    if os.path.exists(db_path):
        os.remove(db_path)

def test_sql_validator(validator):
    # Autofixes invalid query by removing non-SQL stuff
    prompt_obj = PromptObject(
	prompt="test prompt obj",
        data={
            "sql_query": "JUNK SELECT COUNT(product) FROM sales;junk",
        },
    )
    r = validator(prompt_obj)
    assert r.response == {'is_valid': True,
                          'error': None,
                          'explanation': 'Query is valid and can be executed',
                         }
    assert r.data == {'sql_query': 'SELECT COUNT(product) FROM sales;'}

    # Original query is valid, no autofix needed
    prompt_obj = PromptObject(
	prompt="test prompt obj",
        data={
            "sql_query": "SELECT COUNT(product) FROM sales;",
        },
    )
    r = validator(prompt_obj)
    assert r.response == {'is_valid': True,
                          'error': None,
                          'explanation': 'Query is valid and can be executed',
                         }
    assert r.data == {'sql_query': 'SELECT COUNT(product) FROM sales;'}

    # Non-existent column, does not autofix
    prompt_obj = PromptObject(
	prompt="test prompt obj",
        data={
            "sql_query": "SELECT weird FROM sales;",
        },
    )
    r = validator(prompt_obj)
    assert r.response == {'is_valid': False,
                          'error': 'no such column: weird',
                          'explanation': 'Query is invalid: no such column: weird',
                         }
    assert r.data == {'sql_query': 'SELECT weird FROM sales;'}

    # Syntax invalid, validator attempts to autofix the query by removing non-SQL,
    # but the result query is still invalid due to non-existent column,
    # so original query is returned in response
    prompt_obj = PromptObject(
	prompt="test prompt obj",
        data={
            "sql_query": "JUNK SELECT weird FROM sales;",
        },
    )
    r = validator(prompt_obj)
    assert r.response == {'is_valid': False,
                          'error': 'near "JUNK": syntax error',
                          'explanation': 'Query is invalid: near "JUNK": syntax error',
                         }
    assert r.data == {'sql_query': 'JUNK SELECT weird FROM sales;'}

if __name__ == "__main__":
    main()
