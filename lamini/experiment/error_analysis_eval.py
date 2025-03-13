from typing import List, Dict, Optional
from collections import Counter
from lamini.experiment.base_generator import BaseGenerator
from lamini.generation.base_prompt_object import PromptObject

class SQLResultAnalyzerGenerator(BaseGenerator):
    """Analyzes differences between SQL query results to determine functional equivalence."""
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
        instruction = """
            Analyze these two SQL query results:
            
            Gold Query: {gold_query}
            Generated Query: {generated_query}
            
            Gold Result:
            {gold_result}
            
            Generated Result:
            {generated_result}
            
            Determine if these results are functionally equivalent. Consider:
            1. Do they contain the same information, even if formatted differently?
            2. Are differences in ordering significant for the query's purpose?
            3. Are numerical differences within acceptable rounding/precision limits?
            4. Do the results answer the original question equally well?
            
            Original Question: {question}
            
            Format your response as a JSON with these keys:
            - equivalent: boolean indicating if results are functionally equivalent
            - confidence: number between 0-1 indicating confidence in assessment
            - explanation: detailed explanation of your reasoning
            - differences: list of specific differences found (empty if equivalent)
        """

        output_type = {
            "equivalent": "boolean",
            "confidence": "number",
            "explanation": "string",
            "differences": "array"
        }

        super().__init__(
            client=client,
            model=model,
            name=name or "SQLResultAnalyzer",
            role=role or "You are an expert at analyzing SQL query results and determining functional equivalence.",
            instruction=instruction,
            output_type=output_type,
            **kwargs,
        )
class SQLExecutionPipeline:
    """Executes SQL queries and compares results for functional equivalence."""
    
    def __init__(
        self,
        model: str,
        db_type: str = "sqlite",
        db_connection=None,
        result_analyzer: Optional[SQLResultAnalyzerGenerator] = None
    ):
        """
        Initialize the SQL execution pipeline.
        
        Args:
            model: The model to use for analyzing result differences
            db_type: Database type ('sqlite' or 'snowflake')
            db_connection: Database connection object (optional)
            result_analyzer: Optional custom result analyzer
        """
        self.model = model
        self.db_type = db_type.lower()
        self.db_connection = db_connection
        self.result_analyzer = result_analyzer or SQLResultAnalyzerGenerator(model=model)
        
        # Validate db_type
        if self.db_type not in ["sqlite", "snowflake"]:
            raise ValueError("db_type must be either 'sqlite' or 'snowflake'")
    
    def connect_to_db(self, connection_params=None):
        """
        Connect to the database if not already connected.
        
        Args:
            connection_params: Parameters for database connection
        
        Returns:
            Database connection object
        """
        if self.db_connection:
            return self.db_connection
            
        if self.db_type == "sqlite":
            import sqlite3
            db_path = connection_params.get("db_path", ":memory:")
            self.db_connection = sqlite3.connect(db_path)
            
        elif self.db_type == "snowflake":
            import snowflake.connector
            self.db_connection = snowflake.connector.connect(
                user=connection_params.get("user"),
                password=connection_params.get("password"),
                account=connection_params.get("account"),
                warehouse=connection_params.get("warehouse"),
                database=connection_params.get("database"),
                schema=connection_params.get("schema")
            )
            
        return self.db_connection
    
    def execute_query(self, query):
        """
        Execute a SQL query and return results as a pandas DataFrame.
        
        Args:
            query: SQL query to execute
            
        Returns:
            pandas DataFrame containing query results
        """
        import pandas as pd
        
        if not self.db_connection:
            raise ValueError("Database connection not established. Call connect_to_db first.")
            
        try:
            if self.db_type == "sqlite":
                return pd.read_sql_query(query, self.db_connection)
                
            elif self.db_type == "snowflake":
                cursor = self.db_connection.cursor()
                cursor.execute(query)
                columns = [desc[0] for desc in cursor.description]
                data = cursor.fetchall()
                cursor.close()
                return pd.DataFrame(data, columns=columns)
                
        except Exception as e:
            print(f"Error executing query: {str(e)}")
            print(f"Query: {query}")
            return None
    
    def compare_results(self, gold_query, generated_query, gold_result, generated_result, question):
        """
        Compare query results to determine functional equivalence.
        
        Args:
            gold_query: Gold standard SQL query
            generated_query: Generated SQL query
            gold_result: DataFrame with gold query results
            generated_result: DataFrame with generated query results
            question: Original question being answered
            
        Returns:
            Dictionary with comparison results
        """
        import pandas as pd
        
        # Quick exact match check
        exact_match = gold_result.equals(generated_result)
        
        if exact_match:
            return {
                "equivalent": True,
                "confidence": 1.0,
                "explanation": "Results are exactly identical",
                "differences": []
            }
        
        # Check if shapes match
        shape_match = gold_result.shape == generated_result.shape
        
        # Convert DataFrames to string representations for the LLM
        gold_result_str = gold_result.to_string(index=False) if gold_result is not None else "Error: Query failed"
        generated_result_str = generated_result.to_string(index=False) if generated_result is not None else "Error: Query failed"
        
        # Use LLM to analyze differences
        prompt_obj = PromptObject(
            "",
            data={
                "gold_query": gold_query,
                "generated_query": generated_query,
                "gold_result": gold_result_str,
                "generated_result": generated_result_str,
                "question": question
            }
        )
        
        try:
            result = self.result_analyzer(prompt_obj)
            if result and result.response:
                return result.response
        except Exception as e:
            print(f"Error analyzing results: {str(e)}")
        
        # Fallback if LLM analysis fails
        return {
            "equivalent": False,
            "confidence": 0.5,
            "explanation": "Unable to determine equivalence with LLM, shapes match: " + str(shape_match),
            "differences": ["Unable to analyze specific differences"]
        }
    
    def evaluate_queries(self, test_cases, connection_params=None):
        """
        Evaluate a set of test cases by executing both gold and generated queries.
        
        Args:
            test_cases: List of dictionaries with keys:
                - question: Original question
                - gold_query: Gold standard SQL query
                - generated_query: Generated SQL query
            connection_params: Database connection parameters
            
        Returns:
            Dictionary with evaluation results
        """
        # Connect to database
        self.connect_to_db(connection_params)
        
        results = []
        overall_stats = {
            "total": len(test_cases),
            "equivalent": 0,
            "non_equivalent": 0,
            "execution_errors": 0,
            "average_confidence": 0.0
        }
        
        for i, test_case in enumerate(test_cases):
            question = test_case.get("question", "")
            gold_query = test_case.get("gold_query", "")
            generated_query = test_case.get("generated_query", "")
            
            print(f"Evaluating test case {i+1}/{len(test_cases)}...")
            
            # Execute queries
            gold_result = self.execute_query(gold_query)
            generated_result = self.execute_query(generated_query)
            
            # Check for execution errors
            if gold_result is None or generated_result is None:
                result = {
                    "question": question,
                    "gold_query": gold_query,
                    "generated_query": generated_query,
                    "status": "error",
                    "error": "Query execution failed",
                    "equivalent": False,
                    "confidence": 0.0,
                    "explanation": "One or both queries failed to execute"
                }
                overall_stats["execution_errors"] += 1
            else:
                # Store the actual result data
                gold_result_data = gold_result.to_dict(orient='records')
                generated_result_data = generated_result.to_dict(orient='records')

                # Compare results
                comparison = self.compare_results(
                    gold_query, generated_query, gold_result, generated_result, question
                )
                
                result = {
                    "question": question,
                    "gold_query": gold_query,
                    "generated_query": generated_query,
                    "status": "success",
                    "equivalent": comparison.get("equivalent", False),
                    "confidence": comparison.get("confidence", 0.0),
                    "explanation": comparison.get("explanation", ""),
                    "differences": comparison.get("differences", []),
                    "gold_result_data": gold_result_data,
                    "generated_result_data": generated_result_data
                }
                
                if result["equivalent"]:
                    overall_stats["equivalent"] += 1
                else:
                    overall_stats["non_equivalent"] += 1
                
                overall_stats["average_confidence"] += result["confidence"]
            
            results.append(result)
        
        # Calculate final stats
        if len(test_cases) > 0:
            overall_stats["average_confidence"] /= len(test_cases)
            overall_stats["equivalence_rate"] = overall_stats["equivalent"] / overall_stats["total"]
        
        return {
            "results": results,
            "stats": overall_stats
        }
    
    def generate_report(self, evaluation_results):
        """
        Generate a human-readable report from evaluation results.
        
        Args:
            evaluation_results: Output from evaluate_queries
            
        Returns:
            String containing formatted report
        """
        stats = evaluation_results["stats"]
        results = evaluation_results["results"]
        
        report = [
            "# SQL Query Evaluation Report",
            "",
            "## Summary",
            f"- Total test cases: {stats['total']}"
        ]
        
        # Only calculate percentages if there are test cases
        if stats['total'] > 0:
            report.extend([
                f"- Functionally equivalent: {stats['equivalent']} ({stats['equivalent']/stats['total']*100:.1f}%)",
                f"- Non-equivalent: {stats['non_equivalent']} ({stats['non_equivalent']/stats['total']*100:.1f}%)",
                f"- Execution errors: {stats['execution_errors']} ({stats['execution_errors']/stats['total']*100:.1f}%)",
                f"- Average confidence: {stats['average_confidence']:.2f}"
            ])
        else:
            report.extend([
                "- Functionally equivalent: 0 (0.0%)",
                "- Non-equivalent: 0 (0.0%)",
                "- Execution errors: 0 (0.0%)",
                "- Average confidence: 0.00"
            ])
        
        report.extend([
            "",
            "## Detailed Results"
        ])
        
        for i, result in enumerate(results):
            report.extend([
                f"### Test Case {i+1}",
                f"**Question:** {result['question']}",
                "",
                f"**Gold Query:**",
                "```sql",
                result['gold_query'],
                "```",
                "",
                f"**Generated Query:**",
                "```sql",
                result['generated_query'],
                "```",
                "",
                f"**Status:** {result['status']}",
                f"**Equivalent:** {result['equivalent']}",
                f"**Confidence:** {result['confidence']:.2f}",
                f"**Explanation:** {result['explanation']}",
                ""
            ])
            
            if result.get("differences") and len(result["differences"]) > 0:
                report.append("**Differences:**")
                for diff in result["differences"]:
                    report.append(f"- {diff}")
                report.append("")
        
        return "\n".join(report)
