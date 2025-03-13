from typing import List, Dict, Any
from collections import Counter
from lamini.experiment.base_generator import BaseGenerator
from lamini.generation.base_prompt_object import PromptObject

class SQLAnalysisGenerator(BaseGenerator):
    """Generates analysis for SQL query failures."""
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
        As an expert SQL analyst, your task is to analyze the following failed SQL queries and provide detailed insights.

        For each failed query, please include:
        1. Error Explanation: Clearly explain the error encountered in the query.
        2. Root Cause Analysis: Identify the root cause of the error.
        3. Correction Suggestions: Provide specific suggestions on how to correct the query.
        4. Improvement Recommendations: Suggest improvements to prevent similar errors in the future.

        Basic Statistics:
        {basic_stats}

        Failed Queries:
        {sample_queries}

        Your response should be concise yet comprehensive, focusing on actionable insights to improve the SQL generation process.
        """

        output_type = output_type or {"analysis": "string"}

        super().__init__(
            client=client,
            model=model,
            name=name or "SQLAnalysisGenerator",
            role=role or "You are an expert at analyzing SQL query failures and identifying patterns.",
            instruction=instruction,
            output_type=output_type,
            **kwargs,
        )


class SQLErrorAnalysis:
    """A class for analyzing SQL errors in generated queries."""
    
    def __init__(
        self,
        model: str
    ):
        """
        Initialize the SQL error analysis component.
        
        Args:
            model: The model to use for error analysis
        """
        self.model = model
        self.analyzer = SQLAnalysisGenerator(model=model)
    
    def extract_failed_queries(self, results_data: List[Dict]) -> List[Dict]:
        """
        Extract failed SQL queries from results data.
        
        Args:
            results_data: List of result dictionaries from data generation
            
        Returns:
            List of dictionaries containing failed queries and their error information
        """
        failed_queries = []
        
        for result in results_data:
            for variation in result.get("pattern_variations", []):
                self._process_variation(variation, failed_queries, result.get("original_question", ""))
                    
            for variation in result.get("structural_variations", []):
                self._process_variation(variation, failed_queries, result.get("original_question", ""))
                    
            for sub_q in result.get("sub_questions", []):
                self._process_variation(sub_q, failed_queries, result.get("original_question", ""), is_subq=True)
                    
        return failed_queries
    
    def _process_variation(self, variation: Dict, failed_queries: List[Dict], original_question: str, is_subq: bool = False):
        """Process a variation and add to failed_queries if validation failed."""
        validation = variation.get("validation", {})
        if not validation:
            validation = variation.get("final_validation", {})
        
        if validation and not validation.get("is_valid", True):
            # Simplified failed query object with only essential information
            failed_query = {
                "question": variation.get("question", variation.get("sub_question", "")),
                "generated_sql": variation.get("generated_sql", variation.get("original_sql", "")),
                "error_message": validation.get("error", ""),
                "error_explanation": validation.get("explanation", "")
            }
            
            failed_queries.append(failed_query)
    
    def extract_basic_statistics(self, failed_queries: List[Dict]) -> Dict[str, Any]:
        """
        Extract basic statistics about the failed queries.
        
        Args:
            failed_queries: List of failed query dictionaries
            
        Returns:
            Dictionary containing statistics about error types and SQL features
        """
        error_types = Counter()
        sql_features = Counter()
        correction_stats = {"corrected": 0, "not_corrected": 0}
        
        for query in failed_queries:
            error_msg = query["error_message"].lower()
            
            if "syntax error" in error_msg:
                error_types["syntax_error"] += 1
            elif "column" in error_msg and "does not exist" in error_msg:
                error_types["missing_column"] += 1
            elif "invalid identifier" in error_msg:
                error_types["invalid_identifier"] += 1
            elif "schema" in error_msg and "does not exist" in error_msg:
                error_types["schema_error"] += 1
            elif "function" in error_msg:
                error_types["function_error"] += 1
            elif "group by" in error_msg:
                error_types["group_by_error"] += 1
            else:
                error_types["other"] += 1
            
            sql = query["generated_sql"].lower()
            if "join" in sql:
                sql_features["join"] += 1
            if "with" in sql and "as" in sql:
                sql_features["cte"] += 1
            if sql.count("select") > 1:
                sql_features["subquery"] += 1
            if "group by" in sql:
                sql_features["group_by"] += 1
            if any(func in sql for func in ["sum(", "avg(", "count(", "min(", "max("]):
                sql_features["aggregation"] += 1
            
            if query.get("was_corrected", False):
                correction_stats["corrected"] += 1
            else:
                correction_stats["not_corrected"] += 1
        
        return {
            "total_failed_queries": len(failed_queries),
            "error_types": dict(error_types),
            "sql_features": dict(sql_features),
            "correction_stats": correction_stats
        }
    
    def generate_llm_analysis(self, failed_queries: List[Dict], basic_stats: Dict[str, Any]) -> str:
        """
        Use a LLM to analyze the failed queries and generate insights.
        
        Args:
            failed_queries: List of failed query dictionaries
            basic_stats: Dictionary containing basic statistics about the failed queries
            
        Returns:
            String containing the LLM analysis of the failed queries
        """
        try:
            # Use sample queries for analysis
            sample_size = min(5, len(failed_queries))
            sample_queries = failed_queries[:sample_size]
            
            # Format sample queries for the prompt
            formatted_queries = []
            for i, query in enumerate(sample_queries):
                formatted_query = f"""
                Query {i+1}:
                Question: {query.get('question', '')}
                SQL: {query.get('generated_sql', '')}
                Error: {query.get('error_message', '')}
                """
                formatted_queries.append(formatted_query)
            
            # Format basic stats as a string
            stats_str = f"""
            Total Failed Queries: {basic_stats['total_failed_queries']}
            Error Types: {basic_stats['error_types']}
            SQL Features: {basic_stats['sql_features']}
            Correction Stats: {basic_stats['correction_stats']}
            """
            
            # Create a prompt object with exactly the expected keys
            prompt_obj = PromptObject(
                "Analyze these SQL query errors",
                data={
                    "basic_stats": stats_str,
                    "sample_queries": "\n".join(formatted_queries)
                }
            )
            
            result = self.analyzer(prompt_obj)
            
            # Return the response
            if hasattr(result, 'response'):
                if isinstance(result.response, dict):
                    return result.response.get('analysis', str(result.response))
                return str(result.response)
            else:
                return "No analysis was generated."
                
        except Exception as e:
            return f"Error generating analysis: {str(e)}"
