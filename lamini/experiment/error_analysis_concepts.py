from collections import Counter
from typing import List, Dict, Optional, Union

from lamini.experiment.base_generator import BaseGenerator
from lamini.generation.base_prompt_object import PromptObject

class TopicAnalyzerGenerator(BaseGenerator):
    """Analyzes questions to identify their main topics and business concepts."""
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
            Analyze this question:
            {question}

            Using this schema:
            {schema}

            And this glossary:
            {glossary}

            Identify:
            1. The main business topics/concepts it covers
            2. The type of analysis being requested
            3. The key metrics or measurements involved

            Format your response as a JSON with these keys:
            - topics: list of main business topics/concepts
            - analysis_type: type of analysis being performed
            - metrics: list of metrics or measurements involved
        """

        output_type = {
            'topics': 'array', 
            'analysis_type': 'string',
            'metrics': 'array'
        }

        super().__init__(
            client=client,
            model=model,
            name=name or "TopicAnalyzer",
            role=role or "You are an expert at analyzing business questions and identifying their key components.",
            instruction=instruction,
            output_type=output_type,
            **kwargs,
        )


class GapAnalyzerGenerator(BaseGenerator):
    """Identifies gaps in topic coverage and generates additional questions."""
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
            Gold Set Topics: {gold_topics}
            Generated Set Topics: {generated_topics}
            Missing Topics: {missing_topics}
            Schema: {schema}
            Glossary: {glossary}
            
            Generate {num_questions} new questions that cover the missing topics. The questions should:
            1. Focus specifically on the underrepresented topics
            2. Be answerable using the available schema
            3. Use terminology from the glossary where applicable
            4. Vary in complexity and analysis type
            5. Each question should cover 2-3 of the missing topics to maximize coverage
            
            Format your response as a JSON with these keys:
            - questions: list of {num_questions} new questions
            - topics_covered: list of topics each question addresses
        """

        output_type = {
            "questions": "array",
            "topics_covered": "array"
        }

        super().__init__(
            client=client,
            model=model,
            name=name or "GapAnalyzer",
            role=role or "You are an expert at identifying and filling gaps in question coverage.",
            instruction=instruction,
            output_type=output_type,
            **kwargs,
        )


class ErrorAnalysis:
    """A class for analyzing and improving question coverage in datasets."""
    
    def __init__(
        self,
        model: str,
        schema: str,
        glossary: str,
        sql_components: Optional[Dict] = None
    ):
        """
        Initialize the error analysis components.
        
        Args:
            model: The model to use for topic analysis and gap filling
            schema: Database schema 
            glossary: Glossary of domain-specific terms
            sql_components: Optional dict containing SQL generator, validator and debugger
        """
        self.model = model
        self.schema = schema
        self.glossary = glossary
        self.sql_components = sql_components
        
        # Initialize analyzers
        self.topic_analyzer = TopicAnalyzerGenerator(model=model)
        self.gap_analyzer = GapAnalyzerGenerator(model=model)
    
    def extract_topics(
        self, 
        questions: List[Union[Dict, str]]
    ) -> Dict:
        """
        Extract topics and analysis types from a set of questions.
        
        Args:
            questions: List of question dictionaries or strings
            
        Returns:
            Dictionary containing extracted topics, analysis types, and metrics
        """
        topics = Counter()
        analysis_types = Counter()
        metrics = Counter()
        
        for q_data in questions:
            question = None
            if isinstance(q_data, str):
                question = q_data
            else:
                for key in ['input', 'question', 'text']:
                    if key in q_data:
                        question = q_data[key]
                        break
            
            if not question:
                continue

            prompt_obj = PromptObject(
                "", 
                data={
                    "question": question,
                    "schema": self.schema,
                    "glossary": self.glossary
                }
            )
            
            try:
                result = self.topic_analyzer(prompt_obj)
                
                if result and result.response:
                    topics.update(result.response.get("topics", []))
                    analysis_types.update([result.response.get("analysis_type", "")])
                    metrics.update(result.response.get("metrics", []))
            except Exception as e:
                print(f"Error processing question: {question[:50]}... - {str(e)}")
                continue

        return {
            "topics": dict(topics),
            "analysis_types": dict(analysis_types),
            "metrics": dict(metrics)
        }
    
    def analyze_topic_coverage(
        self,
        gold_questions: List[Dict],
        generated_questions: List[Dict]
    ) -> Dict:
        """
        Analyze topic coverage between gold and generated question sets.
        
        Args:
            gold_questions: List of reference questions
            generated_questions: List of generated questions
            
        Returns:
            Dictionary containing analysis results
        """
        gold_analysis = self.extract_topics(gold_questions)
        generated_analysis = self.extract_topics(generated_questions)
        
        # Find missing topics (case-sensitive)
        missing_topics = set(gold_analysis["topics"].keys()) - set(generated_analysis["topics"].keys())
        
        # Find underrepresented topics (case-sensitive)
        underrepresented_topics = {}
        for topic, gold_count in gold_analysis["topics"].items():
            if topic in generated_analysis["topics"]:
                gen_count = generated_analysis["topics"][topic]
                if gold_count > gen_count * 1.5:  
                    underrepresented_topics[topic] = gold_count - gen_count
        
        return {
            "gold_analysis": gold_analysis,
            "generated_analysis": generated_analysis,
            "missing_topics": list(missing_topics),
            "underrepresented_topics": underrepresented_topics
        }
    
    def generate_additional_questions(
        self,
        coverage_analysis: Dict,
        num_questions_per_topic: int = 2
    ) -> List[Dict]:
        """
        Generate additional questions to fill identified gaps.
        
        Args:
            coverage_analysis: Output from analyze_topic_coverage
            num_questions_per_topic: Number of questions to generate per topic
            
        Returns:
            List of generated questions with covered topics
        """
        topics_to_cover = list(coverage_analysis["missing_topics"]) + list(coverage_analysis["underrepresented_topics"].keys())
        
        if not topics_to_cover:
            return []
        
        all_additional_questions = []
        
        # Process topics in batches of 3 to avoid context window issues
        for i in range(0, len(topics_to_cover), 3):
            batch_topics = topics_to_cover[i:i+3]
            
            prompt_obj = PromptObject(
                "",
                data={
                    "gold_topics": coverage_analysis["gold_analysis"]["topics"],
                    "generated_topics": coverage_analysis["generated_analysis"]["topics"],
                    "missing_topics": batch_topics,
                    "schema": self.schema,
                    "glossary": self.glossary,
                    "num_questions": num_questions_per_topic * len(batch_topics)
                }
            )
            
            try:
                result = self.gap_analyzer(prompt_obj)
                
                if not result or not result.response:
                    continue
                
                batch_questions = []
                for q, topics in zip(
                    result.response.get("questions", []),
                    result.response.get("topics_covered", [])
                ):
                    batch_questions.append({
                        "question": q,
                        "covered_topics": topics
                    })
                
                all_additional_questions.extend(batch_questions)
                
            except Exception as e:
                print(f"Error generating questions for topics {batch_topics}: {str(e)}")
        
        return all_additional_questions