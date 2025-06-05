import os
from dotenv import load_dotenv
load_dotenv()

from typing import List, Dict, Any
import logging
from deepeval.metrics import (
    HallucinationMetric,
    AnswerRelevancyMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric
)
from deepeval.test_case import LLMTestCase

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class DeepEvalMetrics:
    """DeepEval metrics implementation for version 3.x."""

    def __init__(self):
        """Initialize metrics with proper thresholds."""
        self.thresholds = {
            "hallucination": 0.7,
            "answer_relevancy": 0.7,
            "context_relevancy": 0.7,
            "faithfulness": 0.65
        }

    async def evaluate_response(
        self,
        query: str,
        response: str,
        context: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Evaluate response using DeepEval metrics.

        :param query: The user query.
        :param response: The generated response.
        :param context: List of context documents.
        :return: Dictionary of metric scores.
        """
        try:
            # Check if response is valid
            if not response or response.strip() == "":
                logger.warning("Empty or None response received")
                return self._get_zero_scores("Empty response")

            # Format context text, skip empty
            context_texts = []
            for doc in context:
                content = doc.get('content', doc.get('page_content', ''))
                if content and content.strip():
                    context_texts.append(content.strip())
            
            if not context_texts:
                logger.warning("No valid context found, using minimal context")
                context_texts = ["Context not available for this query"]

            # Create test case
            test_case = LLMTestCase(
                input=query,
                actual_output=response,
                retrieval_context=context_texts,  # List of strings
                context=context_texts  # Some metrics might need this too
            )

            results = {}

            # Initialize metrics with thresholds
            metrics = {
                "hallucination": HallucinationMetric(threshold=self.thresholds["hallucination"]),
                "answer_relevancy": AnswerRelevancyMetric(threshold=self.thresholds["answer_relevancy"]),
                "context_relevancy": ContextualRelevancyMetric(threshold=self.thresholds["context_relevancy"]),
                "faithfulness": FaithfulnessMetric(threshold=self.thresholds["faithfulness"])
            }

            # Evaluate each metric
            for metric_name, metric in metrics.items():
                try:
                    logger.info(f"Evaluating {metric_name}...")
                    
                    # Measure the metric (this is synchronous in DeepEval 3.x)
                    metric.measure(test_case)
                    
                    # Get the score
                    score = metric.score
                    results[f"{metric_name}_score"] = float(score) if score is not None else 0.0
                    
                    # Get success status
                    success = metric.success if hasattr(metric, 'success') else None
                    if success is not None:
                        results[f"{metric_name}_success"] = success
                    
                    logger.info(f"{metric_name} score: {results[f'{metric_name}_score']:.3f}")
                    
                except Exception as e:
                    logger.error(f"Error evaluating {metric_name}: {str(e)}")
                    results[f"{metric_name}_score"] = 0.0
                    results[f"{metric_name}_success"] = False

            # Calculate overall score
            score_keys = [k for k in results.keys() if k.endswith('_score')]
            scores = [results[k] for k in score_keys]
            results["overall_score"] = sum(scores) / len(scores) if scores else 0.0

            logger.info(f"DeepEval Metrics: {results}")
            return results
            

        except Exception as e:
            logger.error(f"Error evaluating response: {str(e)}")
            return self._get_zero_scores(str(e))

    def _get_zero_scores(self, error_msg: str) -> Dict[str, float]:
        """Return zero scores with error message."""
        return {
            "hallucination_score": 0.0,
            "answer_relevancy_score": 0.0,
            "context_relevancy_score": 0.0,
            "faithfulness_score": 0.0,
            "overall_score": 0.0,
            "error": error_msg
        }

    def get_thresholds(self) -> Dict[str, float]:
        """
        Get threshold values for metrics.
        """
        return self.thresholds