# tests/test_rag_pipeline_summary.py
import unittest
from retrieval.rag_pipeline_gemini import aggregate_entities

class TestRAGPipelineSummary(unittest.TestCase):
    def test_aggregate_entities(self):
        entities = [
            {"text": "sunset"}, {"text": "castle"},
            {"text": "sunset"}, {"text": "village"}
        ]
        aggregated = aggregate_entities(entities)
        # Check deduplication
        self.assertEqual(len(aggregated), 3)
        # Check counts
        counts = {item["entity"]: item["count"] for item in aggregated}
        self.assertEqual(counts["sunset"], 2)
        self.assertEqual(counts["castle"], 1)
        self.assertEqual(counts["village"], 1)

    def test_empty_entities(self):
        aggregated = aggregate_entities([])
        self.assertEqual(aggregated, [])

if __name__ == "__main__":
    unittest.main()
