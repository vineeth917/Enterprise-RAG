# tests/test_cross_modal_linking.py

import unittest
from unittest.mock import MagicMock
from retrieval import cross_modal_linking

class TestCrossModalLinking(unittest.TestCase):
    def setUp(self):
        self.dummy_nlp = MagicMock()
        self.dummy_nlp.return_value = [
            MagicMock(lemma_='castle', text='castle'),
            MagicMock(lemma_='mountain', text='mountain')
        ]

        self.dummy_graph = MagicMock()
        self.dummy_graph.session.return_value.__enter__.return_value.run.return_value = []

        self.dummy_get_embedding = lambda x: [0.1, 0.2, 0.3]

        self.dummy_extract_entities = lambda x: [{"text": "sunset", "label": "TIME"}]

    def test_extract_visual_concepts(self):
        terms = cross_modal_linking.extract_visual_concepts("A castle on a mountain", self.dummy_nlp)
        self.assertIn("castle", terms)
        self.assertIn("mountain", terms)

    def test_enhance_cross_modal_linking_returns_list(self):
        result = cross_modal_linking.enhance_cross_modal_linking(
            self.dummy_graph,
            self.dummy_nlp,
            self.dummy_get_embedding,
            self.dummy_extract_entities,
            "A castle on a mountain",
            [],
            ["castle"]
        )
        self.assertIsInstance(result, list)

if __name__ == "__main__":
    unittest.main()
