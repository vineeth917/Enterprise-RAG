# retrieval/test_entity_extraction.py

import unittest
from entity_relationship.extract_entities import extract_entities, deduplicate_entities, merge_similar_entities


class TestEntityExtraction(unittest.TestCase):
    def test_basic_extraction(self):
        """
        Tests extraction of known domain keywords and phrases.
        Updated to check for 'rugged coastline' instead of 'coastline' separately.
        """
        text = "A medieval castle at the golden hour, overlooking the rugged coastline."
        entities = extract_entities(text)
        extracted_texts = [e["text"] for e in entities]
        extracted_labels = [e["label"] for e in entities]

        # Check key phrases/entities
        self.assertIn("castle", extracted_texts)
        self.assertIn("golden hour", extracted_texts)
        self.assertIn("rugged coastline", extracted_texts)

        # Check domain labels
        self.assertIn("ARCHITECTURE", extracted_labels)
        self.assertIn("TIME", extracted_labels)
        self.assertIn("ENVIRONMENT", extracted_labels)

    def test_single_word_extraction(self):
        """
        Tests that single-word parts of phrases are also captured if needed.
        """
        text = "A rugged coastline with a medieval castle."
        entities = extract_entities(text)
        extracted_texts = [e["text"] for e in entities]

        # Depending on your entity splitting logic, these may or may not be present
        # Here we ensure phrases like 'rugged coastline' are found
        self.assertIn("rugged coastline", extracted_texts)
        self.assertIn("castle", extracted_texts)

    def test_regex_match_integration(self):
        """
        Tests regex-based entity recognition.
        """
        text = "The evening was calm with twilight glow in the city skyline."
        entities = extract_entities(text)
        entity_texts = [e["text"] for e in entities]

        self.assertIn("evening", entity_texts)
        self.assertIn("twilight glow", entity_texts)
        self.assertIn("skyline", entity_texts)

    def test_domain_merging(self):
        """
        Tests that similar entities are merged under the same domain label.
        """
        entities = [
            {"text": "view", "label": "GENERAL"},
            {"text": "landscape", "label": "GENERAL"}
        ]
        merged = merge_similar_entities(entities)
        merged_labels = {e["label"] for e in merged}

        self.assertTrue(all(label == "SCENIC_VIEW" for label in merged_labels))

    def test_deduplication_priority(self):
        """
        Tests deduplication prioritizes higher domain relevance.
        """
        entities = [
            {"text": "sunset", "label": "TIME"},
            {"text": "sunset", "label": "REGEX_MATCH"},
            {"text": "sunset", "label": "GENERAL"}
        ]
        deduped = deduplicate_entities(entities)
        self.assertEqual(len(deduped), 1)
        self.assertEqual(deduped[0]["label"], "TIME")  # TIME has higher priority

    def test_context_disambiguation(self):
        """
        Tests that 'view' is not incorrectly extracted from words like 'review'.
        """
        text = "I will review the details tomorrow."
        entities = extract_entities(text)
        extracted_texts = [e["text"] for e in entities]

        self.assertNotIn("view", extracted_texts)  # Should not extract from "review"

    def test_empty_input(self):
        """
        Tests that empty input returns no entities.
        """
        entities = extract_entities("")
        self.assertEqual(entities, [])

    def test_no_domain_match(self):
        """
        Tests that non-domain text yields no entities.
        """
        text = "The quick brown fox jumps over the lazy dog."
        entities = extract_entities(text)
        self.assertEqual(entities, [])

    def test_multiple_same_entity(self):
        """
        Tests that deduplication handles multiple identical entities.
        """
        entities = [
            {"text": "sunset", "label": "TIME"},
            {"text": "sunset", "label": "REGEX_MATCH"},
            {"text": "sunset", "label": "TIME"}  # Duplicate
        ]
        deduped = deduplicate_entities(entities)
        self.assertEqual(len(deduped), 1)


if __name__ == "__main__":
    unittest.main()
