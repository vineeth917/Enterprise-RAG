import pytest
from unittest.mock import patch, MagicMock
from retrieval import rag_pipeline_gemini

def test_pinecone_query_error_handling():
    with patch("retrieval.rag_pipeline_gemini.index_text.query") as mock_query:
        mock_query.side_effect = Exception("Simulated Pinecone failure!")
        
        result = rag_pipeline_gemini.enhanced_rag_pipeline(query_text="some query")
        assert "error" in result["answer"].lower()

def test_graph_driver_error_handling():
    with patch("retrieval.rag_pipeline_gemini.graph_driver.session") as mock_session:
        mock_session.side_effect = Exception("Simulated Neo4j error!")
        
        result = rag_pipeline_gemini.enhanced_rag_pipeline(query_text="test query")
        assert "error" in result["answer"].lower()
