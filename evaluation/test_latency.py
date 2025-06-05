import time
from retrieval.rag_pipeline_gemini import enhanced_rag_pipeline

def test_pipeline_latency():
    start_time = time.time()
    result = enhanced_rag_pipeline(query_text="a sunset view of a coastal castle")
    end_time = time.time()
    elapsed = end_time - start_time

    print(f"Pipeline latency: {elapsed:.2f} seconds")
    assert elapsed < 5.0, "Pipeline took too long!"