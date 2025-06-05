# Enterprise RAG System with Cross-Modal Integration

A robust enterprise-grade RAG (Retrieval Augmented Generation) system that combines text and image understanding with knowledge graph integration. The system uses CC3M dataset for demonstration purposes.

## Evaluation-First Pipeline Design

We prioritize **robust evaluation** using:
- **DeepEval metrics**: hallucination, faithfulness, answer/context relevancy.
- **Latency** and **error handling** tests.
- **Entity extraction** and **cross-modal linking** unit tests.

## System Performance
- **Precision** in cross-modal context retrieval
- Response **latency** optimization
- Graceful **error** handling
- Consistent output quality

### Evaluation Goals

1. **Retrieval Quality**: How relevant the returned context is to the original query.
2. **Hallucination Control**: Minimize hallucination in final answers.
3. **Latency**: Ensure reasonable runtime for production viability.
4. **Domain-aware Summarization**: Responses are enriched by knowledge graph & multimodal links.

### Tests Implemented

| Test Suite                    | Purpose                                   |
|--------------------------     |-------------------------------------------|
| `deepeval_tests.py`           | End-to-end pipeline + hallucination checks |
| `test_cross_modal_linking.py` | Unit tests for linking (text-image)   |
| `test_entity_extraction.py`   | Entity extraction robustness tests   |
| `test_rag.py`                 | End-to-end RAG flow tests                 |
| `test_latency.py`             | Performance and runtime checks            |
| `test_rag_error_handling.py`  | Fallbacks & error resilience          |


##  How to Use

```bash
# Basic example to run end-to-end tests
python evaluation/deepeval_tests.py

# Performance test
python evaluation/test_latency.py

python evaluation/test_entity_extraction.py

# Check error handling
python evaluation/test_rag_error_handling.py
## Architecture Overview

python evaluation/test_rag.py

```

## Basic Architecture

- **Vector Search**: Dual Pinecone indexes for text (1024d) and image (512d) embeddings
- **Entity Management**: Neo4j graph database for entity relationships
- **LLM Integration**: Google's Gemini for response generation
- **UI**: Streamlit-based interface for query and visualization

## Key Features

- Cross-modal retrieval combining text and image understanding
- Entity extraction and relationship mapping
- Interactive visualization of knowledge graphs
- Anti-hallucination mechanisms
- Real-time performance metrics

## Evaluation Criteria

### System Performance
- Precision in cross-modal context retrieval
- Response latency optimization
- Graceful error handling
- Consistent output quality

### Technical Implementation
- Modular architecture for maintainability
- Clear separation of concerns
- Comprehensive error handling
- Performance monitoring

## Setup Instructions

1. **Environment Setup**
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

#download cc3m dataset
kaggle datasets download -d cc3m/conceptual-captions -p data/ --unzip

# Limit to 1000 Samples
import pandas as pd
df = pd.read_csv("data/Train_GCC-training.tsv", sep="\t", names=["caption", "url"])
df_sample = df.sample(n=1000, random_state=42)
df_sample.to_csv("data/processed/cc3m_sample_1000.csv", index=False)

# Download spaCy model
python -m spacy download en_core_web_sm
```

2. **Configuration**
Create a `.env` file with:
```
PINECONE_API_KEY_TEXT=your_key
PINECONE_API_KEY_IMAGE=your_key
PINECONE_ENVIRONMENT=your_env
NEO4J_URI=your_uri
NEO4J_USER=your_user
NEO4J_PASSWORD=your_password
GEMINI_API_KEY=your_key
```

3. **Data Setup**
- Place CC3M dataset in `data/cc3m/` directory
- Run ingestion script:
```bash
python ingestion/cc3m_ingestion.py
```

## Running the System

1. **Start the Application**
```bash
streamlit run app_ui.py
```

2. **Using the Interface**
- Enter natural language queries
- Upload images for visual similarity search
- View results in Answer and Analysis tabs
- Explore entity relationships and confidence scores

## Implementation Details

### Data Processing
- Text embedding: E5-large-v2 (1024d)
- Image embedding: CLIP (512d)
- Entity extraction: spaCy + custom rules

### Storage
- Pinecone indexes:
  - Text: 1024-dimensional vectors
  - Image: 512-dimensional vectors
- Neo4j graph database for entity relationships

### Query Processing
- Cross-modal context retrieval
- Entity-aware response generation
- Confidence score calculation
- Anti-hallucination checks

## Limitations

- Currently supports only image and text modalities
- Uses CC3M dataset for demonstration
- Requires manual setup of vector stores and graph database

## Future Improvements

- Add support for PDF and document processing
- Implement audio/video processing capabilities
- Enhance entity linking across more modalities
- Add batch processing capabilities
