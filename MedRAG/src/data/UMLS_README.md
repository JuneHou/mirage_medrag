# UMLS Graph Communities Data Source for MedRAG

## Overview

This document describes the creation of the UMLS graph communities data source following the MedRAG database generation pipeline.

## Data Source Analysis

### Input Data
- **Source**: `/data/wang/junh/datasets/umls2text/leiden_only_output/communities/communities.json`
- **Format**: JSON file containing Leiden community detection results from UMLS knowledge graph
- **Size**: 416 MB
- **Total Communities**: 35,797
- **Organization**: Multiple runs (35 runs), multiple levels per run, with community summaries

### Community Structure
Each community contains:
- `run`: Run number (0-34)
- `level`: Hierarchical level within the run
- `community_id`: Unique identifier within the run/level
- `triples`: Knowledge graph relationships
- `summary`: Generated textual summary of the community
- `summaries`: List of all summaries
- `intermediate_summaries`: Additional summary information
- `randomness`: Quality metric

### Data Quality
- **Valid summaries**: 34,620 (96.7%)
- **Invalid summaries**: 1,177 (3.3%)
  - Blank summaries
  - Error messages (e.g., "The community is too large")

## Chunking Strategy Analysis

### Comparison with Existing MedRAG Sources

| Source | Chunking Strategy | Typical Size | Files |
|--------|------------------|--------------|-------|
| **PubMed** | None (abstracts naturally short) | 395-1897 chars | One per XML.gz file |
| **Wikipedia** | RecursiveCharacterTextSplitter(1000, 200) | 559-900 chars | Batched (10k articles/file) |
| **Textbooks** | RecursiveCharacterTextSplitter(1000, 200) | 935-970 chars | One per textbook |
| **StatPearls** | Smart merging (<1000 chars) | 533-634 chars | One per article |
| **UMLS** | RecursiveCharacterTextSplitter(1000, 200) | See below | One per run |

### UMLS Summary Length Statistics
```
Min:     30 chars
Max:     20,541 chars
Mean:    1,230 chars
Median:  952 chars
25th:    649 chars
75th:    1,562 chars
95th:    3,061 chars
99th:    4,457 chars

Summaries > 1000 chars: 16,235 (46.9%)
```

**Decision**: YES, chunking is needed for ~47% of communities with summaries >1000 characters.

## Implementation: umls.py

### Key Features

1. **Validation**: Filters out invalid summaries (blank, errors, "too large")
2. **Organization**: Groups communities by run (35 files total)
3. **Chunking**: Uses `RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)` for long summaries
4. **Hierarchical Titles**: "UMLS -- Run X -- Level Y -- Community Z"
5. **Unique IDs**: "UMLS_RX_LY_CZ" or "UMLS_RX_LY_CZ_chunkN" for multiple chunks

### Output Format

Following MedRAG standard JSONL format:
```json
{
    "id": "UMLS_R0_L0_C22",
    "title": "UMLS -- Run 0 -- Level 0 -- Community 22",
    "content": "The appendix is the anatomical site...",
    "contents": "UMLS -- Run 0 -- Level 0 -- Community 22. The appendix is the anatomical site..."
}
```

### Processing Results

```
Total communities: 35,797
Valid communities: 34,620 (96.7%)
Invalid/skipped: 1,177 (3.3%)
Communities chunked: 16,235 (46.9%)
Total chunks created: 62,340

Output: 35 JSONL files
Total size: 123 MB
Average chunks per file: 1,781
```

## Integration with MedRAG

### Changes to utils.py

Added UMLS to `corpus_names` dictionary:
```python
corpus_names = {
    "PubMed": ["pubmed"],
    "Textbooks": ["textbooks"],
    "StatPearls": ["statpearls"],
    "Wikipedia": ["wikipedia"],
    "UMLS": ["umls"],  # NEW
    "MedText": ["textbooks", "statpearls"],
    "MedCorp": ["pubmed", "textbooks", "statpearls", "wikipedia"],
    "MedCorpUMLS": ["pubmed", "textbooks", "statpearls", "wikipedia", "umls"],  # NEW
}
```

### Usage

#### 1. Generate UMLS chunks (Already done)
```bash
cd /data/wang/junh/githubs/mirage_medrag/MedRAG/src/data
python umls.py
```

#### 2. Create embeddings with MedCPT retriever
```python
from utils import Retriever

# This will automatically generate embeddings and build FAISS index
retriever = Retriever(
    retriever_name='ncbi/MedCPT-Query-Encoder',
    corpus_name='umls',
    db_dir='./corpus'
)
```

#### 3. Use UMLS in MedRAG
```python
from medrag import MedRAG

# Use UMLS corpus only
medrag = MedRAG(
    llm_name="OpenAI/gpt-3.5-turbo-16k",
    rag=True,
    retriever_name="MedCPT",
    corpus_name="UMLS",
    db_dir="./corpus"
)

# Or use combined corpus with UMLS
medrag = MedRAG(
    llm_name="OpenAI/gpt-3.5-turbo-16k",
    rag=True,
    retriever_name="MedCPT",
    corpus_name="MedCorpUMLS",  # All 5 sources
    db_dir="./corpus"
)

# Retrieve and answer
answer, snippets, scores = medrag.answer(
    question="What is the relationship between appendicitis and appendectomy?",
    k=32
)
```

## Indexing Process

The indexing follows the standard MedRAG pipeline:

1. **Embedding Generation** (`embed()` function in utils.py):
   - Loads all JSONL files from `corpus/umls/chunk/`
   - Uses MedCPT-Article-Encoder (or other specified retriever)
   - Generates embeddings for each chunk
   - Saves as `.npy` files in `corpus/umls/index/{retriever}/embedding/`

2. **FAISS Index Construction** (`construct_index()` function):
   - Loads all embedding `.npy` files
   - Builds FAISS index (IndexFlatIP for MedCPT, IndexFlatL2 for SPECTER)
   - Creates metadata mapping (index → source file + line number)
   - Saves to `corpus/umls/index/{retriever}/faiss.index`

3. **Retrieval** (`get_relevant_documents()` method):
   - Encodes query using Query-Encoder
   - Searches FAISS index for top-k similar chunks
   - Returns matched documents with scores

## File Structure

```
corpus/umls/
├── chunk/
│   ├── umls_run00.jsonl (3,976 chunks, 6.7 MB)
│   ├── umls_run01.jsonl (2,711 chunks, 5.1 MB)
│   ├── ...
│   └── umls_run34.jsonl (1,424 chunks, 3.0 MB)
│
└── index/
    ├── ncbi/
    │   └── MedCPT-Article-Encoder/
    │       ├── embedding/
    │       │   ├── umls_run00.npy
    │       │   ├── ...
    │       │   └── umls_run34.npy
    │       ├── faiss.index
    │       └── metadatas.jsonl
    │
    ├── facebook/
    │   └── contriever/
    │       └── ...
    │
    └── allenai/
        └── specter/
            └── ...
```

## Next Steps

1. **Generate Embeddings**:
   - Run the Retriever initialization to generate embeddings
   - This may take time depending on the number of chunks (62,340)
   - Consider using GPU for faster embedding generation

2. **Build Indices for Multiple Retrievers**:
   - MedCPT (recommended for medical domain)
   - Contriever (general domain)
   - SPECTER (scientific documents)
   - BM25 (lexical matching)

3. **Evaluate Retrieval Quality**:
   - Test retrieval with medical questions
   - Compare with other corpora (PubMed, Textbooks, etc.)
   - Analyze if graph community summaries provide unique insights

4. **Consider Hybrid Approaches**:
   - Combine UMLS with existing MedCorp
   - Use RRF (Reciprocal Rank Fusion) with multiple retrievers
   - Leverage both structured knowledge (UMLS) and unstructured text (PubMed, etc.)

## Benefits of UMLS Communities

1. **Structured Knowledge**: Communities represent coherent medical concepts
2. **Relationship Context**: Summaries include relationships between entities
3. **Multi-level Hierarchy**: Different granularities (runs, levels)
4. **Comprehensive Coverage**: Based on entire UMLS knowledge graph
5. **Semantic Coherence**: Communities are semantically related entities

## Notes

- The UMLS corpus complements existing text-based corpora with structured knowledge
- Community summaries provide contextual information about medical concept relationships
- The chunking strategy ensures compatibility with existing MedRAG infrastructure
- All processing follows the established MedRAG patterns for consistency
