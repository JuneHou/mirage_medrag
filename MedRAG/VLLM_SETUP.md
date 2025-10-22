# MedRAG with VLLM and Llama-7B Setup Guide

This guide shows how to run MedRAG with a Llama-7B model using VLLM backend for improved performance.

## Prerequisites

### 1. Install Required Packages
```bash
# Navigate to MedRAG directory
cd /data/wang/junh/githubs/MedRAG

# Install MedRAG requirements
pip install -r requirements.txt

# Install VLLM (you already have this)
# pip install vllm

# Git-lfs is NOT required since you have existing corpus data
# (Only needed if downloading original corpus from repository)
```

### 2. Install Java (Required for BM25 retriever)
```bash
# Ubuntu/Debian
sudo apt-get install openjdk-11-jdk

# macOS
brew install openjdk@11

# Verify installation
java -version
```

### 3. GPU Requirements
- **Minimum**: 8GB GPU memory for Llama-7B
- **Recommended**: 16GB+ GPU memory for optimal performance
- CUDA-compatible GPU (NVIDIA)

## Quick Start

### 1. Run the Example Script
```bash
cd /data/wang/junh/githubs/MedRAG
python run_medrag_vllm.py
```

This will:
- Download and cache the MedCorp corpus (first run only)
- Load Llama-2-7B-chat model with VLLM
- Run a sample medical question
- Show retrieved snippets and generated answer

### 2. Customize the Model
Edit `run_medrag_vllm.py` and change the model name:
```python
# Choose your preferred Llama-7B model:
model_name = "meta-llama/Llama-2-7b-chat-hf"           # Llama 2 7B
# model_name = "meta-llama/Meta-Llama-3-8B-Instruct"   # Llama 3 8B (closest to 7B)
# model_name = "meta-llama/Llama-3.2-8B-Instruct"      # Latest Llama 3.2 8B
```

### 3. Use in Your Code
```python
import sys
sys.path.append("/data/wang/junh/githubs/MedRAG")
from run_medrag_vllm import patch_medrag_for_vllm
from src.medrag import MedRAG

# Enable VLLM backend
patch_medrag_for_vllm()

# Initialize MedRAG
medrag = MedRAG(
    llm_name="meta-llama/Llama-2-7b-chat-hf",
    rag=True,
    retriever_name="MedCPT",
    corpus_name="MedCorp",
    db_dir="./src/data/corpus",  # Use existing corpus location
    corpus_cache=True,
    HNSW=True
)

# Ask questions
question = "What is the treatment for pneumonia?"
answer, snippets, scores = medrag.answer(question=question, k=10)
print(answer)
```

## Available Configurations

### Corpus Options
- `"MedCorp"`: Combined medical corpus (recommended)
- `"PubMed"`: Biomedical abstracts only
- `"Textbooks"`: Medical textbooks only
- `"StatPearls"`: Clinical decision support
- `"Wikipedia"`: General knowledge

### Retriever Options
- `"MedCPT"`: Medical domain retriever (recommended)
- `"Contriever"`: General domain retriever
- `"SPECTER"`: Scientific domain retriever
- `"BM25"`: Lexical retriever (requires Java)

## Performance Optimization

### 1. Memory Optimization
If you have limited GPU memory, edit `run_medrag_vllm.py`:
```python
self.llm = LLM(
    model=model_name,
    max_model_len=2048,        # Reduce from 4096
    tensor_parallel_size=1,     # Use multiple GPUs if available
    gpu_memory_utilization=0.8  # Adjust GPU memory usage
)
```

### 2. Faster Retrieval
```python
medrag = MedRAG(
    # ... other parameters ...
    corpus_cache=True,  # Cache corpus in memory
    HNSW=True,         # Use HNSW for faster search
)
```

## Troubleshooting

### Common Issues

1. **Out of Memory Error**:
   - Reduce `max_model_len` in VLLMWrapper
   - Use smaller model like Llama-3.2-3B-Instruct
   - Reduce `tensor_parallel_size`

2. **Corpus Download Fails**:
   - Check internet connection for SharePoint downloads
   - Your local corpus should work fine without Git LFS
   - Try downloading manually from HuggingFace if needed

3. **Java Error for BM25**:
   - Install OpenJDK: `sudo apt-get install openjdk-11-jdk`
   - Set JAVA_HOME if needed
   - Use different retriever: `retriever_name="MedCPT"`

4. **Model Access Error**:
   - Ensure you have access to Llama models on HuggingFace
   - Login with: `huggingface-cli login`
   - Request access for restricted models

### Performance Tips
- First run will be slower due to corpus and model downloads
- Subsequent runs use cached data and are much faster
- Use `corpus_cache=True` for repeated queries
- Consider using smaller models for development/testing

## Example Output
```
✓ VLLM is available
✓ Git LFS is available
Initializing MedRAG with meta-llama/Llama-2-7b-chat-hf and MedCorp dataset...
Using VLLM for model: meta-llama/Llama-2-7b-chat-hf
Running example question...
Question: What are the main symptoms of acute myocardial infarction?

MedRAG Answer: {
  "step_by_step_thinking": "Based on the medical literature retrieved, acute myocardial infarction commonly presents with chest pain, shortness of breath, and nausea...",
  "answer_choice": "A"
}

Retrieved 10 relevant snippets
✓ MedRAG with VLLM setup completed successfully!
```