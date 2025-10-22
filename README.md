# MIRAGE + MedRAG with VLLM Integration

Combined framework for medical question answering using MIRAGE benchmark and MedRAG retrieval system with VLLM optimization for Llama-3-8B.

## Features

- ✅ **VLLM Integration**: Optimized inference with Llama-3-8B-Instruct
- ✅ **MedRAG Retrieval**: Medical domain-specific retrieval with MedCPT
- ✅ **MIRAGE Benchmark**: Comprehensive evaluation on 5 medical QA datasets
- ✅ **GPU Optimization**: Configured for multi-GPU systems (default: cuda:4)
- ✅ **MedCorp Support**: Full medical corpus (Textbooks, PubMed, StatPearls, Wikipedia)

## Quick Start

### Prerequisites

```bash
# Python 3.10+
conda create -n medrag python=3.10
conda activate medrag

# Install dependencies
pip install torch transformers sentence-transformers faiss-cpu
pip install vllm>=0.11.0
pip install tqdm numpy
```

### Setup

1. **Clone this repository**:
```bash
git clone https://github.com/YOUR_USERNAME/mirage_medrag.git
cd mirage_medrag
```

2. **Download corpus data** (optional, but recommended):
   - Textbooks: Auto-downloaded on first run
   - Full MedCorp: ~61GB, see instructions below

3. **Test the setup**:
```bash
./MIRAGE/run_benchmark.sh test
```

## Usage

### Run Benchmark on Single Dataset

```bash
# MMLU (smallest, good for testing)
python MIRAGE/run_benchmark_vllm.py --dataset mmlu --mode rag --k 32

# MedQA
python MIRAGE/run_benchmark_vllm.py --dataset medqa --mode rag --k 32
```

### Run Full Benchmark

```bash
# All 5 datasets (7,663 questions total)
python MIRAGE/run_benchmark_vllm.py --dataset all --mode rag --k 32
```

### Available Datasets

- `mmlu` - Medical MMLU (1,089 questions)
- `medqa` - MedQA-USMLE (1,273 questions)
- `medmcqa` - MedMCQA (4,183 questions)
- `pubmedqa` - PubMedQA (500 questions)
- `bioasq` - BioASQ (618 questions)

### Options

```bash
# Change model
--llm_name "meta-llama/Meta-Llama-3-8B-Instruct"

# Change retriever
--retriever_name MedCPT  # or RRF-2, RRF-4, BM25, Contriever

# Change corpus
--corpus_name MedCorp  # or Textbooks, PubMed, StatPearls

# Change number of retrieved snippets
--k 32  # default is 32

# Run without RAG (CoT only)
--mode cot
```

## Repository Structure

```
mirage_medrag/
├── MIRAGE/                      # MIRAGE benchmark framework
│   ├── src/
│   │   ├── evaluate.py         # Evaluation script
│   │   └── utils.py            # Dataset utilities
│   ├── data/                   # Benchmark datasets
│   ├── run_benchmark_vllm.py   # Main benchmark runner (VLLM optimized)
│   ├── test_setup.py           # Setup verification
│   └── run_benchmark.sh        # Convenience shell script
│
├── MedRAG/                     # MedRAG retrieval system
│   ├── src/
│   │   ├── medrag.py          # Core MedRAG implementation
│   │   ├── utils.py           # Retrieval utilities
│   │   ├── template.py        # Prompt templates
│   │   └── config.py          # Configuration
│   ├── run_medrag_vllm.py     # VLLM integration layer
│   └── requirements.txt
│
├── .gitignore                  # Git ignore rules
└── README.md                   # This file
```

## Configuration

### GPU Selection

By default, the system uses GPU 4 (cuda:4). To change:

Edit `MedRAG/run_medrag_vllm.py`:
```python
os.environ['CUDA_VISIBLE_DEVICES'] = '4'  # Change to your GPU number
```

### Memory Settings

Current settings (in `MedRAG/run_medrag_vllm.py`):
- `max_model_len`: 4096 tokens
- `gpu_memory_utilization`: 0.5 (50% of GPU memory)

Adjust these based on your GPU capacity.

## Corpus Data

The system supports multiple medical corpora:

### Automatic Download (Recommended for Testing)
- **Textbooks**: Auto-downloads on first run (~341MB embeddings)

### Manual Download (For Full MedCorp)

Full MedCorp includes 4 sources. Place them in `MedRAG/src/data/corpus/`:

```bash
cd MedRAG/src/data/corpus

# Clone from HuggingFace (without LFS, just metadata)
git clone https://huggingface.co/datasets/MedRAG/textbooks textbooks
git clone https://huggingface.co/datasets/MedRAG/pubmed pubmed
git clone https://huggingface.co/datasets/MedRAG/statpearls statpearls
git clone https://huggingface.co/datasets/MedRAG/wikipedia wikipedia
```

The system will automatically download embeddings when needed.

## Evaluation

After running predictions, evaluate results:

```bash
cd MIRAGE

# Evaluate RAG results
python src/evaluate.py \
    --results_dir ./prediction \
    --llm_name meta-llama/Meta-Llama-3-8B-Instruct \
    --rag --k 32 \
    --corpus_name medcorp \
    --retriever_name MedCPT

# Evaluate CoT results
python src/evaluate.py \
    --results_dir ./prediction \
    --llm_name meta-llama/Meta-Llama-3-8B-Instruct
```

## Key Modifications from Original

1. **VLLM Integration**: Added `MedRAG/run_medrag_vllm.py` for efficient inference
2. **Llama-3 Support**: Updated for Llama-3-8B-Instruct model
3. **GPU Configuration**: Optimized for specific GPU selection
4. **Path Fixes**: All paths updated for combined repository structure
5. **Import Fixes**: Resolved conflicts between MIRAGE and MedRAG modules
6. **Memory Optimization**: Reduced memory usage to fit on consumer GPUs

## Troubleshooting

### Out of Memory Error
Reduce memory usage in `MedRAG/run_medrag_vllm.py`:
```python
gpu_memory_utilization=0.4  # Reduce from 0.5
max_model_len=2048           # Reduce from 4096
```

### Import Errors
Ensure you're running from the repository root:
```bash
cd /path/to/mirage_medrag
python MIRAGE/run_benchmark_vllm.py ...
```

### Corpus Not Found
The system will auto-download on first run. If issues occur:
```bash
cd MedRAG/src/data/corpus
git clone https://huggingface.co/datasets/MedRAG/textbooks textbooks
```

## Performance

Expected inference speed with Llama-3-8B on GPU:
- MMLU (1,089 questions): ~2-3 hours
- Full benchmark (7,663 questions): ~24-36 hours

## Citation

If you use this code, please cite the original papers:

```bibtex
@article{mirage2024,
  title={MIRAGE: Multimodal Evaluation Benchmark},
  author={...},
  year={2024}
}

@article{medrag2024,
  title={MedRAG: Medical Retrieval-Augmented Generation},
  author={...},
  year={2024}
}
```

## License

This project combines:
- MIRAGE: [Original License](https://github.com/Teddy-XiongGZ/MIRAGE)
- MedRAG: [Original License](https://github.com/Teddy-XiongGZ/MedRAG)

## Acknowledgments

- Original MIRAGE framework by Teddy-XiongGZ
- Original MedRAG system by Teddy-XiongGZ
- VLLM team for efficient LLM inference

## Contact

For issues related to this integration, please open a GitHub issue.
For questions about the original frameworks, refer to their respective repositories.
