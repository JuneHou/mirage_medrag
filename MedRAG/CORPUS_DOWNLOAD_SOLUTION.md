# MedRAG Corpus Download Solution

## Problem Summary
The MedRAG corpus files are stored using Git LFS on HuggingFace, which requires Git LFS to download. Since you don't have root access to install Git LFS, we need an alternative approach.

## Solution: Direct Download via HuggingFace Hub

We can use the `huggingface_hub` Python library to download the actual content files directly, bypassing Git LFS entirely.

## What We've Verified

✅ **HuggingFace Hub Direct Download Works**: The `huggingface_hub` library can download actual file content
✅ **Test Download Successful**: Successfully downloaded 3 textbook files (Anatomy_Gray.jsonl, Biochemistry_Lippincott.jsonl, Cell_Biology_Alberts.jsonl)
✅ **Content is Valid**: Downloaded files contain proper JSON content, not LFS pointers
✅ **Storage Efficient**: Uses symlinks to cache directory to save space

## Corpus Sizes
- **Textbooks**: 18 files (~20MB) - ✅ Test downloaded
- **Wikipedia**: 646 files (~650MB) - Ready to download  
- **PubMed**: 1166 files (~2GB) - Ready to download
- **StatPearls**: Already available (434MB) - ✅ Has real content

## Download Options

### Option 1: Quick Test (Textbooks Only)
```bash
cd /data/wang/junh/githubs/MedRAG
python download_medcorp.py
# Choose option 1
```
Use with: `corpus_name="Textbooks"`

### Option 2: Medium Dataset (Textbooks + Wikipedia)  
```bash
cd /data/wang/junh/githubs/MedRAG
python download_medcorp.py
# Choose option 2
```
Use with: `corpus_name="Textbooks"` or `corpus_name="Wikipedia"`

### Option 3: Full MedCorp Dataset
```bash
cd /data/wang/junh/githubs/MedRAG
python download_medcorp.py
# Choose option 3
```
Use with: `corpus_name="MedCorp"` (full dataset)

## Usage After Download

1. **Fix Dependencies** (if not already done):
   ```bash
   conda activate medrag  # or your environment
   pip install sentence-transformers==5.1.1
   pip install "numpy<2.0"
   ```

2. **Run MedRAG with VLLM**:
   ```python
   from medrag import MedRAG
   
   medrag = MedRAG(
       llm_name="meta-llama/Llama-2-7b-chat-hf",
       rag=True,
       retriever_name="MedCPT", 
       corpus_name="MedCorp",  # or "Textbooks", "Wikipedia", etc.
       db_dir="./src/data/corpus",
       corpus_cache=True,
       HNSW=False
   )
   ```

## Storage Locations
- **Cache**: `/data/wang/junh/hf_cache/` (actual downloaded files)
- **Corpus**: `./src/data/corpus/*/chunk/` (symlinks to cache)
- **Total Space**: Cache + minimal symlink overhead

## Benefits of This Approach
- ✅ No root access required
- ✅ No Git LFS installation needed
- ✅ Efficient storage (symlinks to shared cache)
- ✅ Can download incrementally 
- ✅ Full MedCorp dataset access
- ✅ Compatible with existing MedRAG code

## Files Created
- `download_medcorp.py` - Interactive corpus downloader
- `test_download.py` - Test script (already successful)
- `run_medrag_vllm.py` - VLLM integration script (update corpus_name as needed)