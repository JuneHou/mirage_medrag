# Debug Configuration Guide for PMC-LLaMA

## ✅ Fixed Issues

### 1. **Correct Python Environment**
- **Before**: Used `kare_env` ❌
- **After**: Uses `medrag` environment ✅

### 2. **Correct Working Directory**
- **Before**: `/data/wang/junh/githubs/KARE/ehr_prepare` ❌
- **After**: `/data/wang/junh/githubs/mirage_medrag/MIRAGE` ✅

### 3. **Proper PYTHONPATH**
- **Before**: `${workspaceFolder}` (ambiguous) ❌
- **After**: Explicit paths to all modules ✅
  ```
  /data/wang/junh/githubs/mirage_medrag
  /data/wang/junh/githubs/mirage_medrag/MedRAG
  /data/wang/junh/githubs/mirage_medrag/MIRAGE/src
  ```

### 4. **Removed Syntax Errors**
- **Before**: Trailing comma in args array ❌
- **After**: Clean JSON syntax ✅

### 5. **Added Fast Testing Options**
- **New**: `--max_questions` parameter to process only N questions
- **New**: Quick test configuration (2 questions)

---

## 🚀 How to Use

### Option 1: Quick Test (2 Questions)
**Perfect for debugging and testing parsing logic**

1. In VS Code, open the Run and Debug panel (Ctrl+Shift+D)
2. Select: **"Debug run_benchmark_vllm.py (Quick Test - 2 questions)"**
3. Press F5 or click the green play button

This will:
- Process only 2 questions from MMLU
- Use PMC-LLaMA-7B model
- Enable RAG with 32 snippets
- Resume from previous runs (skip already processed)

**Estimated time**: ~2-5 minutes (after initial model loading)

### Option 2: Process 10 Questions
**Good for validating the full pipeline**

1. Select: **"Debug run_benchmark_vllm.py"**
2. Press F5

This will process 10 questions.

### Option 3: Full Dataset (No Debug)
**For production runs**

```bash
cd /data/wang/junh/githubs/mirage_medrag/MIRAGE

# Activate environment
source /data/wang/junh/envs/medrag/bin/activate

# Run full benchmark
python run_benchmark_vllm.py \
  --dataset mmlu \
  --llm_name chaoyi-wu/PMC_LLAMA_7B \
  --mode rag \
  --k 32 \
  --resume
```

---

## 🐛 Debug the Answer Parsing

### Add Breakpoints

1. **In `run_medrag_vllm.py`** (line 130):
   ```python
   # DEBUG: Print the response we're trying to parse
   print(f"DEBUG: Parsing response of length {len(raw_response)} for model: {model_name}")
   print(f"DEBUG: Response content: {raw_response[:500]}...")
   ```

2. **In `run_medrag_vllm.py`** (line 136 - PMC-LLaMA specific):
   - Set breakpoint here to see if PMC-LLaMA parsing is triggered
   - Check if `model_name` contains "pmc"
   - Inspect the actual response format

3. **In `run_benchmark_vllm.py`** (line 167):
   - Set breakpoint after getting answer
   - Check what `answer_dict` contains

### Check Output Files

After running, check the generated predictions:
```bash
cd /data/wang/junh/githubs/mirage_medrag/MIRAGE/prediction
ls -la mmlu/rag_32/chaoyi-wu/PMC_LLAMA_7B/medcorp/MedCPT/

# View first prediction
cat test_*.json | head -1 | jq .
```

Expected format:
```json
[
  {
    "step_by_step_thinking": "reasoning text here...",
    "answer_choice": "B"
  }
]
```

---

## 🔍 Troubleshooting

### Issue: "It stops without any message"

**Cause**: Model/corpus initialization takes time

**Solutions**:
1. **Be patient**: First run loads:
   - VLLM model (~30-60 seconds)
   - MedCPT retrievers (~15-30 seconds)
   - Corpus database (~10-20 seconds)
   
2. **Check GPU memory**:
   ```bash
   watch -n 1 nvidia-smi
   ```
   Look for GPU 5 (your CUDA_VISIBLE_DEVICES=5)

3. **Check if it's actually running**:
   ```bash
   ps aux | grep python | grep run_benchmark_vllm
   ```

4. **View real-time logs** (in another terminal):
   ```bash
   tail -f /data/wang/junh/githubs/mirage_medrag/MIRAGE/prediction/*.log
   ```

### Issue: "PMC-LLaMA generates wrong format"

**Debug steps**:

1. Add debug output in `parse_response_standard()`:
   ```python
   # At line 130
   with open('/tmp/pmc_llama_debug.txt', 'a') as f:
       f.write(f"\n{'='*80}\n")
       f.write(f"RAW RESPONSE:\n{raw_response}\n")
       f.write(f"MODEL: {model_name}\n")
   ```

2. Run with 2 questions and check:
   ```bash
   cat /tmp/pmc_llama_debug.txt
   ```

3. Match the actual format in the parsing regex (line 137)

### Issue: "Wrong Python environment being used"

Make sure VS Code is using the correct Python interpreter:

1. Press `Ctrl+Shift+P`
2. Type "Python: Select Interpreter"
3. Choose: `/data/wang/junh/envs/medrag/bin/python3.10`

---

## 📊 Monitor Progress

### Terminal Output
You should see:
```
Setting up VLLM...
Initializing MedRAG with chaoyi-wu/PMC_LLAMA_7B...
  Retriever: MedCPT
  Corpus: MedCorp
  K: 32
Loading model...
Processing MMLU
================================================================================
Saving to: ./prediction/mmlu/rag_32/chaoyi-wu/PMC_LLAMA_7B/medcorp/MedCPT
Processing first 2 questions (max_questions=2)
Processing mmlu: 100%|██████████| 2/2 [00:XX<00:00, X.XXs/it]
Processed: 2, Skipped: 0
```

### Check Results
```bash
# Count processed files
ls -1 prediction/mmlu/rag_32/chaoyi-wu/PMC_LLAMA_7B/medcorp/MedCPT/*.json | wc -l

# View a sample result
cat prediction/mmlu/rag_32/chaoyi-wu/PMC_LLAMA_7B/medcorp/MedCPT/test_*.json | head -1 | python -m json.tool
```

---

## ⚡ Speed Up Tips

### 1. Use Smaller K
```json
"args": [
  "--k", "8"  // Instead of 32
]
```

### 2. Cache Corpus
The code already uses:
- `corpus_cache=True` ✅
- `HNSW=True` ✅

### 3. Skip Already Processed
```json
"args": [
  "--resume"  // Skip existing predictions
]
```

### 4. Test with Single Question First
```json
"args": [
  "--max_questions", "1"
]
```

---

## 📝 Command Line Alternatives

### Quick Test from Terminal
```bash
cd /data/wang/junh/githubs/mirage_medrag/MIRAGE
source /data/wang/junh/envs/medrag/bin/activate

CUDA_VISIBLE_DEVICES=5 python run_benchmark_vllm.py \
  --dataset mmlu \
  --llm_name chaoyi-wu/PMC_LLAMA_7B \
  --mode rag \
  --k 32 \
  --max_questions 2
```

### Check Answer Parsing Only
```bash
# Test the parsing function directly
cd /data/wang/junh/githubs/mirage_medrag/MedRAG
python -c "
from run_medrag_vllm import parse_response_standard

# Test with sample PMC-LLaMA output
sample = '[{\"text\": \"Based on the context\", \"answer_choice\": \"B\"}]'
result = parse_response_standard(sample, 'PMC_LLAMA_7B')
print(result)
"
```

---

## ✨ New Features Added

### 1. `--max_questions` Parameter
Process only N questions (great for debugging):
```bash
python run_benchmark_vllm.py --max_questions 5
```

### 2. Two Debug Configurations
- **Full (10 questions)**: For pipeline validation
- **Quick (2 questions)**: For rapid debugging

### 3. Better Progress Reporting
Now shows:
```
Processing first 10 questions (max_questions=10)
Processed: 8, Skipped: 2
```

---

## 🎯 Recommended Workflow

1. **First Time Setup** (one time):
   - Run with `--max_questions 1` to verify everything loads
   - Check output file format
   - Verify answer parsing works

2. **Debug Answer Parsing** (iterative):
   - Use "Quick Test" configuration (2 questions)
   - Add print statements / breakpoints
   - Check `/tmp/pmc_llama_debug.txt` for raw outputs
   - Adjust parsing logic in `parse_response_standard()`

3. **Validate Full Pipeline** (before production):
   - Run with `--max_questions 10`
   - Check accuracy with evaluate.py
   - Review any errors

4. **Production Run**:
   - Remove `--max_questions`
   - Use `--resume` to continue interrupted runs
   - Monitor with `watch nvidia-smi`

---

## 🆘 Still Having Issues?

Check these files for the actual error messages:
1. VS Code Debug Console
2. Integrated Terminal output
3. `prediction/*.log` files
4. `/tmp/pmc_llama_debug.txt` (if you added debug output)

If the debugger doesn't start at all:
- Make sure `medrag` environment has `debugpy` installed
- Try: `pip install debugpy` in the medrag environment
