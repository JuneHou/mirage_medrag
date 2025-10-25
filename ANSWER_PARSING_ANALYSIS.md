# Answer Parsing Code Analysis for PMC-LLaMA

## Summary
The answer parsing code in your `mirage_medrag` folder is located in **both MedRAG and MIRAGE** components, with **significant modifications** compared to the original repositories.

---

## 1. Where is the Answer Parsing Code?

### In Your Local Repository (`/data/wang/junh/githubs/mirage_medrag`)

#### **A. MedRAG Component** (Primary answer parsing)
- **Location**: `/data/wang/junh/githubs/mirage_medrag/MedRAG/run_medrag_vllm.py`
- **Functions**:
  - `parse_response_standard()` (lines 121-217) - Universal parser for all models
  - `parse_response_vllm()` (lines 114-119) - Legacy wrapper that redirects to standard parser
  - `vllm_medrag_answer()` (lines 219-267) - Wrapper that calls parsing

#### **B. MIRAGE Component** (Evaluation-time parsing)
- **Location**: `/data/wang/junh/githubs/mirage_medrag/MIRAGE/src/evaluate.py`
- **Functions**:
  - `evaluate()` (lines 10-52) - Extracts answer_choice from saved predictions
  - Uses helper functions from `utils.py`:
    - `locate_answer()` - General answer extraction
    - `locate_answer4pub_llama()` - Special PMC-LLaMA answer extraction

#### **C. MIRAGE Utils** (Answer extraction helpers)
- **Location**: `/data/wang/junh/githubs/mirage_medrag/MIRAGE/src/utils.py`
- **Functions**:
  - `locate_answer()` (lines 27-66) - Regex-based answer extraction
  - `locate_answer4pub_llama()` (lines 68-91) - PMC-LLaMA specific extraction

---

## 2. Comparison with Original GitHub Repositories

### **Original Repositories**
1. **MedRAG**: https://github.com/Teddy-XiongGZ/MedRAG
2. **MIRAGE**: https://github.com/Teddy-XiongGZ/MIRAGE

### **Key Differences**

#### **A. MedRAG Differences**

| Aspect | Original Repository | Your Modified Version |
|--------|-------------------|----------------------|
| **Main file** | `src/medrag.py` only | Added `run_medrag_vllm.py` |
| **Answer parsing** | Built into MedRAG class, returns raw strings | External parsing functions in `run_medrag_vllm.py` |
| **VLLM support** | ❌ No VLLM support | ✅ Full VLLM integration with monkey-patching |
| **PMC-LLaMA parsing** | Basic template support only | **CUSTOM PMC-LLaMA parsing logic** (lines 136-149) |

**Your Added PMC-LLaMA Parsing (NOT in original)**:
```python
# Lines 136-149 in run_medrag_vllm.py
# PMC-LLaMA specific: Handle array format like [{"text": "...", "answer_choice": "B"}]
if model_name and "pmc" in model_name.lower():
    array_match = re.search(r'\[\s*\{[^}]*"(?:answer_choice|answer)"\s*:\s*"([ABCD])"[^}]*\}\s*\]', response, re.DOTALL)
    if array_match:
        answer_choice = array_match.group(1)
        # Extract text field if available
        text_match = re.search(r'"text"\s*:\s*"([^"]*)"', array_match.group(0))
        reasoning = text_match.group(1) if text_match else "PMC-LLaMA response parsed"
        result = {
            "step_by_step_thinking": reasoning,
            "answer_choice": answer_choice
        }
        print(f"DEBUG: PMC-LLaMA array format parsed - Answer: {answer_choice}, Reasoning length: {len(reasoning)}")
        return result
```

This **specialized parsing for PMC-LLaMA array format** does NOT exist in the original MedRAG repository!

#### **B. MIRAGE Differences**

| Aspect | Original Repository | Your Modified Version |
|--------|-------------------|----------------------|
| **Main evaluation** | `src/evaluate.py` | Same file, **modified** (lines 27-34) |
| **Answer format** | String parsing only: `it.split('"answer_choice": "')` | ✅ **Handles both dict and string** |
| **PMC-LLaMA support** | Has `locate_answer4pub_llama()` | **Same function preserved** |

**Your Modified evaluate.py (lines 27-34)**:
```python
for it in json.load(open(fpath))[:1]:
    if isinstance(it, dict):
        # New format: JSON object
        answer_choice = it.get("answer_choice", "A")
        answers.append(locate_fun(answer_choice))
    else:
        # Old format: JSON string
        answers.append(locate_fun(it.split('"answer_choice": "')[-1].strip()))
```

The original MIRAGE only has:
```python
for it in json.load(open(fpath))[:1]:
    answers.append(locate_fun(it.split('"answer_choice": "')[-1].strip()))
```

Your version adds **dict format support** which the original lacks.

---

## 3. Is Parsing Under MedRAG or MIRAGE?

### **Answer: BOTH, but with different roles**

```
┌─────────────────────────────────────────────────────┐
│              Answer Parsing Pipeline                 │
└─────────────────────────────────────────────────────┘

1. MODEL GENERATES RESPONSE
   ↓
2. MedRAG Component (run_medrag_vllm.py)
   ├─ parse_response_standard()
   │  └─ PMC-LLaMA specific array parsing (YOUR ADDITION)
   │  └─ JSON extraction
   │  └─ Fallback regex patterns
   ↓
3. SAVES TO FILE (prediction/*.json)
   Format: [{"step_by_step_thinking": "...", "answer_choice": "B"}]
   ↓
4. MIRAGE Component (src/evaluate.py)
   ├─ Reads saved predictions
   ├─ Extracts answer_choice (handles dict AND string - YOUR MOD)
   └─ Uses locate_answer() or locate_answer4pub_llama()
   ↓
5. CALCULATES ACCURACY
```

---

## 4. What's Causing Your PMC-LLaMA Problem?

Based on the code analysis, the issue is likely in **Step 2** (MedRAG parsing):

### **Problem Areas**:

1. **Array Format Mismatch** (lines 136-149 in run_medrag_vllm.py)
   - Your custom parsing expects: `[{"text": "...", "answer_choice": "B"}]`
   - PMC-LLaMA might be generating a different format
   - **Solution**: Add debug prints to see actual output

2. **Model Detection** (line 136)
   ```python
   if model_name and "pmc" in model_name.lower():
   ```
   - Check if your model name contains "pmc"
   - If not, it won't use PMC-LLaMA specific parsing

3. **Fallback Chain** (lines 151-195)
   - If array parsing fails, it tries JSON extraction
   - Then regex patterns
   - Finally defaults to "A"

### **Recommended Debug Steps**:

```bash
# 1. Check what model name is being used
grep -n "llm_name" /data/wang/junh/githubs/mirage_medrag/MedRAG/run_medrag_vllm.py

# 2. Add debug output to see raw responses
# Edit line 130 in run_medrag_vllm.py to print more context

# 3. Check actual PMC-LLaMA output format
# Look at generated prediction files in prediction/ directory
```

---

## 5. Key Modifications Summary

### **Your Additions (Not in Original)**:
1. ✅ VLLM integration (`patch_medrag_for_vllm()`)
2. ✅ PMC-LLaMA array format parsing (lines 136-149)
3. ✅ Universal `parse_response_standard()` function
4. ✅ Dict format support in evaluate.py
5. ✅ Model-specific parsing routing

### **Preserved from Original**:
1. ✅ `locate_answer()` function
2. ✅ `locate_answer4pub_llama()` function
3. ✅ MIRAGE evaluation framework
4. ✅ MedRAG core architecture

---

## 6. Files to Check for Debugging

```bash
# Primary parsing code (YOUR MODIFICATIONS)
/data/wang/junh/githubs/mirage_medrag/MedRAG/run_medrag_vllm.py
  - Lines 121-217: parse_response_standard()
  - Lines 136-149: PMC-LLaMA specific parsing

# Evaluation parsing (MODIFIED from original)
/data/wang/junh/githubs/mirage_medrag/MIRAGE/src/evaluate.py
  - Lines 27-34: Dict/string format handling
  - Line 95: PMC-LLaMA detection

# Answer extraction utilities (FROM ORIGINAL)
/data/wang/junh/githubs/mirage_medrag/MIRAGE/src/utils.py
  - Lines 27-66: locate_answer()
  - Lines 68-91: locate_answer4pub_llama()

# Model configuration (YOUR MODIFICATIONS)
/data/wang/junh/githubs/mirage_medrag/MedRAG/src/medrag.py
  - Lines 103-109: PMC-LLaMA template and config
```

---

## 7. Quick Test

Run this to see what your PMC-LLaMA is actually outputting:

```python
# Add this to run_medrag_vllm.py line 130 (inside parse_response_standard)
print("="*80)
print(f"MODEL: {model_name}")
print(f"RAW RESPONSE:\n{raw_response}")
print("="*80)
```

This will show you the exact format PMC-LLaMA is generating, which you can then match in your parsing logic.

---

## Conclusion

Your answer parsing is **heavily modified** from the original repositories:
- **MedRAG**: Added entire VLLM integration layer with custom PMC-LLaMA parsing
- **MIRAGE**: Enhanced to handle both dict and string formats

The PMC-LLaMA issue is likely due to **format mismatch** between what your custom parser expects and what the model actually generates. Debug the raw output to see the exact format.
