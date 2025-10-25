# PMC-LLaMA Consistency with Original MedRAG

## Changes Made to Match Original MedRAG Repository

Based on analysis of the original MedRAG repository (https://github.com/Teddy-XiongGZ/MedRAG), the following changes were made to ensure consistency with the original paper's methodology:

### Key Findings from Original Repository

1. **No Special PMC-LLaMA Parsing**: The original MedRAG does NOT use any special parsing logic for PMC-LLaMA
2. **Unified JSON Format**: All models (including PMC-LLaMA) are expected to produce the same JSON format:
   ```json
   {"step_by_step_thinking": "...", "answer_choice": "A"}
   ```
3. **Simple Chat Template**: PMC-LLaMA uses `templates/pmc_llama.jinja` which is a simple chat formatting template
4. **Model Configuration**: In original `medrag.py` lines 100-110:
   - PMC-LLaMA max_length: 2048
   - PMC-LLaMA context_length: 1024
   - Uses custom chat template but expects standard JSON output

### Changes Made

#### 1. Removed Custom PMC-LLaMA Parser
- **Before**: Used `parse_pmc_llama_response()` with special PMC-LLaMA handling
- **After**: Use `parse_response_standard()` for ALL models including PMC-LLaMA

#### 2. Unified Response Parsing
- **File**: `run_medrag_vllm.py`
- **Function**: `parse_response_standard()` 
- **Purpose**: Single parser that handles all models consistently

#### 3. Updated Model Detection Logic
- **File**: `run_medrag_vllm.py`
- **Function**: `vllm_medrag_answer()`
- **Change**: Removed model-specific branching, use standard parser for all

### Original MedRAG Template Analysis

From `templates/pmc_llama.jinja`:
```jinja
{% if messages[0]['role'] == 'system' %}
    {% set loop_messages = messages[1:] %}
    {% set system_message = messages[0]['content'].strip() + '\n' %}
{% else %}
    {% set loop_messages = messages %}
    {% set system_message = '' %}
{% endif %}
{% for message in loop_messages %}
    {% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}
        {{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}
    {% endif %}
    {% if loop.index0 == 0 %}
        {% set content = system_message + message['content'] %}
    {% else %}
        {% set content = message['content'] %}
    {% endif %}
    {% if message['role'] == 'user' %}
        {{ content.strip() }}
    {% elif message['role'] == 'assistant' %}
        {{ ' '  + content.strip() + ' ' + eos_token }}
    {% endif %}
{% endfor %}
```

This template is simple and doesn't include any special JSON formatting hints, confirming that PMC-LLaMA should naturally produce JSON through proper system prompts.

### Original MedRAG System Prompts

From `template.py` line 0-15:
```python
general_medrag_system = '''You are a helpful medical expert, and your task is to answer a multi-choice medical question using the relevant documents. Please first think step-by-step and then choose the answer from the provided options. Organize your output in a json formatted as Dict{"step_by_step_thinking": Str(explanation), "answer_choice": Str{A/B/C/...}}. Your responses will be used for research purposes only, so please have a definite answer.'''
```

This confirms that ALL models receive the same system prompt asking for JSON output.

### Consistency Verification

The updated implementation now matches the original MedRAG approach:

1. ✅ Uses same JSON format expectation for all models
2. ✅ No special PMC-LLaMA parsing logic
3. ✅ Relies on proper prompting to generate JSON
4. ✅ Uses standard chat templates for formatting
5. ✅ Maintains consistent model configuration

### Expected Behavior

With these changes:
- PMC-LLaMA should generate JSON responses similar to other models
- Response parsing is consistent across all models
- Results should be more aligned with original MedRAG paper benchmarks
- Accuracy should improve from previous ~25% to expected performance levels

### Testing Recommendations

1. Test PMC-LLaMA with a few sample questions to verify JSON output format
2. Compare accuracy with original MedRAG paper results
3. Ensure VLLM configuration matches original transformers pipeline behavior
4. Validate that chat templates are applied correctly

This approach ensures full consistency with the original MedRAG repository methodology.