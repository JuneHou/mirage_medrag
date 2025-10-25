# Chat Template Explanation for Different Models

## Overview

The code uses **two types of templates**:

1. **Custom Templates** (`.jinja` files in `templates/` directory)
2. **Built-in HuggingFace Templates** (from the model's tokenizer)

---

## 📁 Available Custom Templates

### 1. **PMC-LLaMA** (`pmc_llama.jinja`)
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

**Format Output**:
```
System message content
User message<eos_token>
```

### 2. **Mixtral/Mistral** (`mistral-instruct.jinja`)
```jinja
{{ bos_token }}
{% for message in loop_messages %}
    {% if message['role'] == 'user' %}
        {{ '[INST] ' + content.strip() + ' [/INST]' }}
    {% elif message['role'] == 'assistant' %}
        {{ ' '  + content.strip() + ' ' + eos_token }}
    {% endif %}
{% endfor %}
```

**Format Output**:
```
<bos_token>[INST] System + User message [/INST] assistant response<eos_token>
```

### 3. **Meditron** (`meditron.jinja`)
Similar to PMC-LLaMA (simple format without special tokens)

---

## 🤖 Models Using Built-in Templates

### **Llama-2**
- ❌ **No custom template file**
- ✅ Uses HuggingFace's built-in template from the model

**Default Llama-2 format** (from HuggingFace):
```
<s>[INST] <<SYS>>
System message
<</SYS>>

User message [/INST] Assistant response </s>
```

### **Llama-3** (and Llama-3.1, Llama-3.2)
- ❌ **No custom template file**
- ✅ Uses HuggingFace's built-in template

**Default Llama-3 format** (from HuggingFace):
```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

System message<|eot_id|><|start_header_id|>user<|end_header_id|>

User message<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Assistant response<|eot_id|>
```

### **Qwen** (Qwen2, Qwen3)
- ❌ **No custom template file**
- ✅ Uses HuggingFace's built-in template

**Default Qwen format** (from HuggingFace):
```
<|im_start|>system
System message<|im_end|>
<|im_start|>user
User message<|im_end|>
<|im_start|>assistant
Assistant response<|im_end|>
```

---

## 🔍 How It Works in Code

### Code Structure:
```python
if "mixtral" in llm_name.lower():
    # Load CUSTOM template
    template_path = os.path.join(..., 'templates', 'mistral-instruct.jinja')
    self.tokenizer.chat_template = open(template_path).read()...
    
elif "pmc" in llm_name.lower():
    # Load CUSTOM template
    template_path = os.path.join(..., 'templates', 'pmc_llama.jinja')
    self.tokenizer.chat_template = open(template_path).read()...
    
elif "llama-2" in llm_name.lower():
    # NO template loading - uses model's built-in template
    self.max_length = 4096
    
elif "llama-3" in llm_name.lower():
    # NO template loading - uses model's built-in template
    self.max_length = 8192
    
elif "qwen" in llm_name.lower():
    # NO template loading - uses model's built-in template
    self.max_length = 8192
```

### When Templates Are Applied:
```python
# In generate() method (line ~160):
prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
```

This line:
1. Takes the messages: `[{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]`
2. Applies the template (custom or built-in)
3. Returns formatted string ready for the model

---

## 📊 Template Comparison

| Model | Custom Template? | Template File | Format Style |
|-------|-----------------|---------------|--------------|
| PMC-LLaMA | ✅ Yes | `pmc_llama.jinja` | Simple (no special tokens) |
| Mixtral/Mistral | ✅ Yes | `mistral-instruct.jinja` | `[INST]...[/INST]` style |
| Meditron | ✅ Yes | `meditron.jinja` | Simple (no special tokens) |
| Llama-2 | ❌ No | Built-in | `[INST]<<SYS>>...<</SYS>>[/INST]` |
| Llama-3/3.1/3.2 | ❌ No | Built-in | `<\|start_header_id\|>...<\|eot_id\|>` |
| Qwen 2/3 | ❌ No | Built-in | `<\|im_start\|>...<\|im_end\|>` |

---

## 🆕 Creating Templates for Llama/Qwen (Optional)

If you want to create custom templates for Llama or Qwen (though not necessary), here's how:

### Example: Custom Llama-3 Template
```python
# In medrag.py, add this after line 98:
elif "llama-3" in llm_name.lower():
    template_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'templates', 'llama3.jinja')
    if os.path.exists(template_path):
        self.tokenizer.chat_template = open(template_path).read().replace('    ', '').replace('\n', '')
    # else: use built-in template
    self.max_length = 8192
    self.context_length = 7168
```

### Create `templates/llama3.jinja`:
```jinja
{% for message in messages %}
    {% if message['role'] == 'system' %}
        {{ '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n' + message['content'].strip() + '<|eot_id|>' }}
    {% elif message['role'] == 'user' %}
        {{ '<|start_header_id|>user<|end_header_id|>\n\n' + message['content'].strip() + '<|eot_id|>' }}
    {% elif message['role'] == 'assistant' %}
        {{ '<|start_header_id|>assistant<|end_header_id|>\n\n' + message['content'].strip() + '<|eot_id|>' }}
    {% endif %}
{% endfor %}
{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}
```

### Create `templates/qwen.jinja`:
```jinja
{% for message in messages %}
    {% if message['role'] == 'system' %}
        {{ '<|im_start|>system\n' + message['content'].strip() + '<|im_end|>\n' }}
    {% elif message['role'] == 'user' %}
        {{ '<|im_start|>user\n' + message['content'].strip() + '<|im_end|>\n' }}
    {% elif message['role'] == 'assistant' %}
        {{ '<|im_start|>assistant\n' + message['content'].strip() + '<|im_end|>\n' }}
    {% endif %}
{% endfor %}
{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}
```

---

## 🎯 Why Some Models Have Custom Templates?

### **Reasons for Custom Templates:**

1. **PMC-LLaMA**: Medical domain-specific model, may need simpler format
2. **Mixtral/Mistral**: Ensure consistent `[INST]` format across versions
3. **Meditron**: Medical domain-specific, simple format

### **Why Llama/Qwen Don't Need Custom Templates:**

1. ✅ **HuggingFace provides official templates** with the model
2. ✅ **Templates are well-maintained** and updated by model authors
3. ✅ **Standardized format** across model versions
4. ✅ **No need to maintain separate files**

---

## 🔧 Debugging Template Issues

### To see what template is being used:
```python
# Add after line 85 in medrag.py:
print(f"DEBUG: Chat template for {llm_name}:")
print(self.tokenizer.chat_template)
```

### To see formatted prompt:
```python
# Add in generate() method after line 160:
print(f"DEBUG: Formatted prompt:\n{prompt}")
```

### To test template manually:
```python
from transformers import AutoTokenizer

# Test Llama-3
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is 2+2?"}
]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print(prompt)
```

---

## 📝 Summary

**For your question**: 
- **Llama-2, Llama-3, Qwen**: ❌ No template files exist
- They use **built-in HuggingFace templates** automatically
- The code only sets `max_length` and `context_length` for these models
- **PMC-LLaMA and Mixtral**: ✅ Have custom template files

**This is by design** - HuggingFace's built-in templates are sufficient and well-tested for these popular models!
