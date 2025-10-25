#!/usr/bin/env python3
"""
Minimal launcher script to run MedRAG with vllm backend for Llama-7B models.
This script provides a drop-in replacement for the transformers pipeline using vllm.
"""

import os
import sys

# Set GPU device to cuda:4
os.environ['CUDA_VISIBLE_DEVICES'] = '4'

# Get the absolute path to MedRAG directory
medrag_dir = os.path.dirname(os.path.abspath(__file__))
medrag_src = os.path.join(medrag_dir, 'src')

# Add to path at the very beginning to take precedence
if medrag_src in sys.path:
    sys.path.remove(medrag_src)
sys.path.insert(0, medrag_src)

# Now import medrag - it will use our src/medrag.py
import medrag
from medrag import MedRAG
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

class VLLMWrapper:
    """Wrapper to make VLLM compatible with MedRAG's transformers.pipeline interface"""
    
    def __init__(self, model_name, **kwargs):
        print(f"DEBUG: Initializing VLLMWrapper for model: {model_name}")
        self.model_name = model_name
        
        try:
            # Filter out kwargs that are not supported by VLLM
            vllm_supported_kwargs = {
                'tensor_parallel_size', 'dtype', 'quantization', 
                'gpu_memory_utilization', 'swap_space', 'enforce_eager',
                'max_model_len', 'trust_remote_code', 'download_dir',
                'load_format', 'seed'
            }
            
            # Only pass supported kwargs to VLLM
            vllm_kwargs = {k: v for k, v in kwargs.items() if k in vllm_supported_kwargs}
            
            # Initialize VLLM with optimized settings
            gpu_utilization = 0.7
                
            self.llm = LLM(
                model=model_name,
                tensor_parallel_size=1,  # Adjust based on your GPU count
                trust_remote_code=True,
                gpu_memory_utilization=gpu_utilization,
                **vllm_kwargs
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            print(f"DEBUG: VLLMWrapper initialized successfully for {model_name}")
            
        except Exception as e:
            print(f"ERROR: Failed to initialize VLLMWrapper for {model_name}: {e}")
            print(f"ERROR: Exception type: {type(e).__name__}")
            raise e

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
    def __call__(self, prompt, **kwargs):
        """Make the wrapper callable like transformers.pipeline"""
        # DEBUG: Track parameter flow
        print(f"DEBUG: VLLMWrapper received kwargs: {kwargs}")
        
        # Remove parameters that could conflict with VLLM/transformers
        # These are parameters that the original pipeline might set but we handle differently
        conflicting_params = {'max_new_tokens', 'truncation', 'stopping_criteria', 'return_full_text'}
        clean_kwargs = {k: v for k, v in kwargs.items() if k not in conflicting_params}
        print(f"DEBUG: Removed conflicting params, clean_kwargs: {clean_kwargs}")
        
        # Extract relevant parameters and set defaults
        do_sample = clean_kwargs.get('do_sample', False)
        
        # Original MedRAG uses max_length for total sequence length
        # For VLLM, we need to calculate max_new_tokens from max_length
        max_length = clean_kwargs.get('max_length', 2048)
        print(f"DEBUG: Using max_length={max_length}")
        
        # Calculate how many tokens are already in the prompt
        prompt_tokens = len(self.tokenizer.encode(prompt))
        
        # Calculate max_new_tokens as the difference
        max_new_tokens = max(1, max_length - prompt_tokens)  # At least 1 token
        print(f"DEBUG: Calculated max_new_tokens={max_new_tokens} (max_length={max_length} - prompt_tokens={prompt_tokens})")
        
        # Create sampling parameters for vllm
        sampling_params = SamplingParams(
            temperature=0.0 if not do_sample else 0.7,
            top_p=0.9 if do_sample else 1.0,
            max_tokens=max_new_tokens,  # This is the key - no conflict now
            stop=clean_kwargs.get('stop_sequences', None)
        )
        print(f"DEBUG: SamplingParams created with max_tokens={max_new_tokens}")
        
        # Generate response
        outputs = self.llm.generate([prompt], sampling_params)
        generated_text = outputs[0].outputs[0].text
        print(f"Generated text: {generated_text[:300]}...")  # Print first 300 chars
        
        # DEBUG: Print response details
        print(f"DEBUG: VLLM generated response length: {len(generated_text)}")
        print(f"DEBUG: VLLM raw response: {generated_text[:300]}...")
        
        # Return in the expected format (similar to transformers.pipeline)
        # Note: MedRAG expects to extract the generated part by removing the prompt
        return [{"generated_text": prompt + generated_text}]

def parse_response_vllm(raw_response, model_name=None):
    """
    Legacy function - now redirects to parse_response_standard
    for consistency with original MedRAG approach
    """
    return parse_response_standard(raw_response, model_name=model_name)

def parse_response_standard(raw_response, model_name=None):
    """
    Universal response parser for ALL models with model-specific handling
    This follows the original MedRAG approach with PMC-LLaMA specific extensions
    """
    import json
    import re
    
    # DEBUG: Print the response we're trying to parse
    print(f"DEBUG: Parsing response of length {len(raw_response)} for model: {model_name}")
    print(f"DEBUG: Response content: {raw_response[:500]}...")
    
    # Remove any extra whitespace and newlines
    response = raw_response.strip()
    
    # PMC-LLaMA specific: Handle array format like [{"text": "...", "answer_choice": "B"}] or [{"text": "...", "answer": "B"}]
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
    
    # Try to find JSON content between curly braces (most reliable)
    json_match = re.search(r'\{.*\}', response, re.DOTALL)
    if json_match:
        json_str = json_match.group(0)
        try:
            # Try to parse the JSON
            parsed = json.loads(json_str)
            # Validate required fields
            if "step_by_step_thinking" in parsed and "answer_choice" in parsed:
                return parsed
        except json.JSONDecodeError:
            pass
    
    # Fallback: try to extract answer choice and reasoning separately
    # Look for answer choice patterns
    answer_patterns = [
        r'"answer_choice"\s*:\s*"([ABCD])"',
        r'"answer_choice"\s*:\s*([ABCD])',
        r'answer_choice["\']?\s*:\s*["\']?([ABCD])',
        r'(?:answer is|choice is|answer:|choice:)\s*([ABCD])',
        r'\b([ABCD])\s*(?:is the|would be the)?\s*(?:correct|right|answer)',
    ]
    
    answer_choice = None
    for pattern in answer_patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            answer_choice = match.group(1)
            break
    
    # Look for reasoning patterns
    reasoning_patterns = [
        r'"step_by_step_thinking"\s*:\s*"([^"]*)"',
        r'step_by_step_thinking["\']?\s*:\s*["\']([^"\']*)["\']',
        r'reasoning["\']?\s*:\s*["\']([^"\']*)["\']',
        r'explanation["\']?\s*:\s*["\']([^"\']*)["\']',
    ]
    
    reasoning = None
    for pattern in reasoning_patterns:
        match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
        if match:
            reasoning = match.group(1).strip()
            break
    
    # If we found an answer choice, return it with available reasoning
    if answer_choice:
        result = {
            "step_by_step_thinking": reasoning or "Extracted from model response",
            "answer_choice": answer_choice
        }
        print(f"DEBUG: Successfully parsed - Answer: {answer_choice}, Reasoning length: {len(result['step_by_step_thinking'])}")
        return result
    
    # Last resort: look for any single letter A, B, C, or D
    answer_match = re.search(r'\b([ABCD])\b', response)
    if answer_match:
        return {
            "step_by_step_thinking": "Fallback: extracted single letter answer from response",
            "answer_choice": answer_match.group(1)
        }
    
    # Ultimate fallback
    return {
        "answer_choice": "A", 
        "step_by_step_thinking": "Error: Could not parse model response"
    }

def vllm_medrag_answer(medrag_instance, question, options=None, k=32, **kwargs):
    """
    Wrapper function to handle VLLM-specific response parsing for MedRAG
    Following original MedRAG approach - all models use the same parsing logic
    """
    # Get the raw answer from MedRAG
    try:
        answer, snippets, scores = medrag_instance.answer(question=question, options=options, k=k, **kwargs)
        
        # Parse the VLLM response if it's a string
        if isinstance(answer, str):
            # Use standard parser with model-specific handling
            # Pass the model name for PMC-LLaMA specific parsing
            model_name = getattr(medrag_instance, 'llm_name', None)
            parsed_answer = parse_response_standard(answer, model_name=model_name)
        else:
            parsed_answer = answer
            
        # Ensure snippets have the expected format
        processed_snippets = []
        for snippet in snippets:
            if isinstance(snippet, dict):
                # Handle missing 'contents' field - normalize to 'contents'
                if 'contents' not in snippet:
                    if 'content' in snippet:
                        snippet['contents'] = snippet['content']
                    else:
                        # Construct contents from available fields
                        title = snippet.get('title', '')
                        content = snippet.get('content', '')
                        snippet['contents'] = f"{title}. {content}".strip()
                
                # Ensure other expected fields exist
                if 'title' not in snippet:
                    snippet['title'] = snippet.get('id', 'Unknown')
                    
                processed_snippets.append(snippet)
            else:
                # Handle case where snippet is not a dict
                processed_snippets.append({'contents': str(snippet), 'title': 'Unknown'})
        
        return parsed_answer, processed_snippets, scores
        
    except Exception as e:
        print(f"Error in vllm_medrag_answer: {e}")
        # Return a fallback response
        return {"answer_choice": "A", "error": str(e)}, [], []

def patch_medrag_for_vllm():
    """Monkey patch MedRAG to use VLLM for specific models"""
    import transformers
    original_pipeline = transformers.pipeline
    
    def vllm_pipeline(task, model=None, **kwargs):
        # Use VLLM for supported models (Llama, Qwen, PMC-LLaMA, and other common models)
        supported_models = ["llama", "qwen", "meta-llama", "mistral", "mixtral", "pmc"]
        if task == "text-generation" and model and any(name in model.lower() for name in supported_models):
            print(f"DEBUG: Using VLLM for model: {model}")
            return VLLMWrapper(model, **kwargs)
        else:
            print(f"DEBUG: Using transformers pipeline for model: {model}")
            return original_pipeline(task, model=model, **kwargs)
    
    transformers.pipeline = vllm_pipeline

def run_medrag_with_vllm():
    """Run MedRAG with Llama-7B using vllm and medcorp dataset"""
    
    # Patch transformers.pipeline to use VLLM
    patch_medrag_for_vllm()
    
    # Initialize MedRAG with Llama-7B model and medcorp dataset
    # You can replace this with specific Llama-7B model names like:
    # - "meta-llama/Llama-2-7b-chat-hf"  
    # - "meta-llama/Meta-Llama-3-8B-Instruct" (closest to 7B)
    # - Any other Llama-7B variant
    model_name = "meta-llama/Llama-2-7b-chat-hf"  # Change this to your preferred Llama-7B model
    
    print(f"Initializing MedRAG with {model_name} and MedCorp dataset...")
    medrag = MedRAG(
        llm_name=model_name,
        rag=True,
        retriever_name="MedCPT",  # Medical domain retriever
        corpus_name="MedCorp",    # Combined medical corpus
        db_dir="./src/data/corpus",  # Use your existing corpus directory
        corpus_cache=True,        # Cache corpus in memory for speed
        HNSW=True                # Use HNSW for faster retrieval
    )
    
    # Example medical question
    question = "What are the main symptoms of acute myocardial infarction?"
    options = {
        "A": "Chest pain, shortness of breath, nausea",
        "B": "Headache, dizziness, blurred vision", 
        "C": "Abdominal pain, diarrhea, fever",
        "D": "Joint pain, muscle weakness, rash"
    }
    
    print("Running example question...")
    print(f"Question: {question}")
    print("Options:", options)
    
    # Get answer using MedRAG with VLLM-specific parsing
    answer, snippets, scores = vllm_medrag_answer(medrag, question=question, options=options, k=10)
    
    print(f"\nMedRAG Answer: {answer}")
    print(f"\nRetrieved {len(snippets)} relevant snippets")
    if snippets and len(snippets) > 0:
        first_snippet = snippets[0]
        if isinstance(first_snippet, dict) and 'contents' in first_snippet:
            print(f"Top snippet: {first_snippet['contents'][:200]}...")
        elif isinstance(first_snippet, dict) and 'content' in first_snippet:
            print(f"Top snippet: {first_snippet['content'][:200]}...")
        else:
            print(f"Top snippet: {str(first_snippet)[:200]}...")
    
    return medrag

if __name__ == "__main__":
    # Ensure required packages are available
    try:
        import vllm
        print("✓ VLLM is available")
    except ImportError:
        print("✗ VLLM not found. Install with: pip install vllm")
        sys.exit(1)
    
    # Check if git-lfs is available (optional - only needed if downloading original corpus)
    import subprocess
    try:
        subprocess.run(["git", "lfs", "version"], check=True, capture_output=True)
        print("✓ Git LFS is available")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("ℹ Git LFS not found - not needed since you have existing corpus data")
        print("  (Git LFS is only required for downloading original corpus from repository)")
    
    # Run the example
    try:
        medrag_instance = run_medrag_with_vllm()
        print("\n✓ MedRAG with VLLM setup completed successfully!")
        print("\nYou can now use the medrag_instance for your medical QA tasks.")
        
    except Exception as e:
        print(f"✗ Error running MedRAG: {e}")
        print("\nTroubleshooting tips:")
        print("1. Ensure you have sufficient GPU memory (8GB+ recommended for Llama-7B)")
        print("2. Check that the model name is correct and accessible")
        print("3. Verify git-lfs is properly installed for corpus download")
        sys.exit(1)