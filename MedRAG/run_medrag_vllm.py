#!/usr/bin/env python3
"""
Minimal launcher script to run MedRAG with vllm backend for Llama-7B models.
This script provides a drop-in replacement for the transformers pipeline using vllm.
"""

import os
import sys

# Device assignment
RETRIEVER_DEVICE = "cuda:0"  # First visible GPU (GPU 6) for embedding models
LLM_DEVICE = "cuda:1"        # Second visible GPU (GPU 7) for VLLM LLM

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
            
            # Initialize VLLM with optimized settings on second GPU only
            # Reduce GPU utilization to prevent OOM - leave headroom for KV cache growth
            gpu_utilization = 0.7  # Reduced from 0.7 to prevent memory exhaustion
            
            # Set explicit max_model_len to override any 512 token limits
            # This is critical for PMC-LLaMA models that might have incorrect config
            max_model_length = 2048  # Match what we set in medrag.py
            if "pmc" in model_name.lower():
                max_model_length = 2048  # PMC-LLaMA specific
            elif "qwen" in model_name.lower():
                max_model_length = 8192  # Qwen3-8B supports 8192+ tokens
            elif "llama-3" in model_name.lower():
                max_model_length = 4096
            
            print(f"DEBUG: Model: {model_name}, Setting max_model_length={max_model_length}")
            print(f"DEBUG: LLM will use single GPU: {LLM_DEVICE}")
                
            self.llm = LLM(
                model=model_name,
                tensor_parallel_size=1,  # Using single GPU for LLM (GPU 7)
                trust_remote_code=True,
                gpu_memory_utilization=gpu_utilization,
                max_model_len=max_model_length,  # Explicitly set this to override config
                enforce_eager=True,  # Disable CUDA graphs to avoid 512 token limits
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
        # Extract relevant parameters and set defaults
        do_sample = kwargs.get('do_sample', False)
        temperature = kwargs.get('temperature', 0.7)  # Use passed temperature or default to 0.7
        
        # Support repetition penalty for better generation quality
        repetition_penalty = kwargs.get('repetition_penalty', 1.0)
        
        # Use max_length directly as max_new_tokens (no internal calculation)
        # The calling code should pass the desired number of new tokens to generate
        max_new_tokens = kwargs.get('max_length', 2048)
        
        # Calculate tokens for debugging only
        prompt_tokens = len(self.tokenizer.encode(prompt))
        print(f"DEBUG: VLLMWrapper - max_new_tokens={max_new_tokens} (input max_length={max_new_tokens}, prompt_tokens={prompt_tokens})")
        
        # Extract stop sequences from kwargs
        stop_sequences = kwargs.get('stop_sequences', None)
        if stop_sequences is None:
            # Enhanced stop sequences for medical debate to prevent repetition - prioritize boxed format
            stop_sequences = ["\\boxed{A}\n", "\\boxed{B}\n", "\\boxed{C}\n", "\\boxed{D}\n", "<|im_end|>", "</s>", "###", "\n\n\n"]
        
        # Ensure repetition penalty is meaningful (minimum 1.05 to prevent loops)
        if repetition_penalty < 1.05:
            repetition_penalty = 1.15
        
        # Create sampling parameters for vllm with enhanced anti-repetition settings
        sampling_params = SamplingParams(
            temperature=temperature,  # Use the actual temperature parameter
            top_p=0.95,  # Slightly higher top_p for diversity
            max_tokens=max_new_tokens,
            stop=stop_sequences,
            repetition_penalty=repetition_penalty,
            frequency_penalty=0.1,  # Add frequency penalty to discourage repetition
            presence_penalty=0.1    # Add presence penalty for diversity
        )
        
        print(f"DEBUG: Sampling params - temp={temperature}, rep_penalty={repetition_penalty}, stop={stop_sequences}")
        
        # Generate response
        outputs = self.llm.generate([prompt], sampling_params)
        generated_text = outputs[0].outputs[0].text
        
        # Clean up to prevent memory leaks
        del outputs
        
        # Return format handling
        if kwargs.get('return_format') == 'string':
            return generated_text  # Return plain string for debate code
        else:
            return [{"generated_text": prompt + generated_text}]  # Default format for MedRAG

    def generate_with_system(self, system_message, user_prompt, stop_sequences=None, max_new_tokens=800, **kwargs):
        """
        Generate response with explicit system message handling for verification
        """
        # For Qwen models, format properly
        if "qwen" in self.model_name.lower():
            # Use Qwen's chat format with system and user roles
            formatted_prompt = f"<|im_start|>system\n{system_message}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
        else:
            # Fallback to simple concatenation for other models
            formatted_prompt = f"{system_message}\n\n{user_prompt}"
        
        # Enhanced stop sequences for verification
        if stop_sequences is None:
            stop_sequences = [
                "\\boxed{A}\n", "\\boxed{B}\n", "\\boxed{C}\n", "\\boxed{D}\n",  # Primary boxed format
                "<|im_end|>", "</s>", "###", "\n\n\n",
                "Answer: A\n", "Answer: B\n", "Answer: C\n", "Answer: D\n",
                "The correct answer is A\n", "The correct answer is B\n", 
                "The correct answer is C\n", "The correct answer is D\n",
                "Option A\n", "Option B\n", "Option C\n", "Option D\n"
            ]
        
        # Calculate tokens for the formatted prompt
        prompt_tokens = len(self.tokenizer.encode(formatted_prompt))
        print(f"DEBUG: System message generation - prompt_tokens={prompt_tokens}, max_new_tokens={max_new_tokens}")
        
        # Create constrained sampling parameters for verification
        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=max_new_tokens,
            stop=stop_sequences,
            repetition_penalty=1.2,    # Higher penalty for verification
            frequency_penalty=0.2,    # Stronger frequency penalty
            presence_penalty=0.1      # Encourage diversity
        )
        
        # Generate response
        outputs = self.llm.generate([formatted_prompt], sampling_params)
        generated_text = outputs[0].outputs[0].text
        
        # Clean up
        del outputs
        
        print(f"DEBUG: System message generation completed, response length: {len(generated_text)}")
        return generated_text


def log_pmc_response(raw_response, model_name, question_id=None, log_dir=None):
    """
    Log PMC-LLaMA raw response to file for debugging
    """
    # Create log file path with same name as question_id
    log_file = os.path.join(log_dir, f"{question_id}_raw_response.txt")
    
    # Ensure log directory exists
    os.makedirs(log_dir, exist_ok=True)
    
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Question ID: {question_id}\n")
        f.write(f"Response Length: {len(raw_response)}\n")
        f.write("="*50 + "\n")
        f.write(raw_response)
        f.write("\n" + "="*50 + "\n")
    
    print(f"DEBUG: PMC-LLaMA raw response logged to: {log_file}")

def parse_response_standard(raw_response, model_name=None, question_id=None, log_dir=None):
    """
    Universal response parser for ALL models with model-specific handling
    This follows the original MedRAG approach with PMC-LLaMA specific extensions
    """
    import json
    import re
    
    # DEBUG: Print the response we're trying to parse
    # print(f"DEBUG: Parsing response of length {len(raw_response)} for model: {model_name}")
    # print(f"DEBUG: Response content: {raw_response[:500]}...")
    
    # Remove any extra whitespace and newlines
    response = raw_response.strip()
        
    # Log PMC-LLaMA raw responses for debugging
    if "pmc" or "gemma" in model_name.lower():
        log_pmc_response(response, model_name, question_id, log_dir)
    
    # PMC-LLaMA specific: Handle array format like [{"text": "...", "answer_choice": "B"}] or [{"text": "...", "answer": "B"}]
    if "pmc" in model_name.lower():
        # First try: Look for array format with answer_choice/answer field
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
        
        # Second try: PMC-LLaMA specific format with ###Answer: OPTION X IS CORRECT
        answer_option_match = re.search(r'###\s*Answer:\s*OPTION\s+([ABCD])\s+IS\s+CORRECT', response, re.IGNORECASE)
        if answer_option_match:
            answer_choice = answer_option_match.group(1)
            # Extract rationale if available
            rationale_match = re.search(r'###\s*Rationale:\s*(.*?)###\s*Answer:', response, re.IGNORECASE | re.DOTALL)
            reasoning = rationale_match.group(1).strip() if rationale_match else "PMC-LLaMA rationale extracted"
            
            # Also try to extract the text options for additional context
            text_options = []
            text_matches = re.findall(r'"text"\s*:\s*"([^"]*)"', response)
            if text_matches:
                text_options = text_matches
                reasoning = f"Options considered: {'; '.join(text_options[:3])}. {reasoning}"
            
            result = {
                "step_by_step_thinking": reasoning,
                "answer_choice": answer_choice
            }
            print(f"DEBUG: PMC-LLaMA ###Answer format parsed - Answer: {answer_choice}, Reasoning length: {len(reasoning)}")
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
        r'###\s*Answer:\s*OPTION\s+([ABCD])\s+IS\s+CORRECT',  # PMC-LLaMA specific
        r'OPTION\s+([ABCD])\s+IS\s+CORRECT',  # PMC-LLaMA without ###
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
        r'###\s*Rationale:\s*(.*?)(?:###|$)',  # PMC-LLaMA specific rationale
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

def vllm_medrag_answer(medrag_instance, question, options=None, k=32, question_id=None, log_dir=None, **kwargs):
    """
    Wrapper function to handle VLLM-specific response parsing for MedRAG
    Following original MedRAG approach - all models use the same parsing logic
    """
    import gc
    import torch
    
    # Get the raw answer from MedRAG
    try:
        # Pass save parameters based on follow_up mode
        if log_dir and medrag_instance.follow_up:
            kwargs['save_path'] = os.path.join(log_dir, f"{question_id}_conversation.json")
        else:
            kwargs['save_dir'] = log_dir
        
        answer, snippets, scores = medrag_instance.answer(question=question, options=options, k=k, **kwargs)
        
        # Periodic memory cleanup every 50 iterations to prevent gradual memory buildup
        if question_id and isinstance(question_id, str):
            try:
                # Extract iteration number from question_id (e.g., "test_0938" -> 938)
                iter_num = int(question_id.split('_')[-1])
                if iter_num % 50 == 0:
                    print(f"DEBUG: Performing memory cleanup at iteration {iter_num}")
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            except (ValueError, IndexError):
                pass  # Skip if question_id format is unexpected
        
        # Parse the VLLM response if it's a string
        if isinstance(answer, str):
            # Use standard parser with model-specific handling
            # Pass the model name for PMC-LLaMA specific parsing and logging
            model_name = getattr(medrag_instance, 'llm_name', None)
            parsed_answer = parse_response_standard(answer, model_name=model_name, question_id=question_id, log_dir=log_dir)
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
        return {"answer_choice": "E", "error": str(e)}, [], []

def patch_medrag_for_vllm():
    """Monkey patch MedRAG to use VLLM for specific models"""
    import transformers
    original_pipeline = transformers.pipeline
    
    def vllm_pipeline(task, model=None, **kwargs):
        # Use VLLM for supported models (Llama, Qwen, PMC-LLaMA, and other common models)
        supported_models = ["llama", "qwen", "meta-llama", "mistral", "mixtral", "pmc"]
        if task == "text-generation" and model and any(name in model.lower() for name in supported_models):
            # print(f"DEBUG: Using VLLM for model: {model}")
            # print(f"DEBUG: Model name check - 'qwen' in '{model.lower()}': {'qwen' in model.lower()}")
            return VLLMWrapper(model, **kwargs)
        else:
            # print(f"DEBUG: Using transformers pipeline for model: {model}")
            return original_pipeline(task, model=model, **kwargs)
    
    transformers.pipeline = vllm_pipeline

def init_medrag_with_device_separation(
    llm_name="meta-llama/Llama-2-7b-chat-hf",
    rag=True,
    retriever_name="MedCPT",
    corpus_name="MedCorp",
    db_dir="./src/data/corpus",
    corpus_cache=True,
    HNSW=True,
    retriever_device=None,
    **kwargs
):
    """
    Initialize MedRAG with GPU device separation for all corpus types.
    
    This function works with any corpus (MedCorp, MedCorp2, Textbooks, etc.)
    and ensures retriever embeddings use a separate GPU from the LLM.
    
    Args:
        llm_name: Model name for LLM
        rag: Whether to enable retrieval
        retriever_name: Retriever to use (MedCPT, RRF-2, etc.)
        corpus_name: Corpus to use (MedCorp, MedCorp2, Textbooks, etc.)
        db_dir: Directory containing corpus data
        corpus_cache: Whether to cache corpus in memory
        HNSW: Whether to use HNSW indexing
        retriever_device: Device for retriever (default: RETRIEVER_DEVICE)
        **kwargs: Additional arguments for MedRAG
    
    Returns:
        MedRAG instance with device separation configured
    """
    retriever_device = retriever_device or RETRIEVER_DEVICE
    
    print(f"Initializing MedRAG with device separation:")
    print(f"  LLM: {llm_name}")
    print(f"  Retriever device: {retriever_device}")
    print(f"  LLM device: {LLM_DEVICE} (handled by VLLM)")
    if rag:
        print(f"  Retriever: {retriever_name}")
        print(f"  Corpus: {corpus_name}")
    
    # Initialize MedRAG with device separation for all corpus types
    medrag = MedRAG(
        llm_name=llm_name,
        rag=rag,
        retriever_name=retriever_name if rag else None,
        corpus_name=corpus_name if rag else None,
        db_dir=db_dir,
        corpus_cache=corpus_cache,
        HNSW=HNSW,
        retriever_device=retriever_device,  # Apply to all corpus types
        **kwargs
    )
    
    print(f"✓ MedRAG initialized with device separation")
    return medrag

def run_medrag_with_vllm():
    """Run MedRAG with Llama-7B using vllm and medcorp dataset"""
    
    # Patch transformers.pipeline to use VLLM
    patch_medrag_for_vllm()
    
    model_name = "Qwen/Qwen3-8B"  # Change this to your preferred Llama-7B model
    
    print(f"Initializing MedRAG with {model_name} and MedCorp dataset...")
    print(f"Retriever will use: {RETRIEVER_DEVICE}")
    print(f"LLM will use: {LLM_DEVICE}")
    medrag = MedRAG(
        llm_name=model_name,
        rag=True,
        retriever_name="MedCPT",  # Medical domain retriever
        corpus_name="MedCorp",    # Combined medical corpus
        db_dir="./src/data/corpus",  # Use your existing corpus directory
        corpus_cache=True,        # Cache corpus in memory for speed
        HNSW=True,               # Use HNSW for faster retrieval
        retriever_device=RETRIEVER_DEVICE  # Use first GPU for retriever (GPU 6)
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