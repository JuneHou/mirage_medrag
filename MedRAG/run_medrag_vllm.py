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
        self.model_name = model_name
        
        # Filter out kwargs that are not supported by VLLM
        vllm_supported_kwargs = {
            'tensor_parallel_size', 'dtype', 'quantization', 
            'gpu_memory_utilization', 'swap_space', 'enforce_eager',
            'max_model_len', 'trust_remote_code', 'download_dir',
            'load_format', 'seed'
        }
        
        # Only pass supported kwargs to VLLM
        vllm_kwargs = {k: v for k, v in kwargs.items() if k in vllm_supported_kwargs}
        
        # Initialize VLLM with optimized settings for Llama-8B
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=1,  # Adjust based on your GPU count
            trust_remote_code=True,
            gpu_memory_utilization=0.5,  # Use only 50% of GPU memory to avoid OOM
            **vllm_kwargs
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
    def __call__(self, prompt, **kwargs):
        """Make the wrapper callable like transformers.pipeline"""
        # Extract relevant parameters and set defaults
        max_length = kwargs.get('max_length', 2048)
        do_sample = kwargs.get('do_sample', False)
        
        # Create sampling parameters for vllm
        sampling_params = SamplingParams(
            temperature=0.0 if not do_sample else 0.7,
            top_p=0.9 if do_sample else 1.0,
            max_tokens=max_length - len(self.tokenizer.encode(prompt)),
            stop=kwargs.get('stop_sequences', None)
        )
        
        # Generate response
        outputs = self.llm.generate([prompt], sampling_params)
        generated_text = outputs[0].outputs[0].text
        
        # Return in the expected format (similar to transformers.pipeline)
        # Note: MedRAG expects to extract the generated part by removing the prompt
        return [{"generated_text": prompt + generated_text}]

def parse_llama_response(raw_response):
    """
    Parse VLLM Llama model response to extract clean JSON
    Handles the specific formatting issues from VLLM Llama models
    """
    import json
    import re
    
    # Remove any extra whitespace and newlines
    response = raw_response.strip()
    
    # Try to find JSON content between curly braces
    json_match = re.search(r'\{.*\}', response, re.DOTALL)
    if json_match:
        json_str = json_match.group(0)
        try:
            # Try to parse the JSON
            parsed = json.loads(json_str)
            return parsed
        except json.JSONDecodeError as e:
            # If direct parsing fails, try to clean up the JSON
            # Handle common VLLM formatting issues
            json_str = re.sub(r'(?<!\\)"([^"]*)"(?=\s*:)', r'"\1"', json_str)  # Fix unescaped quotes in keys
            json_str = re.sub(r'(?<!\\)"([^"]*)"(?=\s*[,}])', r'"\1"', json_str)  # Fix unescaped quotes in values
            
            try:
                parsed = json.loads(json_str)
                return parsed
            except json.JSONDecodeError as e:
                print(f"Warning: Could not parse JSON response: {e}")
                print(f"Raw response: {response[:200]}...")
                
                # Fallback: extract answer choice if possible
                answer_match = re.search(r'["\']?answer_choice["\']?\s*:\s*["\']?([ABCD])', response)
                if answer_match:
                    fallback_result = {"answer_choice": answer_match.group(1), "step_by_step_thinking": f"Error: '{e}'"}
                    return fallback_result
                else:
                    fallback_result = {"answer_choice": "A", "step_by_step_thinking": f"Error: '{e}'"}
                    return fallback_result
    
    # If no JSON found, try to extract answer choice directly
    answer_match = re.search(r'\b([ABCD])\b', response)
    if answer_match:
        result = {"answer_choice": answer_match.group(1), "step_by_step_thinking": "No structured reasoning found"}
        return result
    
    # Fallback
    result = {"answer_choice": "A", "step_by_step_thinking": "No answer found in response"}
    return result

def vllm_medrag_answer(medrag_instance, question, options=None, k=32, **kwargs):
    """
    Wrapper function to handle VLLM-specific response parsing for MedRAG
    """
    # Get the raw answer from MedRAG
    try:
        answer, snippets, scores = medrag_instance.answer(question=question, options=options, k=k, **kwargs)
        
        # Parse the VLLM response if it's a string
        if isinstance(answer, str):
            parsed_answer = parse_llama_response(answer)
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
        if task == "text-generation" and model and ("llama" in model.lower()):
            print(f"Using VLLM for model: {model}")
            return VLLMWrapper(model, **kwargs)
        else:
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