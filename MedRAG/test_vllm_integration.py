#!/usr/bin/env python3
"""
Test script for VLLM MedRAG integration with improved parsing
"""

import sys
import os

# Add the current directory to path so we can import run_medrag_vllm
sys.path.append('.')

def test_vllm_medrag():
    try:
        from run_medrag_vllm import patch_medrag_for_vllm, vllm_medrag_answer, parse_llama_response
        from src.medrag import MedRAG
        
        print("Testing VLLM MedRAG integration...")
        
        # Test the response parser first
        print("\n1. Testing response parser...")
        test_responses = [
            '{"answer_choice": "A"}',
            ' { "step_by_step_thinking": "Analysis here", "answer_choice": "B" } ',
            'The answer is A',
            'Choice A is correct because...',
        ]
        
        for i, test_resp in enumerate(test_responses):
            print(f"  Test {i+1}: {test_resp[:50]}...")
            parsed = parse_llama_response(test_resp)
            print(f"  Parsed: {parsed}")
        
        # Test corpus availability
        print("\n2. Testing corpus availability...")
        corpus_dir = "./src/data/corpus"
        
        available_corpora = []
        for corpus in ["textbooks", "statpearls", "wikipedia", "pubmed"]:
            chunk_dir = os.path.join(corpus_dir, corpus, "chunk")
            if os.path.exists(chunk_dir):
                files = [f for f in os.listdir(chunk_dir) if f.endswith('.jsonl')]
                # Check if files have real content (not LFS pointers)
                real_files = 0
                for f in files[:3]:  # Check first 3 files
                    try:
                        size = os.path.getsize(os.path.join(chunk_dir, f))
                        if size > 1000:  # Bigger than LFS pointer
                            real_files += 1
                    except:
                        pass
                
                if real_files > 0:
                    available_corpora.append(corpus)
                    print(f"  ✓ {corpus}: {len(files)} files ({real_files} with content)")
                else:
                    print(f"  ✗ {corpus}: {len(files)} files (LFS pointers only)")
        
        if not available_corpora:
            print("  No corpus data available for testing")
            return False
            
        # Choose the best available corpus
        if "textbooks" in available_corpora:
            corpus_name = "Textbooks"
        elif "statpearls" in available_corpora:
            corpus_name = "StatPearls"
        else:
            corpus_name = available_corpora[0].title()
            
        print(f"  Using corpus: {corpus_name}")
        
        print("\n3. Testing MedRAG initialization...")
        # Use a smaller, simpler model for testing
        test_model = os.environ.get("TEST_MODEL", "microsoft/DialoGPT-medium")
        
        patch_medrag_for_vllm()
        
        medrag = MedRAG(
            llm_name=test_model,
            rag=True,
            retriever_name="MedCPT",
            corpus_name=corpus_name,
            db_dir=corpus_dir,
            corpus_cache=False,  # Don't cache to avoid memory issues
            HNSW=False
        )
        
        print("  ✓ MedRAG initialized successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("VLLM MedRAG Integration Test")
    print("=" * 40)
    
    success = test_vllm_medrag()
    
    if success:
        print("\n✓ All tests passed! The integration should work.")
        print("\nTo run with VLLM:")
        print("  export MODEL_NAME=meta-llama/Llama-2-7b-chat-hf")
        print("  export HF_TOKEN=your_token")
        print("  python run_medrag_vllm.py")
    else:
        print("\n✗ Tests failed. Check the errors above.")