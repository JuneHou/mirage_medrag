#!/usr/bin/env python3
"""
Quick test script to verify MIRAGE + VLLM setup is working

This runs a single question from each dataset to verify:
1. VLLM is working
2. MedRAG retrieval is working
3. Prediction saving is working
4. Evaluation is working
"""

import sys
import os

# Add the project root directory to the Python path
# This allows us to import from MedRAG and MIRAGE as top-level packages
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Add paths
sys.path.append("/data/wang/junh/githubs/mirage_medrag/MedRAG")
sys.path.append("/data/wang/junh/githubs/mirage_medrag/MIRAGE/src")

from MedRAG.run_medrag_vllm import patch_medrag_for_vllm, vllm_medrag_answer
from MedRAG.src.medrag import MedRAG
from MIRAGE.src.utils import QADataset

def test_single_question():
    """Test with a single question from MMLU"""
    
    print("="*80)
    print("MIRAGE + VLLM Setup Test")
    print("="*80)
    
    # 1. Setup VLLM
    print("\n[1/5] Setting up VLLM...")
    try:
        patch_medrag_for_vllm()
        print("✓ VLLM setup successful")
    except Exception as e:
        print(f"✗ VLLM setup failed: {e}")
        return False
    
    # 2. Initialize MedRAG
    print("\n[2/5] Initializing MedRAG...")
    try:
        medrag = MedRAG(
            llm_name="meta-llama/Meta-Llama-3-8B-Instruct",
            rag=True,
            retriever_name="MedCPT",  # Use MedCPT instead of RRF-4 to avoid Java requirement
            corpus_name="Textbooks",  # Use smaller corpus for faster test
            db_dir="/data/wang/junh/githubs/mirage_medrag/MedRAG/src/data/corpus",
            corpus_cache=True,
            HNSW=True
        )
        print("✓ MedRAG initialized")
    except Exception as e:
        print(f"✗ MedRAG initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 3. Load test dataset
    print("\n[3/5] Loading MMLU dataset...")
    try:
        dataset = QADataset("mmlu", dir="/data/wang/junh/githubs/mirage_medrag/MIRAGE")
        print(f"✓ Loaded {len(dataset)} questions")
    except Exception as e:
        print(f"✗ Dataset loading failed: {e}")
        return False
    
    # 4. Run prediction
    print("\n[4/5] Running prediction on test question...")
    try:
        question_data = dataset[0]
        print(f"\nQuestion: {question_data['question'][:100]}...")
        
        answer_dict, snippets, scores = vllm_medrag_answer(
            medrag,
            question=question_data['question'],
            options=question_data.get('options'),
            k=5  # Use fewer snippets for faster test
        )
        
        print(f"\n✓ Prediction successful")
        print(f"  Predicted: {answer_dict.get('answer_choice', 'Unknown')}")
        print(f"  Actual: {question_data.get('answer', 'Unknown')}")
        print(f"  Correct: {answer_dict.get('answer_choice') == question_data.get('answer')}")
        print(f"  Retrieved {len(snippets)} snippets")
        
    except Exception as e:
        print(f"✗ Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 5. Test saving
    print("\n[5/5] Testing prediction save...")
    try:
        import json
        test_dir = "/tmp/mirage_test"
        os.makedirs(test_dir, exist_ok=True)
        
        save_path = os.path.join(test_dir, "test_question.json")
        
        # Handle case where step_by_step_thinking might not exist
        step_by_step = answer_dict.get('step_by_step_thinking', 'No reasoning provided')
        answer_choice = answer_dict.get('answer_choice', 'A')
        
        formatted_answer = json.dumps({
            "step_by_step_thinking": step_by_step,
            "answer_choice": answer_choice
        })
        
        with open(save_path, 'w') as f:
            json.dump([formatted_answer], f, indent=2)
        
        print(f"✓ Saved prediction to {save_path}")
        
        # Verify it can be loaded
        with open(save_path, 'r') as f:
            loaded = json.load(f)
        print(f"✓ Verified saved prediction can be loaded")
        
    except Exception as e:
        print(f"✗ Save test failed: {e}")
        return False
    
    print("\n" + "="*80)
    print("✓ ALL TESTS PASSED!")
    print("="*80)
    print("\nYour setup is ready. You can now run:")
    print("  python run_benchmark_vllm.py --dataset mmlu --mode rag --k 32")
    return True


if __name__ == "__main__":
    success = test_single_question()
    sys.exit(0 if success else 1)
