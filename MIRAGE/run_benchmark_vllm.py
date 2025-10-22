#!/usr/bin/env python3
"""
Benchmark runner for MIRAGE using MedRAG with VLLM (Llama models)

This script generates predictions for all 5 MIRAGE datasets using your VLLM-enabled
Llama-7B MedRAG setup. Results are saved in the format expected by MIRAGE's evaluate.py.

Usage:
    # Run CoT (no RAG)
    python run_benchmark_vllm.py --mode cot --dataset all
    
    # Run MedRAG with 32 snippets
    python run_benchmark_vllm.py --mode rag --k 32 --dataset all
    
    # Run specific dataset only
    python run_benchmark_vllm.py --mode rag --k 32 --dataset mmlu
    
    # Use different retriever/corpus
    python run_benchmark_vllm.py --mode rag --k 32 --retriever_name MedCPT --corpus_name Textbooks
"""

import os
import sys
import json
import argparse
from tqdm import tqdm

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Add specific paths for imports
medrag_path = "/data/wang/junh/githubs/mirage_medrag/MedRAG"
mirage_src_path = "/data/wang/junh/githubs/mirage_medrag/MIRAGE/src"

sys.path.insert(0, medrag_path)
sys.path.insert(0, mirage_src_path)

# Import from MedRAG
from run_medrag_vllm import patch_medrag_for_vllm, vllm_medrag_answer, parse_llama_response
from src.medrag import MedRAG

# Import from MIRAGE - be specific to avoid conflicts
sys.path.insert(0, mirage_src_path)
import importlib.util
spec = importlib.util.spec_from_file_location("mirage_utils", os.path.join(mirage_src_path, "utils.py"))
mirage_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mirage_utils)
QADataset = mirage_utils.QADataset


def save_prediction(question_id, answer_dict, save_path):
    """Save prediction in MIRAGE format"""
    # Format the answer as expected by evaluate.py
    formatted_answer = {
        "step_by_step_thinking": answer_dict["step_by_step_thinking"],
        "answer_choice": answer_dict["answer_choice"]
    }
    
    # Save as list with single element (matches MIRAGE format)
    with open(save_path, 'w') as f:
        json.dump([formatted_answer], f, indent=2)


def run_benchmark(
    dataset_names,
    mode='rag',
    k=32,
    llm_name="meta-llama/Meta-Llama-3-8B-Instruct",
    retriever_name="MedCPT",
    corpus_name="MedCorp",
    results_dir="./prediction",
    resume=True
):
    """
    Run benchmark on specified datasets
    
    Args:
        dataset_names: List of dataset names or ['all']
        mode: 'cot' or 'rag'
        k: Number of snippets to retrieve (only for RAG mode)
        llm_name: Model name for VLLM
        retriever_name: Retriever to use (MedCPT, RRF-2, RRF-4, etc.)
        corpus_name: Corpus to use (MedCorp, Textbooks, PubMed, etc.)
        results_dir: Directory to save results
        resume: Skip already processed questions
    """
    
    # Setup VLLM
    print("Setting up VLLM...")
    patch_medrag_for_vllm()
    
    # Initialize MedRAG
    print(f"Initializing MedRAG with {llm_name}...")
    if mode == 'rag':
        print(f"  Retriever: {retriever_name}")
        print(f"  Corpus: {corpus_name}")
        print(f"  K: {k}")
        
    medrag = MedRAG(
        llm_name=llm_name,
        rag=(mode == 'rag'),
        retriever_name=retriever_name if mode == 'rag' else None,
        corpus_name=corpus_name if mode == 'rag' else None,
        db_dir="/data/wang/junh/githubs/mirage_medrag/MedRAG/src/data/corpus",
        corpus_cache=True,
        HNSW=True
    )
    
    # Determine datasets to run
    all_datasets = ['mmlu', 'medqa', 'medmcqa', 'pubmedqa', 'bioasq']
    if 'all' in dataset_names:
        dataset_names = all_datasets
    
    # Process each dataset
    for dataset_name in dataset_names:
        print(f"\n{'='*80}")
        print(f"Processing {dataset_name.upper()}")
        print(f"{'='*80}")
        
        # Load dataset
        dataset = QADataset(dataset_name, dir="/data/wang/junh/githubs/mirage_medrag/MIRAGE")
        
        # Determine split
        split = "dev" if dataset_name == "medmcqa" else "test"
        
        # Setup save directory
        if mode == 'rag':
            save_dir = os.path.join(
                results_dir,
                dataset_name,
                f"rag_{k}",
                llm_name,
                corpus_name.lower(),
                retriever_name
            )
        else:
            save_dir = os.path.join(
                results_dir,
                dataset_name,
                "cot",
                llm_name
            )
        
        os.makedirs(save_dir, exist_ok=True)
        print(f"Saving to: {save_dir}")
        
        # Process questions
        processed = 0
        skipped = 0
        
        for idx in tqdm(range(len(dataset)), desc=f"Processing {dataset_name}"):
            question_data = dataset[idx]
            question_id = dataset.index[idx]
            
            # Check if already processed
            save_path = os.path.join(save_dir, f"{split}_{question_id}.json")
            if resume and os.path.exists(save_path):
                skipped += 1
                continue
            
            # Get answer
            try:
                if mode == 'rag':
                    answer_dict, snippets, scores = vllm_medrag_answer(
                        medrag,
                        question=question_data['question'],
                        options=question_data.get('options'),
                        k=k
                    )
                else:
                    # CoT mode (no RAG)
                    answer_dict, snippets, scores = vllm_medrag_answer(
                        medrag,
                        question=question_data['question'],
                        options=question_data.get('options'),
                        k=0  # No retrieval
                    )
                
                # Save prediction
                save_prediction(question_id, answer_dict, save_path)
                processed += 1
                
            except Exception as e:
                print(f"\nError processing question {question_id}: {e}")
                # Save a default answer to avoid stopping the entire run
                answer_dict = {
                    "step_by_step_thinking": f"Error: {str(e)}",
                    "answer_choice": "A"
                }
                save_prediction(question_id, answer_dict, save_path)
                continue
        
        print(f"\n{dataset_name} complete:")
        print(f"  Processed: {processed}")
        print(f"  Skipped (already done): {skipped}")
        print(f"  Total: {len(dataset)}")


def main():
    parser = argparse.ArgumentParser(description="Run MIRAGE benchmark with VLLM")
    
    parser.add_argument(
        '--dataset',
        type=str,
        nargs='+',
        default=['all'],
        choices=['all', 'mmlu', 'medqa', 'medmcqa', 'pubmedqa', 'bioasq'],
        help='Dataset(s) to run (default: all)'
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        default='rag',
        choices=['cot', 'rag'],
        help='Mode: cot (no retrieval) or rag (with retrieval) (default: rag)'
    )
    
    parser.add_argument(
        '--k',
        type=int,
        default=32,
        help='Number of snippets to retrieve (default: 32)'
    )
    
    parser.add_argument(
        '--llm_name',
        type=str,
        default='meta-llama/Meta-Llama-3-8B-Instruct',
        help='LLM model name (default:meta-llama/Meta-Llama-3-8B-Instruct )'
    )
    
    parser.add_argument(
        '--retriever_name',
        type=str,
        default='MedCPT',
        choices=['BM25', 'Contriever', 'SPECTER', 'MedCPT', 'RRF-2', 'RRF-4'],
        help='Retriever to use (default: MedCPT)'
    )
    
    parser.add_argument(
        '--corpus_name',
        type=str,
        default='MedCorp',
        choices=['PubMed', 'Textbooks', 'StatPearls', 'Wikipedia', 'MedText', 'MedCorp'],
        help='Corpus to use (default: MedCorp)'
    )
    
    parser.add_argument(
        '--results_dir',
        type=str,
        default='./prediction',
        help='Directory to save results (default: ./prediction)'
    )
    
    parser.add_argument(
        '--no-resume',
        action='store_true',
        help='Reprocess all questions (default: skip already processed)'
    )
    
    args = parser.parse_args()
    
    # Run benchmark
    run_benchmark(
        dataset_names=args.dataset,
        mode=args.mode,
        k=args.k,
        llm_name=args.llm_name,
        retriever_name=args.retriever_name,
        corpus_name=args.corpus_name,
        results_dir=args.results_dir,
        resume=not args.no_resume
    )
    
    print("\n" + "="*80)
    print("BENCHMARK COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: {args.results_dir}")
    print("\nTo evaluate results, run:")
    if args.mode == 'rag':
        print(f"  python src/evaluate.py --results_dir {args.results_dir} "
              f"--llm_name {args.llm_name} --rag --k {args.k} "
              f"--corpus_name {args.corpus_name.lower()} --retriever_name {args.retriever_name}")
    else:
        print(f"  python src/evaluate.py --results_dir {args.results_dir} --llm_name {args.llm_name}")


if __name__ == "__main__":
    main()
