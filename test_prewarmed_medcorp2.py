#!/usr/bin/env python3
"""
Test script for the new pre-warmed MedCorp2 implementation
"""

import sys
import os
import time

# Add MedRAG path
sys.path.append("/data/wang/junh/githubs/mirage_medrag/MedRAG/src")

def test_prewarmed_medcorp2():
    """Test the new pre-warmed MedCorp2 implementation"""
    
    try:
        from medrag import MedRAG
        
        print("=" * 80)
        print("Testing Pre-warmed MedCorp2 Implementation")
        print("=" * 80)
        
        # Initialize MedRAG with MedCorp2 (this should trigger pre-warming)
        print("\n1. Initializing MedRAG with MedCorp2...")
        start_time = time.time()
        
        # Use a local model to avoid OpenAI API key requirement
        medrag = MedRAG(
            llm_name="Qwen/Qwen3-8B",  # Use local model
            rag=True,
            retriever_name="MedCPT",
            corpus_name="MedCorp2",
            db_dir="/data/wang/junh/githubs/mirage_medrag/MedRAG/src/data/corpus",
            corpus_cache=True,
            HNSW=True
        )
        
        init_time = time.time() - start_time
        print(f"Initialization completed in {init_time:.2f} seconds")
        
        # Test question
        question = "What are the common side effects of cisplatin chemotherapy?"
        options = {
            "A": "Nausea, vomiting, and nephrotoxicity",
            "B": "Weight gain and hypoglycemia", 
            "C": "Liver toxicity and bleeding",
            "D": "Headache and confusion"
        }
        
        print(f"\n2. Testing with question: {question}")
        print("Options:")
        for key, value in options.items():
            print(f"  {key}. {value}")
        
        # Test the pre-warmed function
        print("\n3. Running pre-warmed retrieval...")
        start_time = time.time()
        
        try:
            # Check if source retrievers are properly initialized
            if hasattr(medrag, 'source_retrievers'):
                print(f"✓ Source retrievers initialized: {list(medrag.source_retrievers.keys())}")
            else:
                print("✗ Source retrievers not found")
                return
            
            answer, snippets, scores = medrag.medrag_answer_by_source_prewarmed(
                question=question,
                options=options,
                k=20,  # Total of 20 documents (will be distributed across 5 sources)
                save_dir="./test_prewarmed_results"
            )
            
            query_time = time.time() - start_time
            print(f"Query completed in {query_time:.2f} seconds")
            
            print(f"\n4. Results:")
            print(f"Answer: {answer}")
            print(f"Total snippets retrieved: {len(snippets)}")
            
            # Show breakdown by source
            source_counts = {}
            for snippet in snippets:
                source = snippet.get('source_type', 'unknown')
                source_counts[source] = source_counts.get(source, 0) + 1
            
            print("\nDocument distribution by source:")
            for source, count in source_counts.items():
                print(f"  {source}: {count} documents")
            
            # Show some sample queries generated
            print("\nSample tailored queries:")
            shown_queries = set()
            for snippet in snippets[:10]:  # Show first 10
                query = snippet.get('tailored_query', 'No query')
                source = snippet.get('source_type', 'unknown')
                if query not in shown_queries and len(query) < 200:  # Avoid too long queries
                    print(f"  {source}: {query}")
                    shown_queries.add(query)
                    if len(shown_queries) >= 3:  # Limit to 3 examples
                        break
            
            print(f"\n5. Performance Summary:")
            print(f"  Initialization time: {init_time:.2f}s")
            print(f"  Query time: {query_time:.2f}s")
            print(f"  Total time: {init_time + query_time:.2f}s")
            
            print("\n✓ Pre-warmed MedCorp2 test completed successfully!")
            
            # Test a second query to show the speed improvement
            print("\n6. Testing second query (should be faster)...")
            start_time = time.time()
            
            question2 = "What is the mechanism of action of metformin in diabetes?"
            answer2, snippets2, scores2 = medrag.medrag_answer_by_source_prewarmed(
                question=question2,
                options=None,
                k=15,
                save_dir="./test_prewarmed_results_2"
            )
            
            query_time2 = time.time() - start_time
            print(f"Second query completed in {query_time2:.2f} seconds")
            print(f"Speed improvement: {query_time - query_time2:.2f}s faster")
            
        except Exception as e:
            print(f"Error during retrieval: {e}")
            import traceback
            traceback.print_exc()
            
    except Exception as e:
        print(f"Error during initialization: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_prewarmed_medcorp2()