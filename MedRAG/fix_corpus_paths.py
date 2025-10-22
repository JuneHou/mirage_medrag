#!/usr/bin/env python3
"""
Quick fix and verification script for MedRAG corpus paths
"""

import os
import sys
sys.path.append(".")

def check_corpus_structure():
    """Check if corpus data is properly structured"""
    corpus_dir = "./src/data/corpus"
    
    print("🔍 Checking corpus structure...")
    
    if not os.path.exists(corpus_dir):
        print(f"✗ Corpus directory not found: {corpus_dir}")
        return False
    
    print(f"✓ Corpus directory found: {corpus_dir}")
    
    # Check for expected corpus subdirectories
    expected_corpora = ["pubmed", "textbooks", "wikipedia", "statpearls"]
    found_corpora = []
    
    for corpus in expected_corpora:
        corpus_path = os.path.join(corpus_dir, corpus)
        if os.path.exists(corpus_path):
            files = os.listdir(corpus_path)
            print(f"  ✓ {corpus}: {len(files)} files/directories")
            found_corpora.append(corpus)
        else:
            print(f"  ✗ {corpus}: not found")
    
    if len(found_corpora) >= 2:
        print(f"✓ Found {len(found_corpora)} corpora - sufficient for MedCorp")
        return True
    else:
        print(f"✗ Only found {len(found_corpora)} corpora - may need more data")
        return False

def test_retrieval_system():
    """Test if retrieval system can initialize with existing corpus"""
    print("\n🔍 Testing retrieval system initialization...")
    
    try:
        from src.utils import RetrievalSystem
        
        # Try to initialize with textbooks (usually smaller)
        retrieval = RetrievalSystem(
            retriever_name="MedCPT",
            corpus_name="Textbooks", 
            db_dir="./src/data/corpus",
            cache=False
        )
        print("✓ Retrieval system initialized successfully")
        return True
        
    except Exception as e:
        print(f"✗ Retrieval system failed: {e}")
        print("  This might indicate missing index files or incompatible data format")
        return False

def test_medrag_initialization():
    """Test MedRAG initialization with correct paths"""
    print("\n🔍 Testing MedRAG initialization...")
    
    try:
        from src.medrag import MedRAG
        
        # Try to initialize MedRAG with a small corpus first
        medrag = MedRAG(
            llm_name="OpenAI/gpt-3.5-turbo",  # Use API model to avoid GPU issues
            rag=True,
            retriever_name="MedCPT",
            corpus_name="Textbooks",  # Start with smaller corpus
            db_dir="./src/data/corpus",
            corpus_cache=False,  # Don't cache for this test
            HNSW=False  # Don't use HNSW for this test
        )
        print("✓ MedRAG initialized successfully")
        return True
        
    except Exception as e:
        print(f"✗ MedRAG initialization failed: {e}")
        return False

def main():
    print("🔧 MedRAG Corpus Path Fix & Verification")
    print("=" * 50)
    
    # Check current working directory
    print(f"Current directory: {os.getcwd()}")
    
    # Run tests
    tests = [
        ("Corpus Structure", check_corpus_structure),
        ("Retrieval System", test_retrieval_system),
        ("MedRAG Initialization", test_medrag_initialization),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results:")
    
    passed = 0
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {test_name}")
        if result:
            passed += 1
    
    print(f"\nResults: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 All tests passed! Your corpus setup is working correctly.")
        print("\nYou can now run:")
        print("  python run_medrag_vllm.py")
    elif passed >= len(results) - 1:
        print("\n⚠ Most tests passed. Minor issues detected.")
        print("\nYou can try running:")
        print("  python run_medrag_vllm.py")
        print("\nIf it fails, the corpus data might need to be rebuilt.")
    else:
        print("\n❌ Multiple issues detected.")
        print("\nPossible solutions:")
        print("1. Check if corpus data is complete")
        print("2. Try rebuilding corpus indices")
        print("3. Check file permissions")
        print("4. Verify the corpus was downloaded properly with git-lfs")

if __name__ == "__main__":
    main()