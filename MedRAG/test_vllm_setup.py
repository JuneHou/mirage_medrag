#!/usr/bin/env python3
"""
Quick smoke test for MedRAG with VLLM integration.
This script performs lightweight tests to verify the setup without heavy computation.
"""

import sys
import os
sys.path.append(".")

def test_imports():
    """Test if all required packages can be imported"""
    print("🔍 Testing imports...")
    
    try:
        import vllm
        print("  ✓ VLLM imported successfully")
    except ImportError as e:
        print(f"  ✗ VLLM import failed: {e}")
        return False
    
    try:
        import transformers
        print("  ✓ Transformers imported successfully")
    except ImportError as e:
        print(f"  ✗ Transformers import failed: {e}")
        return False
    
    try:
        from src.medrag import MedRAG
        print("  ✓ MedRAG imported successfully")
    except ImportError as e:
        print(f"  ✗ MedRAG import failed: {e}")
        return False
    
    return True

def test_vllm_wrapper():
    """Test the VLLM wrapper without loading large models"""
    print("\n🔍 Testing VLLM wrapper...")
    
    try:
        from run_medrag_vllm import VLLMWrapper, patch_medrag_for_vllm
        print("  ✓ VLLM wrapper imported successfully")
        
        # Test the monkey patch function
        patch_medrag_for_vllm()
        print("  ✓ MedRAG patching completed")
        
        return True
    except Exception as e:
        print(f"  ✗ VLLM wrapper test failed: {e}")
        return False

def test_dependencies():
    """Test external dependencies"""
    print("\n🔍 Testing external dependencies...")
    
    # Test git-lfs
    import subprocess
    try:
        result = subprocess.run(["git", "lfs", "version"], 
                              capture_output=True, text=True, check=True)
        print("  ✓ Git LFS is available")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("  ⚠ Git LFS not found - corpus download may fail")
        print("    Install with: apt-get install git-lfs (Ubuntu) or brew install git-lfs (Mac)")
    
    # Test Java (for BM25)
    try:
        result = subprocess.run(["java", "-version"], 
                              capture_output=True, text=True, check=True)
        print("  ✓ Java is available")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("  ⚠ Java not found - BM25 retriever will not work")
        print("    Install with: apt-get install openjdk-11-jdk (Ubuntu)")
    
    return True

def test_model_access():
    """Test if model can be accessed (without loading)"""
    print("\n🔍 Testing model access...")
    
    try:
        from transformers import AutoTokenizer
        
        model_name = "meta-llama/Llama-2-7b-chat-hf"
        print(f"  Testing access to {model_name}...")
        
        # Just try to load tokenizer (much faster than full model)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        print("  ✓ Model access verified (tokenizer loaded)")
        
        return True
    except Exception as e:
        print(f"  ⚠ Model access test failed: {e}")
        print("    This might be normal if you haven't logged into HuggingFace")
        print("    Try: huggingface-cli login")
        return False

def test_corpus_availability():
    """Test if corpus files are accessible"""
    print("\n🔍 Testing corpus availability...")
    
    try:
        # Check if corpus directory exists
        corpus_dir = "./src/data/corpus"
        if os.path.exists(corpus_dir):
            print(f"  ✓ Corpus directory exists: {corpus_dir}")
        else:
            print(f"  ℹ Corpus directory not found - will be created on first run")
        
        return True
    except Exception as e:
        print(f"  ✗ Corpus test failed: {e}")
        return False

def quick_functionality_test():
    """Quick test of core functionality without heavy models"""
    print("\n🔍 Testing basic functionality...")
    
    try:
        # Test retrieval system initialization (lightweight)
        from src.utils import RetrievalSystem
        
        print("  Testing retrieval system...")
        # This will be lightweight if no corpus is cached
        retrieval = RetrievalSystem("MedCPT", "Textbooks", "./src/data/corpus", cache=False)
        print("  ✓ Retrieval system initialized")
        
        return True
    except Exception as e:
        print(f"  ⚠ Functionality test failed: {e}")
        print("    This is expected if corpus is not yet downloaded")
        return False

def main():
    """Run all smoke tests"""
    print("🚀 MedRAG + VLLM Smoke Test")
    print("=" * 50)
    
    tests = [
        ("Package Imports", test_imports),
        ("VLLM Integration", test_vllm_wrapper),  
        ("External Dependencies", test_dependencies),
        ("Model Access", test_model_access),
        ("Corpus Setup", test_corpus_availability),
        ("Basic Functionality", quick_functionality_test),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"  ✗ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Summary:")
    
    passed = 0
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {test_name}")
        if result:
            passed += 1
    
    print(f"\nResults: {passed}/{len(results)} tests passed")
    
    if passed >= len(results) - 2:  # Allow 2 tests to fail
        print("\n🎉 Setup looks good! Ready to run MedRAG with VLLM.")
        print("\nNext steps:")
        print("1. Run: python run_medrag_vllm.py")
        print("2. Or use the integration in your own scripts")
    else:
        print("\n⚠ Some issues detected. Please check the failed tests above.")
        print("\nCommon fixes:")
        print("- Install missing dependencies")
        print("- Login to HuggingFace: huggingface-cli login") 
        print("- Check GPU memory availability")

if __name__ == "__main__":
    main()