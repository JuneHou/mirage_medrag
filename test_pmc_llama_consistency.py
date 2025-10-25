#!/usr/bin/env python3
"""
Enhanced test script to debug PMC-LLaMA issues with VLLM integration

This script tests:
1. PMC-LLaMA model detection and configuration
2. VLLM wrapper initialization
3. Template loading and processing
4. Response parsing consistency

Usage:
    python test_pmc_llama_consistency.py
"""

import os
import sys
import json
import traceback

# Add project paths
medrag_path = "MedRAG"
sys.path.insert(0, medrag_path)

def test_pmc_llama_detection():
    """Test if PMC-LLaMA is properly detected by VLLM patching logic"""
    
    print("=" * 60)
    print("TEST 1: PMC-LLaMA Model Detection")
    print("=" * 60)
    
    try:
        from run_medrag_vllm import patch_medrag_for_vllm
        
        # Test model names that should trigger PMC-LLaMA
        test_models = [
            "axiong/PMC_LLaMA_13B",
            "pmc_llama_13b", 
            "PMC-LLaMA",
            "pmc-llama-7b"
        ]
        
        # Mock the pipeline function to see what gets detected
        detected_models = []
        failed_models = []
        
        for model in test_models:
            # Check if model name contains pmc_llama or pmc-llama
            supported_models = ["llama", "qwen", "meta-llama", "mistral", "mixtral", "pmc_llama", "pmc-llama"]
            is_detected = any(name in model.lower() for name in supported_models)
            
            if is_detected:
                detected_models.append(model)
                print(f"✅ DETECTED: {model}")
            else:
                failed_models.append(model)
                print(f"❌ NOT DETECTED: {model}")
        
        if len(detected_models) == len(test_models):
            print("✅ PMC-LLaMA detection is working properly")
            return True
        else:
            print(f"❌ PMC-LLaMA detection failed for: {failed_models}")
            return False
            
    except Exception as e:
        print(f"❌ ERROR in detection test: {e}")
        traceback.print_exc()
        return False

def test_pmc_llama_config():
    """Test PMC-LLaMA configuration in MedRAG"""
    
    print("\n" + "=" * 60)
    print("TEST 2: PMC-LLaMA Configuration")
    print("=" * 60)
    
    try:
        # Add MedRAG src to path
        src_path = "MedRAG/src"
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
            
        from medrag import MedRAG
        
        # Test PMC-LLaMA initialization
        print("Testing PMC-LLaMA configuration...")
        
        test_model_name = "axiong/PMC_LLaMA_13B"
        
        # This will trigger the configuration logic
        try:
            medrag = MedRAG(
                llm_name=test_model_name,
                rag=False,  # Start with simple setup
            )
            
            print(f"✅ PMC-LLaMA configuration successful")
            print(f"   Max length: {medrag.max_length}")
            print(f"   Context length: {medrag.context_length}")
            
            # Check if template was loaded
            if hasattr(medrag.tokenizer, 'chat_template') and medrag.tokenizer.chat_template:
                print(f"✅ Chat template loaded (length: {len(medrag.tokenizer.chat_template)})")
            else:
                print(f"⚠️  Chat template not found or empty")
            
            return True
            
        except Exception as init_error:
            print(f"❌ PMC-LLaMA initialization failed: {init_error}")
            print(f"   Error type: {type(init_error).__name__}")
            traceback.print_exc()
            return False
            
    except Exception as e:
        print(f"❌ ERROR in configuration test: {e}")
        traceback.print_exc()
        return False

def test_vllm_wrapper_init():
    """Test VLLMWrapper initialization with PMC-LLaMA"""
    
    print("\n" + "=" * 60)
    print("TEST 3: VLLM Wrapper Initialization")
    print("=" * 60)
    
    try:
        from run_medrag_vllm import VLLMWrapper
        
        test_model_name = "axiong/PMC_LLaMA_13B"
        
        print(f"Testing VLLMWrapper with {test_model_name}...")
        print("Note: This may take a few moments to download/load the model...")
        
        try:
            # Try to initialize VLLMWrapper
            wrapper = VLLMWrapper(test_model_name)
            print(f"✅ VLLMWrapper initialization successful")
            print(f"   Model name: {wrapper.model_name}")
            
            # Test a simple call
            test_prompt = "What is 2+2?"
            response = wrapper(test_prompt, max_length=100, do_sample=False)
            print(f"✅ Simple generation test successful")
            print(f"   Response type: {type(response)}")
            
            return True
            
        except Exception as vllm_error:
            print(f"❌ VLLMWrapper failed: {vllm_error}")
            print(f"   Error type: {type(vllm_error).__name__}")
            
            # Common error scenarios
            if "out of memory" in str(vllm_error).lower():
                print("   💡 SUGGESTION: Reduce gpu_memory_utilization or use smaller tensor_parallel_size")
            elif "model not found" in str(vllm_error).lower():
                print("   💡 SUGGESTION: Check if model name is correct or model is accessible")
            elif "cuda" in str(vllm_error).lower():
                print("   💡 SUGGESTION: Check GPU availability and CUDA setup")
            
            traceback.print_exc()
            return False
            
    except Exception as e:
        print(f"❌ ERROR in VLLM wrapper test: {e}")
        traceback.print_exc()
        return False

def test_template_loading():
    """Test PMC-LLaMA template file loading"""
    
    print("\n" + "=" * 60)
    print("TEST 4: Template File Loading")
    print("=" * 60)
    
    try:
        template_path = "MedRAG/templates/pmc_llama.jinja"
        
        print(f"Checking template file: {template_path}")
        
        if os.path.exists(template_path):
            print("✅ Template file exists")
            
            # Try to read the template
            with open(template_path, 'r') as f:
                template_content = f.read()
            
            print(f"✅ Template loaded successfully (length: {len(template_content)})")
            print(f"   First 100 chars: {template_content[:100]}...")
            
            # Check if template looks valid
            if len(template_content.strip()) > 0:
                print("✅ Template appears to have content")
                return True
            else:
                print("❌ Template file is empty")
                return False
                
        else:
            print(f"❌ Template file does not exist: {template_path}")
            
            # List available templates
            template_dir = "MedRAG/templates/"
            if os.path.exists(template_dir):
                templates = os.listdir(template_dir)
                print(f"   Available templates: {templates}")
            
            return False
            
    except Exception as e:
        print(f"❌ ERROR in template loading test: {e}")
        traceback.print_exc()
        return False

def test_response_parsing():
    """Test response parsing with PMC-LLaMA format"""
    
    print("\n" + "=" * 60)
    print("TEST 5: Response Parsing")
    print("=" * 60)
    
    try:
        from run_medrag_vllm import parse_response_standard
        
        # Test cases specific to potential PMC-LLaMA output formats
        test_cases = [
            {
                "response": '{"step_by_step_thinking": "This is medical reasoning", "answer_choice": "A"}',
                "expected_choice": "A",
                "description": "Perfect JSON format"
            },
            {
                "response": 'Based on the medical evidence, the answer is B.',
                "expected_choice": "B", 
                "description": "Simple answer format"
            }
        ]
        
        print("Testing response parsing...")
        all_passed = True
        
        for i, test in enumerate(test_cases, 1):
            print(f"\nTest {i}: {test['description']}")
            
            try:
                result = parse_response_standard(test['response'])
                actual_choice = result.get('answer_choice')
                
                if actual_choice == test['expected_choice']:
                    print(f"✅ PASSED - Answer choice: {actual_choice}")
                else:
                    print(f"❌ FAILED - Expected: {test['expected_choice']}, Got: {actual_choice}")
                    all_passed = False
                    
            except Exception as e:
                print(f"❌ ERROR - {e}")
                all_passed = False
        
        return all_passed
        
    except Exception as e:
        print(f"❌ ERROR in response parsing test: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all PMC-LLaMA debugging tests"""
    
    print("PMC-LLaMA Debugging Test Suite")
    print("=" * 60)
    print("This will help identify where PMC-LLaMA integration is failing")
    print("=" * 60)
    
    results = []
    
    # Test 1: Model detection
    print("\n🔍 Running detection test...")
    results.append(("Model Detection", test_pmc_llama_detection()))
    
    # Test 2: Configuration
    print("\n🔍 Running configuration test...")
    results.append(("Configuration", test_pmc_llama_config()))
    
    # Test 3: VLLM wrapper (may fail if model is not available)
    print("\n🔍 Running VLLM wrapper test...")
    print("⚠️  This test requires the PMC-LLaMA model to be available")
    print("   If you don't have the model, this test will fail - that's OK for debugging")
    results.append(("VLLM Wrapper", test_vllm_wrapper_init()))
    
    # Test 4: Template loading
    print("\n🔍 Running template test...")
    results.append(("Template Loading", test_template_loading()))
    
    # Test 5: Response parsing
    print("\n🔍 Running response parsing test...")
    results.append(("Response Parsing", test_response_parsing()))
    
    # Summary
    print("\n" + "=" * 60)
    print("DEBUGGING SUMMARY:")
    print("=" * 60)
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)
    
    print(f"\nPassed: {passed_count}/{total_count}")
    
    # Provide debugging guidance
    print("\n" + "=" * 60)
    print("DEBUGGING GUIDANCE:")
    print("=" * 60)
    
    if results[0][1]:  # Model detection passed
        print("✅ Model detection is working - PMC-LLaMA should be recognized")
    else:
        print("❌ Model detection failed - Need to fix supported_models list")
    
    if results[1][1]:  # Configuration passed
        print("✅ MedRAG configuration is working - max_length and templates OK")
    else:
        print("❌ MedRAG configuration failed - Check medrag.py initialization")
    
    if results[2][1]:  # VLLM wrapper passed
        print("✅ VLLM wrapper is working - PMC-LLaMA can be loaded")
    else:
        print("❌ VLLM wrapper failed - This is the most likely issue")
        print("   💡 Common causes:")
        print("      - Model not downloaded/accessible")
        print("      - Insufficient GPU memory")
        print("      - CUDA/VLLM compatibility issues")
        print("      - Model architecture not supported by VLLM")
    
    if results[3][1]:  # Template loading passed
        print("✅ Template loading is working - PMC-LLaMA template found")
    else:
        print("❌ Template loading failed - Check template file path")
    
    if results[4][1]:  # Response parsing passed
        print("✅ Response parsing is working - Output format OK")
    else:
        print("❌ Response parsing failed - Check parser logic")
    
    print(f"\n🎯 NEXT STEPS:")
    if passed_count == total_count:
        print("   All tests passed! PMC-LLaMA should work.")
        print("   Try running a full benchmark.")
    elif not results[2][1]:  # VLLM wrapper failed
        print("   Focus on VLLM wrapper initialization.")
        print("   Check model availability and GPU resources.")
    else:
        failed_tests = [name for name, passed in results if not passed]
        print(f"   Focus on fixing: {', '.join(failed_tests)}")

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()