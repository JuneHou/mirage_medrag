#!/usr/bin/env python3
"""
Test script to verify PMC-LLaMA parsing logic
"""

import re
import json

def parse_response_standard(raw_response, model_name=None):
    """
    Universal response parser for ALL models with model-specific handling
    This follows the original MedRAG approach with PMC-LLaMA specific extensions
    """
    
    # DEBUG: Print the response we're trying to parse
    print(f"DEBUG: Parsing response of length {len(raw_response)} for model: {model_name}")
    print(f"DEBUG: Response content: {raw_response[:500]}...")
    
    # Remove any extra whitespace and newlines
    response = raw_response.strip()
    
    # PMC-LLaMA specific: Handle array format like [{"text": "...", "answer_choice": "B"}] or [{"text": "...", "answer": "B"}]
    if model_name and "pmc" in model_name.lower():
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
            print(f"DEBUG: Found answer using pattern: {pattern}")
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
            print(f"DEBUG: Found reasoning using pattern: {pattern}")
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

# Test with your actual PMC-LLaMA response
test_response = '''[{"text": "A lesion causing compression of the facial nerve at the stylomastoid foramen will cause ipsilateral paralysis of the facial muscles. "}, {"text": "A lesion causing compression of the facial nerve at the stylomastoid foramen will cause ipsilateral paralysis of the facial muscles and loss of taste. "}, {"text": "A lesion causing compression of the facial nerve at the stylomastoid foramen will cause ipsilateral paralysis of the facial muscles, loss of taste, lacrimation and decreased salivation. "}] ### Rationale: The facial nerve passes through the facial canal, which is located in the temporal bone. The facial nerve leaves the facial canal at the stylomastoid foramen and then branches into the temporal, zygomatic, buccal, marginal mandibular, and cervical branches. The facial nerve innervates the muscles of facial expression, which are located in the face. The facial nerve also innervates the anterior two thirds of the tongue, which is responsible for taste sensation. Therefore, a lesion causing compression of the facial nerve at the stylomastoid foramen will cause ipsilateral paralysis of the facial muscles and loss of taste. ###Answer: OPTION B IS CORRECT.'''

if __name__ == "__main__":
    print("Testing PMC-LLaMA parsing...")
    print("="*80)
    
    result = parse_response_standard(test_response, "axiong/PMC_LLaMA_13B")
    
    print("="*80)
    print("FINAL RESULT:")
    print(json.dumps(result, indent=2))