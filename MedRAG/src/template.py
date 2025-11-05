from liquid import Template

general_cot_system = '''You are a helpful medical expert, and your task is to answer a multi-choice medical question. Please first think step-by-step and then choose the answer from the provided options. 

IMPORTANT: You must respond with ONLY a valid JSON object in this exact format:
{"step_by_step_thinking": "your reasoning here", "answer_choice": "A"}

Do not include any other text, explanations, or formatting. Just the JSON object.'''

general_cot = Template('''
Here is the question:
{{question}}

Here are the potential choices:
{{options}}

Please think step-by-step and respond with ONLY a valid JSON object in this exact format:
{"step_by_step_thinking": "your reasoning here", "answer_choice": "A"}

Do not include any other text, explanations, or formatting outside the JSON object:
''')

general_medrag_system = '''You are a helpful medical expert, and your task is to answer a multi-choice medical question using the relevant documents. Please first think step-by-step and then choose the answer from the provided options. 

IMPORTANT: You must respond with ONLY a valid JSON object in this exact format:
{"step_by_step_thinking": "your reasoning here", "answer_choice": "A"}

Do not include any other text, explanations, or formatting. Just the JSON object.'''

general_medrag = Template('''
Here are the relevant documents:
{{context}}

Here is the question:
{{question}}

Here are the potential choices:
{{options}}

Please think step-by-step and respond with ONLY a valid JSON object in this exact format:
{"step_by_step_thinking": "your reasoning here", "answer_choice": "A"}

Do not include any other text, explanations, or formatting outside the JSON object:
''')

meditron_cot = Template('''
### User:
Here is the question:
...

Here are the potential choices:
A. ...
B. ...
C. ...
D. ...
X. ...

Please think step-by-step and generate your output in json.

### Assistant:
{"step_by_step_thinking": ..., "answer_choice": "X"}

### User:
Here is the question:
{{question}}

Here are the potential choices:
{{options}}

Please think step-by-step and generate your output in json.

### Assistant:
''')

meditron_medrag = Template('''
Here are the relevant documents:
{{context}}

### User:
Here is the question:
...

Here are the potential choices:
A. ...
B. ...
C. ...
D. ...
X. ...

Please think step-by-step and generate your output in json.

### Assistant:
{"step_by_step_thinking": ..., "answer_choice": "X"}

### User:
Here is the question:
{{question}}

Here are the potential choices:
{{options}}

Please think step-by-step and generate your output in json.

### Assistant:
''')

simple_medrag_system = '''You are a helpful medical expert, and your task is to answer a medical question using the relevant documents.'''
simple_medrag_prompt = Template('''Here are the relevant documents:\n{{context}}\nHere is the question:\n{{question}}''')

i_medrag_system = '''You are a helpful medical assistant, and your task is to answer the given question following the instructions given by the user. '''

follow_up_instruction_ask = '''Please first analyze all the information in a section named Analysis (## Analysis). Then, use key terms from previous answers to form specific and direct questions. Generate {} concise, context-specific queries to search for additional information in an external knowledge base, in a section named Queries (## Queries). Each query should be simple and focused, directly relating to the key terms used in the answers. Wait for responses from the user before proceeding.'''
follow_up_instruction_answer = '''Please first think step-by-step to analyze all the information in a section named Analysis (## Analysis). Then, please provide your answer choice in a section named Answer (## Answer).'''

# Template for generating dual queries for 2-source MedCorp2 architecture
dual_query_generation = Template('''
Medical question: {{question}}
{{options_text}}

Generate exactly two search queries in this format:

MedCorp: [your query for general medical literature]
UMLS: [your query for medical terminology and concept relationships]

Requirements:
- MedCorp query: Focus on clinical findings, treatments, diagnosis, patient management
- UMLS query: Focus on medical terminology relationships, concept connections, drug-disease associations
- Keep each query under 15 words
- Output only the two lines above, no explanations

MedCorp: 
UMLS: 
''')

# Simplified source descriptions for 2-source architecture
source_descriptions = {
    "medcorp": "Combined medical literature corpus including research papers, textbooks, clinical guidelines, and medical encyclopedias. Focus on clinical findings, treatments, diagnostic criteria, evidence-based medicine, and practical medical knowledge.",
    "umls": "Unified Medical Language System with structured medical terminology, concept relationships, semantic networks, and medical ontology. Focus on medical concept relationships, terminology definitions, drug-disease associations, semantic connections, and hierarchical medical knowledge structures."
}