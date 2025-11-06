import os
import re
import json
import tqdm
import torch
import time
import argparse
import transformers
from transformers import AutoTokenizer
import openai
from transformers import StoppingCriteria, StoppingCriteriaList
import tiktoken
import sys
sys.path.append("src")
from utils import RetrievalSystem, DocExtracter
from template import *

from config import config

openai.api_type = openai.api_type or os.getenv("OPENAI_API_TYPE") or config.get("api_type")
openai.api_version = openai.api_version or os.getenv("OPENAI_API_VERSION") or config.get("api_version")
openai.api_key = openai.api_key or os.getenv('OPENAI_API_KEY') or config["api_key"]

if openai.__version__.startswith("0"):
    openai.api_base = openai.api_base or os.getenv("OPENAI_API_BASE") or config.get("api_base")
    if openai.api_type == "azure":
        openai_client = lambda **x: openai.ChatCompletion.create(**{'engine' if k == 'model' else k: v for k, v in x.items()})["choices"][0]["message"]["content"]
    else:
        openai_client = lambda **x: openai.ChatCompletion.create(**x)["choices"][0]["message"]["content"]
else:
    if openai.api_type == "azure":
        openai.azure_endpoint = openai.azure_endpoint or os.getenv("OPENAI_ENDPOINT") or config.get("azure_endpoint")
        openai_client = lambda **x: openai.AzureOpenAI(
            api_version=openai.api_version,
            azure_endpoint=openai.azure_endpoint,
            api_key=openai.api_key,
        ).chat.completions.create(**x).choices[0].message.content
    else:
        openai_client = lambda **x: openai.OpenAI(
            api_key=openai.api_key,
        ).chat.completions.create(**x).choices[0].message.content

class MedRAG:

    def __init__(self, llm_name="OpenAI/gpt-3.5-turbo-16k", rag=True, follow_up=False, retriever_name="MedCPT", corpus_name="Textbooks", db_dir="./corpus", cache_dir=None, corpus_cache=False, HNSW=False, retriever_device=None):
        self.llm_name = llm_name
        self.rag = rag
        self.retriever_name = retriever_name
        self.corpus_name = corpus_name
        self.db_dir = db_dir
        self.cache_dir = cache_dir
        self.HNSW = HNSW  # Store HNSW setting for pre-warm initialization
        self.corpus_cache = corpus_cache  # Store corpus_cache setting
        self.retriever_device = retriever_device
        self.docExt = None
        if rag:
            # For MedCorp2, we'll use pre-warmed source retrievers instead of a single MedCorp2 retriever
            if corpus_name == "MedCorp2":
                self.retrieval_system = None
            else:
                self.retrieval_system = RetrievalSystem(self.retriever_name, self.corpus_name, self.db_dir, cache=corpus_cache, HNSW=HNSW, device=self.retriever_device)
        else:
            self.retrieval_system = None
        self.templates = {"cot_system": general_cot_system, "cot_prompt": general_cot,
                    "medrag_system": general_medrag_system, "medrag_prompt": general_medrag}
        if self.llm_name.split('/')[0].lower() == "openai":
            self.model = self.llm_name.split('/')[-1]
            if "gpt-3.5" in self.model or "gpt-35" in self.model:
                self.max_length = 16384
                self.context_length = 15000
            elif "gpt-4" in self.model:
                self.max_length = 32768
                self.context_length = 30000
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        elif "gemini" in self.llm_name.lower():
            import google.generativeai as genai
            genai.configure(api_key=os.environ['GOOGLE_API_KEY'])
            self.model = genai.GenerativeModel(
                model_name=self.llm_name.split('/')[-1],
                generation_config={
                    "temperature": 0,
                    "max_output_tokens": 2048,
                }
            )
            if "1.5" in self.llm_name.lower():
                self.max_length = 1048576
                self.context_length = 1040384
            else:
                self.max_length = 30720
                self.context_length = 28672
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        else:
            print(f"Initializing HuggingFace model: {self.llm_name}...")
            if "mixtral" in llm_name.lower():
                self.tokenizer = AutoTokenizer.from_pretrained(self.llm_name, cache_dir=self.cache_dir)
                template_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'templates', 'mistral-instruct.jinja')
                self.tokenizer.chat_template = open(template_path).read().replace('    ', '').replace('\n', '')
                self.max_length = 32768
                self.context_length = 30000
                self.tokenizer.model_max_length = self.max_length
            elif "llama-2" in llm_name.lower():
                self.tokenizer = AutoTokenizer.from_pretrained(self.llm_name, cache_dir=self.cache_dir)
                self.max_length = 4096
                self.context_length = 3072
                self.tokenizer.model_max_length = self.max_length
            elif "llama-3" in llm_name.lower():
                self.tokenizer = AutoTokenizer.from_pretrained(self.llm_name, cache_dir=self.cache_dir)
                self.max_length = 8192
                self.context_length = 7168
                if ".1" in llm_name or ".2" in llm_name:
                    self.max_length = 131072
                    self.context_length = 128000
                self.tokenizer.model_max_length = self.max_length
            elif "pmc" in llm_name.lower():
                self.tokenizer = AutoTokenizer.from_pretrained(self.llm_name, cache_dir=self.cache_dir)
                template_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'templates', 'pmc_llama.jinja')
                self.tokenizer.chat_template = open(template_path).read().replace('    ', '').replace('\n', '')
                self.max_length = 2048
                self.context_length = 1024
                # CRITICAL: Override tokenizer's model_max_length to match our max_length
                self.tokenizer.model_max_length = self.max_length
                print(f"DEBUG: Set PMC-LLaMA max_length={self.max_length}, context_length={self.context_length}")
                print(f"DEBUG: PMC-LLaMA template loaded from: {template_path}")
                print(f"DEBUG: PMC-LLaMA tokenizer.model_max_length set to: {self.tokenizer.model_max_length}")
            elif "qwen" in llm_name.lower():
                self.tokenizer = AutoTokenizer.from_pretrained(self.llm_name, cache_dir=self.cache_dir)
                self.max_length = 8192
                self.context_length = 7168
                self.tokenizer.model_max_length = self.max_length
                # print(f"DEBUG: Set Qwen max_length={self.max_length}, context_length={self.context_length}")
            elif "gemma" in llm_name.lower():
                self.tokenizer = AutoTokenizer.from_pretrained(self.llm_name, cache_dir=self.cache_dir)
                self.max_length = 8192
                self.context_length = 7168
                self.tokenizer.model_max_length = self.max_length
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(self.llm_name, cache_dir=self.cache_dir)
                self.max_length = 2048
                self.context_length = 1024

            self.model = transformers.pipeline(
                "text-generation",
                model=self.llm_name,
                # dtype=torch.float16,
                dtype=torch.bfloat16,
                device_map="auto",
                model_kwargs={"cache_dir":self.cache_dir},
            )
        
        self.follow_up = follow_up
        if self.rag and self.follow_up:
            self.answer = self.i_medrag_answer
            self.templates["medrag_system"] = simple_medrag_system
            self.templates["medrag_prompt"] = simple_medrag_prompt
            self.templates["i_medrag_system"] = i_medrag_system
            self.templates["follow_up_ask"] = follow_up_instruction_ask
            self.templates["follow_up_answer"] = follow_up_instruction_answer
        elif self.corpus_name == "MedCorp2":
            print(f"DEBUG: Using simplified 2-source retrieval for MedCorp2 with {self.retriever_name}")
            print("  - MedCorp: Combined general medical literature (direct question querying)")  
            print("  - UMLS: Medical terminology and relationships (direct question querying)")
            # Pre-warm initialize 2 source retrieval systems to avoid per-query initialization overhead
            self._initialize_source_retrievers()
            self.answer = self.medrag_answer_by_source
        else:
            self.answer = self.medrag_answer

    def _initialize_source_retrievers(self):
        """
        Pre-warm initialization of 2 source retrieval systems for MedCorp2.
        - Uses shared embedding model for GPU memory efficiency  
        - 2 separate GPU-based FAISS indexes (MedCorp on GPU 0, UMLS on GPU 1)
        - MedCorp: Combined general medical literature (pubmed + textbooks + statpearls + wikipedia)
        - UMLS: Medical terminology and concept relationships
        """
        print("Pre-warming 2 source retrievers for MedCorp2 with GPU FAISS...")
        
        # Simplified 2-source architecture
        self.sources = ["medcorp", "umls"]
        self.corpus_name_mapping = {
            "medcorp": "MedCorp",    # Combined general medical literature
            "umls": "UMLS"           # Medical terminology and relationships
        }
        
        # GPU allocation for FAISS indexes to avoid threading conflicts
        self.gpu_allocation = {
            "medcorp": 1,  # GPU 1 for general medical literature
            "umls": 0      # GPU 0 for medical terminology
        }
        
        # Initialize shared embedding model once (GPU optimization)
        from utils import CustomizeSentenceTransformer, retriever_names
        from sentence_transformers import SentenceTransformer
        import torch
        
        # Get the actual model name from retriever_names mapping
        actual_model_name = retriever_names[self.retriever_name][0]  # Take first model for shared embedding
        print(f"  DEBUG: Mapped retriever '{self.retriever_name}' to model '{actual_model_name}'")
        print(f"  Loading shared embedding model: {actual_model_name} on device {self.retriever_device}...")
        
        if "contriever" in actual_model_name.lower():
            self.shared_embedding_function = SentenceTransformer(actual_model_name, device=self.retriever_device)
        else:
            self.shared_embedding_function = CustomizeSentenceTransformer(actual_model_name, device=self.retriever_device)
        self.shared_embedding_function.eval()
        print(f"  ✓ Shared embedding model {actual_model_name} loaded on {self.retriever_device}")
        
        # Initialize 2 separate retrieval systems with GPU-specific FAISS
        self.source_retrievers = {}
        for source in self.sources:
            gpu_id = self.gpu_allocation[source]
            print(f"  Initializing {source} retrieval system on GPU {gpu_id}...")
            try:
                # Create retrieval system with GPU-specific device setting
                gpu_device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
                
                self.source_retrievers[source] = RetrievalSystem(
                    retriever_name=self.retriever_name,
                    corpus_name=self.corpus_name_mapping[source],
                    db_dir=self.db_dir,
                    cache=self.corpus_cache,
                    HNSW=self.HNSW,
                    device=gpu_device  # Use GPU-specific device for FAISS
                )
                
                # Replace individual embedding functions with shared one (GPU memory optimization)
                for retriever_list in self.source_retrievers[source].retrievers:
                    for retriever in retriever_list:
                        if hasattr(retriever, 'embedding_function') and retriever.embedding_function is not None:
                            retriever.embedding_function = self.shared_embedding_function
                
                print(f"  ✓ {source} retriever ready on GPU {gpu_id} (shared embedding model, separate GPU FAISS)")
            except Exception as e:
                print(f"  ✗ Failed to initialize {source} retriever: {e}")
                continue
        
        print(f"Pre-warming complete. {len(self.source_retrievers)}/2 source retrievers ready.")
        print("Architecture: 2 sources + 2 GPU FAISS indexes + 1 shared embedding model")
        print(f"  - MedCorp (general literature) on GPU {self.gpu_allocation['medcorp']}")
        print(f"  - UMLS (medical terminology) on GPU {self.gpu_allocation['umls']}")

    def custom_stop(self, stop_str, input_len=0):
        stopping_criteria = StoppingCriteriaList([CustomStoppingCriteria(stop_str, self.tokenizer, input_len)])
        return stopping_criteria

    def generate(self, messages, **kwargs):
        '''
        generate response given messages
        '''
        if "openai" in self.llm_name.lower():
            ans = openai_client(
                model=self.model,
                messages=messages,
                temperature=0.0,
                **kwargs
            )
        elif "gemini" in self.llm_name.lower():
            response = self.model.generate_content(messages[0]["content"] + '\n\n' + messages[1]["content"], **kwargs)
            ans = response.candidates[0].content.parts[0].text
        else:
            stopping_criteria = None
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            if "meditron" in self.llm_name.lower():
                # stopping_criteria = custom_stop(["###", "User:", "\n\n\n"], self.tokenizer, input_len=len(self.tokenizer.encode(prompt_cot, add_special_tokens=True)))
                stopping_criteria = self.custom_stop(["###", "User:", "\n\n\n"], input_len=len(self.tokenizer.encode(prompt, add_special_tokens=True)))
            if "llama-3" in self.llm_name.lower():
                # Filter out parameters that conflict with max_length
                # The original MedRAG only uses max_length, so we remove any max_new_tokens
                filtered_kwargs = {k: v for k, v in kwargs.items() if k not in ['max_new_tokens']}
                
                response = self.model(
                    prompt,
                    do_sample=False,
                    eos_token_id=[self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|eot_id|>")],
                    pad_token_id=self.tokenizer.eos_token_id,
                    max_length=self.max_length,
                    truncation=True,
                    stopping_criteria=stopping_criteria,
                    **filtered_kwargs
                )
            else:
                # Filter out parameters that conflict with max_length
                # The original MedRAG only uses max_length, so we remove any max_new_tokens
                filtered_kwargs = {k: v for k, v in kwargs.items() if k not in ['max_new_tokens']}
                
                response = self.model(
                    prompt,
                    do_sample=False,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.eos_token_id,
                    max_length=self.max_length,
                    max_new_tokens=None,  # CRITICAL: Explicitly disable pipeline's default of 256
                    truncation=True,
                    stopping_criteria=stopping_criteria,
                    **filtered_kwargs
                )
            # ans = response[0]["generated_text"]
            ans = response[0]["generated_text"][len(prompt):]
        return ans

    def medrag_answer(self, question, options=None, k=32, rrf_k=100, save_dir = None, snippets=None, snippets_ids=None, **kwargs):
        '''
        question (str): question to be answered
        options (Dict[str, str]): options to be chosen from
        k (int): number of snippets to retrieve
        rrf_k (int): parameter for Reciprocal Rank Fusion
        save_dir (str): directory to save the results
        snippets (List[Dict]): list of snippets to be used
        snippets_ids (List[Dict]): list of snippet ids to be used
        '''

        if options is not None:
            options = '\n'.join([key+". "+options[key] for key in sorted(options.keys())])
        else:
            options = ''

        # retrieve relevant snippets
        if self.rag:
            if snippets is not None:
                retrieved_snippets = snippets[:k]
                scores = []
            elif snippets_ids is not None:
                if self.docExt is None:
                    self.docExt = DocExtracter(db_dir=self.db_dir, cache=True, corpus_name=self.corpus_name)
                retrieved_snippets = self.docExt.extract(snippets_ids[:k])
                scores = []
            else:
                assert self.retrieval_system is not None
                retrieved_snippets, scores = self.retrieval_system.retrieve(question, k=k, rrf_k=rrf_k)

            contexts = ["Document [{:d}] (Title: {:s}) {:s}".format(idx, retrieved_snippets[idx]["title"], retrieved_snippets[idx]["content"]) for idx in range(len(retrieved_snippets))]
            if len(contexts) == 0:
                contexts = [""]
            if "openai" in self.llm_name.lower():
                contexts = [self.tokenizer.decode(self.tokenizer.encode("\n".join(contexts))[:self.context_length])]
            elif "gemini" in self.llm_name.lower():
                contexts = [self.tokenizer.decode(self.tokenizer.encode("\n".join(contexts))[:self.context_length])]
            else:
                contexts = [self.tokenizer.decode(self.tokenizer.encode("\n".join(contexts), add_special_tokens=False)[:self.context_length])]
        else:
            retrieved_snippets = []
            scores = []
            contexts = []

        if save_dir is not None and not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # generate answers
        answers = []
        if not self.rag:
            prompt_cot = self.templates["cot_prompt"].render(question=question, options=options)
            messages = [
                {"role": "system", "content": self.templates["cot_system"]},
                {"role": "user", "content": prompt_cot}
            ]
            ans = self.generate(messages, **kwargs)
            answers.append(re.sub("\s+", " ", ans))
        else:
            for context in contexts:
                prompt_medrag = self.templates["medrag_prompt"].render(context=context, question=question, options=options)
                messages=[
                        {"role": "system", "content": self.templates["medrag_system"]},
                        {"role": "user", "content": prompt_medrag}
                ]
                ans = self.generate(messages, **kwargs)
                answers.append(re.sub("\s+", " ", ans))
        
        if save_dir is not None:
            with open(os.path.join(save_dir, "snippets.json"), 'w') as f:
                json.dump(retrieved_snippets, f, indent=4)
            with open(os.path.join(save_dir, "response.json"), 'w') as f:
                json.dump(answers, f, indent=4)
        
        return answers[0] if len(answers)==1 else answers, retrieved_snippets, scores

    def medrag_answer_by_source(self, question, options=None, k=32, rrf_k=100, save_dir=None, **kwargs):
        '''
        Simplified MedCorp2 retrieval with 2 sources and direct question querying.
        
        Features:
        - 2 sources: MedCorp (general literature) + UMLS (medical terminology)
        - Direct question querying (no LLM query generation - token efficient)
        - GPU-based FAISS indexes (MedCorp on GPU 1, UMLS on GPU 0)  
        - 1 shared embedding model (GPU memory efficiency)
        - Original MedRAG design: question used directly as retrieval query
        
        question (str): question to be answered
        options (Dict[str, str]): options to be chosen from
        k (int): total number of snippets to retrieve (distributed across 2 sources)
        rrf_k (int): parameter for Reciprocal Rank Fusion
        save_dir (str): directory to save the results
        '''
        if not self.rag:
            # If RAG is disabled, fall back to regular CoT
            return self.medrag_answer(question, options, k, rrf_k, save_dir, **kwargs)
        
        if self.corpus_name != "MedCorp2":
            raise ValueError("medrag_answer_by_source only works with MedCorp2 corpus")
        
        if options is not None:
            options = '\n'.join([key+". "+options[key] for key in sorted(options.keys())])
        else:
            options = ''

        # Check if source retrievers are pre-initialized
        if not hasattr(self, 'source_retrievers') or not self.source_retrievers:
            raise RuntimeError("Source retrievers not initialized. Please check MedCorp2 initialization.")

        # Use pre-initialized 2 source retrievers (each with own GPU FAISS)
        sources = [src for src in self.sources]  # ["medcorp", "umls"]
        
        # Distribute k across 2 sources (roughly equal with remainder to MedCorp)
        k_medcorp = k // 2 + k % 2  # Give extra to general literature if odd
        k_umls = k // 2
        k_distribution = {"medcorp": k_medcorp, "umls": k_umls}
        
        print(f"Retrieving {k_medcorp} docs from MedCorp, {k_umls} docs from UMLS")
        
        all_retrieved_snippets = []
        all_scores = []
        source_contexts = []
        
        if save_dir is not None and not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # Use original question directly as query for both sources (no LLM generation)        
        # Process both sources with GPU FAISS (no threading conflicts as separate GPUs)
        for source in sources:
            query = question  # Use original question directly
            k_source = k_distribution[source]
            gpu_id = self.gpu_allocation[source]
            
            print(f"Processing {source} (GPU {gpu_id}, {k_source} docs)...")
            
            try:
                source_retrieval_system = self.source_retrievers[source]
                
                # Retrieve documents using original question from GPU FAISS index
                retrieved_snippets, scores = source_retrieval_system.retrieve(
                    query, 
                    k=k_source, 
                    rrf_k=rrf_k
                )
                
                # Add source information to each snippet
                for snippet in retrieved_snippets:
                    snippet['source_type'] = source
                    snippet['query_used'] = query
                
                # Create context for this source
                source_context = f"Source: {source.upper()}\n"
                source_context += f"Query: {query}\n"
                source_context += f"Retrieved Documents:\n"
                
                for idx, snippet in enumerate(retrieved_snippets):
                    source_context += f"Document [{idx}] (Title: {snippet['title']}) {snippet['content']}\n"
                
                all_retrieved_snippets.extend(retrieved_snippets)
                all_scores.extend(scores)
                source_contexts.append(source_context)
                
                print(f"  ✓ Retrieved {len(retrieved_snippets)} documents from {source} GPU FAISS")
                
            except Exception as e:
                print(f"  ✗ Error retrieving from {source}: {e}")
                continue
        
        # Combine all contexts
        if len(source_contexts) == 0:
            contexts = [""]
        else:
            combined_context = "\n\n".join(source_contexts)
            # Truncate to context length if needed
            if "openai" in self.llm_name.lower():
                combined_context = self.tokenizer.decode(self.tokenizer.encode(combined_context)[:self.context_length])
            elif "gemini" in self.llm_name.lower():
                combined_context = self.tokenizer.decode(self.tokenizer.encode(combined_context)[:self.context_length])
            else:
                combined_context = self.tokenizer.decode(self.tokenizer.encode(combined_context, add_special_tokens=False)[:self.context_length])
            contexts = [combined_context]
        
        # Generate answer using combined context
        answers = []
        for context in contexts:
            prompt_medrag = self.templates["medrag_prompt"].render(context=context, question=question, options=options)
            messages = [
                {"role": "system", "content": self.templates["medrag_system"]},
                {"role": "user", "content": prompt_medrag}
            ]
            ans = self.generate(messages, **kwargs)
            answers.append(re.sub(r'\s+', ' ', ans))
        
        # Save results
        if save_dir is not None:
            with open(os.path.join(save_dir, "snippets_by_source.json"), 'w') as f:
                json.dump(all_retrieved_snippets, f, indent=4)
            with open(os.path.join(save_dir, "response_by_source.json"), 'w') as f:
                json.dump(answers, f, indent=4)
            with open(os.path.join(save_dir, "source_contexts.json"), 'w') as f:
                json.dump(source_contexts, f, indent=4)
        
        print(f"✓ Retrieved total {len(all_retrieved_snippets)} documents from 2 sources using direct question querying")
        return answers[0] if len(answers)==1 else answers, all_retrieved_snippets, all_scores

    def i_medrag_answer(self, question, options=None, k=32, rrf_k=100, save_path = None, n_rounds=4, n_queries=3, qa_cache_path=None, **kwargs):
        if options is not None:
            options = '\n'.join([key+". "+options[key] for key in sorted(options.keys())])
        else:
            options = ''
        QUESTION_PROMPT = f"Here is the question:\n{question}\n\n{options}"

        context = ""
        qa_cache = []
        if qa_cache_path is not None and os.path.exists(qa_cache_path):
            qa_cache = eval(open(qa_cache_path, 'r').read())[:n_rounds]
            if len(qa_cache) > 0:
                context = qa_cache[-1]
            n_rounds = n_rounds - len(qa_cache)
        last_context = None

        # Run in loop
        max_iterations = n_rounds + 3
        saved_messages = [{"role": "system", "content": self.templates["i_medrag_system"]}]

        for i in range(max_iterations):
            if i < n_rounds:
                if context == "":
                    messages = [
                        {
                            "role": "system",
                            "content": self.templates["i_medrag_system"],
                        },
                        {
                            "role": "user",
                            "content": f"{QUESTION_PROMPT}\n\n{self.templates['follow_up_ask'].format(n_queries)}",
                        },
                    ]
                else:                
                    messages = [
                        {
                            "role": "system",
                            "content": self.templates["i_medrag_system"],
                        },
                        {
                            "role": "user",
                            "content": f"{context}\n\n{QUESTION_PROMPT}\n\n{self.templates['follow_up_ask'].format(n_queries)}",
                        },
                    ]
            elif context != last_context:
                messages = [
                    {
                        "role": "system",
                        "content": self.templates["i_medrag_system"],
                    },
                    {
                        "role": "user",
                        "content": f"{context}\n\n{QUESTION_PROMPT}\n\n{self.templates['follow_up_answer']}",
                    },
                ]
            elif len(messages) == 1:
                messages = [
                    {
                        "role": "system",
                        "content": self.templates["i_medrag_system"],
                    },
                    {
                        "role": "user",
                        "content": f"{context}\n\n{QUESTION_PROMPT}\n\n{self.templates['follow_up_answer']}",
                    },
                ]
            saved_messages.append(messages[-1])
            if save_path:
                with open(save_path, 'w') as f:
                    json.dump([p if type(p) == dict else p.model_dump() for p in saved_messages], f, indent=4)
            last_context = context
            last_content = self.generate(messages, **kwargs)
            response_message = {"role": "assistant", "content": last_content}
            saved_messages.append(response_message)
            if save_path:
                with open(save_path, 'w') as f:
                    json.dump([p if type(p) == dict else p.model_dump() for p in saved_messages], f, indent=4)       
            if i >= n_rounds and ("## Answer" in last_content or "answer is" in last_content.lower()):
                messages.append(response_message)
                messages.append(
                    {
                        "role": "user",
                        "content": "Output the answer in JSON: {'answer': your_answer (A/B/C/D)}" if options else "Output the answer in JSON: {'answer': your_answer}",
                    }
                )
                saved_messages.append(messages[-1])
                answer_content = self.generate(messages, **kwargs)
                answer_message = {"role": "assistant", "content": answer_content}
                messages.append(answer_message)
                saved_messages.append(messages[-1])
                if save_path:
                    with open(save_path, 'w') as f:
                        json.dump([p if type(p) == dict else p.model_dump() for p in saved_messages], f, indent=4)
                return messages[-1]["content"], messages
            elif "## Queries" in last_content:
                messages = messages[:-1]
                if last_content.split("## Queries")[-1].strip() == "":
                    print("Empty queries. Continue with next iteration.")
                    continue
                try:
                    action_str = self.generate([
                        {
                            "role": "user",
                            "content": f"Parse the following passage and extract the queries as a list: {last_content}.\n\nPresent the queries as they are. DO NOT merge or break down queries. Output the list of queries in JSON format: {{\"output\": [\"query 1\", ..., \"query N\"]}}",
                        }
                    ], **kwargs)
                    action_str = re.search(r"output\": (\[.*\])", action_str, re.DOTALL).group(1)
                    action_list = [re.sub(r'^\d+\.\s*', '', s.strip()) for s in eval(action_str)]
                except Exception as E:
                    print("Error parsing action list. Continue with next iteration.")
                    error_class = E.__class__.__name__
                    error = f"{error_class}: {str(E)}"
                    print(error)
                    if save_path:
                        with open(save_path + ".error", 'a') as f:
                            f.write(f"{error}\n")                
                    continue
                for question in action_list:
                    if question.strip() == "":
                        continue
                    try:
                        rag_result = self.medrag_answer(question, k=k, rrf_k=rrf_k, **kwargs)[0]
                        context += f"\n\nQuery: {question}\nAnswer: {rag_result}"
                        context = context.strip()
                    except Exception as E:
                        error_class = E.__class__.__name__
                        error = f"{error_class}: {str(E)}"
                        print(error)
                        if save_path:
                            with open(save_path + ".error", 'a') as f:
                                f.write(f"{error}\n")
                qa_cache.append(context)
                if qa_cache_path:
                    with open(qa_cache_path, 'w') as f:
                        json.dump(qa_cache, f, indent=4)
            else:
                messages.append(response_message)
                print("No queries or answer. Continue with next iteration.")
                continue
        return messages[-1]["content"], messages

class CustomStoppingCriteria(StoppingCriteria):
    def __init__(self, stop_words, tokenizer, input_len=0):
        super().__init__()
        self.tokenizer = tokenizer
        self.stops_words = stop_words
        self.input_len = input_len

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        tokens = self.tokenizer.decode(input_ids[0][self.input_len:])
        return any(stop in tokens for stop in self.stops_words)