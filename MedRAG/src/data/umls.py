import os
import tqdm
import json
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter

def ends_with_ending_punctuation(s):
    ending_punctuation = ('.', '?', '!')
    return any(s.endswith(char) for char in ending_punctuation)

def concat(title, content):
    if ends_with_ending_punctuation(title.strip()):
        return title.strip() + " " + content.strip()
    else:
        return title.strip() + ". " + content.strip()

if __name__ == "__main__":
    # Load communities data
    communities_path = "/data/wang/junh/datasets/umls2text/leiden_only_output/communities/communities.json"
    
    print("Loading communities data...")
    with open(communities_path) as f:
        communities = json.load(f)
    
    print(f"Total communities: {len(communities)}")
    
    # Initialize text splitter for long summaries (chunk_size=1000, chunk_overlap=200)
    # This follows the same strategy as wikipedia.py and textbooks.py
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    
    # Create output directory following the pattern: corpus/{source}/chunk/
    output_dir = "corpus/umls/chunk"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Group communities by run for batch processing (similar to wikipedia.py batching strategy)
    # This helps organize the data and makes it easier to process in parallel
    communities_by_run = {}
    for comm in communities:
        run = comm.get('run', 0)
        if run not in communities_by_run:
            communities_by_run[run] = []
        communities_by_run[run].append(comm)
    
    print(f"Communities organized into {len(communities_by_run)} runs")
    
    # Statistics
    total_valid = 0
    total_invalid = 0
    total_chunks = 0
    total_chunked_communities = 0
    
    # Process each run separately (creates one JSONL file per run)
    for run in tqdm.tqdm(sorted(communities_by_run.keys())):
        output_file = os.path.join(output_dir, f"umls_run{run:02d}.jsonl")
        
        # Skip if file already exists
        if os.path.exists(output_file):
            print(f"Skipping run {run} - file already exists")
            continue
        
        saved_text = []
        valid_count = 0
        invalid_count = 0
        chunked_count = 0
        
        for comm in communities_by_run[run]:
            summary = comm.get('summary', '').strip()
            
            # Skip invalid summaries (blank or error messages)
            # This follows the validation pattern seen in all 4 sources
            if not summary:
                invalid_count += 1
                continue
            
            # Check for error messages like "The community is too large"
            if 'error' in summary.lower() or 'too large' in summary.lower():
                invalid_count += 1
                continue
            
            # Create hierarchical title similar to StatPearls approach
            # Format: "UMLS -- Run X -- Level Y -- Community Z"
            level = comm.get('level', 0)
            comm_id = comm.get('community_id', 0)
            title = f"UMLS -- Run {run} -- Level {level} -- Community {comm_id}"
            
            # Normalize whitespace in summary (common across all sources)
            summary = re.sub("\s+", " ", summary.strip())
            
            # Check if summary needs chunking
            # Following wikipedia.py and textbooks.py: chunk if text is long
            if len(summary) > 1000:
                # Split long summaries into chunks with overlap
                chunks = text_splitter.split_text(summary)
                chunked_count += 1
                
                for i, chunk in enumerate(chunks):
                    # Create unique ID for each chunk (similar to wikipedia.py pattern)
                    chunk_id = f"UMLS_R{run}_L{level}_C{comm_id}_{i}"
                    
                    saved_text.append(json.dumps({
                        "id": chunk_id,
                        "title": title,
                        "content": chunk,
                        "contents": concat(title, chunk)
                    }))
            else:
                # Keep short summaries as-is (similar to pubmed.py - no chunking for short text)
                chunk_id = f"UMLS_R{run}_L{level}_C{comm_id}"
                
                saved_text.append(json.dumps({
                    "id": chunk_id,
                    "title": title,
                    "content": summary,
                    "contents": concat(title, summary)
                }))
            
            valid_count += 1
        
        # Save to JSONL file (one file per run)
        if len(saved_text) > 0:
            with open(output_file, 'w') as f:
                f.write('\n'.join(saved_text))
        
        # Update global statistics
        total_valid += valid_count
        total_invalid += invalid_count
        total_chunks += len(saved_text)
        total_chunked_communities += chunked_count
        
        print(f"Run {run}: {valid_count} valid, {invalid_count} invalid, {len(saved_text)} chunks, {chunked_count} chunked")
    
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"Total valid communities: {total_valid}")
    print(f"Total invalid/skipped communities: {total_invalid}")
    print(f"Total communities that were chunked: {total_chunked_communities}")
    print(f"Total chunks created: {total_chunks}")
    print(f"Output saved to: {output_dir}/umls_run*.jsonl")
    print(f"{'='*60}")
