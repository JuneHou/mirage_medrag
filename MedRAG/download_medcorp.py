#!/usr/bin/env python3
"""
Progressive corpus downloader for MedRAG
Downloads corpora in order of size: Textbooks -> Wikipedia -> PubMed
"""

import os
import sys
from huggingface_hub import list_repo_files, hf_hub_download
from tqdm import tqdm
import time

def download_full_corpus(repo_name, corpus_dir, cache_dir="./MedRAG/src/data/hf_cache"):
    """Download all files from a HuggingFace corpus repository"""
    
    print(f"\n=== Downloading {repo_name} corpus ===")
    
    # Create target directory
    target_dir = os.path.join(corpus_dir, repo_name.split('/')[-1])
    chunk_dir = os.path.join(target_dir, "chunk")
    os.makedirs(chunk_dir, exist_ok=True)
    
    try:
        # List all files in the repository
        files = list_repo_files(repo_name, repo_type='dataset')
        chunk_files = [f for f in files if f.startswith('chunk/') and f.endswith('.jsonl')]
        
        print(f"Found {len(chunk_files)} chunk files to download")
        
        if not chunk_files:
            print(f"No chunk files found in {repo_name}")
            return False
            
        # Download each file
        success_count = 0
        for filename in tqdm(chunk_files, desc=f"Downloading {repo_name}"):
            try:
                # Download to cache first
                local_file = hf_hub_download(
                    repo_id=repo_name,
                    filename=filename,
                    repo_type='dataset',
                    cache_dir=cache_dir
                )
                
                # Create target path
                basename = os.path.basename(filename)  # Just the filename without 'chunk/'
                target_path = os.path.join(chunk_dir, basename)
                
                # Create symlink to save space
                try:
                    if os.path.exists(target_path):
                        os.remove(target_path)
                    os.symlink(local_file, target_path)
                    success_count += 1
                except OSError:
                    # If symlink fails, copy the file
                    import shutil
                    shutil.copy2(local_file, target_path)
                    success_count += 1
                    
            except Exception as e:
                print(f"Error downloading {filename}: {e}")
                continue
                
        print(f"✓ Successfully downloaded {success_count}/{len(chunk_files)} files from {repo_name}")
        return success_count > 0
        
    except Exception as e:
        print(f"Error with {repo_name}: {e}")
        return False

def main():
    # Configuration
    corpus_base_dir = "./src/data/corpus"
    cache_dir = "/data/wang/junh/hf_cache"
    
    # Create cache directory
    os.makedirs(cache_dir, exist_ok=True)
    
    # List of corpus repositories in order of priority/size
    corpus_repos = [
        ("MedRAG/textbooks", "18 files (~20MB)"),
        ("MedRAG/wikipedia", "646 files (~650MB)"), 
        ("MedRAG/pubmed", "1166 files (~2GB)"),
    ]
    
    print("MedRAG Corpus Downloader")
    print("=" * 50)
    print(f"Cache directory: {cache_dir}")
    print(f"Target directory: {corpus_base_dir}")
    print()
    
    print("Download plan:")
    for i, (repo, size) in enumerate(corpus_repos, 1):
        print(f"{i}. {repo} - {size}")
    print()
    
    # Ask user which corpora to download
    print("Options:")
    print("1. Download textbooks only (quick test)")
    print("2. Download textbooks + wikipedia (medium)")
    print("3. Download all corpora (full MedCorp, large)")
    
    choice = input("Enter choice (1-3): ").strip()
    
    if choice == "1":
        repos_to_download = corpus_repos[:1]
    elif choice == "2":
        repos_to_download = corpus_repos[:2]
    elif choice == "3":
        repos_to_download = corpus_repos
    else:
        print("Invalid choice, downloading textbooks only")
        repos_to_download = corpus_repos[:1]
    
    print(f"\nDownloading {len(repos_to_download)} corpora...")
    
    success_count = 0
    for repo, size in repos_to_download:
        if download_full_corpus(repo, corpus_base_dir, cache_dir):
            success_count += 1
        time.sleep(1)  # Brief pause between repos
    
    print(f"\n=== Download Summary ===")
    print(f"Successfully downloaded: {success_count}/{len(repos_to_download)} corpora")
    
    # Provide usage instructions
    if success_count > 0:
        print("\n✓ Corpus data downloaded successfully!")
        print("\nUsage instructions:")
        if success_count == 1:
            print("- Use corpus_name='Textbooks' for downloaded textbooks")
        elif success_count == 2:
            print("- Use corpus_name='Textbooks' for textbooks only")
            print("- Use corpus_name='Wikipedia' for wikipedia only")  
            print("- Create custom corpus combining both")
        else:
            print("- Use corpus_name='MedCorp' for the full dataset")
            print("- Use individual corpus names for specific datasets")
            
        print("\nNext steps:")
        print("1. Make sure sentence-transformers is version 5.1.1+")
        print("2. Make sure numpy is version < 2.0")
        print("3. Run your VLLM MedRAG script")
    else:
        print("⚠ No corpora downloaded successfully")

if __name__ == "__main__":
    main()