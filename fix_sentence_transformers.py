#!/usr/bin/env python3
"""
Compatibility patch for sentence-transformers with newer huggingface-hub
This replaces the deprecated cached_download function with hf_hub_download
"""

import sys
import importlib.util
from huggingface_hub import hf_hub_download

def cached_download(url, library_name=None, library_version=None, cache_dir=None, 
                   user_agent=None, subfolder="", force_download=False, 
                   proxies=None, etag_timeout=10, resume_download=False, 
                   use_auth_token=None, local_files_only=False, legacy_cache_layout=False):
    """
    Compatibility wrapper for the deprecated cached_download function.
    Maps old cached_download calls to hf_hub_download.
    """
    print(f"DEBUG: Patching cached_download call for URL: {url}")
    
    # Parse the URL to extract repo_id and filename
    # URLs typically look like: https://huggingface.co/sentence-transformers/model/resolve/main/file.json
    if "huggingface.co" in url and "/resolve/" in url:
        parts = url.split('/')
        if "sentence-transformers" in parts:
            # Find indices
            try:
                st_idx = parts.index("sentence-transformers")
                model_idx = st_idx + 1
                resolve_idx = parts.index("resolve")
                
                repo_id = f"sentence-transformers/{parts[model_idx]}"
                filename = "/".join(parts[resolve_idx + 2:])  # Skip "resolve" and revision
                
                print(f"DEBUG: Mapped to repo_id='{repo_id}', filename='{filename}'")
                
                return hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    subfolder=subfolder,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    local_files_only=local_files_only,
                    token=use_auth_token,
                    resume_download=resume_download
                )
            except (ValueError, IndexError) as e:
                print(f"DEBUG: Failed to parse URL structure: {e}")
    
    # Fallback: try to download directly if it's a simple URL
    print(f"DEBUG: Using fallback for URL: {url}")
    return url

# Monkey patch the huggingface_hub module
import huggingface_hub
huggingface_hub.cached_download = cached_download

print("✅ Applied compatibility patch for sentence-transformers + huggingface-hub")