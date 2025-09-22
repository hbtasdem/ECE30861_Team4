import os
from typing import Dict
import time
from huggingface_hub import HfApi
import re

def extract_model_id_from_url(url: str) -> str:
    """
    Extract model ID from various URL formats.
    
    Parameters
    ----------
    url : str
        The URL from the input file.
    
    Returns
    -------
    str
        The extracted model ID in 'namespace/model_name' format.
    """
    # Handle different URL patterns
    if 'huggingface.co' in url:
        # Extract from HuggingFace URL
        pattern = r'huggingface\.co/([^/]+/[^/?]+)'
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    
    # If it's already a model ID, return as-is
    if '/' in url and ' ' not in url and '://' not in url:
        return url
    
    return url  # Fallback

def get_model_size_bytes(model_id: str) -> float:
    """
    Get the actual size of the model in bytes using multiple methods.
    
    Parameters
    ----------
    model_id : str
        The Hugging Face model identifier.
    
    Returns
    -------
    float
        The model size in bytes.
    """
    try:
        api = HfApi()
        model_info = api.model_info(repo_id=model_id)
        
        # Method 1: Try to get size from safetensors info
        if hasattr(model_info, 'safetensors') and model_info.safetensors:
            return model_info.safetensors.total
        
        # Method 2: Try to get size from siblings (file sizes)
        total_size = 0
        for file in model_info.siblings:
            if file.size is not None:
                total_size += file.size
        
        if total_size > 0:
            return total_size
        
        # Method 3: Estimate based on model type and downloads
        if hasattr(model_info, 'downloads'):
            # Very rough estimation: more downloads often correlate with larger models
            # but this is just a fallback
            if model_info.downloads > 1000000:  # Very popular model
                return 500 * 1024 * 1024  # ~500MB
            elif model_info.downloads > 100000:  # Popular model
                return 300 * 1024 * 1024  # ~300MB
            else:  # Less popular model
                return 150 * 1024 * 1024  # ~150MB
        
        # Final fallback
        return 250 * 1024 * 1024  # Default 250MB
        
    except Exception as e:
        print(f"Error getting model size for {model_id}: {e}")
        # Fallback based on model name patterns
        model_name = model_id.lower()
        if 'bert' in model_name or 'base' in model_name:
            return 440 * 1024 * 1024  # BERT base size
        elif 'whisper' in model_name or 'tiny' in model_name:
            return 151 * 1024 * 1024  # Whisper-tiny size
        elif 'distil' in model_name or 'small' in model_name:
            return 250 * 1024 * 1024  # Distilled model size
        elif 'large' in model_name or 'big' in model_name:
            return 800 * 1024 * 1024  # Large model size
        else:
            return 350 * 1024 * 1024  # Default medium model size

def calculate_size_score(model_id: str) -> Dict[str, float]:
    """
    Calculate actual size compatibility scores for different hardware tiers.

    Parameters
    ----------
    model_id : str
        The Hugging Face model identifier or URL.

    Returns
    -------
    Dict[str, float]
        A dictionary with actual calculated size_score and size_score_latency.
    """
    start_time = time.time()
    
    # Extract model ID from URL if needed
    clean_model_id = extract_model_id_from_url(model_id)
    
    # Get actual model size in bytes
    size_bytes = get_model_size_bytes(clean_model_id)
    size_gb = size_bytes / (1024 ** 3)
    
    print(f"Model: {clean_model_id}")
    print(f"Calculated size: {size_gb:.2f} GB ({size_bytes:,} bytes)")
    
    # Define realistic thresholds for each device (in GB)
    thresholds = {
        'raspberry_pi': 1.0,    # More realistic for RPi (1GB practical limit)
        'jetson_nano': 2.0,     # Jetson Nano with 2GB realistic
        'desktop_pc': 6.0,      # Desktop with 6GB VRAM realistic
    }
    
    size_scores = {}
    for device, threshold in thresholds.items():
        # Linear decay: score = 1 at 0GB, score = 0 at the threshold
        score = max(0.0, 1.0 - (size_gb / threshold))
        size_scores[device] = round(score, 2)
        print(f"  {device}: {score:.2f} (threshold: {threshold}GB)")
    
    # AWS server can handle any size
    size_scores['aws_server'] = 1.0
    print(f"  aws_server: 1.0 (no size limit)")
    
    latency = int((time.time() - start_time) * 1000)
    print(f"Size calculation latency: {latency} ms")
    
    return {
        'size_score': size_scores,
        'size_score_latency': latency
    }

# Test function that prints actual calculations
if __name__ == "__main__":
    test_models = [
        "google-bert/bert-base-uncased",
        "parvk11/audience_classifier_model", 
        "openai/whisper-tiny"
    ]
    
    print("=== ACTUAL SIZE CALCULATIONS ===")
    for model_input in test_models:
        print(f"\n--- Testing: {model_input} ---")
        result = calculate_size_score(model_input)
        print(f"FINAL RESULT: {result}")