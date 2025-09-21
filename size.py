import os
from typing import Dict
import time
from huggingface_hub import HfApi

def get_largest_file_size_gb(model_id: str) -> float:
    """
    Get the size of the largest file in a Hugging Face model repository using the API.

    Parameters
    ----------
    model_id : str
        The Hugging Face model identifier (e.g., 'google-bert/bert-base-uncased').

    Returns
    -------
    float
        The largest file size in gigabytes (GB), or 0.0 if an error occurs.
    """
    try:
        api = HfApi()
        model_info = api.model_info(repo_id=model_id)
        
        # Find the largest file size
        largest_size = 0
        for file in model_info.siblings:
            if file.size is not None and file.size > largest_size:
                largest_size = file.size
                
        return largest_size / (1024 ** 3)  # Convert bytes to GB
        
    except Exception as e:
        print(f"Error getting file size for {model_id}: {e}")
        return 0.0

def calculate_size_score(model_id: str) -> Dict[str, float]:
    """
    Calculate size compatibility scores for different hardware tiers.

    Parameters
    ----------
    model_id : str
        The Hugging Face model identifier.

    Returns
    -------
    Dict[str, float]
        A dictionary mapping device names to their size compatibility score (0.0-1.0).
    """
    start_time = time.time()
    
    largest_file_gb = get_largest_file_size_gb(model_id)
    
    # Define thresholds for each device (in GB)
    thresholds = {
        'raspberry_pi': 2.0,
        'jetson_nano': 4.0,
        'desktop_pc': 8.0,
    }
    
    size_scores = {}
    for device, threshold in thresholds.items():
        # Linear decay: score = 1 at 0GB, score = 0 at the threshold
        score = max(0.0, 1.0 - (largest_file_gb / threshold))
        size_scores[device] = round(score, 2)
    
    # AWS server can handle any size
    size_scores['aws_server'] = 1.0
    
    latency = int((time.time() - start_time) * 1000)  # Convert to milliseconds
    
    return {
        'size_score': size_scores,
        'size_score_latency': latency
    }