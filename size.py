import os
from typing import Dict

def get_largest_file_size_gb(model_dir: str) -> float:
    """
    Scans the given directory and returns the largest file size in GB.

    Parameters
    ----------
    model_dir : str
        The path to the model directory.

    Returns
    -------
    float
        The largest file size in the directory, in gigabytes (GB).
    """
    largest_size = 0
    for root, _, files in os.walk(model_dir):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                size = os.path.getsize(file_path)
                if size > largest_size:
                    largest_size = size
            except Exception:
                continue
    return largest_size / (1024 ** 3)  # Convert bytes to GB

def calculate_size_score(model_dir: str) -> Dict[str, float]:
    """
    Returns a dictionary mapping device names to a float score between 0.0 and 1.0.

    Parameters
    ----------
    model_dir : str
        The path to the model directory.

    Returns
    -------
    Dict[str, float]
        A dictionary mapping device names to their size compatibility score.
    """
    largest_file_gb = get_largest_file_size_gb(model_dir)
    
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
    
    # An AWS server can handle any size, so it always gets a 1.0
    size_scores['aws_server'] = 1.0
    
    return size_scores