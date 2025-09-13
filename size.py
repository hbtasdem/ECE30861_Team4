from typing import Dict
# Make sure get_largest_file_size_gb is imported or defined somewhere
# from .some_module import get_largest_file_size_gb

def get_largest_file_size_gb(model_id):
    # Placeholder implementation; replace with actual logic to get file size
    return 3.5  # Example size in GB

def calculate_size_score(model_id: str) -> Dict[str, float]:
    """
    Returns a dictionary mapping device names to a float score between 0.0 and 1.0.
    """
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
        size_scores[device] = round(score, 2) # Keep it to 2 decimal places
    
    # An AWS server can handle any size, so it always gets a 1.0
    size_scores['aws_server'] = 1.0
    
    return size_scores