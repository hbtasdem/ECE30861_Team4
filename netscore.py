from typing import Dict
from size import calculate_size_score
from license import get_license_score

def calculate_net_score(model_id: str) -> float:
    """
    Calculate the overall NetScore using weighted metric scores.

    Parameters
    ----------
    model_id : str
        The Hugging Face model identifier.

    Returns
    -------
    float
        The weighted NetScore between 0.0 and 1.0.
    """
    # Calculate individual metric scores
    license_score = get_license_score(model_id)
    size_scores = calculate_size_score(model_id)
    size_avg = sum(size_scores.values()) / len(size_scores)  # Average of hardware scores

    # Placeholder values for other metrics (replace with actual calculations)
    ramp_up = 0.8  # Example value
    bus_factor = 0.7  # Example value
    dataset_quality = 0.9  # Example value
    dataset_code = 0.85  # Example value
    code_quality = 0.75  # Example value
    performance = 0.6  # Example value

    # Apply the weighted formula from your project plan
    weighted_sum = (
        (0.25 * ramp_up) +
        (0.20 * size_avg) + 
        (0.15 * dataset_quality) +
        (0.15 * bus_factor) +
        (0.07 * dataset_code) +
        (0.12 * code_quality) + 
        (0.06 * performance)
    )
    
    # Multiply by license score (acts as a gatekeeper)
    net_score = license_score * weighted_sum
    
    return round(net_score, 2)

model_id = "google-bert/bert-base-uncased"
net_score = calculate_net_score(model_id)
print(f"NetScore for {model_id}: {net_score}")