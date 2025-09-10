from huggingface_hub import HfApi, ModelInfo
from huggingface_hub import snapshot_download
from pathlib import Path
import re
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

COMPATIBLE_LICENSES = {"apache-2.0", "mit", "bsd-3-clause", "bsd-3", "bsl-1.0"}
INCOMPATIBLE_LICENSES = {"gpl", "agpl", "lgpl", "non-commercial", "creative commons"}

def get_license_from_api(model_id: str) -> str:
    "Will get the license from the HF API model"
    # try and except --> code that may throw an expection and has block for error handling
    try:
        api = HfApi()
        model_info = api.model_info(repo_id=model_id)
        if hasattr(model_info, 'cardData') and model_info.cardData: # check if the model info has the term cardData 
            return model_info.cardData.get('license')
    except Exception as e: #handles erroors such as network issues, model not found, etc
        logger.error(f"Error getting license from API for {model_id}: {e}")
    return None

def is_gated_model(model_id: str) -> bool:
    # Placeholder implementation
    return False

def get_license_from_repo(model_id: str) -> str:
    # Placeholder implementation
    return ""

def contains_compatible_license(license_text: str) -> bool:
    # Placeholder implementation
    return False

def contains_license_keywords(license_text: str) -> bool:
    # Placeholder implementation
    return "license" in license_text.lower() or "copyright" in license_text.lower()

def get_license_score(model_id: str) -> float:
    """
    Returns a float score between 0.0 and 1.0 for license quality.
    """
    # 1. Try HF API for license
    license_name = get_license_from_api(model_id)
    
    if license_name:
        if license_name in COMPATIBLE_LICENSES: # e.g., "apache-2.0"
            return 1.0 # Perfect score
        elif license_name in INCOMPATIBLE_LICENSES: # e.g., "gpl"
            return 0.0
        else:
            return 0.5 # Ambiguous, needs review
    
    # 2. If no API license, check for gated model
    if is_gated_model(model_id):
        return 0.0 # Custom license is a failure for automation
    
    # 3. Clone repo and check README/LICENSE files
    license_text = get_license_from_repo(model_id)
    if not license_text:
        return 0.0 # No license found
    
    if contains_compatible_license(license_text):
        return 1.0
    elif contains_license_keywords(license_text): # "license", "copyright"
        return 0.5 # Unclear license
    else:
        return 0.0