from huggingface_hub import HfApi, ModelInfo
from huggingface_hub import snapshot_download
from pathlib import Path
import re
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

COMPATIBLE_LICENSES = {"apache-2.0", "mit", "bsd-3-clause", "bsd-3", "bsl-1.0"}
INCOMPATIBLE_LICENSES = {"gpl", "agpl", "lgpl", "non-commercial", "creative commons"}

def extract_license_from_readme(readme_text: str) -> Optional[str]:
    """Extract license section from README text."""
    patterns = [
        r'##\s*License[^#]*(.*?)(?=##|$)',
        r'##\s*Licensing[^#]*(.*?)(?=##|$)',
        r'\*\*\s*License\s*\*\*[^*]*(.*?)(?=\*\*|$)',
        r'#\s*License[^#]*(.*?)(?=#|$)',  # Different headingss for licences that could present in README
    ]
    for pattern in patterns:
        match = re.search(pattern, readme_text, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip() # Will iterate through all patterns and return the first match
    return None

def get_license_from_api(model_id: str) -> str:
    "Will get the license from the HF API model"
    try: 
        api = HfApi()
        model_info = api.model_info(repo_id=model_id) # Check cardData for license field as specified in project requirements
        if hasattr(model_info, 'cardData') and model_info.cardData: #has the card data attribute
            return model_info.cardData.get('license')
    except Exception as e:
        logger.error(f"Error getting license from API for {model_id}: {e}")
    return None

def is_gated_model(model_id: str) -> bool:
    try:
        api = HfApi()
        model_info = api.model_info(repo_id=model_id)
        card_data = getattr(model_info, 'cardData', {})
        if card_data.get("extra_gated_prompt"):  # Check for explicit gating indicators in model metadata
            return True
        # Continued gating pattern detection for robustness
        gated_indicators = ["gated", "custom license", "agreement", "click through", "accept terms"]
        for key, value in card_data.items():
            if isinstance(value, str):
                lower_value = value.lower()
                if any(indicator in lower_value for indicator in gated_indicators):
                    return True
    except Exception as e:
        logger.error(f"Error checking if model is gated for {model_id}: {e}")
        return False
    return False

def get_license_from_repo(model_id: str) -> Optional[str]:
    """Clone the repository and extract license information from files."""
    try:
        # Download only necessary files for efficiency (ignore large model weights)
        repo_path = Path(repo_path)

         # Check for license files in priority order (LICENSE > COPYING > LICENCE)
        for filename in ["LICENSE", "COPYING", "LICENCE"]:
            file_path = repo_path / filename
            if file_path.exists():
                return file_path.read_text(encoding='utf-8', errors='ignore')

        # Check README.md for license section if no dedicated license files found
        readme_path = repo_path / "README.md"
        if readme_path.exists():
            readme_text = readme_path.read_text(encoding='utf-8', errors='ignore')
            license_text = extract_license_from_readme(readme_text)
            if license_text:
                return license_text
        return None

    except Exception as e:
        logger.error(f"Error getting license from repo for {model_id}: {e}")
        return None

def contains_compatible_license(license_text: str) -> bool:
    """Check if license text contains a compatible license."""
    if not license_text:
        return False
    lower_text = license_text.lower() #for case-insensitive comparison
    return any(license in lower_text for license in COMPATIBLE_LICENSES)

def contains_license_keywords(license_text: str) -> bool:
    """Check if license text contains keywords like license or copyright."""
    if not license_text:
        return False
    lower_text = license_text.lower()
    keywords = ["license", "licence", "copyright"] # Common license indicator terms
    return any(keyword in lower_text for keyword in keywords)

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