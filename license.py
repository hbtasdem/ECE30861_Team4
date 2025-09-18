from huggingface_hub import HfApi, snapshot_download
from pathlib import Path
import re
from typing import Optional, Dict
import logging
import tempfile
import os
import time

logger = logging.getLogger(__name__)

COMPATIBLE_LICENSES = {"apache-2.0", "mit", "bsd-3-clause", "bsd-3", "bsl-1.0"}
INCOMPATIBLE_LICENSES = {"gpl", "agpl", "lgpl", "non-commercial", "creative commons"}

def extract_license_from_readme(readme_text: str) -> Optional[str]:
    """
    Extract license section from README text.

    Parameters
    ----------
    readme_text : str
        The text content of the README file.

    Returns
    -------
    Optional[str]
        The extracted license section, or None if not found.
    """
    patterns = [
        r'##\s*License[^#]*(.*?)(?=##|$)',
        r'##\s*Licensing[^#]*(.*?)(?=##|$)',
        r'\*\*\s*License\s*\*\*[^*]*(.*?)(?=\*\*|$)',
        r'#\s*License[^#]*(.*?)(?=#|$)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, readme_text, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
    return None

def get_license_from_api(model_id: str) -> Optional[str]:
    """
    Get the license from the Hugging Face API model.

    Parameters
    ----------
    model_id : str
        The model identifier on Hugging Face.

    Returns
    -------
    Optional[str]
        The license name, or None if not found.
    """
    try: 
        api = HfApi()
        model_info = api.model_info(repo_id=model_id)
        if hasattr(model_info, 'cardData') and model_info.cardData:
            return model_info.cardData.get('license')
    except Exception as e:
        logger.error(f"Error getting license from API for {model_id}: {e}")
    return None

def is_gated_model(model_id: str) -> bool:
    """
    Check if the model is gated or has a custom license.

    Parameters
    ----------
    model_id : str
        The model identifier on Hugging Face.

    Returns
    -------
    bool
        True if the model is gated, False otherwise.
    """
    try:
        api = HfApi()
        model_info = api.model_info(repo_id=model_id)
        card_data = getattr(model_info, 'cardData', {})
        if card_data.get("extra_gated_prompt"):
            return True
            
        gated_indicators = ["gated", "custom license", "agreement", "click through", "accept terms"]
        for key, value in card_data.items():
            if isinstance(value, str):
                lower_value = value.lower()
                if any(indicator in lower_value for indicator in gated_indicators):
                    return True
    except Exception as e:
        logger.error(f"Error checking if model is gated for {model_id}: {e}")
    return False

def get_license_from_repo(model_id: str) -> Optional[str]:
    """
    Clone the repository and extract license information from files.

    Parameters
    ----------
    model_id : str
        The model identifier on Hugging Face.

    Returns
    -------
    Optional[str]
        The license text found in the repo, or None if not found.
    """
    try:
        # Create temporary directory for download
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_path = snapshot_download(
                repo_id=model_id,
                allow_patterns=["README.md", "LICENSE", "COPYING", "LICENCE"],
                ignore_patterns=["*.bin", "*.safetensors", "*.h5", "*.pt", "*.gguf"],
                local_dir=os.path.join(temp_dir, model_id.replace('/', '_'))
            )
            repo_path = Path(repo_path)

            # Check for license files in priority order
            for filename in ["LICENSE", "COPYING", "LICENCE"]:
                file_path = repo_path / filename
                if file_path.exists():
                    return file_path.read_text(encoding='utf-8', errors='ignore')

            # Check README.md for license section
            readme_path = repo_path / "README.md"
            if readme_path.exists():
                readme_text = readme_path.read_text(encoding='utf-8', errors='ignore')
                return extract_license_from_readme(readme_text)
                
    except Exception as e:
        logger.error(f"Error getting license from repo for {model_id}: {e}")
    return None

def contains_compatible_license(license_text: str) -> bool:
    """
    Check if license text contains a compatible license.

    Parameters
    ----------
    license_text : str
        The license text to check.

    Returns
    -------
    bool
        True if a compatible license is found, False otherwise.
    """
    if not license_text:
        return False
    lower_text = license_text.lower()
    return any(license in lower_text for license in COMPATIBLE_LICENSES)

def contains_license_keywords(license_text: str) -> bool:
    """
    Check if license text contains keywords like license or copyright.

    Parameters
    ----------
    license_text : str
        The license text to check.

    Returns
    -------
    bool
        True if license keywords are found, False otherwise.
    """
    if not license_text:
        return False
    lower_text = license_text.lower()
    keywords = ["license", "licence", "copyright"]
    return any(keyword in lower_text for keyword in keywords)

def get_license_score(model_id: str) -> Dict[str, float]:
    """
    Calculate license compatibility score and latency.

    Parameters
    ----------
    model_id : str
        The Hugging Face model identifier.

    Returns
    -------
    Dict[str, float]
        Dictionary containing license score and latency in milliseconds.
    """
    start_time = time.time()
    
    license_name = get_license_from_api(model_id)
    
    if license_name:
        license_name_lower = license_name.lower()
        if any(compatible in license_name_lower for compatible in COMPATIBLE_LICENSES):
            score = 1.0
        elif any(incompatible in license_name_lower for incompatible in INCOMPATIBLE_LICENSES):
            score = 0.0
        else:
            score = 0.5
    elif is_gated_model(model_id):
        score = 0.0
    else:
        license_text = get_license_from_repo(model_id)
        if not license_text:
            score = 0.0
        elif contains_compatible_license(license_text):
            score = 1.0
        elif contains_license_keywords(license_text):
            score = 0.5
        else:
            score = 0.0

    latency = int((time.time() - start_time) * 1000)
    
    return {
        'license': score,
        'license_latency': latency
    }