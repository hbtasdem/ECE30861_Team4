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

def extract_model_id_from_url(url: str) -> str:
    """
    Extract model ID from various URL formats.
    """
    if 'huggingface.co' in url:
        pattern = r'huggingface\.co/([^/]+/[^/?]+)'
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    
    if '/' in url and ' ' not in url and '://' not in url:
        return url
    
    return url

def extract_license_from_readme(readme_text: str) -> Optional[str]:
    """
    Extract license section from README text.
    """
    patterns = [
        r'##\s*License[^#]*(.*?)(?=##|$)',
        r'##\s*Licensing[^#]*(.*?)(?=##|$)',
        r'\*\*\s*License\s*\*\*[^*]*(.*?)(?=\*\*|$)',
        r'#\s*License[^#]*(.*?)(?=#|$)',
        r'license:\s*([^\n]+)',  # Simple license: line
    ]
    
    for pattern in patterns:
        match = re.search(pattern, readme_text, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
    return None

def get_license_from_api(model_id: str) -> Optional[str]:
    """
    Get the actual license from the Hugging Face API.
    """
    try: 
        api = HfApi()
        model_info = api.model_info(repo_id=model_id)
        
        # Check multiple possible locations for license
        if hasattr(model_info, 'cardData') and model_info.cardData:
            license_val = model_info.cardData.get('license')
            if license_val:
                return license_val
        
        # Check tags for license information
        if hasattr(model_info, 'tags'):
            for tag in model_info.tags:
                if 'license:' in tag:
                    return tag.split('license:')[-1]
        
        return None
        
    except Exception as e:
        logger.debug(f"API error for {model_id}: {e}")
        return None

def is_gated_model(model_id: str) -> bool:
    """
    Check if the model is actually gated.
    """
    try:
        api = HfApi()
        model_info = api.model_info(repo_id=model_id)
        
        # Check if model is explicitly gated
        if hasattr(model_info, 'gated') and model_info.gated:
            return True
            
        # Check card data for gating indicators
        card_data = getattr(model_info, 'cardData', {})
        if card_data.get("extra_gated_prompt"):
            return True
            
        # Check for other gating indicators
        gated_indicators = ["gated", "custom license", "agreement", "click through", "accept terms"]
        for key, value in card_data.items():
            if isinstance(value, str):
                lower_value = value.lower()
                if any(indicator in lower_value for indicator in gated_indicators):
                    return True
                    
        return False
        
    except Exception:
        return False

def get_license_from_repo(model_id: str) -> Optional[str]:
    """
    Actually clone the repository and extract license information.
    """
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_path = snapshot_download(
                repo_id=model_id,
                allow_patterns=["README.md", "LICENSE", "COPYING", "LICENCE", "*.txt"],
                ignore_patterns=["*.bin", "*.safetensors", "*.h5", "*.pt", "*.gguf"],
                local_dir=os.path.join(temp_dir, model_id.replace('/', '_'))
            )
            repo_path = Path(repo_path)

            # Check for license files
            license_files = ["LICENSE", "COPYING", "LICENCE", "LICENSE.txt", "COPYING.txt"]
            for filename in license_files:
                file_path = repo_path / filename
                if file_path.exists():
                    content = file_path.read_text(encoding='utf-8', errors='ignore')
                    if content.strip():  # Only return if file has content
                        return content

            # Check README for license section
            readme_path = repo_path / "README.md"
            if readme_path.exists():
                readme_text = readme_path.read_text(encoding='utf-8', errors='ignore')
                license_text = extract_license_from_readme(readme_text)
                if license_text:
                    return license_text
                
            return None
                
    except Exception as e:
        logger.debug(f"Repo error for {model_id}: {e}")
        return None

def contains_compatible_license(license_text: str) -> bool:
    """
    Actually check if license text contains compatible license.
    """
    if not license_text:
        return False
        
    lower_text = license_text.lower()
    
    # Check for compatible licenses
    for license in COMPATIBLE_LICENSES:
        if license in lower_text:
            return True
            
    # Also check for common variations
    license_variations = {
        "apache": "apache-2.0",
        "apache 2": "apache-2.0", 
        "bsd": "bsd-3-clause",
        "mit license": "mit",
        "boost": "bsl-1.0"
    }
    
    for variation, actual_license in license_variations.items():
        if variation in lower_text and actual_license in COMPATIBLE_LICENSES:
            return True
            
    return False

def contains_license_keywords(license_text: str) -> bool:
    """
    Actually check for license keywords.
    """
    if not license_text:
        return False
        
    lower_text = license_text.lower()
    keywords = ["license", "licence", "copyright", "terms of use", "terms and conditions"]
    return any(keyword in lower_text for keyword in keywords)

def get_license_score(model_id: str) -> Dict[str, float]:
    """
    Calculate actual license compatibility score and latency.

    Parameters
    ----------
    model_id : str
        The Hugging Face model identifier.

    Returns
    -------
    Dict[str, float]
        Dictionary containing actual calculated license score and latency.
    """
    start_time = time.time()
    
    # Extract model ID from URL if needed
    clean_model_id = extract_model_id_from_url(model_id)
    
    print(f"Model: {clean_model_id}")
    print("Calculating license score...")
    
    # Step 1: Try API first
    license_name = get_license_from_api(clean_model_id)
    if license_name:
        print(f"Found license in API: {license_name}")
        license_name_lower = license_name.lower()
        
        if any(compatible in license_name_lower for compatible in COMPATIBLE_LICENSES):
            score = 1.0
            print("Score: 1.0 (compatible license)")
        elif any(incompatible in license_name_lower for incompatible in INCOMPATIBLE_LICENSES):
            score = 0.0
            print("Score: 0.0 (incompatible license)")
        else:
            score = 0.5
            print("Score: 0.5 (ambiguous license)")
    else:
        # Step 2: Check if gated
        if is_gated_model(clean_model_id):
            score = 0.0
            print("Score: 0.0 (gated model)")
        else:
            # Step 3: Check repository
            print("Checking repository for license...")
            license_text = get_license_from_repo(clean_model_id)
            
            if not license_text:
                score = 0.0
                print("Score: 0.0 (no license found)")
            elif contains_compatible_license(license_text):
                score = 1.0
                print("Score: 1.0 (compatible license in repo)")
            elif contains_license_keywords(license_text):
                score = 0.5
                print("Score: 0.5 (license keywords found)")
            else:
                score = 0.0
                print("Score: 0.0 (no recognizable license)")

    latency = int((time.time() - start_time) * 1000)
    print(f"License calculation latency: {latency} ms")
    
    return {
        'license': score,
        'license_latency': latency
    }

# Test function that prints actual calculations
if __name__ == "__main__":
    test_models = [
        "google-bert/bert-base-uncased",
        "parvk11/audience_classifier_model", 
        "openai/whisper-tiny"
    ]
    
    print("=== ACTUAL LICENSE CALCULATIONS ===")
    for model_input in test_models:
        print(f"\n--- Testing: {model_input} ---")
        result = get_license_score(model_input)
        print(f"FINAL RESULT: {result}")