# Evidence of claims (benchmarks, evals)
from urllib.parse import urlparse
import time
from huggingface_hub import model_info, hf_hub_download
import requests

"""
    Fetches the model card JSON from Hugging Face given a model URL.

    Parameters
    ----------
    model_url : str
        Full Hugging Face model URL.

    Returns
    -------
    dict
        Parsed JSON from Hugging Face API.
    """
def fetch_model_card(model_url: str) -> dict:
    parsed = urlparse(model_url)
    path = parsed.path.strip("/")
    parts = path.split("/")

    if "tree" in parts:
        tree_index = parts.index("tree")
        model_id = "/".join(parts[:tree_index])  # everything before "tree"
    else:
        model_id = path

    info = model_info(model_id)
    return model_id, info


"""
    Computes a score 0-1 based on evidence supporting model performance.

    Parameters
    ----------
    model_url : str
        URL to Hugging Face model.

    Returns
    -------
    float
        Score in range 0-1.
    float 
        latency in seconds.
    """
def performance_claims(model_url):
    #start latency timer 
    start = time.time()
    score = 0

    model_id, info = fetch_model_card(model_url)

    # look for results in the model info
    # if every metric has a value and is verified it gets a 1
    # 10% is are they verified. 
    # if all the values are None it gets 0 
    total_vals = 0
    verified = 0
    if info.model_index:
        for entry in info.model_index:
            for result in entry.get("results",[]):
                for metric in result.get("metrics",[]):
                    if metric["value"] != None:
                        total_vals += 1
                        if metric["verified"] == False:
                            verified += 1
    
    if total_vals != 0:
        score = 1 - 0.1 * (verified / total_vals)

    else:
        # model_index not in the model info, so have to search the readme to evaluation metrics
        path = hf_hub_download(repo_id=model_id, filename="README.md")
        with open(path, "r") as f:
            readme = f.read()
        # LLM REQUIREMENT FULFILL HERE. complete later. piazza post is unclear how we handle keys. 

        if "[More Information Needed]" in readme:
            score = 0
        else:
            if "Evaluation results" in readme:
                score += .7

    end = time.time()
    latency = end - start
    return score, latency


# UNIT TEST
class Test_performanceclaims:
    def test1(self):
        model_url = "https://huggingface.co/google-bert/bert-base-uncased"
        score,latency = performance_claims(model_url)
        assert (score >= .7 and score <= 1)

    def test2(self):
        model_url = "https://huggingface.co/parvk11/audience_classifier_model"
        score,latency = performance_claims(model_url)
        assert (score <= .3 and score >= 0)
    
    def test3(self):
        model_url = "https://huggingface.co/openai/whisper-tiny/tree/main"
        score,latency = performance_claims(model_url)
        assert (score >= .7 and score <= 1)
