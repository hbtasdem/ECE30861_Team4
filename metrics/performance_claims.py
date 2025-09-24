# Evidence of claims (benchmarks, evals)
from urllib.parse import urlparse
import time
from huggingface_hub import model_info, hf_hub_download
import os
import requests


"""
Use Purdue GenAI Studio to measure performance claims.

Parameters
----------
prompt : str
    Request for information and the README for the model.

Returns
-------
string
    Response from LLM. Should be just a float in string format
"""
def query_genai_studio(prompt: str) -> str:
    # get api key from environment variable
    api_key = os.environ.get("GEN_AI_STUDIO_API_KEY")
    if not api_key:
        print("Error: GEN_AI_STUDIO_API_KEY environment variable not found")

    url = "https://genai.rcac.purdue.edu/api/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    body = {
        "model": "llama3.1:latest",
        "messages": [{"role": "user", "content": prompt}],
        "stream": False
    }

    response = requests.post(url, headers=headers, json=body)
    if response.status_code != 200:
        raise Exception(f"GenAI Studio API error: {response.status_code}, {response.text}")

    data = response.json()
    # OpenAI-style completion
    return data["choices"][0]["message"]["content"]


"""
Fetches the model card JSON from Hugging Face given a model URL.

Parameters
----------
model_url : str
    Hugging Face model URL.

Returns
-------
tuple (str, object)
str
    The model id parsed from the url 
object
    model_info object from Hugging Face
"""
def fetch_model_card(model_url: str) -> tuple[str, object]:
    parsed = urlparse(model_url)
    path = parsed.path.strip("/")
    parts = path.split("/")

    if "tree" in parts:
        tree_index = parts.index("tree")
        model_id = "/".join(parts[:tree_index])
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
tuple(float, float)
float
    Score in range 0-1.
float 
    Latency in seconds.
"""
def performance_claims(model_url: str) -> tuple[str, str]:
    #start latency timer 
    start = time.time()
    score = 0

    model_id, info = fetch_model_card(model_url)

    # look for results in the model info
    # if every metric has a value and is verified it gets a 1
    # 10%  of score is based on are they verified. 
    # if all the values are None, must check README
    total_vals = 0
    verified = 0
    if info.model_index:
        for entry in info.model_index:
            for result in entry.get("results",[]):
                for metric in result.get("metrics",[]):
                    if metric["value"] != None:
                        total_vals += 1
                        if metric["verified"] == True:
                            verified += 1
    
    if total_vals != 0: # some values found
        score = 0.9 + 0.1 * (verified / total_vals)

    else: # no metric values found in model_info
        # Have to search the readme for evaluation metrics
        path = hf_hub_download(repo_id=model_id, filename="README.md")
        with open(path, "r") as f:
            readme = f.read()

        # LLM REQUIREMENT FULFILLED HERE.
        prompt = ( f"Analyze the following README text for evidence of evaluation results or benchmarks "
                   f"supporting the model's performance. Return a score between 0 and 1. I am using this in "
                   f"code, so do not return ANYTHING but the float score. \n\nREADME:\n{readme}" )
        valid_llm_output = False
        while valid_llm_output == False:
            llm_score_str = query_genai_studio(prompt)
            # Get float score from string
            llm_score = float(llm_score_str.strip())
            if (llm_score >= 0) and (llm_score <= 1):
                valid_llm_output = True
                score = llm_score
            else:
                print("Invalid llm output. Retrying.")

    end = time.time()
    latency = end - start
    return score, latency


# UNIT TESTS
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
