import requests
import time

"""
Fetches the dataset card (README.md) from Hugging Face.

Parameters
----------
dataset_url : str
    Full Hugging Face dataset URL.

Returns
-------
string
    README text
"""
def fetch_dataset_card(dataset_url: str) -> dict:
    dataset_id = dataset_url.split("datasets/")[1].strip("/")
    raw_url = f"https://huggingface.co/datasets/{dataset_id}/raw/main/README.md"

    # Fetch the README
    resp = requests.get(raw_url)
    if resp.status_code == 200:
        return resp.text


"""
    Indicates if the dataset used for training and benchmarking
    is well documented, along with any example code.

    Parameters
    ----------
    code_link :   string
    dataset_link: string

    Returns
    -------
    float
        Score in range 0 (bad) - 1 (good)
    float
        latency of metric in seconds
"""
# def dataset_and_code_score(code_link, dataset_link):
def dataset_and_code_score(code_link, dataset_link):

    # start latency timer 
    start = time.time()

    # Max score = 5
    # Each worth +1 score:
    # Code, Dataset sources, uses, data collection and processing, bias
    score = 0
    max_score = 5

    # Assume if no dataset or code link given, then it doesn't exist
    if code_link:
        score += 1

    if dataset_link:
        dataset_card = fetch_dataset_card(dataset_link) 
        # Dataset sources
        if any(key in dataset_card for key in ["source", "sources"]):
            score += 1
        # Uses
        if any(key in dataset_card for key in ["uses", "usage", "use case"]):
            score += 1
        # Data collection and processing
        if any(key in dataset_card for key in ["creators", "split","collection","curation"]):
            score += 1
        # Bias / ethical considerations
        if any(key in dataset_card for key in ["bias", "considerations"]):
            score += 1

    score = score / max_score
    end = time.time()
    latency = end - start

    return score, latency

# UNIT TEST
class Test_datasetandcodescore:
    def testbert(self):
        code_link = "https://github.com/google-research/bert"
        dataset_link = "https://huggingface.co/datasets/bookcorpus/bookcorpus"
        score,latency = dataset_and_code_score(code_link, dataset_link)
        assert (score == 1)
    