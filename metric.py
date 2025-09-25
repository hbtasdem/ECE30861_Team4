
# from metrics.data_quality import data_quality
# from metrics.code_quality import code_quality
from metrics.dataset_and_code_score import dataset_and_code_score
from metrics.performance_claims import performance_claims
# from metrics.size import size_calculator
# from metrics.bus_factor import bus_factor
# from metrics.ramp_up_time import ramp_up_time

import sys
import os
import requests
from urllib.parse import urlparse
from logger import log_error

"""
If no dataset in input line, see if model was trained on
a previously seen dataset. 

Parameters
----------
model_url : str
    Raw model url. 
    NOTE: THIS COULD BE model_id if already parsed elsewhere

Returns
-------
string
    The dataset link found, or None. 
"""
def find_dataset(model_url: str, seen_datasets: set) -> str:
    parsed = urlparse(model_url)
    path_parts = parsed.path.strip("/").split("/")
    model_id = "/".join(path_parts[-2:])
    readme_url = f"https://huggingface.co/{model_id}/resolve/main/README.md"

    resp = requests.get(readme_url, timeout=30)
    if resp.status_code != 200:
        return None
    readme_text = resp.text.lower()

    for dataset_url in seen_datasets:
        dataset = dataset_url.split("/")[-1].lower()
        if dataset in readme_text:
            return dataset_url
    # None found
    return None


def main():
    # ----- example parsing input -----
    LOG_FILE = os.environ.get("LOG_FILE")
    if not LOG_FILE:
        print("LOG_FILE not set", file=sys.stderr)
        sys.exit(1)
    log_error("yello")
    url_file = sys.argv[1]
    with open(url_file, "r", encoding="ascii") as input:
        # loop for reading each line:
        # for line in input:
        #     if line:  # skip blank lines
        #         print("\n",line)
        #         urls = [url.strip() for url in line.split(',')]
        #         code_url = urls[0]
        #         dataset_url = urls[1]
        #         model_url = urls[2]
        #         print("code:",code_url)
        #         print("dataset:",dataset_url)
        #         print("model:",model_url)
        # but for now I just want one at a time:
        line = input.readline().strip()
        
    line = line.split(",")
    code_url = line[0].strip()
    dataset_url = line[1].strip()
    model_url = line[2].strip()

    print("Sample input:")
    print("code:",code_url)
    print("dataset:",dataset_url)
    print("model:",model_url)

    # ----- FIND MISSING DATASET -----
    seen_datasets = set()
    if dataset_url:
        seen_datasets.add(dataset_url)
    else:
        dataset_url = find_dataset(model_url, seen_datasets)
    # --------------------------------
        
    # ----- Used for my own debugging/ testing -----
    # print("Seen_datasets: ",seen_datasets)
    # Give another model trained on bookcorpus to test find_dataset    
    # model_url = "https://huggingface.co/AiresPucrs/bert-base-bookcorpus"
    # seen_datasets.add(dataset_url)
    # dataset_url = find_dataset(model_url, seen_datasets)
    
    # print("\nTest with new input:")
    # print(dataset_url)
    # print(model_url)
    # print("Seen datasets: ",seen_datasets)
    # ---------------------------------------------
    
    # ----- Call my metrics -----
    dc_score, dc_latency = dataset_and_code_score(code_url,dataset_url)
    print("\ndc score:",dc_score)
    print("dc latency:",dc_latency)

    perf_score, perf_latency = performance_claims(model_url) 
    print("\nperf score:",perf_score)
    print("perf latency:",perf_latency)
    
    
if __name__ == "__main__":
    main() 
    