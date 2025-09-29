from metrics.performance_claims import performance_claims
from metrics.dataset_and_code_score import dataset_and_code_score
from input import find_dataset
import requests as rq

class Test_performanceclaims:
    def test_bert(self):
        model_url = "https://huggingface.co/google-bert/bert-base-uncased"
        score,latency = performance_claims(model_url)
        # sample output : 0.92
        assert ((0.92-.15) <= score <= 1)

    def test_audience(self):
        model_url = "https://huggingface.co/parvk11/audience_classifier_model"
        score,latency = performance_claims(model_url)
        # sample output: 0.15
        assert (0 <= score <= (0.15+.15))
    
    def test_whispertiny(self):
        model_url = "https://huggingface.co/openai/whisper-tiny/tree/main"
        score,latency = performance_claims(model_url)
        # sample output: 0.80
        assert ((0.80-0.15) <= score <= (0.80+0.15))

class Test_datasetandcodescore:
    def test_bert(self):
        code_url = "https://github.com/google-research/bert"
        dataset_url = "https://huggingface.co/datasets/bookcorpus/bookcorpus"
        score,latency = dataset_and_code_score(code_url, dataset_url)
        # sample output: 1
        assert((1-0.15) <= score <= 1)
    def test_no_urls(self):
        code_url = ""
        dataset_url = ""
        score,latency = dataset_and_code_score(code_url, dataset_url)
        # sample output: 0
        assert (score == 0)

class Test_inputmissingdataset:
    def test_seen(self):
        # Use bert-base-uncased from sample_input
        # Put the known dataset into seen 
        seen_set = {"https://huggingface.co/datasets/bookcorpus/bookcorpus"}
        # Model readme
        model_path = "google-bert/bert-base-uncased"
        model_rm_url = f"https://huggingface.co/{model_path}/raw/main/README.md"
        model_readme = rq.get(model_rm_url, timeout=50)
        if model_readme.status_code == 200:
            model_readme = model_readme.text.lower()
        # Look for dataset in readme
        found = find_dataset(model_readme, seen_set)
        assert(found == "https://huggingface.co/datasets/bookcorpus/bookcorpus")
    
    def test_notseen(self):
        seen_set = {"https://huggingface.co/datasets/bookcorpus/bookcorpus"}
        readme = "This is a readme. We used dataset image-net.org"
        found = find_dataset(readme,seen_set)
        assert(found == "")
