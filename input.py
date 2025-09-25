
'''
Main function to get & condition the user input url

Parameters
----------
user_url: str
    user input pre-parsed

Returns
-------
    None
'''    
def main():
    
    from urllib.parse import urlparse
    import requests as rq
    import re
    import json
    import sys
    import metric
    
    model_readme = ""
    dataset_readme = ""
    code_readme = ""
    model_info = {}
    dataset_info = {}
    code_info = {}
    
    # user_input = input() #see if returns user input from cli
        
    url_file = sys.argv[1]
    with open(url_file, "r", encoding="ascii") as input:
        # loop for reading each line:
        for line in input:
            # if line:  # skip blank lines
            print("\n",line)

            urls = [url.strip() for url in line.split(',')]

            raw_code_url = urls[0]
            raw_dataset_url = urls[1]
            raw_model_url = urls[2]

            print("code:", raw_code_url)
            print("dataset:", raw_dataset_url)
            print("model:", raw_model_url)
                
            parsed_model = urlparse(raw_model_url)
            model_path = parsed_model.path.strip('/')
            parsed_dataset = urlparse(raw_dataset_url)
            dataset_path = parsed_dataset.path.strip('/')
            parsed_code = urlparse(raw_code_url)
            code_path = parsed_code.path.strip('/')
            
            model_url = f'https://huggingface.co/api/models/{path}'
    
            try:
                api_response = rq.get(model_url)
                if api_response.status_code == 200:
                    model_info = api_response.json() #api info for dataset
            except:
                model_info = {}
            
            model_rm_url = f"https://huggingface.co/{path}/raw/main/README.md"
            
            try:
                model_readme = rq.get(model_rm_url, timeout=50)
                if model_readme.status_code == 200:
                    model_readme = model_readme.text.lower()
            except:
                model_readme = ""
    
        #might not need
            dataset_url = f'https://huggingface.co/api/models/{path}'
            try:
                api_response = rq.get(dataset_url)
                if api_response.status_code == 200:
                    dataset_info = api_response.json() #api info for dataset
            except:
                dataset_info = {}
            
            dataset_rm_url = f"https://huggingface.co/{path}/raw/main/README.md"
            
            try:
                dataset_readme = rq.get(dataset_rm_url, timeout=50)
                if dataset_readme.status_code == 200:
                    dataset_readme = dataset_readme.text.lower()
            except:
                dataset_readme = ""
                
                
            match = re.search(r'github\.com/([^/]+)/([^/]+)', raw_code_url)
            
            if match:
                owner, repo = match.groups()
                repo = repo.replace('.git', '')
                code_url = f"https://api.github.com/repos/{owner}/{repo}" #api url for code
                
                try:
                    api_response = rq.get(code_url)
                    if api_response.status_code == 200:
                        code_info = api_response.json()
                except:
                    code_info = {}
                
                try:
                    code_readme = rq.get(f"{code_url}/readme", headers={'Accept': 'application/vnd.github.v3.raw'})
                    if code_readme.status_code == 200:
                        code_readme = code_readme.text.lower()
                except:
                    code_readme = ""
            
            net_score = metric.main(model_info, model_readme, raw_model_url, code_info, code_readme, raw_dataset_url)
            print(net_score)
                    
    
if __name__ == "__main__":
    main() 
    


