
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
    
    model_readme = ""
    dataset_readme = ""
    code_readme = ""
    model_info = {}
    dataset_info = {}
    code_info = {}
    
    user_input = input() #see if returns user input from cli
    
    urls = [url.strip() for url in user_input.split(',')]
    
    for i in urls:
        if "huggingface.co/datasets" in i: #dataset condition check
            type = "DATASET"
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
                
        elif "huggingface.co/" in i: #model condition check
            type = "MODEL"
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
                
        elif "github.com" in i: #code condition check
            type = "CODE"
            match = re.search(r'github\.com/([^/]+)/([^/]+)', user_input)
            
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
        else:
            raise ValueError("Unknown URL type") #debug statement
    
    print(type)    

    
if __name__ == "__main__":
    main() 
    


