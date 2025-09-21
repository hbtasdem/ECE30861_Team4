'''
pip install requests
pip install
'''

'''
Scoring function to call all the metric calc functions

Parameters
----------
user_url: str
    user input pre-parsed
type: str
    type of user_url evaluated in main

Returns
-------
    None
'''

def masterScoring(user_url: str, type: str) -> None:
    from urllib.parse import urlparse
    import requests as rq
    import re
    import json
    
    parsed_url = urlparse(user_url)
    path = parsed_url.path.strip('/')
    
    api_info = {}
    readme = ""
    
    if type == "MODEL" or type == "DATASET":
        # HF logic
        api_url = f'https://huggingface.co/api/models/{path}' if type == "MODEL" else f'https://huggingface.co/api/datasets/{path}'
        try:
            api_response = rq.get(api_url)
            if api_response.status_code == 200:
                api_info = api_response.json()
        except:
            api_info = {}
        
        readme_url = f"https://huggingface.co/{path}/raw/main/README.md"
        try:
            readme = rq.get(readme_url, timeout=50)
            if readme.status_code == 200:
                readme = readme.text.lower()
        except:
            readme = ""
    
    elif type == "CODE":
        # GitHub logic
        match = re.search(r'github\.com/([^/]+)/([^/]+)', user_url)
        if match:
            owner, repo = match.groups()
            repo = repo.replace('.git', '')
            api_url = f"https://api.github.com/repos/{owner}/{repo}"
            
            try:
                api_response = rq.get(api_url)
                if api_response.status_code == 200:
                    api_info = api_response.json()
            except:
                api_info = {}
            
            try:
                readme_response = rq.get(f"{api_url}/readme", headers={'Accept': 'application/vnd.github.v3.raw'})
                if readme_response.status_code == 200:
                    readme = readme_response.text.lower()
            except:
                readme = ""

    ''' -- SCORE FUNC CALLS --'''
    
    # DATA QUALITY
    #complete_checker(api_info, readme)
    #correct_checker(readme)
    #coverage_checker(api_info, readme)
    #relevance_checker(api_info)
    
    # CODE QUALITY
    #code_quality(type, api_info, readme)

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
    
    user_url = input() #see if returns user input from cli
    
    if "huggingface.co/datasets" in user_url: #dataset condition check
        type = "DATASET"
    elif "huggingface.co/" in user_url: #model condition check
        type = "MODEL"
    elif "github.com" in user_url: #code condition check
        type = "CODE"
    else:
        raise ValueError("Unknown URL type") #debug statement

    print(type)

# if type == "MODEL" or type == "DATASET":
    masterScoring(user_url, type)

if __name__ == "__main__":
    main() 
    