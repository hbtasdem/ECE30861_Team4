
def masterScoring(user_url: str) -> None: #call scoring functions
   
    from urllib.parse import urlparse
    import requests as rq
    import json
    
    parsed_url = urlparse(user_url) #needed to translate user url to HF api url
    path = parsed_url.path.strip('/') 
    
    api_url = f'https://huggingface.co/api/models/{path}' #HF api url
    api_response = rq.get(api_url) #rest api GET
    api_info = api_response.json()
    
    print(api_info) #debug -using api to get info

    
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

    if type == "MODEL":
        masterScoring(user_url)

if __name__ == "__main__":
    main() 
    