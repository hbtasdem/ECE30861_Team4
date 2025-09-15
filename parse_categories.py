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

Returns
-------
    None
'''

def masterScoring(user_url: str) -> None: #call scoring functions
   
    from urllib.parse import urlparse
    import requests as rq
    import json
    
    
    parsed_url = urlparse(user_url) #needed to translate user url to HF api url
    path = parsed_url.path.strip('/') 
    
    api_url = f'https://huggingface.co/api/models/{path}' #HF api url
    api_response = rq.get(api_url) #rest api GET
    api_info = api_response.json()
    readme_url = f"https://huggingface.co/{user_url}/raw/main/README.md"
    
    try:
        readme = rq.get(readme_url, timeout= 50)
        if readme.status_code == 200:
            readme = readme.text
        else:
            print(f"Readme not found. Error code: {readme.status_code}")
    except rq.RequestException as e:
        print(f"Error fetching Readme: {e}")
    
    readme = readme.text.lower()
    # license = license_check(api_info)
    #print(api_info) #debug -using api to get info
    
    #relevance_checker(api_info)
    coverage_checker(api_info, readme)
    # complete_checker(api_info, readme)


def coverage_checker(api_info: str, readme: str) -> int:
    
    # import requests
    
    # readme_url = f"https://huggingface.co/{user_url}/raw/main/README.md"
    # print(readme_url)
    
    # try:
    #     readme = requests.get(readme_url, timeout= 50)
    #     if readme.status_code == 200:
    #         readme = readme.text
    #     else:
    #         print("Readme not found. Error code: {readme.status_code}")
    # except requests.RequestException as e:
    #     print(f"Error fetching Readme: {e}")
    
    # readme = readme.content.lower()
    
    
    # Following list of words are used as a filter on the readme file. This lis it curated by 
    # Claude Sonnet 4 with the following prompts: 
    '''
    "Give me a list of words that can be used on Hugging Face model readme content to check if the model
    can be credited for data coverage. remember that these words will be used as a filter to score the data quality 
    of the model, so try to have a comprehensive list." and
    "not geographic coverage, we are testing whether the data is diverse on samples 
    so it represents the general population of a specific purpose well."
    ''' 
    
    checked_words = ['diverse', 'diversity', 'varied', 'variety', 'various', 'different',
    'heterogeneous', 'mixed', 'multiple', 'range', 'spectrum',
    
    # Representativeness
    'representative', 'represents', 'representative sample', 'cross-section',
    'reflects', 'mirrors', 'captures', 'encompasses', 'covers',
    'comprehensive', 'extensive', 'broad', 'wide', 'spanning',
    
    # Balance and distribution
    'balanced', 'well-balanced', 'evenly distributed', 'uniform',
    'stratified', 'proportional', 'equal representation', 'fair distribution',
    'well-distributed', 'equally represented', 'balanced across']
    

    coverage_count = sum(1 for word in checked_words if word in readme)
    
    # Simple scoring based on coverage word frequency
    if coverage_count >= 5:
        return 1.0  # Highly descriptive of coverage
    elif coverage_count >= 3:
        return 0.7  # Good coverage description
    elif coverage_count >= 1:
        return 0.5  # Some coverage mentioned
    else:
        return 0.3  # No clear coverage description


# def relevance_checker(api_info: str) -> int:
    
#     from datetime import date
#     from dateutil import parser
      
#     today = date.today() #todays date
    
#     date_creation = api_info['createdAt'] #extract creation date from json, returns date & time format
#     print(date_creation)
    
#     date_creation = parser.parse(date_creation) #parse the date info wo the time/time zone
#     print(date_creation)
#     date_creation = date_creation.date()
#     print(date_creation)
    
#     days_passed = (today - date_creation).days
#     print(days_passed)

#     if days_passed > 360:
#         relevance_score = 0.01
#     elif days_passed > 180:
#         relevance_score = 0.03
#     elif days_passed > 90:
#         relevance_score = 0.06
#     else:
#         relevance_score = 0.1
    
#     print(relevance_score)
#     return relevance_score 

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

    if type == "MODEL":
        masterScoring(user_url)

if __name__ == "__main__":
    main() 
    