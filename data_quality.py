'''
pip install huggingface_hub
'''
from huggingface_hub import DatasetCard, ModelCard
from huggingface_hub.utils import EntryNotFoundError 

'''
Completeness
comment out once completed after readme info can be verified
pip3 install datasets

Try #1: Check a list of completeness keywords in the card_data and readme, not an ideal indicator of completeness
Try #2: Use pandas isnull to check if there's a missing value in the dataset
'''
def complete_checker(api_info: str, readme: str) -> int:
    
    from urllib.parse import urlparse
    
    complete_score = 0.0
    
    
    card_data = api_info.get('cardData', {})
    
    #the following list of completeness keywords was created using Claude Sonnet 4
    complete_kw= {
    # Essential metadata completeness
    'license': 'License information present',
    'description': 'Clear dataset description',
    'use': 'Uses',
    'limitation': 'Limitations',
    'tags': 'Keywords for model categorization',
    'citation': 'Proper citation information',
    'source': 'Data source attribution',
    'language': 'Data languages available'
    }
    
    check1 = sum(1 for item in complete_kw if item in card_data)
    check2 = sum(1 for item in complete_kw if item in readme)
    checklist = check1 + check2
    
    print(card_data)
    # print(readme)
    print(checklist)
    
   
    if checklist >= 7: 
        return 1.0   
    elif checklist >= 4:
        return 0.5
    else:
        return 0.1  



'''
Correctness

(09/16) Decided to take the correctness metric down as it is impractical to check during run time
(09/21) Found Accuracy tag in one of the readmes so added a regex expression to extract it in case most models have it
    (IMP) Will lower the accuracy weight since it might not be a common practice to have accuracy value displayed in readme.
'''
def correct_checker(readme: str) -> int:
    import re
    
    if not readme:
        return 0.0
    
    acc_pattern = [
        r'type:\s*accuracy\s*value:\s*([\d.]+)',
        r'accuracy:\s*([\d.]+)',
        r'"accuracy":\s*([\d.]+)',
    ]

    for pattern in acc_pattern:
       match = re.search(pattern, readme, re.IGNORECASE)
       if match:
           accuracy_val = float(match.group(1))
           return accuracy_val

'''
Try #1: treats more data labels = more coverage
Coverage calculator -> readme content search to analyze coverage

'''
def coverage_checker(api_info: str, readme: str) -> int:
    
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
    

'''
Relevance calculator

Parameters
----------
api_info: str
    api info in json format, extracted from user input post-parsing

Returns
-------
relevance_score : int
    parameter that stores the calculated relevance information
        0.1 if the model 0-90 days old
        0.06 if the model 90-180 days old
        0.03 if the model 180-360 days old
        0.01 if the model +360 days old

'''
def relevance_checker(api_info: str) -> int:
    
    from datetime import date
    from dateutil import parser
      
    today = date.today() #todays date
    
    date_creation = api_info['createdAt'] #extract creation date from json, returns date & time format
    date_creation = parser.parse(date_creation) #format the date/time info, make it easy to extract date
    date_creation = date_creation.date() #extract date wo the time/time zone
    
    days_passed = (today - date_creation).days #get number of days passed in int
    
    #used for debug:
    # print(today)
    # print(date_creation)
    # print(days_passed)

    #categorize relevance based on the number of days passed 
        #might need to change the thresholds depending on the testcases fed, relevance of 1 year might be too ambitious
    if days_passed > 360: #720 maybe?
        relevance_score = 0.01
    elif days_passed > 180:
        relevance_score = 0.03
    elif days_passed > 90:
        relevance_score = 0.06
    else:
        relevance_score = 0.1
    
    return relevance_score 

def data_quality_calc(complete_score: int, correct_score: int, coverage_score: int, relevance_score: int) -> int:
    
    data_quality_score = 0.0
    
    return data_quality_score


'''add a new function data_quality which will be the main function here'''