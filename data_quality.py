'''
pip install huggingface_hub
'''
from huggingface_hub import DatasetCard, ModelCard
from huggingface_hub.utils import EntryNotFoundError 

'''
Completeness
'''
def complete_checker(api_info: str, readme_) -> int:
    
    complete_score = 0.0
    
    card_data = api_info.get('cardData', {})
        
    return complete_score


'''
Correctness
'''
def correct_checker(api_info: str) -> int:
    
    correct_score = 0.0

    return correct_score


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
    if days_passed > 360: 
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