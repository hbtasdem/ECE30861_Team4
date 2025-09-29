'''
Code quality calculator using popularity as a reliability factor (assuming the number of people using a model)

Parameters
----------
type: str
    type of the user url input based on the main function categorization
api_info: str
    api info in json format, extracted from user input post-parsing
readme: str
    readme from the model or github based on the user url type

Returns
-------
code_quality_score : int
    calculated with the following metrics:
        - reusability -> checks the documentation lenght of the readme, weight: 0.4x
        - reliability -> for model input, # of downloads; for github code, # of stars + # of forks, weight: 0.4x
        - testability -> readme filter words count, weight: 0.2x
    
'''

import logger
  
# def code_quality_calc(type: str, api_info: str, readme: str) -> int:
def code_quality(model_info: str, code_info: str, model_readme: str, code_readme: str):
    import requests as rq
    import time
    
    start = time.time()
    logger.info("Calculating code_quality metric")

    # init metrics used in final score calculation to avoid bugs
    len_score = 0.0
    pop_score = 0.0
    
    # reusability check
    doc_length = len(model_readme.split()) if model_readme else 0
    if doc_length > 1000:
        len_score = 1.0
    elif doc_length > 500:
        len_score = 0.7
    elif doc_length > 200:
        len_score = 0.4
    else:
        len_score = 0.1
        
    logger.debug(f"Documentation length: {doc_length}, Reusability score: {len_score}")
    
    # reliability check - determine type based on available data
    if model_info and model_info.get('downloads') is not None:
        # MODEL case - downloads check from card_data
        popularity = model_info.get('downloads', 0) 
        if popularity > 700000:
            pop_score = 1.0
        elif popularity > 500000:
            pop_score = 0.7
        elif popularity > 100000:
            pop_score = 0.3
        else:
            pop_score = 0.1
        logger.debug(f"Model downloads: {popularity}, Reliability score: {pop_score}")
            
    elif code_info:
        # CODE case - stars + forks check of github repo
        stars = code_info.get('stargazers_count', 0)
        forks = code_info.get('forks_count', 0)
        total_engagement = stars + forks
        
        if total_engagement > 90000:
            pop_score = 1.0
        elif total_engagement > 60000:
            pop_score = 0.7
        elif total_engagement > 30000:
            pop_score = 0.5
        else:
            pop_score = 0.1
        logger.debug(f"GitHub engagement: {total_engagement}, Reliability score: {pop_score}")
    
    # test keywords from readme
    testability_indicators = ['test', 'tested', 'testing', 'pytest', 'unittest', 'unit test', 'ci', 'continuous integration']
    readme_to_check = code_readme if code_readme else model_readme
        
    if readme_to_check:
        test_mentions = sum(1 for indicator in testability_indicators if indicator in readme_to_check.lower())
        test_score = min(test_mentions / 5, 1)  # 3 keywords = full points
        logger.debug(f"Test mentions: {test_mentions}, Testability score: {test_score}")
    else:
        logger.debug("No readme available for testability check")
        test_score = 0.0
        
    code_quality_score = min(len_score * 0.4 + pop_score * 0.4 + test_score * 0.2, 1)
    logger.info(f"Final code quality score: {code_quality_score}")

    end = time.time()
    latency = end - start
    return code_quality_score, latency