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
# def code_quality_calc(type: str, api_info: str, readme: str) -> int:
def code_quality_calc(model_info: str, code_info: str, model_readme: str, code_readme: str):

    import requests as rq
    import time
    
    start = time.time()

    # init metrics used in final score calculation to avoid bugs
    len_score = 0.0
    pop_score = 0.0
    
    # reusability check
    doc_length = len(model_readme.split()) # documentation lenght in words
    if doc_length > 1000:
        len_score = 1.0
    elif doc_length > 500:
        len_score = 0.7
    elif doc_length > 200:
        len_score = 0.4
    else:
        len_score = 0.1
        
    print("len:", len_score) # debug
    
    # reliability check based on type
    if type == 'MODEL':
        
        # downloads check from card_data (for models)
        popularity = model_info.get('downloads', 0) 
        if popularity > 700000:
            pop_score = 1.0
        elif popularity > 500000:
            pop_score = 0.7
        elif popularity > 100000:
            pop_score = 0.3
        else:
            pop_score = 0.1
            
        print("pop:", pop_score) 
    
    elif type == 'CODE': 
        stars = code_info.get('stargazers_count', 0)
        forks = code_info.get('forks_count', 0)
        
        # stars + forks check of github repo (for code)
        if stars + forks > 90000:
            pop_score = 1.0
        elif stars + forks > 60000:
            pop_score = 0.7
        elif stars + forks > 30000:
            pop_score = 0.5
        else:
            pop_score = 0.1
        
        print("pop:", pop_score) 
    
    # test keywords from readme
    testability_indicators = ['test', 'tested', 'testing', 'pytest', 'unittest', 'unit test', 'ci', 'continuous integration']
    test_mentions = sum(1 for indicator in testability_indicators if indicator in code_readme)
    test_score = min(test_mentions / 3, 1)  # 3 keywords = full points
    
    code_quality_score = min(len_score * 0.4 + pop_score * 0.4 + test_score * 0.2, 1)
    print("code:", code_quality_score)
    end = time.time()
    
    latency = end - start
    print(latency)
    return code_quality_score, latency


# def code_quality(type: str, api_info: str, readme: str) -> int:
#     code_quality_score = code_quality_calc(type, api_info, readme)
    
#     return code_quality_score

def code_quality(model_info: str, code_info: str, model_readme: str, code_readme: str) -> int:
    
    code_quality_score = code_quality_calc(model_info, code_info, model_readme, code_readme)
    
    return code_quality_score