# If the dataset used for training and benchmarking is well documented, along with any example code

"""
    Indicates if the dataset used for training and benchmarking
    is well documented, along with any example code.

    Parameters
    ----------
    dataset_card : type??
        Data obtained from HuggingFace (API? idk)

    Returns
    -------
    float
        Score in range 0 (bad) - 1 (good)
"""
def dataset_and_code_score(dataset_card):
    # Max score = 5
    # Each worth +1 score:
    # Dataset sources, uses, data collection and processing, bias, example code
    score = 0
    max_score = 5

    # Dataset sources 
    # Uses
    # Data collection and processing
    # Bias 
    # Code example 

    return (score / max_score)