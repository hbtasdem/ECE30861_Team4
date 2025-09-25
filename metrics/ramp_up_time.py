from parse_categories import masterScoring

def calculate_api_complexity_score(api_info) -> float: 
    #having a pipeline task for a specific purpose makes it easier 
    
    tags = api_info.get("tags", [])
    pipeline_tag = api_info.get("pipeline_tag", None)

    score = 0.5  

    if pipeline_tag in {"text-classification", "translation", "summarization", "fill-mask"}:
        score = 0.9
    elif pipeline_tag in {"token-classification", "question-answering"}:
        score = 0.8
    elif pipeline_tag in {"text-generation", "image-classification"}:
        score = 0.7
    elif pipeline_tag is None:
        score = 0.4

    if "multimodal" in tags or "large" in tags or "gpt" in tags:
        score -= 0.2

    return score
    

def calculate_documentation_score(api_info) -> float: 
    """
    Calculate documentation score based on if it has a README.md, summary, or usage information

    Parameters
    ----------
    api_info : dict
        Dictionary with information about the API

    Returns
    -------
    float
        Community support availability score (0-1)
    """

    card_data = api_info.get("cardData", {})
    siblings = [s.get("rfilename", "").lower() for s in api_info.get("siblings", [])]

    has_readme = any("readme.md" in f for f in siblings)
    has_summary = bool(card_data.get("summary"))
    has_usage = "usage" in str(card_data).lower()

    if has_readme or has_summary or has_usage: 
        score = 1
    else: 
        score = 0

    return score

def calculate_community_support_score(api_info) -> float: 
    """
    Calculate community support score based on number of likes and downloads

    Parameters
    ----------
    api_info : dict
        Dictionary with information about the API

    Returns
    -------
    float
        Community support availability score (0-1)
    """

    likes = api_info.get("likes", 0)
    downloads = api_info.get("downloads", 0)

    score = 0

    # Likes weight
    if likes > 1000:
        score += 0.5
    elif likes > 100:
        score += 0.4
    elif likes > 10:
        score += 0.2
    elif likes > 0:
        score += 0.1

    # Downloads weight
    if downloads > 1000000:
        score += 0.5
    elif downloads > 100000:
        score += 0.4
    elif downloads > 10000:
        score += 0.2
    elif downloads > 1000:
        score += 0.1

    return score


def calculate_quick_start_availability_score(api_info) -> float: 
    """
    Calculate quick start availability score based on availability of a quickstart guide, example script, or notebook

    Parameters
    ----------
    api_info : dict
        Dictionary with information about the API

    Returns
    -------
    float
        Quick start availability score (0-1)
    """

    siblings = [s.get("rfilename", "").lower() for s in api_info.get("siblings", [])]
    card_data = str(api_info.get("cardData", {})).lower()

    has_notebook = any(f.endswith(".ipynb") for f in siblings)
    has_example_script = any("example" in f for f in siblings)
    has_quickstart = "quickstart" in card_data or "usage" in card_data

    score = 0
    if has_quickstart:
        score += 0.5
    if has_notebook:
        score += 0.3
    if has_example_script:
        score += 0.2

    return score

def ramp_up_time(api_info : dict) -> float: 
    """
    Calculate ramp-up time score (higher score means faster to ramp up).

    Parameters
    ----------
    api_info : dict
        Dictionary with information about the API

    Returns
    -------
    float
        Ramp-up time metric score (0-1)
    """

    api_complexity_score = calculate_api_complexity_score(api_info)
    documentation_score = calculate_documentation_score(api_info)
    community_support_score = calculate_community_support_score(api_info)
    quick_start_availability_score = calculate_quick_start_availability_score(api_info)

    ramp_up_time_metric_score = (0.35 * api_complexity_score + 
             0.35 * documentation_score + 
             0.25 * community_support_score + 
             0.05 * quick_start_availability_score)
    
    return ramp_up_time_metric_score


# def main(): 
#     api_info = masterScoring('https://huggingface.co/google/gemma-3-270m')

#     ramp_up_time_score = calculate_ramp_up_time(api_info)

#     # print(api_info.get("cardData"))
#     # print(api_info.get("siblings"))

#     print("Ramp up Time Score")
#     print(str(ramp_up_time_score))


# if __name__ == "__main__":
#     main()