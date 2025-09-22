from parse_categories import masterScoring
from datetime import datetime, timezone

def calculate_active_maintenance_score(api_info) -> float: 
    
    created_at = api_info.get("createdAt")
    last_modified = api_info.get("lastModified")

    if not created_at or not last_modified:
        return 0
    
    created_date = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    modified_date = datetime.fromisoformat(last_modified.replace("Z", "+00:00"))
    now = datetime.now(timezone.utc)

    age_in_days = (now - created_date).days
    days_since_modified = (now - modified_date).days

    if age_in_days < 180:
        age_score = 1
    elif age_in_days < 365:
        age_score = 0.8
    elif age_in_days < 730:
        age_score = 0.6
    else:
        age_score = 0.4


    if days_since_modified < 30:
        update_score = 1
    elif days_since_modified < 90:
        update_score = 0.8
    elif days_since_modified < 180:
        update_score = 0.6
    elif days_since_modified < 365:
        update_score = 0.4
    else:
        update_score = 0.2

    
    return update_score * 0.75 + age_score * 0.25

def calculate_contributor_diversity_score(api_info) -> float:

    num_contributors = len(api_info.get('spaces', []))

    if num_contributors == 0 or num_contributors == 1: 
        score = 0
    elif num_contributors == 2 or num_contributors == 3: 
        score = 0.2
    elif num_contributors == 4 or num_contributors == 5: 
        score = 0.4
    elif num_contributors == 6 or num_contributors == 7: 
        score = 0.6
    elif num_contributors == 8 or num_contributors == 9: 
        score = 0.8
    else: 
        score = 1
    
    return score


def calculate_org_backing_score(api_info) -> float:
    KNOWN_ORGS = {"google", "meta", "microsoft", "openai", "apple", "ibm", "huggingface"}

    author = api_info.get("author", "").lower()
    if author in KNOWN_ORGS: 
        return 1
    else: 
        return 0

def bus_factor(api_info) -> float: 
    """
    Calculate bus factor (higher score is better).

    Parameters
    ----------
    contributor_diversity_score : float
        Contributor diversity score (0-1)
    active_maintenance_score : float
        Active maintenance score (0-1)
    org_backing_score : float
        Org backing score (0-1)

    Returns
    -------
    float
        Bus Factor metric score (0-1)
    """

    contributor_diversity_score = calculate_contributor_diversity_score(api_info)
    active_maintenance_score = calculate_active_maintenance_score(api_info)
    org_backing_score = calculate_org_backing_score(api_info)

    bus_factor_metric_score = (0.55 * contributor_diversity_score + 
             0.35 * active_maintenance_score + 
             0.1 * org_backing_score)
    
    return bus_factor_metric_score


# def main(): 
#     api_info = masterScoring('https://huggingface.co/google/gemma-3-270m')

#     bus_factor_score = calculate_bus_factor(api_info)
#     print(str(bus_factor_score))


# if __name__ == "__main__":
#     main()