
from metrics.data_quality import data_quality
from metrics.code_quality import code_quality
from metrics.dataset_and_code_score import dataset_and_code_score
from metrics.performance_claims import performance_claims
from metrics.size import size_calculator
from metrics.bus_factor import bus_factor
from metrics.ramp_up_time import ramp_up_time

def main():
    
    data_quality_score = data_quality(api_info, readme)
    code_quality_score = code_quality(type, api_info, readme)
    dc_score = dataset_and_code_score(code_link,dataset_readme) #HILAL FIX THIS IN MAIN
    perf_score = performance_claims(model_url) #HILAL CHECK THE PASSED VALUE - user url not api
    size_score = size_calculator(api_info) # make sure api info = model dir
    #license = 
    bus_score = bus_factor(api_info)
    ramp_score = ramp_up_time(api_info)
    
    
if __name__ == "__main__":
    main() 
    