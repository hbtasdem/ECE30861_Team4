
from metrics.data_quality import data_quality
from metrics.code_quality import code_quality
from metrics.dataset_and_code_score import dataset_and_code_score
from metrics.performance_claims import performance_claims
from metrics.size import size_calculator
from metrics.bus_factor import bus_factor
from metrics.ramp_up_time import ramp_up_time
import input

def main(model_info, model_readme, raw_model_url, code_info, code_readme, raw_dataset_url):
    
    # data_quality_score = data_quality(api_info, readme)
    data_quality_score, dq_latency = data_quality(model_info, model_readme)
    # code_quality_score = code_quality(type, api_info, readme)
    code_quality_score, cq_latency = code_quality(model_info, code_info, model_readme, code_readme)
    
    dc_score, dc_latency = dataset_and_code_score(code_info, raw_dataset_url) # what is code link
    
    perf_score, perf_latency = performance_claims(raw_model_url) #HILAL CHECK THE PASSED VALUE - user url not api
    size_score, size_latency = size_calculator(model_info) # make sure api info = model dir
    license_score, license_latency = 0 #FIX
    #license = 
    bus_score, bus_latency = bus_factor(model_info) # this use model info or other?
    ramp_score, ramp_score = ramp_up_time(model_info)  
    
    #add in size
    net_score = license * (0.25 + ramp_score + 0.15 * data_quality_score + 0.15 * bus_score + 
                           0.07 * dc_score + 0.12 * code_quality_score + 0.06 * perf_score) 
    
    return net_score
if __name__ == "__main__":
    main() 
    