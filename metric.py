
from metrics.data_quality import data_quality
from metrics.code_quality import code_quality
from metrics.dataset_and_code_score import dataset_and_code_score
from metrics.performance_claims import performance_claims
from metrics.size import calculate_size_score
from metrics.bus_factor import bus_factor
from metrics.ramp_up_time import ramp_up_time
from metrics.license import get_license_score
from print_metrics import print_model_evaluation
import logger

import time

def main(model_info, model_readme, raw_model_url, code_info, code_readme, raw_dataset_url):
    start = time.time()
    logger.info("Begin processing metrics.")
    
    data_quality_score, dq_latency = data_quality(model_info, model_readme)
    code_quality_score, cq_latency = code_quality(model_info, code_info, model_readme, code_readme)
    
    dc_score, dc_latency = dataset_and_code_score(code_info, raw_dataset_url)
    
    perf_score, perf_latency = performance_claims(raw_model_url)
    size_score, size_latency = calculate_size_score(raw_model_url)
    license_score, license_latency = get_license_score(raw_model_url)
    bus_score, bus_latency = bus_factor(model_info)
    ramp_score, ramp_latency = ramp_up_time(model_info)  
    
    #add in size 
    net_score = 0.25 * ramp_score + 0.15 * data_quality_score + 0.15 * bus_score + 0.07 * dc_score + 0.12 * code_quality_score + 0.06 * perf_score

    end = time.time()
    net_latency = end - start

    print_model_evaluation(
        model_info, 
        size_score, size_latency, 
        license_score, license_latency,
        ramp_score, ramp_latency, 
        bus_score, bus_latency, 
        dc_score, dc_latency, 
        data_quality_score, dq_latency, 
        code_quality_score, cq_latency, 
        perf_score, perf_latency, 
        net_score,net_latency
        )
    
if __name__ == "__main__":
    main() 
    