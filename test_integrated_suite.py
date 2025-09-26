"""
Integrated test suite for the complete model evaluation system.
"""
import pytest
import json
from size import calculate_size_score
from license import get_license_score
from netscore import calculate_net_score

def test_complete_model_evaluation():
    """
    Test complete evaluation pipeline for a known model.
    """
    model_id = "google-bert/bert-base-uncased"
    
    # Calculate all metrics
    size_result = calculate_size_score(model_id)
    license_result = get_license_score(model_id)
    
    # Create mock other metrics (you'll integrate these later)
    mock_metrics = {
        'ramp_up_time': 0.9,
        'bus_factor': 0.95,
        'performance_claims': 0.92,
        'dataset_and_code_score': 1.0,
        'dataset_quality': 0.95,
        'code_quality': 0.93
    }
    
    # Combine all metrics
    all_metrics = {
        'license': license_result['license'],
        'size_score': size_result['size_score'],
        **mock_metrics
    }
    
    # Calculate NetScore
    net_result = calculate_net_score(all_metrics)
    
    # Verify output structure matches requirements
    expected_structure = {
        'name': model_id.split('/')[-1],
        'category': 'MODEL',
        'net_score': float,
        'net_score_latency': int,
        'ramp_up_time': float,
        'ramp_up_time_latency': int,
        'bus_factor': float,
        'bus_factor_latency': int,
        'performance_claims': float,
        'performance_claims_latency': int,
        'license': float,
        'license_latency': int,
        'size_score': dict,
        'size_score_latency': int,
        'dataset_and_code_score': float,
        'dataset_and_code_score_latency': int,
        'dataset_quality': float,
        'dataset_quality_latency': int,
        'code_quality': float,
        'code_quality_latency': int
    }
    
    # Test that we can create the required JSON output
    output_data = {
        'name': model_id.split('/')[-1],
        'category': 'MODEL',
        'net_score': net_result['net_score'],
        'net_score_latency': net_result['net_score_latency'],
        'ramp_up_time': mock_metrics['ramp_up_time'],
        'ramp_up_time_latency': 45,  # Mock latency
        'bus_factor': mock_metrics['bus_factor'],
        'bus_factor_latency': 25,    # Mock latency
        'performance_claims': mock_metrics['performance_claims'],
        'performance_claims_latency': 35,  # Mock latency
        'license': license_result['license'],
        'license_latency': license_result['license_latency'],
        'size_score': size_result['size_score'],
        'size_score_latency': size_result['size_score_latency'],
        'dataset_and_code_score': mock_metrics['dataset_and_code_score'],
        'dataset_and_code_score_latency': 15,  # Mock latency
        'dataset_quality': mock_metrics['dataset_quality'],
        'dataset_quality_latency': 20,  # Mock latency
        'code_quality': mock_metrics['code_quality'],
        'code_quality_latency': 22   # Mock latency
    }
    
    # Verify all required fields are present
    for key, expected_type in expected_structure.items():
        assert key in output_data
        assert isinstance(output_data[key], expected_type)
    
    # Test JSON serialization (required for stdout output)
    json_output = json.dumps(output_data)
    assert isinstance(json_output, str)
    assert len(json_output) > 0

def test_multiple_models():
    """
    Test evaluation of multiple models to ensure consistency.
    """
    test_models = [
        "google-bert/bert-base-uncased",
        "facebook/bart-base"
    ]
    
    for model_id in test_models:
        size_result = calculate_size_score(model_id)
        license_result = get_license_score(model_id)
        
        # Basic validation
        assert 0.0 <= license_result['license'] <= 1.0
        assert all(0.0 <= score <= 1.0 for score in size_result['size_score'].values())
        assert size_result['size_score_latency'] >= 0
        assert license_result['license_latency'] >= 0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])