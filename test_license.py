from metrics.license import get_license_score

max_deviation = 0.15

class Test_License: 
    def test_bert_base_uncased(self): 
        expected_license = 1.0  # Apache 2.0 license - compatible
        # Pass just the model ID string instead of the full API info
        model_id = "google-bert/bert-base-uncased"
        actual_license, actual_latency = get_license_score(model_id)
        print(f"Score: {actual_license}")
        print(f"Latency: {actual_latency}")
        assert actual_license <= (min(1, expected_license + max_deviation)) and actual_license >= (max(0, expected_license - max_deviation))
    
    def test_audience_classifier_model(self):
        expected_license = 0.0  # No clear license found - incompatible
        # Pass just the model ID string instead of the full API info
        model_id = "parvk11/audience_classifier_model"
        actual_license, actual_latency = get_license_score(model_id)
        print(f"Score: {actual_license}")
        print(f"Latency: {actual_latency}")
        assert actual_license <= (min(1, expected_license + max_deviation)) and actual_license >= (max(0, expected_license - max_deviation))    
    
    def test_whisper_tiny(self): 
        expected_license = 1.0  # Apache 2.0 license - compatible
        # Pass just the model ID string instead of the full API info
        model_id = "openai/whisper-tiny"
        actual_license, actual_latency = get_license_score(model_id)
        print(f"Score: {actual_license}")
        print(f"Latency: {actual_latency}")
        assert actual_license <= (min(1, expected_license + max_deviation)) and actual_license >= (max(0, expected_license - max_deviation))