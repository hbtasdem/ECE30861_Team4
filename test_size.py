from metrics.size import calculate_size_score

max_deviation = 0.15

class Test_Size: 
    def test_bert_base_uncased(self): 
        # Expected NET score (weighted average), not just raspberry_pi score
        # raspberry_pi: 0.2 * 0.35 = 0.07
        # jetson_nano: 0.6 * 0.25 = 0.15  
        # desktop_pc: 0.9 * 0.20 = 0.18
        # aws_server: 1.0 * 0.20 = 0.20
        # Total net score: 0.07 + 0.15 + 0.18 + 0.20 = 0.60
        expected_size = 0.60
        model_id = "google-bert/bert-base-uncased"
        size_scores, actual_size, actual_latency = calculate_size_score(model_id)
        print(f"Net Score: {actual_size}")
        print(f"Detailed Scores: {size_scores}")
        print(f"Latency: {actual_latency}")
        assert actual_size <= (min(1, expected_size + max_deviation)) and actual_size >= (max(0, expected_size - max_deviation))
    
    def test_audience_classifier_model(self):
        # Expected NET score (weighted average)
        # raspberry_pi: 0.75 * 0.35 = 0.2625
        # jetson_nano: 0.875 * 0.25 = 0.21875
        # desktop_pc: 0.96875 * 0.20 = 0.19375
        # aws_server: 1.0 * 0.20 = 0.20
        # Total net score: ~0.875
        expected_size = 0.875
        model_id = "parvk11/audience_classifier_model"
        size_scores, actual_size, actual_latency = calculate_size_score(model_id)
        print(f"Net Score: {actual_size}")
        print(f"Detailed Scores: {size_scores}")
        print(f"Latency: {actual_latency}")
        assert actual_size <= (min(1, expected_size + max_deviation)) and actual_size >= (max(0, expected_size - max_deviation))    
    
    def test_whisper_tiny(self): 
        # Expected NET score (weighted average)
        # raspberry_pi: 0.9 * 0.35 = 0.315
        # jetson_nano: 0.95 * 0.25 = 0.2375
        # desktop_pc: 0.9875 * 0.20 = 0.1975
        # aws_server: 1.0 * 0.20 = 0.20
        # Total net score: ~0.95
        expected_size = 0.95
        model_id = "openai/whisper-tiny"
        size_scores, actual_size, actual_latency = calculate_size_score(model_id)
        print(f"Net Score: {actual_size}")
        print(f"Detailed Scores: {size_scores}")
        print(f"Latency: {actual_latency}")
        assert actual_size <= (min(1, expected_size + max_deviation)) and actual_size >= (max(0, expected_size - max_deviation))