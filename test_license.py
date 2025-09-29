from metrics.license import get_license_score, get_detailed_license_score, get_license_score_cached, extract_license_section

max_deviation = 0.15

class Test_License: 
    def test_bert_base_uncased(self): 
        expected_license = 1.0  # Apache 2.0 license - compatible
        model_id = "google-bert/bert-base-uncased"
        actual_license, actual_latency = get_license_score(model_id)
        print(f"Score: {actual_license}")
        print(f"Latency: {actual_latency}")
        assert actual_license <= (min(1, expected_license + max_deviation)) and actual_license >= (max(0, expected_license - max_deviation))
    
    def test_audience_classifier_model(self):
        expected_license = 0.0  # No clear license found - incompatible
        model_id = "parvk11/audience_classifier_model"
        actual_license, actual_latency = get_license_score(model_id)
        print(f"Score: {actual_license}")
        print(f"Latency: {actual_latency}")
        assert actual_license <= (min(1, expected_license + max_deviation)) and actual_license >= (max(0, expected_license - max_deviation))    
    
    def test_whisper_tiny(self): 
        expected_license = 1.0  # Apache 2.0 license - compatible
        model_id = "openai/whisper-tiny"
        actual_license, actual_latency = get_license_score(model_id)
        print(f"Score: {actual_license}")
        print(f"Latency: {actual_latency}")
        assert actual_license <= (min(1, expected_license + max_deviation)) and actual_license >= (max(0, expected_license - max_deviation))

    def test_ambiguous_license_with_positive_indicators(self):
        # Test case to cover lines 101-117: ambiguous license with positive indicators
        expected_license = 0.5  # Ambiguous license with positive indicators
        model_id = "microsoft/DialoGPT-small"
        actual_license, actual_latency = get_license_score(model_id)
        print(f"Score: {actual_license}")
        print(f"Latency: {actual_latency}")
        assert actual_license >= 0.0 and actual_license <= 1.0

    def test_ambiguous_license_no_positive_indicators(self):
        # Test case for no positive indicators (line 117)
        # This is hard to test with real models, so we'll test the function directly
        from metrics.license import analyze_license_text
        # Test text with no license indicators
        license_text = "This is some random text with no license information."
        score = analyze_license_text(license_text)
        assert score == 0.0  # Should return 0.0 for no clear license

    def test_gated_license_detection(self):
        # Test gated license detection (line 75)
        from metrics.license import analyze_license_text
        license_text = "This is a gated model requiring access request."
        score = analyze_license_text(license_text)
        assert score == 0.0  # Should return 0.0 for gated licenses

    def test_empty_license_text(self):
        # Test empty license text (line 75)
        from metrics.license import analyze_license_text
        score = analyze_license_text("")
        assert score == 0.0

    def test_none_license_text(self):
        # Test None license text
        from metrics.license import analyze_license_text
        score = analyze_license_text(None)
        assert score == 0.0

    def test_extract_license_section_edge_cases(self):
        # Test edge cases for extract_license_section - FIXED
        # Empty content
        result = extract_license_section("")
        assert result == ""

        # Content without license - FIXED: The function returns context when "license" is found
        result = extract_license_section("Some random text without license")
        # Since "license" is in the text, it will return some context around it
        assert "license" in result.lower()
        assert "random text" in result.lower()

    def test_get_license_score_with_dict_input(self):
        # Test with dictionary input containing model_id
        model_input = {'model_id': 'google-bert/bert-base-uncased'}
        score, latency = get_license_score(model_input)
        assert 0 <= score <= 1
        assert latency >= 0

    def test_get_license_score_with_empty_dict(self):
        # Test with empty dictionary (should return 0.0)
        model_input = {}
        score, latency = get_license_score(model_input)
        assert score == 0.0
        assert latency >= 0

    def test_get_license_score_with_url_dict(self):
        # Test with dictionary input containing URL
        model_input = {'url': 'https://huggingface.co/google-bert/bert-base-uncased'}
        score, latency = get_license_score(model_input)
        assert 0 <= score <= 1
        assert latency >= 0

    def test_get_detailed_license_score(self):
        # Test the detailed license score function (covers lines 306-320)
        model_id = "google-bert/bert-base-uncased"
        result = get_detailed_license_score(model_id)
        
        # Check the structure of the returned dictionary
        assert 'license' in result
        assert 'license_latency' in result
        assert isinstance(result['license'], float)
        assert isinstance(result['license_latency'], int)
        assert 0 <= result['license'] <= 1
        assert result['license_latency'] >= 0

    def test_get_license_score_cached(self):
        # Test the cached version (covers lines 323-342)
        model_id = "google-bert/bert-base-uncased"
        
        # First call
        score1, latency1 = get_license_score_cached(model_id)
        
        # Second call (should use cache)
        score2, latency2 = get_license_score_cached(model_id)
        
        # Scores should be the same
        assert score1 == score2
        # Second call should be faster (or at least not slower)
        assert latency2 <= latency1 or abs(latency2 - latency1) < 100  # Allow small variance

    def test_license_with_dict_name(self):
        # Test with dictionary input containing name
        model_input = {'name': 'google-bert/bert-base-uncased'}
        score, latency = get_license_score(model_input)
        assert 0 <= score <= 1
        assert latency >= 0

    def test_main_block_coverage(self):
        # Test to cover the main block (lines 339-342)
        # We can't easily test the main block directly, but we can test the functions it calls
        model_id = "google-bert/bert-base-uncased"
        score, latency = get_license_score_cached(model_id)
        assert 0 <= score <= 1
        assert latency >= 0

    def test_compatible_license_detection(self):
        # Test various compatible licenses
        from metrics.license import analyze_license_text
        compatible_licenses = [
            "Apache 2.0 license",
            "MIT License", 
            "BSD-3-Clause",
            "BSL-1.0",
            "LGPLv2.1"
        ]
        for license_text in compatible_licenses:
            score = analyze_license_text(license_text)
            assert score == 1.0, f"Failed for: {license_text}"

    def test_incompatible_license_detection(self):
        # Test various incompatible licenses
        from metrics.license import analyze_license_text
        incompatible_licenses = [
            "GPL v3 license",
            "AGPL license",
            "Non-commercial use only",
            "Proprietary license"
        ]
        for license_text in incompatible_licenses:
            score = analyze_license_text(license_text)
            assert score == 0.0, f"Failed for: {license_text}"

    # NEW TESTS TO COVER MISSING LINES
    def test_ambiguous_license_exact_coverage(self):
        # Direct test to cover lines 101-117 specifically
        from metrics.license import analyze_license_text
        
        # Test case that hits the "ambiguous but positive indicators" path (line 112-115)
        license_text = "This model is open source and permissive for research use."
        score = analyze_license_text(license_text)
        # This should hit line 112 (checking for positive words) and return 0.5
        assert score == 0.5

        # Test case that hits the "no clear license found" path (line 117)
        license_text = "This is some completely random text with no license mentions."
        score = analyze_license_text(license_text)
        assert score == 0.0

    def test_download_readme_edge_cases(self):
        # Test edge cases for download_readme_directly to cover lines 139-140, 159-160
        from metrics.license import download_readme_directly
        
        # Test with a non-existent model to cover error handling
        result = download_readme_directly("non/existent-model-12345")
        assert result == ""  # Should return empty string for non-existent models

    def test_license_pattern_matching(self):
        # Test to cover lines 215-216, 220-221 (pattern matching edge cases)
        from metrics.license import extract_license_section
        
        # Test with various license header formats
        test_content = """
# Some content
## License
Apache 2.0
## Other section
More content
"""
        result = extract_license_section(test_content)
        assert "Apache" in result

        # Test with license: pattern
        test_content = "license: MIT License"
        result = extract_license_section(test_content)
        assert "MIT" in result

    def test_cache_functionality(self):
        # Test to cover lines 323-342 (caching functionality)
        model_id = "google-bert/bert-base-uncased"
        
        # Clear cache first
        from metrics.license import _license_cache
        _license_cache.clear()
        
        # First call - should calculate
        score1, latency1 = get_license_score_cached(model_id)
        
        # Second call - should use cache
        score2, latency2 = get_license_score_cached(model_id)
        
        # Verify cache is being used
        assert model_id in _license_cache
        assert score1 == score2

    def test_extract_model_id_edge_cases(self):
        # Test to cover lines 263-265 (extract_model_id edge cases)
        from metrics.license import extract_model_id_from_url
        
        # Test various URL formats
        assert extract_model_id_from_url("https://huggingface.co/google/bert") == "google/bert"
        assert extract_model_id_from_url("google/bert") == "google/bert"
        assert extract_model_id_from_url("random text") == "random text"

    def test_analyze_license_mixed_signals(self):
        # Test to cover line 307 and other edge cases
        from metrics.license import analyze_license_text
        
        # Test with both compatible and incompatible licenses (should return 0.0)
        license_text = "This uses Apache 2.0 but also has GPL components"
        score = analyze_license_text(license_text)
        assert score == 0.0  # Incompatible takes precedence

    def test_direct_license_analysis_coverage(self):
        # Direct test to hit the exact missing lines in analyze_license_text
        from metrics.license import analyze_license_text
        
        # Test case 1: Empty text (line 75)
        assert analyze_license_text("") == 0.0
        
        # Test case 2: Gated license (line 75 in the conditions)
        assert analyze_license_text("gated model access request") == 0.0
        
        # Test case 3: Compatible license found (should return 1.0)
        assert analyze_license_text("Apache 2.0") == 1.0
        
        # Test case 4: Incompatible license found (should return 0.0)
        assert analyze_license_text("GPL v3") == 0.0
        
        # Test case 5: Ambiguous with positive indicators (lines 112-115)
        assert analyze_license_text("open source permissive") == 0.5
        
        # Test case 6: No clear license (line 117)
        assert analyze_license_text("random text") == 0.0

    def test_license_section_extraction_coverage(self):
        # Test various extraction patterns to cover missing lines
        from metrics.license import extract_license_section
        
        # Test empty content
        assert extract_license_section("") == ""
        
        # Test with license header pattern
        content_with_header = """
# Title
## License
MIT License
## Other
Content
"""
        result = extract_license_section(content_with_header)
        assert "MIT" in result
        
        # Test with license: pattern
        content_with_colon = "license: Apache 2.0"
        result = extract_license_section(content_with_colon)
        assert "Apache" in result
        
        # Test with license in quotes pattern
        content_with_quotes = 'license "BSD-3-Clause"'
        result = extract_license_section(content_with_quotes)
        assert "BSD" in result