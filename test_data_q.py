
# import parse_categories
import data_quality
import code_quality
from datetime import datetime, timedelta

"""
Usage Instructions: 

Dependencies: 
Install pytest in terminal: pip install pytest 

Commands: 
Run a certain test (ex: just test_add_1):           pytest test_example.py::Test_Add::test_add_1 -v
Run a certain class of tests (ex: just Test_Add):   pytest test_example.py::Test_Add -v
Run all the tests in the file:                      pytest test_example.py -v

Test Suite for Data Quality and Code Quality Functions

Disclaimer: 
Test cases were developed with Claude Sonnet 4 assistance due to the complexity of manually 
constructing realistic test data and calculating accurate scoring thresholds. Since our code 
structure doesn't support passing in real URLs in a test suite, AI assistance was used to:
    1. Generate realistic API response structures mirroring HF and GitHub data
    2. Calculate expected scores based on weighted metrics in our quality functions  
"""

class Test_Model: # Model tests
    def test_model_good(self):  # Good data quality case
        # assert parse_categories.masterScoring('https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english', 'MODEL') == 
        api_info = {'cardData': {'license': 'MIT', 'tags': ['test']}}
        readme = "description citation language source"  # Just the keywords we're testing for
        type_val = "MODEL"
        api_info = {
        'cardData': {
            'license': 'MIT',
            'description': 'Clear model description',
            'tags': ['nlp', 'classification'], 
            'citation': 'Please cite our paper',
            'source': 'Original research data',
            'language': ['en', 'es'],
            'use': 'Text classification tasks',
            'limitation': 'May not work on domain-specific text'
        },
        'createdAt': (datetime.now() - timedelta(days=30)).isoformat() + 'Z'  # 30 days old = recent
        }
    
        # Perfect README with all keywords
        readme = """
        This model demonstrates diverse and varied approaches to text classification.
        The training data represents a comprehensive range of different text types.
        
        Our dataset is well-balanced and evenly distributed across multiple categories.
        The representative sample captures extensive coverage of the target domain.
        
        type: accuracy
        value: 0.95
        
        Additional keywords: heterogeneous, broad, spanning, proportional coverage
        with citation information, source attribution, and language specifications.
        Uses include various limitation scenarios with proper description.
        """
        
        result = data_quality.data_quality(api_info, readme)
        
        print(f"Data Quality Score: {result}")
        
        assert result >= 0.98
    
    
    def test_dataset_poor(self):  # Poor data quality case
   
        api_info = {
            'cardData': {},
            'createdAt': (datetime.now() - timedelta(days=800)).isoformat() + 'Z'
        }
        
        readme = "Model for stuff. It works okay I guess. No specific details provided."
        
        assert data_quality.data_quality(api_info, readme) <= 0.2

class Test_Dataset: # Dataset input tests
    def test_dataset_good(self):  # Good data quality case
        
        api_info = {
            'cardData': {
                'license': 'MIT',
                'description': 'Comprehensive dataset description',
                'tags': ['dataset', 'nlp'],
                'citation': 'Please cite our dataset',
                'source': 'Academic research',
                'language': ['en'],
                'use': 'Text classification',
                'limitation': 'Limited to English text'
            },
            'createdAt': (datetime.now() - timedelta(days=60)).isoformat() + 'Z'
        }
        
        readme = """
        This dataset contains diverse and representative samples from various sources.
        The data is well-balanced across different categories and comprehensive in scope.
        type: accuracy
        value: 0.92
        """
        
        assert data_quality.data_quality(api_info, readme) >= 0.8


    def test_dataset_bad(self):  # Poor data quality case
        api_info = {
            'cardData': {},
            'createdAt': (datetime.now() - timedelta(days=900)).isoformat() + 'Z'
        }
            
        readme = "Old dataset with minimal info."
            
        assert data_quality.data_quality(api_info, readme) <= 0.3
        
class Test_Code: # Github repo tests 
# NOTES: do we need to have a test suite for dataset quality given a code URL input?
    
    def test_code_good(self):  # Good code quality case
        api_info = {'stargazers_count': 50000, 'forks_count': 15000}  # 65k total > 60k threshold
        readme = """
        This is a comprehensive library for machine learning tasks.
        Installation instructions are provided below with detailed examples.
        The code is thoroughly tested using pytest and unittest frameworks.
        Continuous integration ensures testing reliability across versions.
        Complete documentation with usage examples and API reference.
        """ * 30  # Make it long (>1000 words for full reusability score)
        
        assert code_quality.code_quality("CODE", api_info, readme) >= 0.8

    def test_code_bad(self):  # Poor code quality case
        api_info = {'stargazers_count': 5, 'forks_count': 1} 
        readme = "Basic repo"  
        
        assert code_quality.code_quality("CODE", api_info, readme) <= 0.2
        
