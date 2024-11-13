import sys
from pathlib import Path
from tests.test_story_validation import ParameterOptimizer

if __name__ == '__main__':
    optimizer = ParameterOptimizer()
    optimizer.setUpClass()
    best_params, best_score = optimizer.optimize_parameters() 