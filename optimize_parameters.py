import numpy as np
from typing import Dict, List, Tuple
from src.engine.vector_engine import VectorEngine
import itertools
import logging

logger = logging.getLogger(__name__)

class ParameterOptimizer:
    def __init__(self):
        # Core threshold parameters
        self.threshold_ranges = {
            'oracle_threshold': np.arange(0.4, 0.7, 0.05),
            'failure_threshold': np.arange(0.2, 0.4, 0.05),
            'distance_multiplier': np.arange(0.8, 1.2, 0.1)
        }
        
        # Phase space parameters
        self.phase_ranges = {
            'beta': np.linspace(0.5, 2.0, 4),
            'memory_weight': np.linspace(0.1, 0.5, 5),
            'temporal_decay': np.linspace(0.05, 0.2, 4),
            'resonance_bandwidth': np.linspace(0.1, 0.3, 3),
            'phase_transition_threshold': np.linspace(0.3, 0.6, 4),
            'coherence_threshold': np.linspace(0.6, 0.8, 3)
        }
        self.vector_engine = VectorEngine()

    def _generate_test_pairs(self) -> List[Tuple[str, str, float]]:
        """Generate test pairs of oracle/failure responses."""
        # Example test pairs
        test_pairs = [
            ("mama", "fire hot burn me now", 0.8),  # Basic test case
            ("The sun feels warm", "The physics here make no sense", 0.6),  # Phase transition
            ("I understand now", "Error cascade detected", 0.7),  # Stability test
        ]
        return test_pairs

    def _generate_param_combinations(self, param_ranges: Dict[str, np.ndarray]) -> List[Dict[str, float]]:
        """Generate all possible parameter combinations."""
        # Get all parameter names and their possible values
        param_names = list(param_ranges.keys())
        param_values = list(param_ranges.values())
        
        # Generate all combinations
        combinations = list(itertools.product(*param_values))
        
        # Convert to list of dictionaries
        param_dicts = []
        for combo in combinations:
            param_dict = {name: float(value) for name, value in zip(param_names, combo)}
            param_dicts.append(param_dict)
            
        return param_dicts

    def evaluate_parameters(self, vector_engine: VectorEngine, 
                          test_data: List[Tuple[str, str, float]]) -> float:
        """Evaluate parameter set using theoretical framework metrics."""
        scores = []
        for oracle, failure, expected_stability in test_data:
            # Use cached embeddings
            oracle_embedding = vector_engine.embed_response(oracle)
            oracle_stability = vector_engine.calculate_stability(oracle_embedding)
            oracle_score = oracle_stability if oracle_stability > 0.7 else 0
            
            # Use cached embeddings
            failure_embedding = vector_engine.embed_response(failure)
            failure_stability = vector_engine.calculate_stability(failure_embedding)
            failure_score = (1 - failure_stability) if failure_stability < 0.3 else 0
            
            scores.append((oracle_score + failure_score) / 2)
            
        return np.mean(scores)

    def optimize_parameters(self) -> Tuple[Dict[str, float], float]:
        """Grid search optimization using both threshold and phase space parameters."""
        best_params = {}
        best_score = float('-inf')
        test_data = self._generate_test_pairs()
        
        # Combine all parameter ranges
        all_params = {**self.threshold_ranges, **self.phase_ranges}
        
        # Generate parameter combinations
        param_combinations = self._generate_param_combinations(all_params)
        total_combinations = len(param_combinations)
        
        for i, params in enumerate(param_combinations):
            self.vector_engine.set_parameters(**params)
            score = self.evaluate_parameters(self.vector_engine, test_data)
            
            if score > best_score:
                best_score = score
                best_params = params
                logger.info(f"New best parameters found: {best_params}")
                logger.info(f"Score: {best_score}")
            
            if i % 10 == 0:
                logger.info(f"Progress: {i}/{total_combinations} combinations tested")
        
        # Save final best parameters
        if best_params:
            self.vector_engine.set_parameters(**best_params)
            self.vector_engine.save_parameters()
            
            logger.info("\nBest Parameters Found and Saved:")
            for param, value in best_params.items():
                logger.info(f"{param}: {value:.3f}")
            logger.info(f"Score: {best_score}")
        
        return best_params, best_score

if __name__ == '__main__':
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Run optimization
    optimizer = ParameterOptimizer()
    best_params, best_score = optimizer.optimize_parameters()