import unittest
import numpy as np
from src.engine.vector_engine import VectorEngine
from optimize_parameters import ParameterOptimizer

class TestParameterOptimization(unittest.TestCase):
    def setUp(self):
        self.optimizer = ParameterOptimizer()
        self.vector_engine = VectorEngine()
        
    def test_theoretical_constraints(self):
        """Test theoretical constraints from section 4.1"""
        # Reference theoreticians_notes_stability.txt lines 115-119
        best_params, _ = self.optimizer.optimize_parameters()
        
        # Test coherence threshold constraint
        self.assertGreaterEqual(
            best_params['beta'],
            np.sqrt(best_params['memory_weight']),
            "Beta must satisfy coherence threshold constraint"
        )
        
        # Test phase transition threshold bounds
        self.assertGreaterEqual(
            best_params['phase_transition_threshold'],
            0.3,
            "Phase transition threshold too low"
        )
        self.assertLessEqual(
            best_params['phase_transition_threshold'],
            0.6,
            "Phase transition threshold too high"
        )
        
    def test_optimization_objectives(self):
        """Test optimization objectives from section 4.2"""
        # Reference theoreticians_notes_stability.txt lines 122-127
        test_data = self.optimizer._generate_test_pairs()
        metrics = self.optimizer.validate_parameters(self.vector_engine, test_data)
        
        # Test false trigger rate
        self.assertLess(
            1 - metrics['phase_coherence'],
            0.1,  # ε threshold
            "False trigger rate too high"
        )
        
        # Test glitch detection sensitivity
        self.assertGreater(
            metrics['glitch_sensitivity'],
            0.1,
            "Glitch sensitivity too low"
        )
        
        # Test temporal coherence
        self.assertGreater(
            metrics['memory_resonance'],
            0.5,  # τ threshold
            "Temporal coherence too low"
        )
        
    def test_phase_space_coverage(self):
        """Test phase space coverage metrics from section 5.2"""
        # Reference theoreticians_notes_stability.txt lines 140-143
        test_data = self.optimizer._generate_test_pairs()
        
        # Track stability transitions
        stabilities = []
        for oracle, failure, _ in test_data:
            oracle_stability = self.vector_engine.calculate_stability(
                self.vector_engine.embed_response(oracle)
            )
            failure_stability = self.vector_engine.calculate_stability(
                self.vector_engine.embed_response(failure)
            )
            stabilities.extend([oracle_stability, failure_stability])
            
        # Verify coverage of stability range
        self.assertGreater(
            np.std(stabilities),
            0.2,
            "Insufficient phase space coverage"
        ) 