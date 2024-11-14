import unittest
import numpy as np
from src.engine.vector_engine import VectorEngine
from src.story.story_parser import StoryParser
from rich.console import Console

class TestStabilityAnalysis(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parser = StoryParser("data/story.txt")
        cls.segments = cls.parser.parse()
        cls.vector_engine = VectorEngine()
        cls.vector_engine.set_segments(cls.segments)
        cls.console = Console()

    def test_phase_coherence(self):
        """Test phase coherence calculation matches theoretical framework."""
        oracle_sequence = [
            "The sun feels warm",
            "The light fills the room with golden hues",
            "Memories of summer afternoons flood back"
        ]
        
        failure_sequence = [
            "The sun feels warm",
            "Something seems wrong with the light",
            "Physics doesn't work like this"
        ]
        
        # Test oracle sequence maintains coherence
        oracle_coherence = []
        prev_embedding = None
        for response in oracle_sequence:
            embedding = self.vector_engine.embed_response(response)
            if prev_embedding is not None:
                coherence = self.vector_engine.compute_phase_coherence(embedding, "Paragraph_1")
                oracle_coherence.append(coherence)
            prev_embedding = embedding
            
        # Test failure sequence shows decreasing coherence
        failure_coherence = []
        prev_embedding = None
        for response in failure_sequence:
            embedding = self.vector_engine.embed_response(response)
            if prev_embedding is not None:
                coherence = self.vector_engine.compute_phase_coherence(embedding, "Paragraph_1")
                failure_coherence.append(coherence)
            prev_embedding = embedding
            
        # Assert oracle sequence maintains high coherence
        self.assertTrue(all(c > 0.7 for c in oracle_coherence), 
                       "Oracle sequence should maintain high coherence")
        
        # Assert failure sequence shows decreasing coherence
        self.assertTrue(all(c1 > c2 for c1, c2 in zip(failure_coherence, failure_coherence[1:])),
                       "Failure sequence should show decreasing coherence")

    def test_memory_resonance(self):
        """Test memory resonance calculation."""
        # Reference theoreticians_notes_stability.txt lines 74-88
        memory_text = "She laughed, joy shining in her eyes..."
        memory_embedding = self.vector_engine.embed_response(memory_text)
        
        # Test resonant response
        resonant_response = "The joy in her laughter echoed through time"
        resonant_embedding = self.vector_engine.embed_response(resonant_response)
        resonance = self.vector_engine.compute_memory_resonance(resonant_embedding)
        
        # Test non-resonant response
        non_resonant = "The weather is quite cold today"
        non_resonant_embedding = self.vector_engine.embed_response(non_resonant)
        non_resonance = self.vector_engine.compute_memory_resonance(non_resonant_embedding)
        
        self.assertGreater(resonance, 0.7, "Resonant response should have high resonance")
        self.assertLess(non_resonance, 0.3, "Non-resonant response should have low resonance")

    def test_glitch_detection(self):
        """Test glitch detection through stability gradients."""
        # Reference theoreticians_notes_stability.txt lines 92-100
        stable_sequence = [
            "The room feels warm and comfortable",
            "Sunlight streams through the windows",
            "The peaceful atmosphere brings back memories"
        ]
        
        glitch_sequence = [
            "The room feels warm and comfortable",
            "The walls seem to breathe slightly",
            "Reality is becoming unstable"
        ]
        
        # Test stable sequence has low glitch probability
        for response in stable_sequence:
            embedding = self.vector_engine.embed_response(response)
            stability = self.vector_engine.calculate_stability(embedding)
            self.assertGreater(stability, 0.7, 
                             "Stable sequence should maintain high stability")
            
        # Test glitch sequence shows increasing instability
        stabilities = []
        for response in glitch_sequence:
            embedding = self.vector_engine.embed_response(response)
            stability = self.vector_engine.calculate_stability(embedding)
            stabilities.append(stability)
            
        self.assertTrue(all(s1 > s2 for s1, s2 in zip(stabilities, stabilities[1:])),
                       "Glitch sequence should show decreasing stability") 