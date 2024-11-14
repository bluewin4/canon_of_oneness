import unittest
import numpy as np
from src.engine.vector_engine import VectorEngine
from src.story.story_parser import StoryParser
from rich.console import Console
from rich.table import Table

class TestPhaseSpace(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parser = StoryParser("data/story.txt")
        cls.segments = cls.parser.parse()
        cls.vector_engine = VectorEngine()
        cls.vector_engine.set_segments(cls.segments)
        cls.console = Console()

    def test_phase_space_coverage(self):
        """Test phase space coverage metrics from section 5.2"""
        table = Table(title="Phase Space Coverage Analysis")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right")
        table.add_column("Status", style="bold")
        
        # Test sequence from theoretical notes
        sequence = [
            "The sun feels warm",
            "The light seems wrong",
            "Why can't I see the source?",
            "The physics here make no sense"
        ]
        
        stabilities = []
        coherence_scores = []
        resonance_scores = []
        
        for response in sequence:
            embedding = self.vector_engine.embed_response(response)
            stability = self.vector_engine.calculate_stability(embedding)
            coherence = self.vector_engine.compute_phase_coherence(embedding, "Paragraph_1")
            resonance = self.vector_engine.compute_memory_resonance(embedding)
            
            stabilities.append(stability)
            coherence_scores.append(coherence)
            resonance_scores.append(resonance)
            
        # Calculate metrics from section 5.1
        stability_distribution = np.std(stabilities)
        avg_coherence = np.mean(coherence_scores)
        avg_resonance = np.mean(resonance_scores)
        
        # Add results to table
        table.add_row(
            "Stability Distribution",
            f"{stability_distribution:.3f}",
            "✅ Pass" if stability_distribution > 0.2 else "❌ Fail"
        )
        table.add_row(
            "Phase Coherence",
            f"{avg_coherence:.3f}",
            "✅ Pass" if avg_coherence > 0.5 else "❌ Fail"
        )
        table.add_row(
            "Memory Resonance",
            f"{avg_resonance:.3f}",
            "✅ Pass" if avg_resonance > 0.3 else "❌ Fail"
        )
        
        self.console.print(table)
        
        # Validate metrics against theoretical constraints
        self.assertGreater(stability_distribution, 0.2, "Insufficient phase space coverage") 