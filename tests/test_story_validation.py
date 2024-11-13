import unittest
from src.engine.vector_engine import VectorEngine
from src.story.story_parser import StoryParser
import numpy as np
from rich.console import Console
from rich.table import Table

class TestStoryValidation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parser = StoryParser("data/story.txt")
        cls.segments = cls.parser.parse()
        cls.vector_engine = VectorEngine()
        cls.vector_engine.set_segments(cls.segments)
        cls.console = Console()
        
    def test_oracle_failure_pairs(self):
        # Create results table
        table = Table(title="Oracle/Failure Pair Analysis")
        table.add_column("Section", style="cyan")
        table.add_column("Oracle Stability", justify="right")
        table.add_column("Failure Stability", justify="right")
        table.add_column("Status", style="bold")
        
        # Track overall results
        failures = []
        pairs = self._get_oracle_failure_pairs()
        
        for section_id, (oracle, failure) in pairs.items():
            oracle_embedding = self.vector_engine.embed_response(oracle.content)
            oracle_stability = self.vector_engine.calculate_stability(oracle_embedding)
            
            failure_embedding = self.vector_engine.embed_response(failure.content)
            failure_stability = self.vector_engine.calculate_stability(failure_embedding)
            
            # Validate and collect results
            status = []
            if oracle_stability <= failure_stability:
                status.append("❌ Oracle <= Failure")
                failures.append(f"{section_id}: Oracle ({oracle_stability:.2f}) <= Failure ({failure_stability:.2f})")
            
            if oracle_stability <= 0.5:
                status.append("❌ Oracle too low")
                failures.append(f"{section_id}: Oracle stability too low ({oracle_stability:.2f})")
                
            if failure_stability >= 0.3:
                status.append("❌ Failure too high")
                failures.append(f"{section_id}: Failure stability too high ({failure_stability:.2f})")
                
            status_text = " | ".join(status) if status else "✅ Pass"
            
            # Add row to table
            table.add_row(
                section_id,
                f"{oracle_stability:.3f}",
                f"{failure_stability:.3f}",
                status_text
            )
            
            # Add detailed content for failed pairs
            if status:
                self.console.print(f"\n[red]Failed Pair Details - {section_id}:[/red]")
                self.console.print("[yellow]Oracle:[/yellow]", oracle.content)
                self.console.print("[yellow]Failure:[/yellow]", failure.content)
        
        # Print results table
        self.console.print("\n")
        self.console.print(table)
        
        # If any failures, print summary and fail test
        if failures:
            self.console.print("\n[red]Failed Validations:[/red]")
            for failure in failures:
                self.console.print(f"❌ {failure}")
            self.fail(f"\n{len(failures)} validation(s) failed")
    
    def test_memory_triggers(self):
        """Test that memory triggers are properly detected"""
        table = Table(title="Memory Trigger Analysis")
        table.add_column("Memory ID", style="cyan")
        table.add_column("Trigger", style="yellow")
        table.add_column("Test Response", style="blue")
        table.add_column("Status", style="bold")
        
        failures = []
        
        for segment_id, segment in self.segments.items():
            if segment_id.startswith('Memory_') and segment.triggers:
                for trigger in segment.triggers:
                    test_response = f"I want to {trigger}"
                    response_embedding = self.vector_engine.embed_response(test_response)
                    
                    triggered_memories = self.vector_engine.check_memory_trigger(
                        response_embedding,
                        test_response
                    )
                    
                    status = "✅ Pass" if segment_id in triggered_memories else "❌ Fail"
                    
                    table.add_row(
                        segment_id,
                        trigger,
                        test_response,
                        status
                    )
                    
                    if segment_id not in triggered_memories:
                        failures.append(f"{segment_id}: Failed to trigger with phrase '{trigger}'")
                        # Print the failed memory content
                        self.console.print(f"\n[red]Failed Memory Details - {segment_id}:[/red]")
                        self.console.print("[yellow]Content:[/yellow]", segment.content)
                        self.console.print("[yellow]Trigger:[/yellow]", trigger)
        
        # Print results table
        self.console.print("\n")
        self.console.print(table)
        
        # If any failures, print summary and fail test
        if failures:
            self.console.print("\n[red]Failed Memory Triggers:[/red]")
            for failure in failures:
                self.console.print(f"❌ {failure}")
            self.fail(f"\n{len(failures)} memory trigger(s) failed")
    
    def _get_oracle_failure_pairs(self):
        pairs = {}
        for segment_id, segment in self.segments.items():
            if segment.segment_type == 'oracle' and segment.oracle_pair_id:
                pairs[segment_id] = (segment, self.segments[segment.oracle_pair_id])
        return pairs
    
    def _get_parent_paragraph(self, segment):
        if not segment.parent_paragraph:
            return None
        return self.segments.get(segment.parent_paragraph)

if __name__ == '__main__':
    unittest.main(verbosity=2) 