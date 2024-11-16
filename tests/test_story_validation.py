import unittest
from src.engine.stability_engine import StabilityEngine
from src.story.story_parser import StoryParser
from rich.console import Console
from rich.table import Table
from concurrent.futures import ThreadPoolExecutor
import random
from src.engine.llm_handler import LLMHandler

class TestStoryValidation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        cls.parser = StoryParser("data/story.txt")
        cls.segments = cls.parser.parse()
        cls.llm_handler = LLMHandler()
        cls.stability_engine = StabilityEngine(llm_handler=cls.llm_handler)
        cls.stability_engine.set_segments(cls.segments)
        cls.stability_engine.set_parameters(
            oracle_threshold=0.7,
            failure_threshold=0.3,
            distance_multiplier=1.0
        )
        cls.console = Console()

    def test_critical_pairs(self):
        """Test a representative sample of oracle/failure pairs."""
        table = Table(title="Critical Pair Analysis")
        table.add_column("Section", style="cyan")
        table.add_column("Oracle Stability", justify="right")
        table.add_column("Oracle Memory", justify="right")
        table.add_column("Failure Stability", justify="right")
        table.add_column("Failure Memory", justify="right")
        table.add_column("Status", style="bold")
        
        # Get all oracle/failure pairs directly
        oracle_segments = {
            id: seg for id, seg in self.segments.items() 
            if id.startswith('Oracle_')
        }
        
        # Sample 3 random oracle segments
        sample_oracles = random.sample(list(oracle_segments.items()), 3)
        failures = []
        
        for oracle_id, oracle in sample_oracles:
            # Get corresponding failure ID
            section_num = oracle_id.split('_', 2)[1]  # Handle both Oracle_1 and Oracle_1_specific
            base_failure_id = f"Failure_{section_num}"
            
            # Find matching failure segment
            failure_id = next(
                (id for id in self.segments.keys() 
                 if id.startswith(base_failure_id)), 
                None
            )
            
            if not failure_id:
                continue
                
            failure = self.segments[failure_id]
            paragraph_id = f"Paragraph_{section_num}"
            
            try:
                # Calculate stability scores (these are floats)
                oracle_stability = self.stability_engine.calculate_stability(
                    response_text=oracle.content,
                    segment_id=paragraph_id
                )
                failure_stability = self.stability_engine.calculate_stability(
                    response_text=failure.content,
                    segment_id=paragraph_id
                )
                
                # Get available memories for this section (as a set of strings)
                available_memories = set(
                    memory_id for memory_id in self.segments.keys()
                    if memory_id.startswith(f'Memory_{section_num}')
                )
                
                # Get triggered memories (as sets of strings)
                oracle_memories = set(self.stability_engine.check_memory_trigger(
                    response_text=oracle.content,
                    segment_id=paragraph_id
                ))
                failure_memories = set(self.stability_engine.check_memory_trigger(
                    response_text=failure.content,
                    segment_id=paragraph_id
                ))
                
                # Validate results
                status = []
                
                # Stability checks (comparing floats)
                if oracle_stability <= failure_stability:
                    status.append("❌ Oracle <= Failure")
                    failures.append(f"{oracle_id}: Invalid stability relationship")
                    
                if oracle_stability < 0.7:  # Oracle threshold
                    status.append("❌ Oracle too low")
                    failures.append(f"{oracle_id}: Stability {oracle_stability:.2f} < 0.7")
                    
                if failure_stability > 0.3:  # Failure threshold
                    status.append("❌ Failure too high")
                    failures.append(f"{failure_id}: Stability {failure_stability:.2f} > 0.3")
                
                # Memory trigger checks (comparing sets of strings)
                memory_overlap = oracle_memories.intersection(available_memories)
                if not memory_overlap:
                    status.append("❌ No memories triggered")
                    failures.append(f"{oracle_id}: Should trigger at least one memory")
                    
                failure_overlap = failure_memories.intersection(available_memories)
                if failure_overlap:
                    status.append("❌ Failure triggered memory")
                    failures.append(f"{failure_id}: Should not trigger memories")
                
                status_text = " | ".join(status) if status else "✅ Pass"
                
                table.add_row(
                    oracle_id,
                    f"{oracle_stability:.3f}",
                    str(len(memory_overlap)),  # Show number of triggered memories
                    f"{failure_stability:.3f}",
                    str(len(failure_overlap)),  # Show number of triggered memories
                    status_text
                )
                
            except Exception as e:
                failures.append(f"{oracle_id}: Error - {str(e)}")
        
        # Print results
        self.console.print("\n")
        self.console.print(table)
        
        if failures:
            self.fail("\n".join(failures))

    def test_memory_trigger_sample(self):
        """Test a sample of memory triggers."""
        table = Table(title="Memory Trigger Analysis")
        table.add_column("Memory ID", style="cyan")
        table.add_column("Trigger", style="yellow")
        table.add_column("Similarity", style="magenta")
        table.add_column("Status", style="bold")
        
        # Select a representative sample of memories
        memories = [(id, seg) for id, seg in self.segments.items() 
                   if id.startswith('Memory_') and seg.triggers]
        sample_size = min(3, len(memories))
        sample_memories = random.sample(memories, sample_size)
        
        failures = []
        
        for memory_id, memory in sample_memories:
            # Test one random trigger for each memory
            trigger = random.choice(list(memory.triggers))
            similarity = self.stability_engine.calculate_trigger_similarity(
                trigger,
                memory_id,
                trigger
            )
            
            status = "✅ Pass" if similarity >= 0.5 else "❌ Fail"
            
            table.add_row(
                memory_id,
                trigger,
                f"{similarity:.3f}",
                status
            )
            
            if similarity < 0.5:
                failures.append(f"{memory_id}: Failed to trigger")
        
        self.console.print("\n")
        self.console.print(table)
        
        if failures:
            self.fail(f"\n{len(failures)} trigger test(s) failed")

    def _select_representative_pairs(self, pairs, sample_size=3):
        """Select representative pairs from different parts of the story."""
        if len(pairs) <= sample_size:
            return pairs
            
        # Get early, middle, and late story pairs
        sorted_pairs = sorted(pairs.items())
        indices = [
            0,  # Early story
            len(sorted_pairs) // 2,  # Middle story
            len(sorted_pairs) - 1  # Late story
        ]
        return {k: pairs[k] for k in [sorted_pairs[i][0] for i in indices]}

    def _calculate_pair_stability(self, section_id, oracle, failure):
        """Calculate stability for an oracle/failure pair."""
        oracle_stability = self.stability_engine.calculate_stability(
            oracle.content,
            oracle.segment_id
        )
        failure_stability = self.stability_engine.calculate_stability(
            failure.content,
            failure.segment_id
        )
        return oracle_stability, failure_stability

    def _get_oracle_failure_pairs(self):
        """Get oracle/failure pairs from segments."""
        return {
            segment_id: (segment, self.segments[segment.oracle_pair_id])
            for segment_id, segment in self.segments.items()
            if segment.segment_type == 'oracle' and segment.oracle_pair_id
        }
