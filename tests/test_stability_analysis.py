import pytest
from src.engine.stability_engine import StabilityEngine
from src.engine.llm_handler import LLMHandler
from src.story.story_parser import StoryParser

class TestStabilityAnalysis:
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment."""
        self.parser = StoryParser("data/story.txt")
        self.segments = self.parser.parse()
        self.llm_handler = LLMHandler()
        self.stability_engine = StabilityEngine(llm_handler=self.llm_handler)
        self.stability_engine.set_segments(self.segments)

    def test_stability_calculation(self):
        """Test that stability calculation works correctly."""
        # Test with paragraph from story.txt
        paragraph_id = "Paragraph_8"
        
        test_cases = [
            {
                "response": "I must destroy all traces of my research to prevent this from happening again",
                "expected_min": 0.7,
                "description": "Highly coherent response"
            },
            {
                "response": "Let's go get ice cream and watch movies",
                "expected_max": 0.3,
                "description": "Thematically disconnected response"
            },
            {
                "response": "The purple elephants are dancing on Jupiter's rings",
                "expected_max": 0.1,
                "description": "Completely incoherent response"
            }
        ]
        
        for case in test_cases:
            stability = self.stability_engine.calculate_stability(
                response_text=case["response"],
                segment_id=paragraph_id
            )
            
            if "expected_min" in case:
                assert stability >= case["expected_min"], \
                    f"{case['description']}: Stability {stability} should be >= {case['expected_min']}"
            if "expected_max" in case:
                assert stability <= case["expected_max"], \
                    f"{case['description']}: Stability {stability} should be <= {case['expected_max']}"

    def test_memory_trigger_detection(self):
        """Test that memory triggers are properly detected."""
        memory_id = "Memory_8_mounmentalist"
        
        test_cases = [
            {
                "response": "I painted my memories onto the walls, trying to externalize them through art",
                "should_trigger": True,
                "description": "Strong trigger match"
            },
            {
                "response": "The genetic patterns in nature inspired my paintings",
                "should_trigger": True,
                "description": "Partial trigger match"
            },
            {
                "response": "I went for a walk in the park",
                "should_trigger": False,
                "description": "No trigger match"
            }
        ]
        
        for case in test_cases:
            triggered_memories = self.stability_engine.check_memory_trigger(
                response_text=case["response"],
                segment_id="Paragraph_8"
            )
            
            if case["should_trigger"]:
                assert memory_id in triggered_memories, \
                    f"{case['description']}: Should have triggered memory"
            else:
                assert memory_id not in triggered_memories, \
                    f"{case['description']}: Should not have triggered memory"

    def test_invalid_inputs(self):
        """Test handling of invalid inputs."""
        # Test with invalid segment ID
        stability = self.stability_engine.calculate_stability(
            response_text="Any response",
            segment_id="Invalid_ID"
        )
        assert stability == 0.0, "Invalid segment ID should return 0.0 stability"
        
        # Test with empty response
        stability = self.stability_engine.calculate_stability(
            response_text="",
            segment_id="Paragraph_8"
        )
        assert stability == 0.0, "Empty response should return 0.0 stability" 