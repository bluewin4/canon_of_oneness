from src.engine.llm_handler import LLMHandler
from typing import List

class VectorEngine:
    def __init__(self, llm_handler=None):
        self.segments = {}
        self.oracle_threshold = 0.7
        self.failure_threshold = 0.4
        self.distance_multiplier = 1.0
        self.position = None
        self.llm_handler = llm_handler or LLMHandler()

    def set_segments(self, segments):
        self.segments = segments

    def embed_response(self, text):
        # No longer needed - return dummy value
        return text

    def calculate_stability(self, response_text, segment_id=None):
        if not segment_id or segment_id not in self.segments:
            return 0.0
            
        current_segment = self.segments[segment_id]
        
        # Get available memories for this segment
        available_memories = [
            memory.content 
            for memory_id, memory in self.segments.items()
            if memory_id.startswith('Memory_') and 
            memory.parent_paragraph == segment_id
        ]
        
        return self.llm_handler.calculate_stability(
            current_paragraph=current_segment.content,
            player_input=response_text,
            available_memories=available_memories
        )

    def find_nearest_segments(self, response_text):
        # Return empty list of nearby segments
        return []

    def check_memory_trigger(self, response_text: str, segment_id: str) -> List[str]:
        """Check if response triggers any memories."""
        triggered_memories = []
        
        if not segment_id or segment_id not in self.segments:
            return triggered_memories
        
        current_segment = self.segments[segment_id]
        
        # Get available memories for this section
        available_memories = [
            memory_id for memory_id, memory in self.segments.items()
            if memory_id.startswith('Memory_') and 
            memory.parent_paragraph == segment_id
        ]
        
        for memory_id in available_memories:
            memory = self.segments[memory_id]
            
            trigger_score = self.llm_handler.calculate_memory_trigger(
                response_text=response_text,
                memory_text=memory.content,
                triggers=memory.triggers,
                current_paragraph=current_segment.content
            )
            
            if trigger_score >= self.oracle_threshold:
                triggered_memories.append(memory_id)
                
        return triggered_memories

    def update_position(self, response_text):
        # Do nothing
        pass

    def calculate_trigger_similarity(self, response_text, memory_id, trigger):
        """Calculate similarity between response and a specific memory trigger."""
        if memory_id not in self.segments:
            return 0.0
        
        memory = self.segments[memory_id]
        current_segment = self.segments[memory.parent_paragraph]
        
        return self.llm_handler.calculate_memory_trigger(
            response_text=response_text,
            memory_text=memory.content,
            triggers=[trigger],
            current_paragraph=current_segment.content
        )

    def compute_phase_coherence(self, text, segment_id):
        # Return default medium coherence
        return 0.6

    def compute_memory_resonance(self, text):
        # Return default medium resonance
        return 0.5

    def set_parameters(self, oracle_threshold: float, failure_threshold: float, distance_multiplier: float):
        """Set engine parameters."""
        self.oracle_threshold = oracle_threshold
        self.failure_threshold = failure_threshold
        self.distance_multiplier = distance_multiplier

    def save_parameters(self, filepath):
        # Do nothing for now
        pass

    def save_embeddings(self):
        # Do nothing for now
        pass
