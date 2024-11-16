from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class StabilityEngine:
    def __init__(self, llm_handler):
        self.llm_handler = llm_handler
        self.segments = {}
        self.context = None
        self.oracle_threshold = 0.7
        self.failure_threshold = 0.3
        self.distance_multiplier = 1.0

    def set_segments(self, segments: Dict):
        """Store story segments for reference."""
        self.segments = segments

    def calculate_stability(self, response_text: str, segment_id: Optional[str] = None) -> float:
        """Calculate stability based on LLM analysis and memory triggers."""
        # Handle empty input or invalid segment
        if not response_text.strip() or not segment_id or segment_id not in self.segments:
            return 0.0
        
        current_segment = self.segments[segment_id]
        
        # Get available memories for this segment
        available_memories = [
            memory.content 
            for memory_id, memory in self.segments.items()
            if memory_id.startswith('Memory_') and 
            memory.parent_paragraph == segment_id
        ]
        
        # Calculate base stability from LLM
        base_stability = self.llm_handler.calculate_stability(
            current_paragraph=current_segment.content,
            player_input=response_text,
            available_memories=available_memories
        )
        
        # Check if any memories are triggered
        triggered_memories = self.check_memory_trigger(
            response_text=response_text,
            segment_id=segment_id
        )
        
        # Adjust stability based on memory triggers
        if triggered_memories:
            # Boost stability if memories are triggered, but don't exceed 1.0
            base_stability = min(1.0, base_stability * self.distance_multiplier)
        
        return base_stability

    def check_memory_trigger(self, response_text: str, segment_id: str) -> List[str]:
        """Check if response triggers any memories."""
        triggered_memories = []
        
        if not segment_id or segment_id not in self.segments:
            logger.debug(f"Invalid segment_id: {segment_id}")
            return triggered_memories
        
        current_segment = self.segments[segment_id]
        
        # Calculate base stability first
        base_stability = self.llm_handler.calculate_stability(
            current_paragraph=current_segment.content,
            player_input=response_text,
            available_memories=[]  # No memories yet, just checking base stability
        )
        
        # If base stability is too low, don't bother checking memories
        if base_stability < self.failure_threshold:
            logger.debug(f"Base stability {base_stability} below threshold {self.failure_threshold}")
            return triggered_memories
        
        # Get available memories for this section
        available_memories = [
            memory_id for memory_id, memory in self.segments.items()
            if memory_id.startswith('Memory_') and 
            memory.parent_paragraph == segment_id
        ]
        
        logger.debug(f"Found {len(available_memories)} available memories for segment {segment_id}")
        
        for memory_id in available_memories:
            memory = self.segments[memory_id]
            
            trigger_score = self.llm_handler.calculate_memory_trigger(
                response_text=response_text,
                memory_text=memory.content,
                triggers=memory.triggers,
                current_paragraph=current_segment.content
            )
            
            logger.debug(f"Memory {memory_id} trigger score: {trigger_score}")
            
            if trigger_score >= self.oracle_threshold:
                triggered_memories.append(memory_id)
                logger.info(f"Triggered memory {memory_id} with score {trigger_score}")
        
        return triggered_memories

    def calculate_trigger_similarity(self, response_text: str, memory_id: str, trigger: str) -> float:
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

    def compute_coherence(self, response: str, segment_id: str) -> float:
        """Calculate narrative coherence between response and segment."""
        if not segment_id or segment_id not in self.segments:
            return 0.0
            
        current_segment = self.segments[segment_id]
        coherence = self.llm_handler.calculate_coherence(
            current_paragraph=current_segment.content,
            response_text=response
        )
        return coherence

    def set_parameters(self, oracle_threshold: float, failure_threshold: float, distance_multiplier: float):
        """Set stability calculation parameters."""
        self.oracle_threshold = oracle_threshold
        self.failure_threshold = failure_threshold
        self.distance_multiplier = distance_multiplier