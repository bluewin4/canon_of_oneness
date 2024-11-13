import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from .vector_engine import VectorEngine
from .state_machine import StateMachine, GameState
import re
from .llm_handler import LLMHandler

# Set the environment variable before other imports
os.environ["TOKENIZERS_PARALLELISM"] = "false"

@dataclass
class GameResponse:
    content: str
    state: GameState
    stability: float
    nearby_content: List[Tuple[str, float]]
    stats: Dict
    can_progress: bool
    # New fields for enhanced feedback
    message: Optional[str] = None
    discovered_memory: Optional[str] = None
    triggered_glitch: Optional[str] = None

class ResponseHandler:
    def __init__(self, 
                 vector_engine: VectorEngine, 
                 state_machine: StateMachine,
                 min_response_length: int = 3,
                 max_response_length: int = 500):  # Added max length
        self.vector_engine = vector_engine
        self.state_machine = state_machine
        self.min_response_length = min_response_length
        self.max_response_length = max_response_length
        self.llm_handler = LLMHandler(
            cache_dir="cache",
            cache_ttl_hours=24,
            max_retries=3,
            retry_delay=1.0
        )
        
    def _clean_input(self, text: str) -> str:
        """Clean and normalize player input text.
        
        Args:
            text: Raw player input text
            
        Returns:
            Cleaned and normalized text string
        """
        # Remove extra whitespace
        cleaned = ' '.join(text.split())
        # Remove any special characters that might cause issues
        cleaned = ''.join(char for char in cleaned if char.isprintable())
        return cleaned.strip()
        
    def process_response(self, player_input: str) -> GameResponse:
        """Process a player's response and return game state updates."""
        try:
            # Clean and validate input
            cleaned_input = self._clean_input(player_input)
            validation_result = self._validate_input(cleaned_input)
            if validation_result is not True:
                return self._create_invalid_response(validation_result)
            
            # Generate embeddings and calculate metrics
            response_embedding = self.vector_engine.embed_response(cleaned_input)
            stability = self.vector_engine.calculate_stability(response_embedding)
            nearest_segments = self.vector_engine.find_nearest_segments(response_embedding)
            triggered_memories = self.vector_engine.check_memory_trigger(response_embedding)
            
            # Update vector position
            self.vector_engine.update_position(response_embedding)
            
            # Get previous stats for comparison
            previous_memories = set(self.state_machine.context.discovered_memories)
            previous_glitches = set(self.state_machine.context.triggered_glitches)
            
            # Update game state
            new_state, content = self.state_machine.update_state(
                stability,
                nearest_segments,
                triggered_memories
            )
            
            # Check what's new
            new_memory = None
            new_glitch = None
            current_memories = set(self.state_machine.context.discovered_memories)
            current_glitches = set(self.state_machine.context.triggered_glitches)
            
            if len(current_memories) > len(previous_memories):
                new_memory = (current_memories - previous_memories).pop()
            if len(current_glitches) > len(previous_glitches):
                new_glitch = (current_glitches - previous_glitches).pop()
            
            # Generate appropriate feedback message
            feedback_message = self._generate_feedback(
                stability,
                new_state,
                nearest_segments
            )
            
            # Check for progression possibility
            next_paragraph = self.state_machine.check_progression()
            
            narrative_response = self.llm_handler.generate_response(
                current_paragraph=self.state_machine.segments[self.state_machine.context.current_paragraph].content,
                player_input=cleaned_input,
                stability=stability,
                nearby_segments=nearest_segments,
                state_history=[state.value for state in self.state_machine.context.state_history[-3:]]
            )
            
            # Override narrative response if we hit critical instability
            if stability < 0.1:
                narrative_response = content  # Use the reset content from state machine
            
            return GameResponse(
                content=narrative_response,
                state=new_state,
                stability=stability,
                nearby_content=self._format_nearby_content(nearest_segments),
                stats=self.state_machine.get_progress_stats(),
                can_progress=bool(next_paragraph),
                message=feedback_message,
                discovered_memory=new_memory,
                triggered_glitch=new_glitch
            )
            
        except Exception as e:
            # Use a default stability value for error cases
            return GameResponse(
                content="I apologize, but I'm having trouble processing your response. Please try again.",
                state=GameState.INVALID,
                stability=0.0,  # Default stability for errors
                nearby_content=[],
                stats=self.state_machine.get_progress_stats(),
                can_progress=False,
                message=f"Error: {str(e)}",
                discovered_memory=None,
                triggered_glitch=None
            )
    
    def _validate_input(self, player_input: str) -> bool | str:
        """Validate player input with enhanced feedback."""
        if len(player_input) < self.min_response_length:
            return f"Response too short. Please write at least {self.min_response_length} characters."
        if len(player_input) > self.max_response_length:
            return f"Response too long. Please keep it under {self.max_response_length} characters."
        return True
    
    def _generate_feedback(self, 
                          stability: float, 
                          state: GameState,
                          nearest_segments: List[Tuple[str, float]]) -> str:
        """Generate contextual feedback based on player's response."""
        if stability < 0.1:
            return "[CRITICAL INSTABILITY DETECTED] The narrative is collapsing. Returning to last stable point..."
        elif stability < 0.3:
            return "⚠️ WARNING: Narrative stability critical. Choose your next words carefully."
        elif stability < 0.5:
            return "The narrative feels unstable. Try staying closer to the current context."
        elif stability < 0.7:
            return "You're maintaining narrative coherence, but there's room for stronger connections."
        else:
            return "Your response resonates well with the narrative."
    
    def _format_nearby_content(self, 
                             nearest_segments: List[Tuple[str, float]], 
                             threshold: float = 0.7) -> List[Tuple[str, float]]:
        """Format nearby content with relevance filtering."""
        formatted = [(segment[0], 1 - segment[1]) for segment in nearest_segments]
        # Only return segments above relevance threshold
        return [item for item in formatted if item[1] >= threshold] 
    
    def _create_invalid_response(self, message: str) -> GameResponse:
        """Create an invalid response."""
        return GameResponse(
            content="",
            state=GameState.INVALID,
            stability=0.0,
            nearby_content=[],
            stats=self.state_machine.get_progress_stats(),
            can_progress=False,
            message=message
        )