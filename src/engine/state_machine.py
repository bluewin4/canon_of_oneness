from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional
from enum import Enum
import random
from ..story.story_parser import StorySegment

class GameState(Enum):
    STABLE = "stable"
    UNSTABLE = "unstable"
    GLITCHING = "glitching"
    MEMORY = "memory"
    INVALID = "invalid"

@dataclass
class GameContext:
    current_paragraph: str
    stability: float
    discovered_memories: Set[str] = field(default_factory=set)
    triggered_glitches: Set[str] = field(default_factory=set)
    available_segments: Dict[str, StorySegment] = field(default_factory=dict)
    state_history: List[GameState] = field(default_factory=list)
    
    def add_memory(self, memory_id: str) -> None:
        self.discovered_memories.add(memory_id)
    
    def add_glitch(self, glitch_id: str) -> None:
        self.triggered_glitches.add(glitch_id)

class StateMachine:
    def __init__(self, 
                 segments: Dict[str, StorySegment],
                 stability_threshold: float = 0.3,
                 glitch_threshold: float = 0.1):
        """Initialize the state machine.
        
        Args:
            segments: Dictionary of all story segments
            stability_threshold: Threshold below which system becomes unstable
            glitch_threshold: Threshold below which glitches can occur
        """
        self.segments = segments
        self.stability_threshold = stability_threshold
        self.glitch_threshold = glitch_threshold
        self.context = self._initialize_context()
        
    def _initialize_context(self) -> GameContext:
        """Set up initial game context starting at Paragraph_1."""
        return GameContext(
            current_paragraph="Paragraph_1",
            stability=1.0,
            available_segments={k: v for k, v in self.segments.items() 
                              if v.parent_paragraph == "Paragraph_1"}
        )
    
    def update_state(self, 
                    stability: float,
                    nearest_segments: List[tuple[str, float]],
                    triggered_memories: List[str]) -> tuple[GameState, str]:
        """Update game state based on current conditions."""
        self.context.stability = stability
        
        # Check for triggered memories first, before any stability checks
        if triggered_memories:
            for memory_id in triggered_memories:
                if memory_id not in self.context.discovered_memories:
                    self.context.add_memory(memory_id)
                    self.context.state_history.append(GameState.MEMORY)
                    return GameState.MEMORY, self.segments[memory_id].content
        
        # Critical instability check - reset to start of current paragraph
        if stability < 0.1:  # Critical threshold
            self.context.state_history.append(GameState.UNSTABLE)
            self._reset_current_progress()
            return GameState.UNSTABLE, (
                "The narrative has become critically unstable. Reality shifts violently...\n\n" +
                self.segments[self.context.current_paragraph].content
            )
        
        # Rest of the existing state update logic...
        if stability < self.glitch_threshold:
            if random.random() < 0.7:
                available_glitches = [
                    seg_id for seg_id, seg in self.context.available_segments.items()
                    if seg_id.startswith("Glitch_") and seg_id not in self.context.triggered_glitches
                ]
                if available_glitches:
                    glitch_id = random.choice(available_glitches)
                    self.context.add_glitch(glitch_id)
                    self.context.state_history.append(GameState.GLITCHING)
                    return GameState.GLITCHING, self.segments[glitch_id].content
        
        if stability < self.stability_threshold:
            self.context.state_history.append(GameState.UNSTABLE)
            return GameState.UNSTABLE, self._get_unstable_message()
        
        self.context.state_history.append(GameState.STABLE)
        return GameState.STABLE, self.segments[self.context.current_paragraph].content
    
    def _get_unstable_message(self) -> str:
        """Generate message for unstable state."""
        messages = [
            "The narrative feels uncertain...",
            "Reality seems to shift slightly...",
            "The story wavers at the edges of coherence...",
            "Something feels off about this moment..."
        ]
        return random.choice(messages)
    
    def check_progression(self) -> Optional[str]:
        """Check if conditions are met to progress to next paragraph.
        
        Returns:
            Next paragraph ID if progression conditions met, None otherwise
        """
        current_num = int(self.context.current_paragraph.split('_')[1])
        required_memories = {
            seg_id for seg_id, seg in self.segments.items()
            if seg_id.startswith('Memory_') and 
               seg.parent_paragraph == self.context.current_paragraph
        }
        
        # Progress if all memories for current paragraph are found
        if required_memories.issubset(self.context.discovered_memories):
            next_paragraph = f"Paragraph_{current_num + 1}"
            if next_paragraph in self.segments:
                return next_paragraph
        return None
    
    def advance_paragraph(self, new_paragraph_id: str) -> None:
        """Advance to the next paragraph and update available segments."""
        self.context.current_paragraph = new_paragraph_id
        self.context.available_segments = {
            k: v for k, v in self.segments.items() 
            if v.parent_paragraph == new_paragraph_id
        }
    
    def get_progress_stats(self) -> Dict:
        """Get current progress statistics."""
        return {
            "current_paragraph": self.context.current_paragraph,
            "stability": self.context.stability,
            "discovered_memories": len(self.context.discovered_memories),
            "total_memories": len([s for s in self.segments if s.startswith("Memory_")]),
            "triggered_glitches": len(self.context.triggered_glitches),
            "state_history": [state.value for state in self.context.state_history[-5:]]
        } 
    
    def _reset_current_progress(self) -> None:
        """Reset progress for the current paragraph when stability is critical."""
        current_para = self.context.current_paragraph
        # Remove memories and glitches from current paragraph
        self.context.discovered_memories = {
            memory_id for memory_id in self.context.discovered_memories
            if self.segments[memory_id].parent_paragraph != current_para
        }
        self.context.triggered_glitches = {
            glitch_id for glitch_id in self.context.triggered_glitches
            if self.segments[glitch_id].parent_paragraph != current_para
        } 