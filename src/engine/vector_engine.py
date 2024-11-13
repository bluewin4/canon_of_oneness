from sentence_transformers import SentenceTransformer
import numpy as np
from typing import Dict, List, Tuple
from ..story.story_parser import StorySegment

class VectorEngine:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """Initialize the vector engine with a specified model.
        
        Args:
            model_name: The name of the sentence-transformer model to use.
                       Default is all-MiniLM-L6-v2 for good balance of speed/quality.
        """
        self.model = SentenceTransformer(model_name)
        self.segment_embeddings: Dict[str, np.ndarray] = {}
        self.current_position: np.ndarray = None
        
    def embed_segments(self, segments: Dict[str, StorySegment]) -> None:
        """Compute embeddings for all story segments."""
        for segment_id, segment in segments.items():
            self.segment_embeddings[segment_id] = self.model.encode(
                segment.content,
                normalize_embeddings=True
            )
    
    def embed_response(self, text: str) -> np.ndarray:
        """Compute embedding for a player's response."""
        return self.model.encode(text, normalize_embeddings=True)
    
    def calculate_distances(self, response_embedding: np.ndarray) -> Dict[str, float]:
        """Calculate cosine distances between response and all segments."""
        distances = {}
        for segment_id, segment_embedding in self.segment_embeddings.items():
            distance = 1 - np.dot(response_embedding, segment_embedding)
            distances[segment_id] = float(distance)  # Convert to float for JSON serialization
        return distances
    
    def find_nearest_segments(self, 
                            response_embedding: np.ndarray, 
                            n: int = 3) -> List[Tuple[str, float]]:
        """Find the n nearest segments to the response."""
        distances = self.calculate_distances(response_embedding)
        sorted_segments = sorted(distances.items(), key=lambda x: x[1])
        return sorted_segments[:n]
    
    def update_position(self, response_embedding: np.ndarray, 
                       learning_rate: float = 0.5) -> None:
        """Update the current position in the vector space.
        
        Args:
            response_embedding: The embedding of the player's response
            learning_rate: How much to move toward the new position (0-1)
        """
        if self.current_position is None:
            self.current_position = response_embedding
        else:
            self.current_position = (1 - learning_rate) * self.current_position + \
                                  learning_rate * response_embedding
            # Renormalize
            self.current_position = self.current_position / np.linalg.norm(self.current_position)
    
    def calculate_stability(self, response_embedding: np.ndarray) -> float:
        """Calculate system stability (0-1) based on distances to known segments."""
        distances = self.calculate_distances(response_embedding)
        min_distance = min(distances.values())
        # Convert distance to stability (closer = more stable)
        # Using a sigmoid-like curve for smooth transition
        stability = 1 / (1 + np.exp(10 * (min_distance - 0.5)))
        return float(stability)
    
    def check_memory_trigger(self, 
                           response_embedding: np.ndarray, 
                           threshold: float = 0.2) -> List[str]:
        """Check if any memories should be triggered based on proximity."""
        triggered_memories = []
        distances = self.calculate_distances(response_embedding)
        
        for segment_id, distance in distances.items():
            if segment_id.startswith('Memory_') and distance < threshold:
                triggered_memories.append(segment_id)
                
        return triggered_memories 