from sentence_transformers import SentenceTransformer
import numpy as np
from typing import Dict, List, Tuple
from ..story.story_parser import StorySegment
from sklearn.metrics.pairwise import cosine_similarity
import logging
import os

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

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
        self.segments: Dict[str, StorySegment] = {}
        self.oracle_threshold = 0.5
        self.failure_threshold = 0.3
        self.distance_multiplier = 1.0
        
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
    
    def calculate_cosine_similarity(self, response_embedding: np.ndarray) -> Dict[str, float]:
        """Calculate cosine similarity between response and all segments."""
        similarities = {}
        for segment_id, segment_embedding in self.segment_embeddings.items():
            similarity = cosine_similarity(
                response_embedding.reshape(1, -1),
                segment_embedding.reshape(1, -1)
            )[0][0]
            similarities[segment_id] = float(similarity)
        return similarities
    
    def find_nearest_segments(self, 
                              response_embedding: np.ndarray, 
                              n: int = 3) -> List[Tuple[str, float]]:
        """Find the n nearest segments to the response based on cosine similarity."""
        similarities = self.calculate_cosine_similarity(response_embedding)
        sorted_segments = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
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
        """Calculate system stability (0-1) based on cosine similarities."""
        similarities = self.calculate_cosine_similarity(response_embedding)
        max_similarity = max(similarities.values())
        
        # Adjust these parameters to tune the stability curve
        stability = 1 / (1 + np.exp(-5 * (max_similarity - 0.7)))  # Sigmoid centered at 0.7
        
        # Clip to ensure we stay in [0,1] range
        stability = stability * self.distance_multiplier
        return float(np.clip(stability, 0.0, 1.0))
    
    def check_memory_trigger(self, 
                             response_embedding: np.ndarray, 
                             response_text: str,
                             threshold: float = 0.7) -> List[str]:
        """Check if any memories should be triggered based on explicit triggers and semantic similarity.
        
        Args:
            response_embedding: The embedding of the player's response
            response_text: The actual text of the player's response
            threshold: Cosine similarity threshold for triggering memories
        """
        triggered_memories = []
        
        # First, check for trigger phrases
        for segment_id, segment_embedding in self.segment_embeddings.items():
            if not segment_id.startswith('Memory_'):
                continue
            
            segment = self.segments.get(segment_id)
            if segment and segment.triggers:
                for trigger in segment.triggers:
                    if trigger.lower() in response_text.lower():
                        triggered_memories.append(segment_id)
                        break
        
        # If no triggers matched, fallback to semantic similarity
        if not triggered_memories:
            similarities = self.calculate_cosine_similarity(response_embedding)
            for segment_id, similarity in similarities.items():
                if segment_id.startswith('Memory_') and similarity >= threshold:
                    triggered_memories.append(segment_id)
        
        return triggered_memories

    def set_segments(self, segments: Dict[str, StorySegment]) -> None:
        """Set the story segments and compute their embeddings.
        
        Args:
            segments: Dictionary of story segments
        """
        cache_file = "cache/embeddings.npy"
        
        # Try to load from cache first
        if self.load_embeddings(cache_file):
            return
        
        # If cache doesn't exist or failed to load, compute embeddings
        logger.info("Computing embeddings for segments...")
        self.segments = segments
        self.embed_segments(segments)
        
        # Save to cache for future use
        self.save_embeddings(cache_file)

    def save_embeddings(self, cache_file: str = "cache/embeddings.npy") -> None:
        """Save segment embeddings and parameters to a cache file."""
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        np.save(cache_file, {
            'embeddings': self.segment_embeddings,
            'segments': self.segments,
            'parameters': {
                'oracle_threshold': self.oracle_threshold,
                'failure_threshold': self.failure_threshold,
                'distance_multiplier': self.distance_multiplier
            }
        }, allow_pickle=True)

    def load_embeddings(self, cache_file: str = "cache/embeddings.npy") -> bool:
        """Load segment embeddings and parameters from cache file if available."""
        try:
            if os.path.exists(cache_file):
                cached_data = np.load(cache_file, allow_pickle=True).item()
                self.segment_embeddings = cached_data['embeddings']
                self.segments = cached_data['segments']
                
                # Load parameters if they exist in cache
                if 'parameters' in cached_data:
                    params = cached_data['parameters']
                    self.oracle_threshold = params['oracle_threshold']
                    self.failure_threshold = params['failure_threshold']
                    self.distance_multiplier = params['distance_multiplier']
                    logger.info("Loaded parameters from cache")
                
                logger.info("Loaded embeddings from cache")
                return True
            return False
        except Exception as e:
            logger.error(f"Error loading embeddings cache: {e}")
            return False

    def set_parameters(self, oracle_threshold, failure_threshold, distance_multiplier):
        self.oracle_threshold = oracle_threshold
        self.failure_threshold = failure_threshold
        self.distance_multiplier = distance_multiplier