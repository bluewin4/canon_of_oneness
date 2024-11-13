from sentence_transformers import SentenceTransformer
import numpy as np
from typing import Dict, List, Tuple
from ..story.story_parser import StorySegment
from sklearn.metrics.pairwise import cosine_similarity
import logging
import os
import json

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
        self.trigger_embeddings: Dict[str, Dict[str, np.ndarray]] = {}
        self.tag_embeddings: Dict[str, np.ndarray] = {}
        self.prompt_embeddings: Dict[str, np.ndarray] = {}
        self.current_position: np.ndarray = None
        self.segments: Dict[str, StorySegment] = {}
        
        # Load parameters from file if they exist, otherwise use defaults
        self.load_parameters()

    def load_parameters(self, param_file: str = "cache/parameters.json") -> None:
        """Load parameters from JSON file if it exists, otherwise use defaults."""
        try:
            if os.path.exists(param_file):
                with open(param_file, 'r') as f:
                    params = json.load(f)
                self.oracle_threshold = params['oracle_threshold']
                self.failure_threshold = params['failure_threshold']
                self.distance_multiplier = params['distance_multiplier']
                logger.info("Loaded parameters from parameters file")
            else:
                # Default values
                self.oracle_threshold = 0.5
                self.failure_threshold = 0.3
                self.distance_multiplier = 1.0
        except Exception as e:
            logger.error(f"Error loading parameters: {e}")
            # Fall back to defaults
            self.oracle_threshold = 0.5
            self.failure_threshold = 0.3
            self.distance_multiplier = 1.0

    def save_parameters(self, param_file: str = "cache/parameters.json") -> None:
        """Save current parameters to a JSON file."""
        os.makedirs(os.path.dirname(param_file), exist_ok=True)
        params = {
            'oracle_threshold': float(self.oracle_threshold),
            'failure_threshold': float(self.failure_threshold),
            'distance_multiplier': float(self.distance_multiplier)
        }
        with open(param_file, 'w') as f:
            json.dump(params, f, indent=4)
        logger.info("Saved parameters to file")

    def embed_segments(self, segments: Dict[str, StorySegment]) -> None:
        """Compute embeddings for all story segments and their triggers/tags/prompts."""
        for segment_id, segment in segments.items():
            # Embed main content
            self.segment_embeddings[segment_id] = self.model.encode(
                segment.content,
                normalize_embeddings=True
            )
            
            # Debug logging
            logger.info(f"Embedding segment {segment_id}")
            
            # Embed triggers if they exist
            if segment.triggers:
                logger.info(f"Found triggers for {segment_id}: {segment.triggers}")
                self.trigger_embeddings[segment_id] = {}
                for trigger in segment.triggers:
                    trigger_embedding = self.model.encode(
                        trigger,
                        normalize_embeddings=True
                    )
                    self.trigger_embeddings[segment_id][trigger] = trigger_embedding
                    logger.info(f"Embedded trigger '{trigger}' for {segment_id}")
            
            # Embed tags if they exist
            if hasattr(segment, 'tags') and segment.tags:
                for tag in segment.tags:
                    if tag not in self.tag_embeddings:
                        self.tag_embeddings[tag] = self.model.encode(
                            tag,
                            normalize_embeddings=True
                        )
            
            # Embed prompts if they exist
            if hasattr(segment, 'prompts') and segment.prompts:
                for prompt in segment.prompts:
                    if prompt not in self.prompt_embeddings:
                        self.prompt_embeddings[prompt] = self.model.encode(
                            prompt,
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
    
    def calculate_trigger_similarity(self,
                                   response_embedding: np.ndarray,
                                   segment_id: str,
                                   trigger: str) -> float:
        """Calculate similarity between response and a specific trigger."""
        # Debug logging
        logger.info(f"Calculating similarity for {segment_id}, trigger: {trigger}")
        logger.info(f"Trigger embeddings available: {list(self.trigger_embeddings.keys())}")
        
        if segment_id in self.trigger_embeddings:
            logger.info(f"Triggers for {segment_id}: {list(self.trigger_embeddings[segment_id].keys())}")
            
            if trigger in self.trigger_embeddings[segment_id]:
                trigger_embedding = self.trigger_embeddings[segment_id][trigger]
                similarity = cosine_similarity(
                    response_embedding.reshape(1, -1),
                    trigger_embedding.reshape(1, -1)
                )[0][0]
                logger.info(f"Calculated similarity: {similarity}")
                return float(similarity)
            else:
                logger.warning(f"Trigger '{trigger}' not found in embeddings for {segment_id}")
        else:
            logger.warning(f"No trigger embeddings found for {segment_id}")
        return 0.0

    def check_memory_trigger(self, 
                             response_embedding: np.ndarray,
                             threshold: float = 0.5) -> List[str]:
        """Check if any memories should be triggered based on semantic similarity."""
        triggered_memories = []
        
        # Debug logging
        logger.info(f"Checking memory triggers. Available segments: {list(self.trigger_embeddings.keys())}")
        
        for segment_id, triggers in self.trigger_embeddings.items():
            if not segment_id.startswith('Memory_'):
                continue
                
            logger.info(f"Checking triggers for {segment_id}")
            # Check similarity against each trigger embedding
            trigger_similarities = []
            for trigger_text, trigger_embedding in triggers.items():
                similarity = cosine_similarity(
                    response_embedding.reshape(1, -1),
                    trigger_embedding.reshape(1, -1)
                )[0][0]
                trigger_similarities.append(similarity)
                logger.info(f"Trigger '{trigger_text}' similarity: {similarity}")
            
            # If any trigger similarity exceeds threshold, add the memory
            if any(sim >= threshold for sim in trigger_similarities):
                triggered_memories.append(segment_id)
                logger.info(f"Memory triggered: {segment_id}")
        
        return triggered_memories

    def check_tag_matches(self, 
                         response_embedding: np.ndarray,
                         threshold: float = 0.7) -> List[str]:
        """Check for matching tags based on semantic similarity."""
        matched_tags = []
        
        for tag, tag_embedding in self.tag_embeddings.items():
            similarity = cosine_similarity(
                response_embedding.reshape(1, -1),
                tag_embedding.reshape(1, -1)
            )[0][0]
            
            if similarity >= threshold:
                matched_tags.append(tag)
        
        return matched_tags

    def check_prompt_matches(self, 
                           response_embedding: np.ndarray,
                           threshold: float = 0.7) -> List[str]:
        """Check for matching prompts based on semantic similarity."""
        matched_prompts = []
        
        for prompt, prompt_embedding in self.prompt_embeddings.items():
            similarity = cosine_similarity(
                response_embedding.reshape(1, -1),
                prompt_embedding.reshape(1, -1)
            )[0][0]
            
            if similarity >= threshold:
                matched_prompts.append(prompt)
        
        return matched_prompts

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
        """Save all embeddings and parameters to a cache file."""
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        
        # Debug logging
        logger.info(f"Saving embeddings to {cache_file}")
        logger.info(f"Number of trigger embeddings: {len(self.trigger_embeddings)}")
        
        np.save(cache_file, {
            'embeddings': self.segment_embeddings,
            'trigger_embeddings': self.trigger_embeddings,
            'tag_embeddings': self.tag_embeddings,
            'prompt_embeddings': self.prompt_embeddings,
            'segments': self.segments,
            'parameters': {
                'oracle_threshold': self.oracle_threshold,
                'failure_threshold': self.failure_threshold,
                'distance_multiplier': self.distance_multiplier
            }
        }, allow_pickle=True)

    def load_embeddings(self, cache_file: str = "cache/embeddings.npy") -> bool:
        """Load all embeddings and parameters from cache file if available."""
        try:
            if os.path.exists(cache_file):
                cached_data = np.load(cache_file, allow_pickle=True).item()
                self.segment_embeddings = cached_data['embeddings']
                self.trigger_embeddings = cached_data.get('trigger_embeddings', {})
                self.tag_embeddings = cached_data.get('tag_embeddings', {})
                self.prompt_embeddings = cached_data.get('prompt_embeddings', {})
                self.segments = cached_data['segments']
                
                if 'parameters' in cached_data:
                    params = cached_data['parameters']
                    self.oracle_threshold = params.get('oracle_threshold', 0.5)
                    self.failure_threshold = params.get('failure_threshold', 0.3)
                    self.distance_multiplier = params.get('distance_multiplier', 1.0)
                    logger.info("Loaded parameters from cache")
                
                logger.info("Loaded embeddings from cache")
                return True
            return False
        except Exception as e:
            logger.error(f"Error loading embeddings cache: {e}")
            if os.path.exists(cache_file):
                os.remove(cache_file)
                logger.info("Removed corrupted cache file")
            return False

    def set_parameters(self, oracle_threshold, failure_threshold, distance_multiplier):
        self.oracle_threshold = oracle_threshold
        self.failure_threshold = failure_threshold
        self.distance_multiplier = distance_multiplier