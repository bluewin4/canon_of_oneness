from sentence_transformers import SentenceTransformer
import numpy as np
from typing import Dict, List, Tuple, Optional
from ..story.story_parser import StorySegment
from sklearn.metrics.pairwise import cosine_similarity
import logging
import os
import json

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorEngine:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """Initialize the vector engine with a specified model."""
        super().__init__()
        self.model = SentenceTransformer(model_name)
        self.segment_embeddings: Dict[str, np.ndarray] = {}
        self.trigger_embeddings: Dict[str, Dict[str, np.ndarray]] = {}
        self.memory_embeddings: Dict[str, np.ndarray] = {}
        self.tag_embeddings: Dict[str, np.ndarray] = {}
        self.prompt_embeddings: Dict[str, np.ndarray] = {}
        self.current_position: Optional[np.ndarray] = None
        self.segments: Dict[str, StorySegment] = {}
        self.phase_history: List[np.ndarray] = []
        self.reference_points: Dict[str, np.ndarray] = {}
        
        # Phase space parameters
        self.beta = 1.0  # Phase coherence sensitivity
        self.memory_weight = 0.3  # Memory resonance weight
        self.temporal_decay = 0.1  # Temporal decay rate
        self.resonance_bandwidth = 0.2  # Allowable frequency band
        self.phase_transition_threshold = 0.4  # Threshold for phase transitions
        self.coherence_threshold = 0.7  # Minimum coherence requirement
        
        # Load parameters from file if they exist, otherwise use defaults
        self.load_parameters()

    def load_parameters(self, param_file: str = "cache/parameters.json") -> None:
        """Load parameters from JSON file if it exists, otherwise use defaults."""
        try:
            if os.path.exists(param_file):
                with open(param_file, 'r') as f:
                    params = json.load(f)
                self.oracle_threshold = params.get('oracle_threshold', 0.5)
                self.failure_threshold = params.get('failure_threshold', 0.3)
                self.distance_multiplier = params.get('distance_multiplier', 1.0)
                self.beta = params.get('beta', 1.0)
                self.memory_weight = params.get('memory_weight', 0.3)
                self.temporal_decay = params.get('temporal_decay', 0.1)
                self.resonance_bandwidth = params.get('resonance_bandwidth', 0.2)
                self.phase_transition_threshold = params.get('phase_transition_threshold', 0.4)
                self.coherence_threshold = params.get('coherence_threshold', 0.7)
                logger.info("Loaded parameters from parameters file")
            else:
                self._set_default_parameters()
                logger.info("Using default parameters")
        except Exception as e:
            logger.error(f"Error loading parameters: {e}")
            self._set_default_parameters()

    def _set_default_parameters(self):
        """Set default values for all parameters."""
        self.oracle_threshold = 0.5
        self.failure_threshold = 0.3
        self.distance_multiplier = 1.0
        self.beta = 1.0
        self.memory_weight = 0.3
        self.temporal_decay = 0.1
        self.resonance_bandwidth = 0.2
        self.phase_transition_threshold = 0.4
        self.coherence_threshold = 0.7

    def save_parameters(self, param_file: str = "cache/parameters.json") -> None:
        """Save current parameters to a JSON file."""
        os.makedirs(os.path.dirname(param_file), exist_ok=True)
        params = {
            'oracle_threshold': float(self.oracle_threshold),
            'failure_threshold': float(self.failure_threshold),
            'distance_multiplier': float(self.distance_multiplier),
            'beta': float(self.beta),
            'memory_weight': float(self.memory_weight),
            'temporal_decay': float(self.temporal_decay),
            'resonance_bandwidth': float(self.resonance_bandwidth),
            'phase_transition_threshold': float(self.phase_transition_threshold),
            'coherence_threshold': float(self.coherence_threshold)
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
            logger.info(f"Embedded content for segment {segment_id}")

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
                        logger.info(f"Embedded tag '{tag}'")
            
            # Embed prompts if they exist
            if hasattr(segment, 'prompts') and segment.prompts:
                for prompt in segment.prompts:
                    if prompt not in self.prompt_embeddings:
                        self.prompt_embeddings[prompt] = self.model.encode(
                            prompt,
                            normalize_embeddings=True
                        )
                        logger.info(f"Embedded prompt '{prompt}'")

    def embed_response(self, text: str) -> np.ndarray:
        """Compute embedding for a player's response."""
        # Check if we have this embedding cached
        if hasattr(self, 'response_embeddings') and text in self.response_embeddings:
            return self.response_embeddings[text]
        
        # If not cached, compute it
        embedding = self.model.encode(text, normalize_embeddings=True)
        
        # Cache it
        if not hasattr(self, 'response_embeddings'):
            self.response_embeddings = {}
        self.response_embeddings[text] = embedding
        
        logger.info(f"Embedded response: '{text}'")
        return embedding

    def calculate_cosine_similarity(self, response_embedding: np.ndarray) -> Dict[str, float]:
        """Calculate cosine similarity between response and all segments."""
        similarities = {}
        for segment_id, segment_embedding in self.segment_embeddings.items():
            # Don't reshape since vectors are already normalized
            similarity = np.dot(response_embedding, segment_embedding)
            similarities[segment_id] = float(similarity)
            logger.debug(f"Similarity with {segment_id}: {similarity}")
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
            logger.info("Initialized current position")
        else:
            self.current_position = (1 - learning_rate) * self.current_position + \
                                      learning_rate * response_embedding
            # Renormalize
            self.current_position = self.current_position / np.linalg.norm(self.current_position)
            logger.info("Updated current position")

    def compute_phase_coherence(self, response_embedding: np.ndarray, context_id: str) -> float:
        """Compute phase coherence using three-point normalization.
        
        Args:
            response_embedding: Current response vector
            context_id: ID of the context segment
        
        Returns:
            float: Phase coherence score between 0 and 1
        """
        if context_id not in self.segment_embeddings:
            return 0.0
        
        canon_embed = self.segment_embeddings[context_id]
        
        # Get memory reference points for this context
        memory_refs = [
            mem_embed for mem_id, mem_embed in self.memory_embeddings.items()
            if mem_id.split('_')[1] == context_id.split('_')[1]  # Match context number
        ]
        
        if not memory_refs:
            reference_embed = np.zeros_like(response_embedding)
        else:
            # Use weighted average of relevant memories as reference
            weights = [np.exp(-0.1 * i) for i in range(len(memory_refs))]  # Temporal decay
            reference_embed = np.average(memory_refs, axis=0, weights=weights)
        
        # Calculate normalized angle using three-point method
        v1 = canon_embed - reference_embed
        v2 = response_embedding - reference_embed
        
        # Normalize vectors
        v1_norm = v1 / np.linalg.norm(v1)
        v2_norm = v2 / np.linalg.norm(v2)
        
        # Calculate phase angle with stability weighting
        cos_theta = np.dot(v1_norm, v2_norm)
        phase_angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))
        
        # Apply non-linear scaling from theoretical framework
        coherence = np.exp(-self.beta * phase_angle)
        
        return float(coherence)

    def calculate_stability(self, response_embedding: np.ndarray, segment_id: Optional[str] = None) -> float:
        """Enhanced stability calculation using phase space analysis.
        
        Implements S(r,g|C,M) = exp(-β|D(r,g,C,M)|) × ∏ᵢ cos(θᵢ(t) - θc(t))
        """
        # Phase coherence (local structure)
        phase_coherence = self.compute_phase_coherence(response_embedding, segment_id) if segment_id else 0.0
        
        # Memory resonance (global context)
        memory_resonance = self.compute_memory_resonance(response_embedding)
        
        # Track phase history for transition detection
        self.phase_history.append(response_embedding)
        if len(self.phase_history) > 10:
            self.phase_history.pop(0)
        
        # Calculate phase transition probability
        phase_transition = 0.0
        if len(self.phase_history) >= 2:
            prev_embedding = self.phase_history[-2]
            delta_phase = np.arccos(np.clip(
                np.dot(response_embedding, prev_embedding), -1.0, 1.0
            ))
            phase_transition = np.exp(-delta_phase / self.phase_transition_threshold)
        
        # Combine components using theoretical framework formula
        coherence_term = np.exp(-self.beta * (1 - phase_coherence))
        resonance_term = np.power(memory_resonance, self.memory_weight)
        transition_term = np.exp(-phase_transition)
        
        # Calculate final stability score
        stability = coherence_term * resonance_term * transition_term
        
        # Apply bandwidth-based scaling
        bandwidth_factor = 1.0 - (1.0 - self.resonance_bandwidth) * phase_transition
        stability *= bandwidth_factor
        
        # Normalize to [0,1] range with sharper discrimination
        stability = 1 / (1 + np.exp(-12 * (stability - 0.5)))
        
        return float(np.clip(stability, 0.0, 1.0))

    def compute_memory_resonance(self, response_embedding: np.ndarray) -> float:
        """Compute memory resonance using theoretical μ(r,M) function."""
        if not self.memory_embeddings:
            return 1.0
            
        resonances = []
        current_time = len(self.phase_history)
        
        for memory_id, memory_embedding in self.memory_embeddings.items():
            # Calculate temporal component
            memory_time = int(memory_id.split('_')[1])  # Extract memory sequence number
            temporal_decay = np.exp(-0.1 * abs(current_time - memory_time))
            
            # Calculate correlation component using normalized dot product
            correlation = np.dot(response_embedding, memory_embedding) / \
                         (np.linalg.norm(response_embedding) * np.linalg.norm(memory_embedding))
            
            # Combine using theoretical formula
            resonance = temporal_decay * (0.5 * (correlation + 1))  # Map to [0,1]
            resonances.append(resonance)
        
        # Use product formula from theoretical framework
        if resonances:
            return float(np.exp(np.mean(np.log(np.abs(resonances) + 1e-8))))
        return 1.0

    def calculate_trigger_similarity(self,
                                   response_embedding: np.ndarray,
                                   segment_id: str,
                                   trigger: str) -> float:
        """Calculate similarity between response and a specific trigger."""
        logger.info(f"Calculating similarity for {segment_id}, trigger: '{trigger}'")
        if segment_id in self.trigger_embeddings:
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
                             response_text: str,
                             threshold: float = 0.5) -> List[str]:
        """Check if any memories should be triggered based on semantic similarity."""
        triggered_memories = []
        
        logger.info(f"Checking memory triggers. Available segments: {list(self.trigger_embeddings.keys())}")
        
        for segment_id, triggers in self.trigger_embeddings.items():
            if not segment_id.startswith('Memory_'):
                continue
                
            logger.info(f"Checking triggers for {segment_id}")
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
                logger.info(f"Matched tag '{tag}' with similarity {similarity}")
        
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
                logger.info(f"Matched prompt '{prompt}' with similarity {similarity}")
        
        return matched_prompts

    def set_segments(self, segments: Dict[str, StorySegment]) -> None:
        """Set story segments and compute embeddings."""
        self.segments = segments
        
        # Compute embeddings for all segments
        for segment_id, segment in segments.items():
            # Regular segment embedding
            self.segment_embeddings[segment_id] = self.embed_response(segment.content)
            
            # Memory-specific embeddings
            if segment_id.startswith('Memory_'):
                self.memory_embeddings[segment_id] = self.embed_response(segment.content)
                
                # Handle triggers for memories
                if segment.triggers:
                    self.trigger_embeddings[segment_id] = {}
                    for trigger in segment.triggers:
                        self.trigger_embeddings[segment_id][trigger] = self.embed_response(trigger)

    def save_embeddings(self, cache_file: str = "cache/embeddings.npy") -> None:
        """Save all embeddings and parameters to a cache file."""
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        
        # Add response_embeddings to the save data
        save_data = {
            'embeddings': self.segment_embeddings,
            'trigger_embeddings': self.trigger_embeddings,
            'tag_embeddings': self.tag_embeddings,
            'prompt_embeddings': self.prompt_embeddings,
            'response_embeddings': getattr(self, 'response_embeddings', {}),
            'segments': self.segments,
            'parameters': {
                'oracle_threshold': self.oracle_threshold,
                'failure_threshold': self.failure_threshold,
                'distance_multiplier': self.distance_multiplier
            }
        }
        
        np.save(cache_file, save_data, allow_pickle=True)
        logger.info("Embeddings saved successfully.")

    def load_embeddings(self, cache_file: str = "cache/embeddings.npy") -> bool:
        """Load all embeddings and parameters from cache file if available."""
        try:
            if os.path.exists(cache_file):
                cached_data = np.load(cache_file, allow_pickle=True).item()
                self.segment_embeddings = cached_data['embeddings']
                self.trigger_embeddings = cached_data.get('trigger_embeddings', {})
                self.tag_embeddings = cached_data.get('tag_embeddings', {})
                self.prompt_embeddings = cached_data.get('prompt_embeddings', {})
                self.response_embeddings = cached_data.get('response_embeddings', {})
                self.segments = cached_data['segments']
                
                if 'parameters' in cached_data:
                    params = cached_data['parameters']
                    self.oracle_threshold = params.get('oracle_threshold', 0.5)
                    self.failure_threshold = params.get('failure_threshold', 0.3)
                    self.distance_multiplier = params.get('distance_multiplier', 1.0)
                    logger.info("Loaded parameters from cache")
                
                logger.info("Loaded embeddings from cache")
                return True
            logger.info("No cache file found. Need to compute embeddings.")
            return False
        except Exception as e:
            logger.error(f"Error loading embeddings cache: {e}")
            if os.path.exists(cache_file):
                os.remove(cache_file)
                logger.info("Removed corrupted cache file")
            return False

    def set_parameters(self, **params):
        """Set all engine parameters."""
        # Core thresholds
        self.oracle_threshold = params.get('oracle_threshold', self.oracle_threshold)
        self.failure_threshold = params.get('failure_threshold', self.failure_threshold)
        self.distance_multiplier = params.get('distance_multiplier', self.distance_multiplier)
        
        # Phase space parameters
        self.beta = params.get('beta', self.beta)
        self.memory_weight = params.get('memory_weight', self.memory_weight)
        self.temporal_decay = params.get('temporal_decay', self.temporal_decay)
        self.resonance_bandwidth = params.get('resonance_bandwidth', self.resonance_bandwidth)
        self.phase_transition_threshold = params.get('phase_transition_threshold', 
                                                   self.phase_transition_threshold)
        self.coherence_threshold = params.get('coherence_threshold', self.coherence_threshold)
        
        logger.info(f"Parameters updated: {params}")