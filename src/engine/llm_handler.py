from anthropic import Anthropic, APIError, APITimeoutError, RateLimitError
import os
import json
import time
import hashlib
from typing import Dict, List, Optional
from pathlib import Path
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

class LLMCacheEntry:
    def __init__(self, response: str, timestamp: float):
        self.response = response
        self.timestamp = timestamp
        
    def is_expired(self, ttl_hours: int = 24) -> bool:
        """Check if cache entry has expired."""
        return time.time() - self.timestamp > (ttl_hours * 3600)
    
    def to_dict(self) -> dict:
        return {
            "response": self.response,
            "timestamp": self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'LLMCacheEntry':
        return cls(data["response"], data["timestamp"])

class LLMHandler:
    def __init__(self, 
                 cache_dir: str = "cache",
                 cache_ttl_hours: int = 24,
                 max_retries: int = 3,
                 retry_delay: float = 1.0):
        """Initialize the LLM handler with caching and retry logic.
        
        Args:
            cache_dir: Directory to store response cache
            cache_ttl_hours: Hours before cache entries expire
            max_retries: Maximum number of API retry attempts
            retry_delay: Delay between retries in seconds
        """
        self.client = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
        self.cache_dir = Path(cache_dir)
        self.cache_ttl_hours = cache_ttl_hours
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.cache: Dict[str, LLMCacheEntry] = {}
        
        # Create cache directory if it doesn't exist
        self.cache_dir.mkdir(exist_ok=True)
        self._load_cache()
        
    def generate_response(self, 
                         current_paragraph: str,
                         player_input: str,
                         stability: float,
                         nearby_segments: List[tuple[str, float]],
                         state_history: List[str],
                         use_cache: bool = True) -> str:
        """Generate a contextual response using Claude with caching and error handling."""
        
        # Generate cache key from input parameters
        cache_key = self._generate_cache_key(
            current_paragraph, 
            player_input, 
            stability, 
            nearby_segments, 
            state_history
        )
        
        # Check cache if enabled
        if use_cache:
            cached_response = self._get_cached_response(cache_key)
            if cached_response:
                return cached_response
        
        # Prepare prompt
        prompt = self._create_prompt(
            current_paragraph,
            player_input,
            stability,
            nearby_segments,
            state_history
        )
        
        # Generate response with retry logic
        for attempt in range(self.max_retries):
            try:
                response = self.client.messages.create(
                    model="claude-3-sonnet-20240229",
                    max_tokens=1000,
                    temperature=self._calculate_temperature(stability),
                    messages=[{
                        "role": "user",
                        "content": prompt
                    }]
                )
                
                response_text = response.content[0].text
                
                # Cache successful response
                self._cache_response(cache_key, response_text)
                return response_text
                
            except RateLimitError:
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(self.retry_delay * (attempt + 1))  # Exponential backoff
                
            except APITimeoutError:
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(self.retry_delay)
                
            except APIError as e:
                # Don't retry on authentication or invalid request errors
                if e.status_code in (401, 400):
                    raise
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(self.retry_delay)
                
            except Exception as e:
                # Log unexpected errors and raise
                self._log_error(e)
                raise
    
    def _create_prompt(self, 
                      current_paragraph: str,
                      player_input: str,
                      stability: float,
                      nearby_segments: List[tuple[str, float]],
                      state_history: List[str]) -> str:
        """Create the prompt for Claude."""
        return f"""You are an interactive narrative engine. Generate a response to the player's input that maintains narrative coherence while reflecting the current stability level ({stability:.2f}).

Current narrative context:
{current_paragraph}

Relevant narrative elements (with relevance scores):
{self._format_nearby_segments(nearby_segments)}

Player input: {player_input}

Recent state history: {', '.join(state_history[-3:])}

Rules:
- If stability is high ({stability:.2f} > 0.7), maintain clear narrative coherence
- If stability is medium (0.3 < {stability:.2f} < 0.7), introduce subtle uncertainty
- If stability is low ({stability:.2f} < 0.3), incorporate narrative distortions
- Keep the response length similar to the original paragraph
- Maintain the same writing style and tone as the original
- Reference nearby narrative elements when relevant

Generate a response paragraph:"""
    
    def _generate_cache_key(self, *args) -> str:
        """Generate a unique cache key from input parameters."""
        combined_input = json.dumps(args, sort_keys=True)
        return hashlib.md5(combined_input.encode()).hexdigest()
    
    def _get_cached_response(self, cache_key: str) -> Optional[str]:
        """Get response from cache if available and not expired."""
        if cache_key in self.cache:
            entry = self.cache[cache_key]
            if not entry.is_expired(self.cache_ttl_hours):
                return entry.response
            else:
                del self.cache[cache_key]
        return None
    
    def _cache_response(self, cache_key: str, response: str) -> None:
        """Cache a response and save to disk."""
        self.cache[cache_key] = LLMCacheEntry(response, time.time())
        self._save_cache()
    
    def _load_cache(self) -> None:
        """Load cache from disk."""
        cache_file = self.cache_dir / "response_cache.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                    self.cache = {
                        k: LLMCacheEntry.from_dict(v) 
                        for k, v in data.items()
                    }
            except Exception as e:
                self._log_error(f"Error loading cache: {e}")
                self.cache = {}
    
    def _save_cache(self) -> None:
        """Save cache to disk."""
        cache_file = self.cache_dir / "response_cache.json"
        try:
            with open(cache_file, 'w') as f:
                json.dump({
                    k: v.to_dict() 
                    for k, v in self.cache.items()
                }, f)
        except Exception as e:
            self._log_error(f"Error saving cache: {e}")
    
    def _log_error(self, error: Exception) -> None:
        """Log errors to file."""
        log_file = self.cache_dir / "error.log"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            with open(log_file, 'a') as f:
                f.write(f"[{timestamp}] {str(error)}\n")
        except:
            pass  # If we can't log, continue silently
    
    def _calculate_temperature(self, stability: float) -> float:
        """Calculate LLM temperature based on stability."""
        return 0.3 + ((1 - stability) * 0.7)
    
    def _format_nearby_segments(self, 
                              segments: List[tuple[str, float]]) -> str:
        """Format nearby segments for prompt context."""
        formatted = []
        for segment_id, relevance in segments:
            formatted.append(f"- {segment_id} (relevance: {relevance:.2f})")
        return "\n".join(formatted) 