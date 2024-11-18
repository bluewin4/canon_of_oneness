from aijson import register_action
import re

@register_action
async def extract_stability_score(response: str) -> float:
    """
    Extract a stability score from the response. The score should be between 0.0 and 1.0.
    
    Args:
        response: The full response from the LLM containing analysis and a stability score
        
    Returns:
        float: Extracted stability score between 0.0 and 1.0
        
    Example response format:
        Analysis of the response...
        Multiple lines of text...
        0.7
    """
    # Find the last line that contains a number between 0 and 1
    lines = response.strip().split('\n')
    
    for line in reversed(lines):
        # Look for a floating point number between 0 and 1
        match = re.search(r'\b(1|0?\.[0-9]+)\b', line)
        if match:
            try:
                score = float(match.group(1))
                if 0 <= score <= 1:
                    return score
            except ValueError:
                continue
    
    # Default to 0.5 if no valid score is found
    return 0.5
