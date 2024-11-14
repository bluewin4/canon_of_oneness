from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set
import logging

logger = logging.getLogger(__name__)

@dataclass
class StorySegment:
    segment_type: str  # 'paragraph', 'memory', 'glitch', 'oracle', or 'failure'
    segment_id: str
    content: str
    parent_paragraph: Optional[str] = None
    triggers: Set[str] = field(default_factory=set)
    oracle_pair_id: Optional[str] = None  # New field to link oracle/failure pairs

class StoryParser:
    def __init__(self, story_file: str):
        self.story_file = story_file
        self.segments: Dict[str, StorySegment] = {}
        
    def parse(self) -> Dict[str, StorySegment]:
        current_paragraph = None
        current_triggers = set()
        
        with open(self.story_file, 'r') as f:
            content = f.read()
            
        lines = content.split('\n')
        current_content = []
        current_id = None
        
        for line in lines:
            line = line.strip()
            
            if not line:
                if current_id and current_content:
                    self._add_segment(current_id, current_content, current_paragraph, current_triggers)
                current_content = []
                current_triggers = set()
                continue
                
            # Parse trigger line
            if line.startswith('[TRIGGERS:'):
                triggers = line.replace('[TRIGGERS:', '').replace(']', '').strip()
                current_triggers = {t.strip().lower() for t in triggers.split(',')}
                continue
                
            if line.startswith(('Paragraph_', 'Memory_', 'Glitch_', 'Oracle_', 'Failure_')):
                if current_id and current_content:
                    self._add_segment(current_id, current_content, current_paragraph, current_triggers)
                current_id = line
                current_content = []
                current_triggers = set()
                
                if line.startswith('Paragraph_'):
                    current_paragraph = line
            else:
                # Remove quotes from oracle/failure content
                if current_id and (current_id.startswith(('Oracle_', 'Failure_'))):
                    line = line.strip('"')
                current_content.append(line)
                
        # Add final segment
        if current_id and current_content:
            self._add_segment(current_id, current_content, current_paragraph, current_triggers)
            
        # Link oracle/failure pairs after all segments are parsed
        self._link_oracle_failure_pairs()
            
        return self.segments
    
    def _add_segment(self, segment_id: str, content: List[str], parent_paragraph: Optional[str], triggers: Set[str]):
        """Add a segment to the story with debug logging."""
        segment_type = segment_id.split('_')[0].lower()
        
        # Debug logging
        print(f"Adding segment {segment_id}")
        if triggers:
            print(f"  with triggers: {triggers}")
        
        self.segments[segment_id] = StorySegment(
            segment_type=segment_type,
            segment_id=segment_id,
            content='\n'.join(content),
            parent_paragraph=parent_paragraph if segment_type not in ['paragraph', 'oracle', 'failure'] else None,
            triggers=triggers if triggers else set(),  # Ensure triggers is never None
            oracle_pair_id=None  # Will be set later for oracle/failure pairs
        )
    
    def _link_oracle_failure_pairs(self):
        """Link oracle and failure segments that form pairs."""
        for segment_id, segment in self.segments.items():
            if segment.segment_type == 'oracle':
                # Get corresponding failure ID
                try:
                    failure_id = f"Failure_{segment_id.split('Oracle_')[1]}"
                    if failure_id in self.segments:
                        # Link both segments to each other
                        segment.oracle_pair_id = failure_id
                        self.segments[failure_id].oracle_pair_id = segment_id
                        print(f"Linked oracle {segment_id} with failure {failure_id}")
                    else:
                        logger.warning(f"Failure segment {failure_id} not found for oracle {segment_id}")
                except IndexError:
                    logger.error(f"Invalid Oracle segment ID format: {segment_id}")