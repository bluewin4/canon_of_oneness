from dataclasses import dataclass
from typing import List, Dict, Optional, Set

@dataclass
class StorySegment:
    segment_type: str  # 'paragraph', 'memory', or 'glitch'
    segment_id: str
    content: str
    parent_paragraph: Optional[str] = None
    triggers: Set[str] = None  # New field for triggers

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
                
            if line.startswith(('Paragraph_', 'Memory_', 'Glitch_')):
                if current_id and current_content:
                    self._add_segment(current_id, current_content, current_paragraph, current_triggers)
                current_id = line
                current_content = []
                current_triggers = set()
                
                if line.startswith('Paragraph_'):
                    current_paragraph = line
            else:
                current_content.append(line)
                
        # Add final segment
        if current_id and current_content:
            self._add_segment(current_id, current_content, current_paragraph, current_triggers)
            
        return self.segments
    
    def _add_segment(self, segment_id: str, content: List[str], parent_paragraph: Optional[str], triggers: Set[str]):
        segment_type = segment_id.split('_')[0].lower()
        self.segments[segment_id] = StorySegment(
            segment_type=segment_type,
            segment_id=segment_id,
            content='\n'.join(content),
            parent_paragraph=parent_paragraph if segment_type != 'paragraph' else None,
            triggers=triggers
        ) 