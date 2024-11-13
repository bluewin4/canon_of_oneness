from dataclasses import dataclass
from typing import List, Dict, Optional

@dataclass
class StorySegment:
    segment_type: str  # 'paragraph', 'memory', or 'glitch'
    segment_id: str
    content: str
    parent_paragraph: Optional[str] = None

class StoryParser:
    def __init__(self, story_file: str):
        self.story_file = story_file
        self.segments: Dict[str, StorySegment] = {}
        
    def parse(self) -> Dict[str, StorySegment]:
        current_paragraph = None
        
        with open(self.story_file, 'r') as f:
            content = f.read()
            
        lines = content.split('\n')
        current_content = []
        current_id = None
        
        for line in lines:
            line = line.strip()
            
            if not line:
                if current_id and current_content:
                    self._add_segment(current_id, current_content, current_paragraph)
                current_content = []
                continue
                
            if line.startswith(('Paragraph_', 'Memory_', 'Glitch_')):
                if current_id and current_content:
                    self._add_segment(current_id, current_content, current_paragraph)
                current_id = line
                current_content = []
                
                if line.startswith('Paragraph_'):
                    current_paragraph = line
            else:
                current_content.append(line)
                
        # Add final segment
        if current_id and current_content:
            self._add_segment(current_id, current_content, current_paragraph)
            
        return self.segments
    
    def _add_segment(self, segment_id: str, content: List[str], parent_paragraph: Optional[str]):
        segment_type = segment_id.split('_')[0].lower()
        self.segments[segment_id] = StorySegment(
            segment_type=segment_type,
            segment_id=segment_id,
            content='\n'.join(content),
            parent_paragraph=parent_paragraph if segment_type != 'paragraph' else None
        ) 