from pydantic import BaseModel


class Memory(BaseModel):
    content: str
    triggers: list[str]
    oracles: list[str] = []
    failures: list[str] = []

class Glitch(BaseModel):
    content: str

class Paragraph(BaseModel):
    content: str
    memories: list[Memory] = []
    glitches: list[Glitch] = []

class Story(BaseModel):
    paragraphs: list[Paragraph]
    
    def get_paragraph(self, index: int) -> Paragraph | None:
        """Get paragraph at index."""
        if 0 <= index < len(self.paragraphs):
            return self.paragraphs[index]
        return None
    
    def get_memories_for_paragraph(self, index: int) -> list[Memory]:
        """Get all memories for a specific paragraph."""
        paragraph = self.get_paragraph(index)
        return paragraph.memories if paragraph else []
    
    def get_glitches_for_paragraph(self, index: int) -> list[str]:
        """Get all glitches for a specific paragraph."""
        paragraph = self.get_paragraph(index)
        return paragraph.glitches if paragraph else []
    
    def get_all_memories(self) -> list[Memory]:
        """Get all memories across all paragraphs."""
        return [
            memory 
            for para in self.paragraphs 
            for memory in para.memories
        ]
    
    def total_paragraphs(self) -> int:
        """Get total number of paragraphs."""
        return len(self.paragraphs)
