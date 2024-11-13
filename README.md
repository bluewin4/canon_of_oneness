# Interactive Narrative Engine

A dynamic interactive fiction engine that uses AI embeddings to create a responsive narrative experience where player choices affect story stability and progression.

## Overview

This system creates an interactive narrative experience where:
- Player responses influence the story's stability
- Hidden memories can be discovered through contextually relevant responses
- Narrative glitches may occur during unstable states
- Progress is gated by memory discovery

## Core Mechanics

### Narrative Stability
- Each player response is analyzed for its semantic relevance to the current context
- Stability ranges from 0-1 (0% to 100%)
- Different stability levels trigger different states:
  - 0-10%: Critical instability (resets current paragraph progress)
  - 10-30%: Dangerous (may trigger glitches)
  - 30-50%: Unstable
  - 50-70%: Stable
  - 70-100%: Optimal

### Memory Discovery
- Each story section contains hidden memories
- Players can discover these through contextually relevant responses
- All memories in a section must be found to progress
- Discovered memories persist even if stability drops

### Glitch System
- Low stability (below 30%) may trigger narrative glitches
- Glitches provide alternate story perspectives
- Glitches don't block progression but add depth to the narrative

### Progression System
- Story is divided into paragraphs
- Each paragraph has:
  - Main narrative content
  - Hidden memories to discover
  - Potential glitches
- Progress to next paragraph requires:
  - Discovering all memories in current section
  - Using the 'next' command when available

## Technical Architecture

### Vector Engine (`VectorEngine`)
- Uses sentence transformers for semantic analysis
- Embeds story segments and player responses
- Calculates narrative stability
- Detects memory triggers through semantic proximity

### State Machine (`StateMachine`)
- Manages game state transitions
- Tracks discovered memories and glitches
- Handles paragraph progression
- Manages narrative stability effects

### Response Handler (`ResponseHandler`)
- Processes player input
- Coordinates between Vector Engine and State Machine
- Generates appropriate feedback
- Manages narrative coherence

### Story Parser (`StoryParser`)
- Parses story content from text files
- Organizes content into segments:
  - Main paragraphs
  - Hidden memories
  - Potential glitches

## Usage

## Story File Format

Stories are structured in `story.txt` with segments marked by IDs:

## Dependencies

- sentence-transformers
- numpy
- rich (for CLI interface)
- python-dotenv
- requests

## Installation

1. Clone the repository
2. Install dependencies using either:
   ```bash
   pip install -r requirements.txt
   ```
   OR
   ```bash
   conda env create -f environment.yml
   ```

### Running the Game

1. Start the game by running:
```bash
python main.py
```

2. Basic commands:
- Type responses naturally to interact with the story
- Use 'next' to progress when available
- Use 'quit' to exit
- Use 'help' for command list

3. Interface elements:
- Stability meter shows narrative coherence
- Progress stats track:
  - Current paragraph
  - Discovered memories
  - Encountered glitches
- Warning messages indicate stability issues