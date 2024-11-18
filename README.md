# Interactive Narrative Engine

A dynamic interactive fiction engine that uses AI embeddings to create a responsive narrative experience where player choices affect story stability and progression.

Made a new branch that has a brain-dead implementation. I'm working on an evolutionary Kodoku to get a better metric for the embedding stuff later. In the meantime just use this branch and install the requirements, make a .env with your anthropic key and run main.py

feature/simple-llm-stability is the branch
using python
3.10.15


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

## Usage

## Story File Format

Stories are structured in `story.txt` with segments marked by IDs:

## Dependencies

- sentence-transformers
- numpy
- rich (for CLI interface)
- python-dotenv
- requests

## Environment

Create a `.env` file with the following variables:
- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `DEBUG` (optional, if truthy shows debug panel with logs)
- `LOG_CONFIG` (optional, defaults to `warning`)

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
