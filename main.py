from pathlib import Path
import sys
from dotenv import load_dotenv
import tomli
from src.story import Story
from src.cli import GameCLI

load_dotenv()


def load_story(story_file: Path) -> Story:
    """Load story from TOML file."""
    with open(story_file, 'rb') as f:
        story_data = tomli.load(f)
    return Story.model_validate(story_data)

def main():
    story_file = Path("data/story.toml")
    story = load_story(story_file)
    cli = GameCLI(story)
    cli.cmdloop()

if __name__ == "__main__":
    main()
