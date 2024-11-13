from src.interface.cli import GameCLI

def main():
    story_file = "data/story.txt"
    cli = GameCLI(story_file)
    cli.cmdloop()

if __name__ == "__main__":
    main() 