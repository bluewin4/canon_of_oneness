import cmd
import os
from typing import Optional
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress_bar import ProgressBar
from ..engine.response_handler import ResponseHandler
from ..engine.vector_engine import VectorEngine
from ..engine.state_machine import StateMachine
from ..story.story_parser import StoryParser

class GameCLI(cmd.Cmd):
    intro = """
    Welcome to the Interactive Story System
    =====================================
    Type 'help' for a list of commands.
    Type your responses to progress through the story.
    Type 'quit' to exit.
    """
    prompt = "\n> "

    def __init__(self, story_file: str):
        super().__init__()
        self.console = Console()
        
        # Initialize game components
        self.console.print("Initializing story system...", style="yellow")
        self.parser = StoryParser(story_file)
        self.segments = self.parser.parse()
        
        self.console.print("Loading vector engine...", style="yellow")
        self.vector_engine = VectorEngine()
        self.vector_engine.set_segments(self.segments)
        
        self.state_machine = StateMachine(self.segments)
        self.response_handler = ResponseHandler(self.vector_engine, self.state_machine)
        
        # Display initial content
        self._display_current_paragraph()

    def default(self, line: str) -> None:
        """Handle player input."""
        if not line:
            return
            
        response = self.response_handler.process_response(line)
        self._display_response(response)
        
        if response.can_progress:
            self.console.print("\n[Ready to progress! Type 'next' to continue]", 
                             style="green bold")

    def _display_response(self, response) -> None:
        """Display the game's response with formatting."""
        # Clear screen for better readability
        os.system('cls' if os.name == 'nt' else 'clear')
        
        # Display stability meter with enhanced warnings
        self._display_stability_meter(response.stability)
        
        # Display feedback message with appropriate styling
        if response.message:
            if response.stability < 0.1:
                self.console.print(Panel(
                    response.message,
                    style="bold red",
                    title="CRITICAL STABILITY WARNING"
                ))
            elif response.stability < 0.3:
                self.console.print(Panel(
                    response.message,
                    style="bold yellow",
                    title="WARNING"
                ))
            else:
                self.console.print(Panel(response.message, style="blue"))
        
        # Display main content
        self.console.print("\n" + response.content + "\n")
        
        # Show any discoveries
        if response.discovered_memory:
            self.console.print("[Memory Discovered!]", style="bright_yellow bold")
            memory_content = self.segments[response.discovered_memory].content
            self.console.print(Panel(memory_content, style="yellow"))
            
        if response.triggered_glitch:
            self.console.print("[Glitch Detected!]", style="bright_red bold")
            glitch_content = self.segments[response.triggered_glitch].content
            self.console.print(Panel(glitch_content, style="red"))
            
        # Display stats
        self._display_stats(response.stats)

    def _display_stability_meter(self, stability: float) -> None:
        """Display a visual stability meter with enhanced warnings."""
        stability_percentage = int(stability * 100)
        
        # Determine color and warning level
        if stability < 0.1:
            color = "red bold blink"
            warning = "CRITICAL"
        elif stability < 0.3:
            color = "red"
            warning = "DANGEROUS"
        elif stability < 0.5:
            color = "yellow"
            warning = "UNSTABLE"
        elif stability < 0.7:
            color = "green"
            warning = "STABLE"
        else:
            color = "bright_green"
            warning = "OPTIMAL"
        
        self.console.print("\nNarrative Stability:", style=color)
        self.console.print(f"[{color}]{warning}[/{color}] ({stability_percentage}%)")
        
        # Visual meter
        meter_width = 40
        filled = int(meter_width * stability)
        meter = "█" * filled + "░" * (meter_width - filled)
        self.console.print(f"[{color}]{meter}[/{color}]")
        
        if stability < 0.3:
            self.console.print("\n⚠️  Low stability may cause narrative reset!", 
                             style="bold yellow")

    def _display_stats(self, stats: dict) -> None:
        """Display game statistics."""
        self.console.print("\nProgress:", style="cyan")
        self.console.print(f"📖 Current Paragraph: {stats['current_paragraph']}")
        self.console.print(f"💭 Memories: {stats['discovered_memories']}/{stats['total_memories']}")
        self.console.print(f"⚡ Glitches Encountered: {stats['triggered_glitches']}")

    def _display_current_paragraph(self) -> None:
        """Display the current paragraph content."""
        current_content = self.segments[self.state_machine.context.current_paragraph].content
        self.console.print("\n" + current_content + "\n")

    def do_next(self, arg: str) -> None:
        """Progress to next paragraph if conditions are met."""
        next_paragraph = self.state_machine.check_progression()
        if next_paragraph:
            self.state_machine.advance_paragraph(next_paragraph)
            os.system('cls' if os.name == 'nt' else 'clear')
            self.console.print("[Progressing to next section...]", style="green bold")
            self._display_current_paragraph()
        else:
            self.console.print(
                "Cannot progress yet. Try exploring more memories in this section.",
                style="yellow"
            )

    def do_quit(self, arg: str) -> bool:
        """Exit the game."""
        self.console.print("\nThank you for playing!", style="cyan bold")
        return True

    def do_help(self, arg: str) -> None:
        """Display help information."""
        help_text = """
        Available Commands:
        ------------------
        - Just type your response to interact with the story
        - 'next' - Progress to next paragraph (if available)
        - 'quit' - Exit the game
        - 'help' - Show this help message
        
        Tips:
        -----
        - Watch the stability meter to gauge your responses
        - Try to discover all memories in each section
        - Pay attention to narrative cues
        """
        self.console.print(Markdown(help_text)) 