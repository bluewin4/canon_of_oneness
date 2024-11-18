import asyncio
import cmd
import io
from rich.console import Console
from ..engine.response_handler import ResponseHandler
from ..models.story import Story

class GameCLI(cmd.Cmd):
    intro = """
    Welcome to the Interactive Story System
    =====================================
    Type 'help' for a list of commands.
    Type your responses to progress through the story.
    Type 'quit' to exit.
    """
    prompt = "\n> "

    def __init__(self, story: Story):
        super().__init__()
        self.story = story

        self.console = Console()

        # redirect stdout to a buffer
        self.log_buffer = io.StringIO()
        self._setup_logging()
        
        # Initialize game components
        self.response_handler = ResponseHandler(
            console=self.console,
            story=self.story,
            stdout_buffer=self.log_buffer,
        )
    
    def onecmd(self, line: str) -> None:
        if line.strip() == 'next':
            self.do_next('')
        elif line.strip() in ('quit', 'EOF'):
            self.do_quit('')
        else:
            self.default(line)

    def _setup_logging(self) -> None:
        # Redirect structlog and basic logging to the buffer
        import logging
        import structlog

        # Create a logging handler that writes to the buffer
        class BufferingHandler(logging.Handler):
            def __init__(self, buffer):
                super().__init__()
                self.buffer = buffer

            def emit(self, record):
                log_entry = self.format(record)
                self.buffer.write(log_entry + '\n')

        # Set up the buffer for logging
        buffer_handler = BufferingHandler(self.log_buffer)
        buffer_handler.setFormatter(logging.Formatter('%(message)s'))

        # Configure basic logging
        logging.basicConfig(handlers=[buffer_handler])

        # Configure structlog to use the same handler
        structlog.configure(
            processors=[
                structlog.processors.JSONRenderer()
            ],
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
        # from aijson.log_config import configure_logging
        # configure_logging(pretty=False)
        structlog.get_logger().addHandler(buffer_handler)

    def default(self, line: str) -> None:
        """Handle player input."""
        if not line:
            return
        asyncio.run(self.response_handler.process_response(line))

    def do_next(self, arg: str) -> None:
        """Progress to next paragraph if conditions are met."""
        self.response_handler.next_paragraph()

    def do_quit(self, arg: str) -> bool:
        """Exit the game."""
        self.console.print("\nThank you for playing!", style="cyan bold")
        return True

    def do_help(self, arg: str) -> None:
        """Display help information."""
        self.response_handler.show_help()
