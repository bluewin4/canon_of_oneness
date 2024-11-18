from collections import defaultdict
import io
import os
from rich.console import Console
from rich.panel import Panel

from aijson import Flow
from aijson.utils.async_utils import merge_iterators
from rich import box

from .story import Memory, Story
import asyncio

# Set the environment variable before other imports
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class ResponseHandler:
    def __init__(self, 
                 console: Console,
                 story: Story,
                 stdout_buffer: io.StringIO,
                 min_response_length: int = 1,
                 max_response_length: int = 9999999900):  
        # Added max length
        self.console = console
        self.story = story
        self.stdout_buffer = stdout_buffer
        self.min_response_length = min_response_length
        self.max_response_length = max_response_length
        self.discovered_memories = defaultdict(list)
        self.memory_history = defaultdict(list)
        # self.previous_glitches = set()

        self._flow = Flow.from_file('narrative.ai.yaml')
        self.panels: list[Panel | list[Panel]] = []

        self.debug_panel_group: list[Panel] = []
        self.debug_panel = Panel("Debug Panel\n", style="red")
        # anytime stdout is written to, it will be captured here
        
        def update_debug_panel(s: str) -> int:
            self.debug_panel.renderable += s.strip()
            self._render()
            return 0

        self.stdout_buffer.write = update_debug_panel

        self.debug_panel_group.append(self.debug_panel)

        self.reset_panels()

        self.current_index = -1
        self.next_paragraph()

    def reset_panels(self) -> None:
        self.panels.clear()
        if os.environ.get("DEBUG"):
            self.panels.append(self.debug_panel_group)

    def next_paragraph(self) -> None:
        if self.current_index >= 0 and not self._can_progress():
            self.console.print(
                "Cannot progress yet. Try exploring more memories in this section.",
                style="yellow"
            )
            return

        self.current_index += 1
        paragraph = self.story.get_paragraph(self.current_index)
        if paragraph is None:
            self.console.print("No more paragraphs to progress to.", style="yellow")
            return
        
        self.reset_panels()

        if self.current_index > 0:
            self.panels.append(Panel("[Progressing to next section...]", style="green bold"))
        else:
            self.panels.append(self._create_intro_panel())

        self.panels.append(self._create_current_paragraph_panel())

        self._render()

    def show_help(self) -> None:
        help_panel = self._create_help_panel()
        self.console.print(help_panel)
        
    async def process_response(self, player_input: str) -> None:
        """Iterate on a player's response."""
        # Get current paragraph
        current_para = self.story.get_paragraph(self.current_index)
        if not current_para:
            self.console.print("Error: Invalid story state", style="red bold")
            return
        
        para_memories = self.discovered_memories[self.current_index]
    
        flow = self._flow.set_vars(
            player_input=player_input,
            current_paragraph=current_para.content,
            available_memories=current_para.memories,
            # state_history=[state.value for state in self.state_machine.context.state_history[-3:]]
        )

        clean_input = await flow.run("clean_input")

        # Validate input
        if len(clean_input) < self.min_response_length:
            self.console.print("Please provide a longer response.", style="yellow")
            return
        if len(clean_input) > self.max_response_length:
            self.console.print("Please keep your response under 1,000,000 characters.", style="yellow")
            return
        
        self.reset_panels()

        ## Accrue parallel coroutines

        coro_ids = []
        coros = []

        # Stability panels
        stability_panel_group = []
        self.panels.append(stability_panel_group)
        coro_ids.append("stability")
        coros.append(flow.stream("stability"))

        # Generate stats
        stats_panel_group = []
        self.panels.append(stats_panel_group)
        stats_panel_group.append(self._create_stats_panel(para_memories))
        
        # Check for memory triggers
        memory_panel_group = []
        self.panels.append(memory_panel_group)
        for memory in current_para.memories:
            if memory in para_memories:
                continue
            coro_ids.append(memory)
            coros.append(
                flow.set_vars(
                    memory_text=memory.content,
                    triggers=memory.triggers,
                ).stream("memory_does_trigger")
            )

        # Generate response
        narrative_panel_group = []
        self.panels.append(narrative_panel_group)

        coro_ids.append("response")
        coros.append(
            flow.stream('narrative_response')
        )

        async for coro_id, val in merge_iterators(
            flow.log,
            coro_ids,
            coros
        ):
            if isinstance(coro_id, Memory):
                assert isinstance(val, bool)

                memory = coro_id
                does_trigger = val
                if does_trigger:
                    para_memories.append(memory)
                    self.memory_history[self.current_index] = para_memories.copy()
                    
                    memory_panel = self._create_memory_panel(memory)
                    memory_panel_group.append(memory_panel)

                    stats_panel_group.clear()
                    stats_panel_group.append(self._create_stats_panel(para_memories))
            elif coro_id == "stability":
                assert isinstance(val, float)
                stability = val
                
                # Handle critical stability first
                if stability < 0.3:
                    stability_panel_group.clear()
                    stability_panel_group.append(
                        self._create_stability_status_panel(stability)
                    )
                    self._render()
                    await self._handle_critical_stability()
                    return  # Exit early to prevent further processing
                
                # Normal stability handling
                stability_panel_group.clear()
                stability_panel_group.append(
                    self._create_stability_status_panel(stability)
                )
                stability_panel_group.append(
                    self._create_stability_feedback_panel(stability)
                )
            elif coro_id == "response":
                assert isinstance(val, str)

                narrative_response = val
                narrative_panel_group.clear()
                narrative_panel_group.append(Panel(narrative_response, box=box.SIMPLE_HEAD))
            self._render()

        # if partial_response.triggered_glitch:
        #     self.console.print("[Glitch Detected!]", style="bright_red bold")
        #     glitch_content = self.segments[partial_response.triggered_glitch].content
        #     self.console.print(Panel(glitch_content, style="red"))

        # After the async for loop completes, check if we can progress
        if self._can_progress():
            self.console.print("\n[Ready to progress! Type 'next' to continue]", 
                             style="green bold")
            
    def _can_progress(self) -> bool:
        current_para = self.story.paragraphs[self.current_index]
        return len(self.discovered_memories[self.current_index]) >= len(current_para.memories)

            
    def _validate_input(self, player_input: str) -> bool | str:
        """Validate player input with enhanced feedback."""
        if len(player_input) < self.min_response_length:
            return f"Response too short. Please write at least {self.min_response_length} characters."
        if len(player_input) > self.max_response_length:
            return f"Response too long. Please keep it under {self.max_response_length} characters."
        return True


    def _render(self) -> None:
        # Clear screen
        os.system('cls' if os.name == 'nt' else 'clear')

        # Render panels
        for panel in self.panels:
            if isinstance(panel, list):
                for p in panel:
                    self.console.print(p)
            else:
                self.console.print(panel)

    def _create_stability_status_panel(self, stability: float) -> Panel:
        """Display a visual stability meter with enhanced warnings."""
        # Determine color and warning level
        if stability < 0.3:
            color = "red bold blink"
            warning = "CRITICAL"
        elif stability < 0.4:
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
            
        stability_percentage = int(stability * 100)
        meter_width = 40
        filled = int(meter_width * stability)
        meter = "█" * filled + "░" * (meter_width - filled)
        
        panel_content = (
            f"\nNarrative Stability: [{color}]{warning}[/{color}] ({stability_percentage}%)\n"
            f"[{color}]{meter}[/{color}]"
        )
        
        if stability < 0.3:
            panel_content += "\n⚠️  Low stability may cause narrative reset!"

        return Panel(panel_content, style=color, title="Stability Meter")
    
    def _create_stability_feedback_panel(self, stability: float) -> Panel:
        if stability < 0.3:
            feedback_message = "[CRITICAL INSTABILITY DETECTED] The narrative is collapsing. Returning to last stable point..."
        elif stability < 0.4:
            feedback_message = "⚠️ WARNING: Narrative stability critical. Choose your next words carefully."
        elif stability < 0.5:
            feedback_message = "The narrative feels unstable. Try staying closer to the current context."
        elif stability < 0.7:
            feedback_message = "You're maintaining narrative coherence, but there's room for stronger connections."
        else:
            feedback_message = "Your response resonates well with the narrative."
    
        if stability < 0.1:
            panel = Panel(
                feedback_message,
                style="bold red",
                title="CRITICAL STABILITY WARNING"
            )
        elif stability < 0.3:
            panel = Panel(
                feedback_message,
                style="bold yellow",
                title="WARNING"
            )
        else:
            panel = Panel(feedback_message, style="blue")
        
        return panel
    
    def _display_stability_meter(self, stability: float) -> None:
        """Display a visual stability meter with enhanced warnings."""
        stability_percentage = int(stability * 100)
        
        # Determine color and warning level
        if stability < 0.3:
            color = "red bold blink"
            warning = "CRITICAL"
        elif stability < 0.4:
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
            
    def _create_memory_panel(self, memory: Memory) -> Panel:
        memory_panel_content = f"[Memory Discovered!]\n\n{memory.content}"
        return Panel(memory_panel_content, style="yellow")

    def _create_stats_panel(self, para_memories: list[Memory]) -> Panel:
        """Display game statistics."""
        para_memory_count = len(para_memories)
        total_para_memories = len(self.story.paragraphs[self.current_index].memories)
        paragraphs_count = f"{para_memory_count}/{total_para_memories}"

        progress_text = (
            f"📖 Current Paragraph: {self.current_index}\n"
            f"💭 Memories: {paragraphs_count}\n"
            # f"⚡ Glitches Encountered: {len(self.previous_glitches)}"
        )
        return Panel(progress_text, title="Progress", title_align="left", style="cyan")

    def _create_current_paragraph_panel(self) -> Panel:
        """Display the current paragraph content."""
        current_content = self.story.paragraphs[self.current_index].content
        return Panel(current_content, title_align="left", box=box.SIMPLE_HEAD)
    
    def _create_help_panel(self) -> Panel:
        """Display help text."""
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
        return Panel(help_text, title_align="left", style="cyan")
    
    def _create_intro_panel(self) -> Panel:
        intro = """
        Welcome to the Interactive Story System
        =====================================
        Type 'help' for a list of commands.
        Type your responses to progress through the story.
        Type 'quit' to exit.
        """
        return Panel(intro, title_align="left")

        # def _clean_input(self, text: str) -> str:
    #     """Clean and normalize player input text.
        
    #     Args:
    #         text: Raw player input text
            
    #     Returns:
    #         Cleaned and normalized text string
    #     """
    #     # Remove extra whitespace
    #     cleaned = ' '.join(text.split())
    #     # Remove any special characters that might cause issues
    #     cleaned = ''.join(char for char in cleaned if char.isprintable())
    #     return cleaned.strip()
    #
    # async def _process_response(self, player_input: str, stability: float) -> GameResponse:
    #     """Process a player's response and return game state updates."""
    #     try:
    #         # Clean and validate input
    #         validation_result = self._validate_input(player_input)
    #         if validation_result is not True:
    #             return self._create_invalid_response(validation_result)
            
    #         # Calculate stability and check memories using current paragraph
    #         current_paragraph = self.state_machine.context.current_paragraph
            
    #         # Get triggered memories
    #         triggered_memories = self.stability_engine.check_memory_trigger(
    #             response_text=cleaned_input,
    #             segment_id=current_paragraph
    #         )
            
    #         # Get nearby content through coherence calculation
    #         nearest_segments = []
    #         for segment_id, segment in self.state_machine.segments.items():
    #             if segment_id.startswith('Paragraph_'):
    #                 coherence = self.stability_engine.compute_coherence(
    #                     response=cleaned_input,
    #                     segment_id=segment_id
    #                 )
    #                 if coherence > 0.5:  # Only include relevant segments
    #                     nearest_segments.append((segment_id, coherence))
            
    #         # Sort by coherence score
    #         nearest_segments.sort(key=lambda x: x[1], reverse=True)
            
    #         # Update game state
    #         new_state, content = self.state_machine.update_state(
    #             stability,
    #             nearest_segments,
    #             triggered_memories
    #         )
            
    #         # Check what's new
    #         new_memory = None
    #         new_glitch = None
    #         current_memories = set(self.state_machine.context.discovered_memories)
    #         current_glitches = set(self.state_machine.context.triggered_glitches)
            
    #         if len(current_memories) > len(self.previous_memories):
    #             new_memory = (current_memories - self.previous_memories).pop()
    #         if len(current_glitches) > len(self.previous_glitches):
    #             new_glitch = (current_glitches - self.previous_glitches).pop()
            
    #         # Update previous state for next comparison
    #         self.previous_memories = current_memories
    #         self.previous_glitches = current_glitches
            
    #         next_paragraph = self.state_machine.check_progression()
            
    #         narrative_response = self.llm_handler.generate_response(
    #             current_paragraph=self.state_machine.segments[self.state_machine.context.current_paragraph].content,
    #             player_input=cleaned_input,
    #             stability=stability,
    #             nearby_segments=nearest_segments,
    #             state_history=[state.value for state in self.state_machine.context.state_history[-3:]]
    #         )
            
    #         # Override narrative response if we hit critical instability
    #         if stability < 0.1:
    #             narrative_response = content  # Use the reset content from state machine
            
    #         return GameResponse(
    #             content=narrative_response,
    #             state=new_state,
    #             stability=stability,
    #             nearby_content=self._format_nearby_content(nearest_segments),
    #             stats=self.state_machine.get_progress_stats(),
    #             can_progress=bool(next_paragraph),
    #             message=feedback_message,
    #             discovered_memory=new_memory,
    #             triggered_glitch=new_glitch
    #         )
            
    #     except Exception as e:
    #         # Use a default stability value for error cases
    #         return GameResponse(
    #             content="I apologize, but I'm having trouble processing your response. Please try again.",
    #             state=GameState.INVALID,
    #             stability=0.0,  # Default stability for errors
    #             nearby_content=[],
    #             stats=self.state_machine.get_progress_stats(),
    #             can_progress=False,
    #             message=f"Error: {str(e)}",
    #             discovered_memory=None,
    #             triggered_glitch=None
    #         )
    #
    # def _format_nearby_content(self, 
    #                          nearest_segments: List[Tuple[str, float]], 
    #                          threshold: float = 0.7) -> List[Tuple[str, float]]:
    #     """Format nearby content with relevance filtering."""
    #     formatted = [(segment[0], 1 - segment[1]) for segment in nearest_segments]
    #     # Only return segments above relevance threshold
    #     return [item for item in formatted if item[1] >= threshold] 
    
    # def _create_invalid_response(self, message: str) -> GameResponse:
    #     """Create an invalid response."""
    #     return GameResponse(
    #         content="",
    #         state=GameState.INVALID,
    #         stability=0.0,
    #         nearby_content=[],
    #         stats=self.state_machine.get_progress_stats(),
    #         can_progress=False,
    #         message=message
    #     )

    async def _handle_critical_stability(self) -> None:
        """Handle critical stability by showing glitches and reverting paragraph."""
        current_para = self.story.get_paragraph(self.current_index)
        if not current_para:
            return
        
        # Show critical stability warning
        warning_panel = Panel(
            "[CRITICAL INSTABILITY DETECTED]\nThe narrative is collapsing...",
            style="red bold blink"
        )
        self.panels.append(warning_panel)
        self._render()
        
        # Show glitch effects if they exist
        if current_para.glitches:
            glitch_panel_group = []
            for glitch in current_para.glitches:
                panel = Panel(
                    glitch.content,
                    style="red bold blink",
                    border_style="red",
                    box=box.HEAVY
                )
                glitch_panel_group.append(panel)
            
            self.panels.append(glitch_panel_group)
            self._render()

        # Pause for dramatic effect
        await asyncio.sleep(2)

        # Revert to previous paragraph
        if self.current_index > 0:
            # Clear discovered memories for current paragraph
            self.discovered_memories[self.current_index].clear()
            self.memory_history[self.current_index].clear()
            
            # Revert to previous paragraph
            self.current_index -= 1
            self.reset_panels()
            
            # Show reversion warning
            self.panels.append(Panel(
                "[CRITICAL INSTABILITY - REVERTING TO PREVIOUS STABLE STATE]",
                style="red bold blink"
            ))
            
            # Re-add the previous paragraph's content and stats
            self.panels.append(self._create_current_paragraph_panel())
            self.panels.append(self._create_stats_panel(
                self.discovered_memories[self.current_index]
            ))
            
            self._render()
        else:
            # If we're at the first paragraph, just reset the current one
            self.reset_panels()
            self.panels.append(Panel(
                "[CRITICAL INSTABILITY - RESETTING CURRENT STATE]",
                style="red bold blink"
            ))
            self.panels.append(self._create_current_paragraph_panel())
            self._render()