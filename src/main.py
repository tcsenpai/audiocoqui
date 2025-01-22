import lib.tts as tts
import PyPDF2
import os
from lib.ebook_reader import (
    split_into_chunks,
    detect_section_break,
    add_silence_between_sections,
)
import shutil
import rich_click as click
from dotenv import load_dotenv
from pydub import AudioSegment
import glob
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, Group
from rich.logging import RichHandler
import logging
import time
from lib.progress_tracker import ProgressTracker, ProcessingState
from rich.status import Status
from enum import Enum, auto
import psutil

# Initialize Rich console and configure logging with Rich handler
console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)],
)
log = logging.getLogger("rich")

SECTION_BREAK_THRESHOLD = float(os.getenv("SECTION_BREAK_THRESHOLD", 0.3))
SECTION_BREAK_SILENCE = int(os.getenv("SECTION_BREAK_SILENCE", 2000))


class ProcessingStatus(Enum):
    INITIALIZING = auto()
    PROCESSING = auto()
    PAUSED = auto()
    CONCATENATING = auto()
    COMPLETED = auto()
    ERROR = auto()


def create_layout() -> Layout:
    """
    Creates the TUI layout with three main sections:
    - Header: Shows the current file being processed
    - Progress: Displays the progress bar
    - Main content: Split between current text preview and statistics
    - Errors: Displays error messages
    """
    layout = Layout(name="root")
    layout.split(
        Layout(name="header", size=3),
        Layout(name="main", ratio=1),
    )
    layout["main"].split(
        Layout(name="progress", size=3),
        Layout(name="current_text", ratio=1),
        Layout(name="stats", size=8),
        Layout(name="errors", size=3),
    )
    return layout


def concatenate_audio_files(output_dir, pdf_path):
    """Combines all generated WAV files into a single audiobook file."""
    try:
        pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
        audio_files = sorted(glob.glob(os.path.join(output_dir, "*.wav")))

        if not audio_files:
            log.warning("No audio files found to concatenate")
            return

        # Verify all files exist and are readable
        for audio_file in audio_files:
            if not os.path.exists(audio_file):
                raise FileNotFoundError(f"Missing audio file: {audio_file}")

        # Concatenate all audio files
        combined = AudioSegment.from_wav(audio_files[0])
        for audio_file in audio_files[1:]:
            sound = AudioSegment.from_wav(audio_file)
            combined += sound

        # Ensure output directory exists
        final_output_dir = os.getenv(
            "FINAL_OUTPUT_DIR", os.path.dirname(os.path.abspath(__file__))
        )
        os.makedirs(final_output_dir, exist_ok=True)

        audiobook_path = os.path.join(final_output_dir, f"{pdf_name}_audiobook.wav")
        combined.export(audiobook_path, format="wav")
        log.info(f"Created audiobook: {audiobook_path}")

    except Exception as e:
        log.error(f"Failed to concatenate audio files: {e}")
        raise


def clean_output_directory(directory: str) -> None:
    """Safely clean the output directory."""
    try:
        if os.path.exists(directory):
            for file in glob.glob(os.path.join(directory, "*.wav")):
                try:
                    os.remove(file)
                except OSError as e:
                    log.warning(f"Failed to remove file {file}: {e}")
            try:
                os.rmdir(directory)
            except OSError:
                log.warning(f"Directory {directory} not empty or locked")
    except Exception as e:
        log.error(f"Failed to clean directory {directory}: {e}")
        raise


def create_progress_columns():
    """Create enhanced progress display"""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=None),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("[cyan]{task.completed}/{task.total}"),
        expand=True,
    )


def update_status(layout: Layout, status: ProcessingStatus, details: str = ""):
    """Update processing status display"""
    status_styles = {
        ProcessingStatus.INITIALIZING: "yellow",
        ProcessingStatus.PROCESSING: "green",
        ProcessingStatus.PAUSED: "yellow",
        ProcessingStatus.CONCATENATING: "blue",
        ProcessingStatus.COMPLETED: "green",
        ProcessingStatus.ERROR: "red",
    }

    status_text = f"[{status_styles[status]}]{status.name}[/] {details}"
    layout["header"].update(Panel(status_text, style="bold"))


def get_resource_usage():
    """Get current resource usage"""
    process = psutil.Process()
    memory_info = process.memory_info()
    return {
        "memory_percent": process.memory_percent(),
        "memory_mb": memory_info.rss / 1024 / 1024,
        "cpu_percent": process.cpu_percent(),
    }


def update_stats_display(layout: Layout, stats_data: dict):
    """Update statistics with resource usage"""
    resources = get_resource_usage()
    stats = f"""[bold green]Statistics[/bold green]
Pages processed: {stats_data['current_page']}/{stats_data['total_pages']}
Total chunks: {stats_data['total_chunks']}
Current page chunks: {stats_data['chunks_in_page']}
Time elapsed: {stats_data['elapsed']:.1f}s
Average time per page: {stats_data['avg_time']:.1f}s

[bold blue]Resource Usage[/bold blue]
Memory: {resources['memory_mb']:.1f}MB ({resources['memory_percent']:.1f}%)
CPU: {resources['cpu_percent']:.1f}%"""

    layout["stats"].update(Panel(stats))


def read_pdf_to_audio(pdf_path, output_dir="audio_pages", clean=True):
    """
    Main function to convert PDF to audio. Processes the PDF page by page,
    converts text chunks to speech, and displays progress in a TUI.

    Args:
        pdf_path: Path to the PDF file
        output_dir: Directory for output audio files
        clean: Whether to clean the output directory before starting
    """
    # Initialize progress tracker
    progress_tracker = ProgressTracker(pdf_path, output_dir)

    # Clean output and progress if requested
    if clean:
        clean_output_directory(output_dir)
        progress_tracker.clear_progress()
    os.makedirs(output_dir, exist_ok=True)

    # Check for existing progress
    saved_state = progress_tracker.load_progress()
    start_page = 0
    total_chunks = 0

    if saved_state and progress_tracker.verify_chunk_files(saved_state):
        start_page = saved_state.current_page
        total_chunks = saved_state.chunks_processed
        log.info(f"Resuming from page {start_page + 1}")

    # Initialize TUI components
    layout = create_layout()
    start_time = time.time()
    current_page = start_page
    last_stats_update = time.time()
    stats_update_interval = 1.0  # Update every second

    update_status(
        layout, ProcessingStatus.INITIALIZING, f"Loading {os.path.basename(pdf_path)}"
    )

    error_messages = []

    with open(pdf_path, "rb") as file:
        pdf_reader = PyPDF2.PdfReader(file)
        total_pages = len(pdf_reader.pages)

        # Configure progress bar
        progress = create_progress_columns()
        page_task = progress.add_task(
            "[red]Pages...", total=total_pages, start=start_page
        )
        chunk_task = progress.add_task(
            "[yellow]Chunks...", total=None, visible=False  # Will be updated per page
        )

        # Initialize TUI layout
        layout["header"].update(
            Panel(f"Processing: {os.path.basename(pdf_path)}", style="bold blue")
        )
        layout["progress"].update(
            Panel(Group(progress, "[cyan]Press Ctrl+C to pause processing"))
        )

        def update_display(
            current_text: str,
            current_chunk: int = 0,
            total_chunks: int = 0,
            is_section_break: bool = False,
            stats_data: dict = None,
        ):
            """Batch update all UI elements to reduce flickering"""
            # Update text preview
            if is_section_break:
                layout["current_text"].update(
                    Panel(
                        f"[yellow]Section break detected - adding pause[/yellow]\n{current_text[:200]}"
                    )
                )
            else:
                layout["current_text"].update(
                    Panel(
                        f"[yellow]Current chunk ({current_chunk}/{total_chunks}):[/yellow]\n{current_text[:200]}"
                    )
                )

            # Update statistics
            if stats_data:
                stats = f"""[bold green]Statistics[/bold green]
Pages processed: {stats_data['current_page']}/{stats_data['total_pages']}
Total chunks: {stats_data['total_chunks']}
Current page chunks: {stats_data['chunks_in_page']}
Time elapsed: {stats_data['elapsed']:.1f}s
Average time per page: {stats_data['avg_time']:.1f}s"""
                layout["stats"].update(Panel(stats))

        def update_error_panel(layout: Layout, error_messages: list):
            """Update error panel with latest messages"""
            if error_messages:
                error_text = "\n".join(
                    f"[red]• {msg}[/red]" for msg in error_messages[-3:]
                )
                layout["errors"].update(Panel(error_text, title="[red]Warnings/Errors"))
            else:
                layout["errors"].update(Panel("No errors", style="green"))

        # Main processing loop with live UI updates
        with Live(layout, refresh_per_second=10, screen=True):
            update_status(layout, ProcessingStatus.PROCESSING)

            try:
                for page_num in range(start_page, total_pages):
                    current_page = page_num + 1
                    page = pdf_reader.pages[page_num]
                    text = page.extract_text().strip()

                    if not text.strip():
                        error_messages.append(f"Empty text on page {page_num + 1}")
                        update_error_panel(layout, error_messages)
                        continue

                    # Check for section break
                    is_section_break = detect_section_break(
                        text, SECTION_BREAK_THRESHOLD
                    )
                    chunks = split_into_chunks(text)

                    # Update page progress
                    progress.update(page_task, advance=1)

                    # Reset chunk progress for new page
                    progress.update(
                        chunk_task, total=len(chunks), visible=True, completed=0
                    )

                    for chunk_num, chunk in enumerate(chunks):
                        # Update chunk progress
                        progress.update(chunk_task, advance=1)

                        audio_file = (
                            f"page_{page_num+1:04d}_chunk_{chunk_num+1:03d}.wav"
                        )
                        audio_path = os.path.join(output_dir, audio_file)

                        # Generate audio
                        tts.tts_audio(chunk, audio_path)
                        total_chunks += 1

                        # Save progress after each chunk
                        state = ProcessingState(
                            current_page=page_num,
                            total_pages=total_pages,
                            chunks_processed=total_chunks,
                            last_chunk_file=audio_file,
                        )
                        progress_tracker.save_progress(state)

                        # If it's the last chunk of a page with a section break,
                        # add extra silence
                        if is_section_break and chunk_num == len(chunks) - 1:
                            audio = AudioSegment.from_wav(audio_path)
                            audio = add_silence_between_sections(
                                audio, SECTION_BREAK_SILENCE
                            )
                            audio.export(audio_path, format="wav")

                        # Batch update UI
                        update_display(
                            current_text=chunk,
                            current_chunk=chunk_num + 1,
                            total_chunks=len(chunks),
                            is_section_break=is_section_break
                            and chunk_num == len(chunks) - 1,
                            stats_data={
                                "current_page": current_page,
                                "total_pages": total_pages,
                                "total_chunks": total_chunks,
                                "chunks_in_page": len(chunks),
                                "elapsed": time.time() - start_time,
                                "avg_time": (time.time() - start_time) / current_page,
                            },
                        )

                    # Update error panel periodically
                    update_error_panel(layout, error_messages)

                    current_time = time.time()
                    if current_time - last_stats_update >= stats_update_interval:
                        update_stats_display(
                            layout,
                            {
                                "current_page": current_page,
                                "total_pages": total_pages,
                                "total_chunks": total_chunks,
                                "chunks_in_page": len(chunks),
                                "elapsed": current_time - start_time,
                                "avg_time": (current_time - start_time) / current_page,
                            },
                        )
                        last_stats_update = current_time

                # Clear progress file when complete
                progress_tracker.clear_progress()

                # Final processing: combine all audio files
                layout["current_text"].update(
                    Panel("[yellow]Finalizing: Concatenating audio files...[/yellow]")
                )
                concatenate_audio_files(output_dir, pdf_path)

                # Display final statistics
                elapsed = time.time() - start_time
                final_stats = f"""[bold green]Complete![/bold green]
Total pages: {total_pages}
Total chunks: {total_chunks}
Total time: {elapsed:.1f}s
Average time per page: {elapsed/total_pages:.1f}s"""
                layout["stats"].update(Panel(final_stats))

                update_status(layout, ProcessingStatus.COMPLETED)

            except Exception as e:
                error_messages.append(str(e))
                update_error_panel(layout, error_messages)
                raise


if __name__ == "__main__":
    load_dotenv()
    click.rich_click.USE_RICH_MARKUP = True

    @click.command()
    @click.option(
        "--pdf",
        default=os.getenv("PDF_PATH", "ebook/book.pdf"),
        help="Path to the PDF file",
    )
    @click.option(
        "--clean/--no-clean",
        default=True,
        help="Clean output directory before starting",
    )
    @click.option(
        "--output-dir",
        default=os.getenv("OUTPUT_DIR", "audio_pages"),
        help="Output directory",
    )
    def main(pdf, clean, output_dir):
        """Convert PDF to audiobook using TTS"""
        read_pdf_to_audio(pdf, output_dir, clean)

    main()
