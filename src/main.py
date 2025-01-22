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
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
from rich.logging import RichHandler
import logging
import time

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


def create_layout() -> Layout:
    """
    Creates the TUI layout with three main sections:
    - Header: Shows the current file being processed
    - Progress: Displays the progress bar
    - Main content: Split between current text preview and statistics
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
    )
    return layout


def concatenate_audio_files(output_dir, pdf_path):
    """
    Combines all generated WAV files into a single audiobook file.

    Args:
        output_dir: Directory containing the individual WAV files
        pdf_path: Path to the original PDF (used for naming the final file)
    """
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    audio_files = sorted(glob.glob(os.path.join(output_dir, "*.wav")))

    if not audio_files:
        print("No audio files found to concatenate")
        return

    # Concatenate all audio files
    combined = AudioSegment.from_wav(audio_files[0])
    for audio_file in audio_files[1:]:
        sound = AudioSegment.from_wav(audio_file)
        combined += sound

    # Export final audiobook
    script_dir = os.path.dirname(os.path.abspath(__file__))
    audiobook_path = os.path.join(
        os.getenv("FINAL_OUTPUT_DIR", script_dir), f"{pdf_name}_audiobook.wav"
    )
    combined.export(audiobook_path, format="wav")
    print(f"Created audiobook: {audiobook_path}")


def read_pdf_to_audio(pdf_path, output_dir="audio_pages", clean=True):
    """
    Main function to convert PDF to audio. Processes the PDF page by page,
    converts text chunks to speech, and displays progress in a TUI.

    Args:
        pdf_path: Path to the PDF file
        output_dir: Directory for output audio files
        clean: Whether to clean the output directory before starting
    """
    # Clean output directory if requested
    if clean and os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Initialize TUI components
    layout = create_layout()
    start_time = time.time()
    total_chunks = 0
    current_page = 0

    with open(pdf_path, "rb") as file:
        pdf_reader = PyPDF2.PdfReader(file)
        total_pages = len(pdf_reader.pages)

        # Configure progress bar
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        )
        task = progress.add_task("[red]Processing...", total=total_pages)

        # Initialize TUI layout
        layout["header"].update(
            Panel(f"Processing: {os.path.basename(pdf_path)}", style="bold blue")
        )
        layout["progress"].update(Panel(progress))

        # Main processing loop with live UI updates
        with Live(layout, refresh_per_second=10, screen=True):
            for page_num in range(total_pages):
                current_page = page_num + 1
                page = pdf_reader.pages[page_num]
                text = page.extract_text().strip()

                if text:
                    # Check for section break
                    is_section_break = detect_section_break(
                        text, SECTION_BREAK_THRESHOLD
                    )
                    chunks = split_into_chunks(text)
                    total_chunks += len(chunks)

                    for chunk_num, chunk in enumerate(chunks):
                        audio_file = os.path.join(
                            output_dir,
                            f"page_{page_num+1:04d}_chunk_{chunk_num+1:03d}.wav",
                        )

                        # Generate audio
                        tts.tts_audio(chunk, audio_file)

                        # If it's the last chunk of a page with a section break,
                        # add extra silence
                        if is_section_break and chunk_num == len(chunks) - 1:
                            audio = AudioSegment.from_wav(audio_file)
                            audio = add_silence_between_sections(
                                audio, SECTION_BREAK_SILENCE
                            )
                            audio.export(audio_file, format="wav")

                            # Update UI to show section break
                            layout["current_text"].update(
                                Panel(
                                    f"[yellow]Section break detected - adding pause[/yellow]\n{chunk[:200]}"
                                )
                            )
                        else:
                            layout["current_text"].update(
                                Panel(
                                    f"[yellow]Current chunk ({chunk_num + 1}/{len(chunks)}):[/yellow]\n{chunk[:200]}"
                                )
                            )

                        # Update statistics display
                        elapsed = time.time() - start_time
                        stats = f"""[bold green]Statistics[/bold green]
Pages processed: {current_page}/{total_pages}
Total chunks: {total_chunks}
Current page chunks: {len(chunks)}
Time elapsed: {elapsed:.1f}s
Average time per page: {elapsed/current_page:.1f}s"""
                        layout["stats"].update(Panel(stats))

                        # Update progress bar
                        progress.update(
                            task,
                            description=f"Page {current_page}/{total_pages}",
                        )

                progress.update(task, advance=1)

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
