import json
import os
from dataclasses import dataclass, asdict
from typing import Optional, List
import logging
import glob
import re
import time

log = logging.getLogger("rich")


@dataclass
class ProcessingState:
    """
    Represents the current state of PDF processing.

    Attributes:
        current_page: The page currently being processed (0-based index)
        total_pages: Total number of pages in the PDF
        chunks_processed: Number of text chunks processed so far
        last_chunk_file: Name of the last audio file created
    """

    current_page: int
    total_pages: int
    chunks_processed: int
    last_chunk_file: Optional[str] = None

    def __post_init__(self):
        """Validates state attributes after initialization"""
        if self.current_page >= self.total_pages:
            raise ValueError("Current page cannot exceed total pages")
        if self.chunks_processed < 0:
            raise ValueError("Chunks processed cannot be negative")
        if self.current_page < 0:
            raise ValueError("Current page cannot be negative")

    def to_dict(self):
        """Converts state to dictionary for JSON serialization"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        """Creates state instance from dictionary"""
        return cls(**data)


@dataclass
class ProcessingStats:
    """
    Tracks processing statistics for analysis and reporting.

    Attributes:
        start_time: Timestamp when processing started
        chunks_per_page: List tracking number of chunks for each page
        processing_times: List of processing times for each page
    """

    start_time: float
    chunks_per_page: List[int]
    processing_times: List[float]


class ProgressTracker:
    """
    Manages progress tracking and recovery for PDF to audio conversion.

    Handles saving and loading processing state, allowing for crash recovery
    and progress resumption.
    """

    def __init__(self, pdf_path: str, output_dir: str):
        """
        Initialize progress tracker.

        Args:
            pdf_path: Path to the PDF being processed
            output_dir: Directory where audio files are saved
        """
        self.pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
        self.output_dir = output_dir
        self.journal_path = os.path.join(output_dir, f"{self.pdf_name}_progress.json")
        self.stats = ProcessingStats(
            start_time=time.time(), chunks_per_page=[], processing_times=[]
        )

    def save_progress(self, state: ProcessingState):
        """
        Save current processing state to journal file with error handling.

        Args:
            state: Current processing state to save
        """
        temp_path = f"{self.journal_path}.tmp"
        try:
            # Write to temporary file first
            with open(temp_path, "w") as f:
                json.dump(state.to_dict(), f)
            # Atomic rename for safer file writing
            os.replace(temp_path, self.journal_path)
        except Exception as e:
            log.error(f"Failed to save progress: {e}")
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def load_progress(self) -> Optional[ProcessingState]:
        """
        Load previous processing state if it exists.

        Returns:
            ProcessingState if found and valid, None otherwise
        """
        if not os.path.exists(self.journal_path):
            return None

        try:
            with open(self.journal_path, "r") as f:
                data = json.load(f)
                return ProcessingState.from_dict(data)
        except (json.JSONDecodeError, FileNotFoundError):
            log.warning("Failed to load progress file, starting from beginning")
            return None

    def clear_progress(self):
        """Remove the progress journal file"""
        if os.path.exists(self.journal_path):
            os.remove(self.journal_path)

    def verify_chunk_files(self, state: ProcessingState) -> bool:
        """Verify that all chunk files up to the saved state exist."""
        if not state.last_chunk_file:
            return False

        # Get the page and chunk numbers from the last file
        match = re.match(r"page_(\d+)_chunk_(\d+)\.wav", state.last_chunk_file)
        if not match:
            return False

        last_page, last_chunk = map(int, match.groups())

        # Check all files exist up to this point
        for page in range(1, last_page + 1):
            page_files = glob.glob(
                os.path.join(self.output_dir, f"page_{page:04d}_*.wav")
            )
            if page < last_page:
                # All previous pages should have their chunks
                if not page_files:
                    return False
            else:
                # Last page should have chunks up to last_chunk
                expected_chunks = {
                    f"page_{page:04d}_chunk_{i:03d}.wav"
                    for i in range(1, last_chunk + 1)
                }
                actual_chunks = {os.path.basename(f) for f in page_files}
                if not expected_chunks.issubset(actual_chunks):
                    return False

        return True

    def find_last_valid_chunk(self) -> Optional[str]:
        """
        Find the last valid chunk file in case of corruption.

        Returns:
            str: Filename of last valid chunk, or None if none found
        """
        all_chunks = sorted(glob.glob(os.path.join(self.output_dir, "*.wav")))
        return all_chunks[-1] if all_chunks else None

    def recover_from_last_valid(self) -> Optional[ProcessingState]:
        """
        Attempt to recover from the last valid chunk file.

        Returns:
            ProcessingState: Recovered state if possible, None otherwise
        """
        last_chunk = self.find_last_valid_chunk()
        if not last_chunk:
            return None

        # Parse page number from filename (e.g., "page_0001_chunk_001.wav")
        match = re.match(r"page_(\d+)_chunk_\d+\.wav", os.path.basename(last_chunk))
        if not match:
            return None

        page_num = int(match.group(1)) - 1  # Convert to 0-based index
        return ProcessingState(
            current_page=page_num,
            total_pages=self.get_total_pages(),
            chunks_processed=self.count_existing_chunks(),
            last_chunk_file=os.path.basename(last_chunk),
        )

    def update_stats(self, chunks_in_page: int, page_time: float):
        """
        Update processing statistics.

        Args:
            chunks_in_page: Number of chunks in current page
            page_time: Time taken to process the page
        """
        self.stats.chunks_per_page.append(chunks_in_page)
        self.stats.processing_times.append(page_time)
