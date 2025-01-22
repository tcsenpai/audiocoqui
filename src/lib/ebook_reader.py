import re
import warnings
from pydub import AudioSegment


def clean_text(text):
    """
    Cleans and normalizes input text from PDF ebooks by handling common formatting issues.

    Args:
        text (str): Raw text input from PDF, potentially containing various formatting artifacts

    Returns:
        str: Cleaned text optimized for TTS processing
    """
    # Handle common PDF extraction artifacts
    replacements = {
        # Join hyphenated words at line breaks
        "-\n": "",
        # Fix common PDF artifacts
        "ﬁ": "fi",  # Common ligature
        "ﬀ": "ff",  # Common ligature
        "…": "...",  # Ellipsis
        "–": "-",  # En dash
        "—": "-",  # Em dash
        """: '"',    # Smart quotes
        """: '"',
        "'": "'",
        "'": "'",
        # Add more replacements as needed
    }

    # Apply all replacements
    for old, new in replacements.items():
        text = text.replace(old, new)

    # Handle multiple spaces and line breaks
    text = " ".join(text.split())

    # Remove any remaining control characters
    text = "".join(char for char in text if char.isprintable() or char.isspace())

    # Optional: Handle common abbreviations for better TTS
    text = text.replace(" Dr.", " Doctor")
    text = text.replace(" Mr.", " Mister")
    text = text.replace(" Mrs.", " Misses")
    text = text.replace(" vs.", " versus")

    return text.strip()


def split_into_chunks(text, chunk_size=200, max_size=250):
    """
    Split text into chunks while trying to preserve sentence boundaries.
    Will only split sentences if absolutely necessary to meet max_size constraint.

    Args:
        text (str): Input text to be split
        chunk_size (int): Target size for chunks (default: 200)
        max_size (int): Maximum allowed size for any chunk (default: 250)

    Returns:
        list: List of text chunks, each trying to respect sentence boundaries

    Note:
        - Attempts to keep sentences together if under max_size
        - Falls back to word splitting if a sentence exceeds max_size
        - Issues warning when forced to split mid-sentence
    """
    text = clean_text(text)
    chunks = []

    # Split into sentences using regex that handles common abbreviations
    sentence_pattern = r"[.!?]+[\s]+|[.!?]+$"
    sentences = re.split(sentence_pattern, text)
    sentences = [s.strip() for s in sentences if s.strip()]

    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence_len = len(sentence) + 1  # +1 for the space/period

        # Handle sentences that exceed max_size
        if sentence_len > max_size:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_length = 0

            warnings.warn(
                f"Found sentence longer than {max_size} characters. Splitting mid-sentence."
            )
            # Fall back to word-by-word splitting for this sentence
            words = sentence.split()
            temp_chunk = []
            temp_length = 0

            for word in words:
                word_len = len(word) + 1
                if temp_length + word_len > max_size and temp_chunk:
                    chunks.append(" ".join(temp_chunk))
                    temp_chunk = []
                    temp_length = 0
                temp_chunk.append(word)
                temp_length += word_len

            if temp_chunk:
                chunks.append(" ".join(temp_chunk))
            continue

        # Try to keep sentences together if possible
        if current_length + sentence_len > chunk_size:
            # Check if exceeding max_size
            if current_length + sentence_len > max_size:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_length = 0

        current_chunk.append(sentence)
        current_length += sentence_len

        # Create new chunk if we're over the target size
        if current_length >= chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_length = 0

    # Add any remaining text as the final chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def detect_section_break(text, threshold=0.3):
    """
    Detects if a page ends with significant whitespace, indicating a section break.

    Args:
        text (str): The text from the current page
        threshold (float): Ratio of trailing whitespace to consider as section break (0.0-1.0)

    Returns:
        bool: True if a section break is detected
    """
    # Count trailing newlines/whitespace at end of page
    trailing_space = len(text) - len(text.rstrip())
    text_length = len(text)

    if text_length == 0:
        return False

    # Calculate ratio of trailing whitespace
    whitespace_ratio = trailing_space / text_length
    return whitespace_ratio > threshold


def add_silence_between_sections(audio_segment, silence_duration=2000):
    """
    Adds silence between sections based on detected breaks.

    Args:
        audio_segment (AudioSegment): The audio segment to process
        silence_duration (int): Duration of silence in milliseconds (default: 2000ms)

    Returns:
        AudioSegment: Audio with appropriate silences added
    """
    # Create silence segment
    silence = AudioSegment.silent(duration=silence_duration)
    return audio_segment + silence
