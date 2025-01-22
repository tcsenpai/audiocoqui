import torch
from TTS.api import TTS
import os
import dotenv

# Load environment variables from .env file
dotenv.load_dotenv()

# Determine if CUDA is available for GPU acceleration
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize the TTS model
# Using XTTS v2 model which supports multiple languages
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

# Load configuration from environment variables
SPEAKER_WAV = os.getenv("SPEAKER_WAV")  # Path to speaker voice sample
LANGUAGE = os.getenv("LANGUAGE")  # Target language for TTS


def tts_audio(text: str, output_path: str):
    """
    Converts text to speech using the XTTS v2 model.

    Args:
        text (str): The text to convert to speech
        output_path (str): Path where the output WAV file will be saved

    Note:
        Uses environment variables:
        - SPEAKER_WAV: Path to a reference audio file for voice cloning
        - LANGUAGE: Target language code (e.g., "en", "es", "fr")
    """
    tts.tts_to_file(
        text=text,
        speaker_wav=SPEAKER_WAV,
        language=LANGUAGE,
        file_path=output_path,
    )
