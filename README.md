# audiocoqui

A tool to convert PDF ebooks to audiobooks using XTTS v2 TTS model by cloning a speaker voice.

## Requirements

- Python 3.10+
- pip install -r requirements.txt

## Setup

- Copy .env.example to .env and fill in the missing values
- Download a speaker voice sample and put it in the source_audio folder (any .wav file of more than 10 seconds should work)
- Put your PDF in the proper folder as specified in the .env file

## Usage

NOTE: You can and should clean the output by removing the audio_pages folder after you're done (example in `clean_output` file)

- python src/main.py

## Expected output

- A folder with all the audio pages of the PDF and their chunks if splitted.
- A final audiobook file as specified in the .env file.

## Features

- [x] Clone speaker voice
- [x] Keep sentences together
- [x] Clean text input (remove line breaks, etc.)
- [x] Split PDF into pages
- [x] Convert pages to audio
- [x] Add silence between sections
- [x] Concatenate audio files
