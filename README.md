# Voice-Enabled AI Chatbot

## About
A Python chatbot prototype that supports text and voice interaction.

## Technologies
- Python
- OpenAI Whisper for speech-to-text
- Ollama with Llama 3.2 (3B) for local response generation
- pyttsx3 for text-to-speech

## How it works
1. The user enters text or provides voice input.
2. Whisper transcribes audio into text.
3. Ollama generates a response using the local language model.
4. pyttsx3 speaks the response aloud.

## Setup
1. Install Python.
2. Install the Python dependencies listed in requirements.txt.
3. Install and configure Ollama and download the required model.
4. Run the main Python file.

## Current limitations
This is a learning project. Speech transcription accuracy and audio dependency setup may require further improvement.
