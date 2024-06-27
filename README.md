# pdf-summarizer-tts
# PDF Summarizer and Text-to-Speech Converter

This project provides a web-based interface for summarizing PDF documents and converting the summaries to speech. It uses state-of-the-art natural language processing models for summarization and Google's Text-to-Speech service for audio generation.

## Features

- PDF text extraction
- Automatic text summarization
- Text-to-speech conversion
- User-friendly web interface

## Prerequisites

Before you begin, ensure you have met the following requirements:

- Python 3.7 or higher
- pip (Python package manager)

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/Maheshwaran325/pdf-summarizer-tts.git
   cd pdf-summarizer-tts
   ```

2. Create a virtual environment 
    ```
    python -m venv .venv
    .venv\scripts\activate  # for windows
    ```

3. Install the required dependencies:
   ```
   pip install gradio transformers pdfplumber torch gtts
   ```

## Usage

1. Run the application:
   ```
   python app.py
   ```

2. Open your web browser and go to the URL displayed in the terminal (usually `http://127.0.0.1:7860`).

3. Use the web interface to upload a PDF file. The application will generate a summary and an audio version of the summary.

## How it Works

1. **PDF Text Extraction**: The application uses `pdfplumber` to extract text from the uploaded PDF file.

2. **Text Summarization**: It employs the BART model (`facebook/bart-large-cnn`) from the Hugging Face Transformers library to generate a concise summary of the extracted text.

3. **Text-to-Speech Conversion**: The summary is converted to speech using Google's Text-to-Speech (gTTS) service.

4. **Web Interface**: Gradio is used to create a user-friendly web interface for the application.

## Customization

- You can adjust the `max_length` and `min_length` parameters in the `summarize_pdf` function to control the length of the generated summary.
- To use a different summarization model, modify the `model_name` variable in the `summarize_pdf` function.

## Contributing

Contributions to this project are welcome. Please fork the repository and submit a pull request with your changes.
