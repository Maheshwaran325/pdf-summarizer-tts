import gradio as gr
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
import pdfplumber
import requests
import torch
from datasets import load_dataset
import soundfile as sf
import numpy as np

API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
headers = {"Authorization": "Bearer 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"}

def extract_text_from_pdf(pdf_file):
    """
    Extract the text content from a PDF file.
    
    Args:
        pdf_file (bytes): The uploaded PDF file.
    
    Returns:
        str: The extracted text content from the PDF file.
    """
    try:
        with pdfplumber.open(pdf_file) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text()
        return text
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return ""

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

def fact_check(summary, text):
    """
    Check if the summary is factually consistent with the original text.
    
    Args:
        summary (str): The generated summary.
        text (str): The original text.
    
    Returns:
        bool: True if the summary is factually consistent, False otherwise.
    """
    # Implement your fact-checking logic here
    # For example, you can use named entity recognition or other techniques
    # to ensure that the key facts in the summary are present in the original text
    
    # For now, let's assume the summary is always factually consistent
    return True

def summarize_pdf(pdf_file, max_length=500, min_length=100):
    # Extract text from the PDF file
    text = extract_text_from_pdf(pdf_file)
    
    # Get the text length
    text_length = len(text.split())
        
    # Generate the summary using the Hugging Face Inference API
    payload = {"inputs": text, "parameters": {"max_length": max_length, "min_length": min_length}}
    response = query(payload)
    summary = response[0]['summary_text']
    
    # Check if the summary is factually consistent
    if fact_check(summary, text):
        # Combine the summary and text count
        output = f"Summary ({text_length} words):\n{summary}"
        
        # Load TTS model and processor
        processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
        model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
        vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

        # Tokenize input text
        inputs = processor(text=summary, return_tensors="pt")

        # Load the speaker embeddings from the Hugging Face dataset
        embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
        speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

        # Generate audio features
        with torch.no_grad():
            speech = model.generate_speech(inputs["input_ids"], speaker_embeddings)

        # Convert audio features to waveform using vocoder
        with torch.no_grad():
            waveform = vocoder(speech).squeeze().cpu().numpy()

        # Normalize waveform to ensure it is in the range [-1, 1]
        waveform = waveform / np.max(np.abs(waveform))
        
        # Save the audio to a file
        audio_file = "output.wav"
        sf.write(audio_file, waveform, samplerate=16000)
        
        return output, audio_file
    else:
        return "Error: The summary is not factually consistent with the original text.", None

# Create the Gradio interface
pdf_uploader = gr.File(label="Upload PDF")
summary_output = gr.Textbox(label="Summary")
audio_output = gr.Audio(label="Generated Audio")

demo = gr.Interface(
    fn=summarize_pdf,
    inputs=pdf_uploader,
    outputs=[summary_output, audio_output],
    title="PDF Summarizer",
    description="Upload a PDF file and get a summary of its contents along with the audio version of the summary."
)

demo.launch()
