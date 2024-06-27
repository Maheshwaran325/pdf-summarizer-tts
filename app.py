import gradio as gr
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import pdfplumber
import torch
from gtts import gTTS
import os

def extract_text_from_pdf(pdf_file):
    try:
        with pdfplumber.open(pdf_file) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text()
        return text
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return ""

def summarize_pdf(pdf_file, max_length=500, min_length=100):
    # Extract text from the PDF file
    text = extract_text_from_pdf(pdf_file)
    
    # Get the text length
    text_length = len(text.split())
    
    # Load the summarization model and tokenizer
    model_name = "facebook/bart-large-cnn"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    # Create a summarization pipeline
    summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)
    
    # Generate the summary
    summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)[0]['summary_text']
    
    # Combine the summary and text count
    output = f"Summary ({text_length} words):\n{summary}"
    
    # Generate audio using gTTS
    audio_file = "output.mp3"
    tts = gTTS(text=summary, lang='en')
    tts.save(audio_file)
    
    return output, audio_file

# Create the Gradio interface
pdf_uploader = gr.File(label="Upload PDF")
summary_output = gr.Textbox(label="Summary")
audio_output = gr.Audio(label="Generated Audio")

demo = gr.Interface(
    fn=summarize_pdf,
    inputs=pdf_uploader,
    outputs=[summary_output, audio_output],
    title="Improved PDF Summarizer",
    description="Upload a PDF file and get a summary of its contents along with the audio version of the summary."
)

demo.launch()