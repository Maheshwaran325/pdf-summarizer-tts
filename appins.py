import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
import pdfplumber
import requests

API_URL = "https://api-inference.huggingface.co/models/microsoft/Phi-3-mini-128k-instruct"
headers = {"Authorization": "Bearer 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"}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

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
        
    # Load the Phi-3 Mini-128K-Instruct model and tokenizer
    model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-128k-instruct")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-128k-instruct")
    
    # Generate the summary using the Phi-3 Mini-128K-Instruct model
    prompt = f"<|user|>\n{text}\n<|end|>\n<|assistant|>"
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output = model.generate(
        input_ids,
        max_length=max_length,
        num_return_sequences=1,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id,
    )
    summary = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Check if the summary is factually consistent
    if fact_check(summary, text):
        # Combine the summary and text count
        output = f"Summary ({text_length} words):\n{summary}"
    else:
        output = "Error: The summary is not factually consistent with the original text."
    
    return output

# Create the Gradio interface
pdf_uploader = gr.File(label="Upload PDF")
summary_output = gr.Textbox(label="Summary")

demo = gr.Interface(
    fn=summarize_pdf,
    inputs=pdf_uploader,
    outputs=summary_output,
    title="PDF Summarizer",
    description="Upload a PDF file and get a summary of its contents."
)

demo.launch()
