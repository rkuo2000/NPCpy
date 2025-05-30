from npcpy.llm_funcs import get_llm_response

response = get_llm_response(
    prompt="What is machine learning?",
    model="llama3.2",
    provider="ollama"
)

print("Response:", response['response'])



from npcpy.llm_funcs import get_llm_response

response = get_llm_response(
    prompt="Describe what you see in this image.",
    model="gemma3:4b",
    provider="ollama",
    images=["test_data/markov_chain.png"]
)

print("Response:", response['response'])



from npcpy.llm_funcs import get_llm_response

response = get_llm_response(
    prompt="Summarize the key points in this document.",
    model="llava:latest",
    provider="ollama",
    attachments=["test_data/yuan2004.pdf"]
)

print("Response:", response['response'])



from npcpy.llm_funcs import get_llm_response

response = get_llm_response(
    prompt="Extract data from these files and highlight the most important information.",
    model="llava:7b",
    provider="ollama",
    attachments=["test_data/yuan2004.pdf", "test_data/markov_chain.png", "test_data/sample_data.csv"]
)

print("Response:", response['response'])




from npcpy.llm_funcs import get_llm_response
from npcpy.npc_compiler import NPC

# Create a simple NPC with custom system message
npc = NPC(
    name="OCR_Assistant",
    description="An assistant specialized in document processing and OCR.",
    model="llava:7b",
    provider="ollama",
    directive="You are an expert at analyzing documents and extracting valuable information."
)

response = get_llm_response(
    prompt="What do you see in this diagram?",
    images=["test_data/markov_chain.png"],
    npc=npc
)

print("Response:", response['response'])
print("System message used:", response['messages'][0]['content'] if response['messages'] else "No system message")


from npcpy.llm_funcs import get_llm_response

# Create a conversation with history
messages = [
    {"role": "system", "content": "You are a document analysis assistant."},
    {"role": "user", "content": "I have some engineering diagrams I need to analyze."},
    {"role": "assistant", "content": "I'd be happy to help analyze your engineering diagrams. Please share them with me."}
]

response = get_llm_response(
    prompt="Here's the diagram I mentioned earlier.",
    model="llava:7b",
    provider="ollama",
    messages=messages,
    attachments=["test_data/markov_chain.png"]
)

print("Response:", response['response'])
print("Updated message history length:", len(response['messages']))



from npcpy.llm_funcs import get_llm_response

# Create a conversation with history
messages = [
    {"role": "system", "content": "You are a document analysis assistant."},
    {"role": "user", "content": "I have some engineering diagrams I need to analyze."},
    {"role": "assistant", "content": "I'd be happy to help analyze your engineering diagrams. Please share them with me."}
]

response = get_llm_response(
    prompt="Here's the diagram I mentioned earlier.",
    model="llava:7b",
    provider="ollama",
    messages=messages,
    attachments=["test_data/markov_chain.png"]
)

print("Response:", response['response'])
print("Updated message history length:", len(response['messages']))



from npcpy.llm_funcs import get_llm_response

response = get_llm_response(
    prompt="Analyze this image and give a detailed explanation.",
    model="llava:7b",
    provider="ollama",
    images=["test_data/markov_chain.png"],
    stream=True
)

# For streaming responses, you'd typically iterate through them
print("Streaming response object type:", type(response['response']))


from npcpy.llm_funcs import get_llm_response
from npcpy.data.load import load_pdf, load_image
import os
import pandas as pd
import json
from PIL import Image
import io
import numpy as np

# Example paths
pdf_path = 'test_data/yuan2004.pdf'
image_path = 'test_data/markov_chain.png'
csv_path = 'test_data/sample_data.csv'

# Method 1: Simple attachment-based approach
response = get_llm_response(
    'Extract and analyze all text and images from these files. What are the key concepts presented?', 
    model='llava:7b', 
    provider='ollama',
    attachments=[pdf_path, image_path, csv_path]
)

print("\nResponse from attachment-based approach:")
print(response['response'])