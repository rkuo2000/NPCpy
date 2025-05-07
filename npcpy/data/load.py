import fitz  # PyMuPDF
import pandas as pd
import json
import io
from PIL import Image
import numpy as np
from typing import Optional

import os

def load_csv(file_path):
    df = pd.read_csv(file_path)
    return df


def load_json(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    return df


def load_txt(file_path):
    with open(file_path, "r") as f:
        text = f.read()
    df = pd.DataFrame({"text": [text]})
    return df


def load_excel(file_path):
    df = pd.read_excel(file_path)
    return df


def load_image(file_path):
    img = Image.open(file_path)
    img_array = np.array(img)
    df = pd.DataFrame(
        {
            "image_array": [img_array.tobytes()],
            "shape": [img_array.shape],
            "dtype": [img_array.dtype.str],
        }
    )
    return df


def load_pdf(file_path):
    pdf_document = fitz.open(file_path)
    texts = []
    images = []

    for page_num, page in enumerate(pdf_document):
        # Extract text
        text = page.get_text()
        texts.append({"page": page_num + 1, "content": text})

        # Extract images
        image_list = page.get_images(full=True)
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = pdf_document.extract_image(xref)
            image_bytes = base_image["image"]

            # Convert image to numpy array
            image = Image.open(io.BytesIO(image_bytes))
            img_array = np.array(image)

            images.append(
                {
                    "page": page_num + 1,
                    "index": img_index + 1,
                    "array": img_array.tobytes(),
                    "shape": img_array.shape,
                    "dtype": str(img_array.dtype),
                }
            )

    # Create DataFrame
    df = pd.DataFrame(
        {"texts": json.dumps(texts), "images": json.dumps(images)}, index=[0]
    )

    return df


extension_map = {
    "PNG": "images",
    "JPG": "images",
    "JPEG": "images",
    "GIF": "images",
    "SVG": "images",
    "MP4": "videos",
    "AVI": "videos",
    "MOV": "videos",
    "WMV": "videos",
    "MPG": "videos",
    "MPEG": "videos",
    "DOC": "documents",
    "DOCX": "documents",
    "PDF": "documents",
    "PPT": "documents",
    "PPTX": "documents",
    "XLS": "documents",
    "XLSX": "documents",
    "TXT": "documents",
    "CSV": "documents",
    "ZIP": "archives",
    "RAR": "archives",
    "7Z": "archives",
    "TAR": "archives",
    "GZ": "archives",
    "BZ2": "archives",
    "ISO": "archives",
}


def load_file_contents(file_path, chunk_size=250):
    """
    Load and format the contents of a file based on its extension.
    Returns a list of chunks from the file content.
    """
    file_ext = os.path.splitext(file_path)[1].upper().lstrip('.')
    chunks = []
    
    try:
        if file_ext == 'PDF':
            # Load PDF content
            pdf_document = fitz.open(file_path)
            full_text = ""
            
            # Extract text from each page
            for page in pdf_document:
                full_text += page.get_text() + "\n\n"
            
            # Chunk the text
            for i in range(0, len(full_text), chunk_size):
                chunk = full_text[i:i+chunk_size].strip()
                if chunk:  # Skip empty chunks
                    chunks.append(chunk)
                    
        elif file_ext == 'CSV':
            df = pd.read_csv(file_path)
            # Add metadata as first chunk
            meta = f"CSV Columns: {', '.join(df.columns)}\nRows: {len(df)}"
            chunks.append(meta)
            
            # Convert sample data to string and chunk it
            sample = df.head(20).to_string()
            for i in range(0, len(sample), chunk_size):
                chunk = sample[i:i+chunk_size].strip()
                if chunk:
                    chunks.append(chunk)
                    
        elif file_ext in ['XLS', 'XLSX']:
            df = pd.read_excel(file_path)
            # Add metadata as first chunk
            meta = f"Excel Columns: {', '.join(df.columns)}\nRows: {len(df)}"
            chunks.append(meta)
            
            # Convert sample data to string and chunk it
            sample = df.head(20).to_string()
            for i in range(0, len(sample), chunk_size):
                chunk = sample[i:i+chunk_size].strip()
                if chunk:
                    chunks.append(chunk)
                    
        elif file_ext == 'TXT':
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Chunk the text
            for i in range(0, len(content), chunk_size):
                chunk = content[i:i+chunk_size].strip()
                if chunk:
                    chunks.append(chunk)
                    
        elif file_ext == 'JSON':
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            content = json.dumps(data, indent=2)
            
            # Chunk the JSON
            for i in range(0, len(content), chunk_size):
                chunk = content[i:i+chunk_size].strip()
                if chunk:
                    chunks.append(chunk)
                    
        else:
            chunks.append(f"Unsupported file format: {file_ext}")
            
        return chunks
            
    except Exception as e:
        return [f"Error loading file {file_path}: {str(e)}"]