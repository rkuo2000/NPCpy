from npcpy.llm_funcs import get_llm_response
from npcpy.data.load import load_pdf, load_image
import os
import pandas as pd
import json
from PIL import Image
import io
import numpy as np
import argparse
from typing import List, Dict, Any, Optional, Union
import time
import sys

def process_pdf(pdf_path: str, extract_images: bool = True, extract_tables: bool = False) -> Dict[str, Any]:
    """
    Process PDF file to extract text, images, and optionally tables
    
    Args:
        pdf_path: Path to the PDF file
        extract_images: Whether to extract images from PDF
        extract_tables: Whether to extract tables from PDF
        
    Returns:
        Dictionary containing extracted content
    """
    result = {"text": [], "images": [], "tables": []}
    
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file not found at {pdf_path}")
        return result
    
    try:
        pdf_df = load_pdf(pdf_path)
        
        
        if 'texts' in pdf_df.columns:
            texts = json.loads(pdf_df['texts'].iloc[0])
            for item in texts:
                result["text"].append({
                    "page": item.get('page', 0),
                    "content": item.get('content', ''),
                    "bbox": item.get('bbox', None)
                })
        
        
        if extract_images and 'images' in pdf_df.columns:
            images_data = json.loads(pdf_df['images'].iloc[0])
            temp_paths = []
            
            for idx, img_data in enumerate(images_data):
                if 'array' in img_data and 'shape' in img_data and 'dtype' in img_data:
                    shape = img_data['shape']
                    dtype = img_data['dtype']
                    img_array = np.frombuffer(img_data['array'], dtype=np.dtype(dtype))
                    img_array = img_array.reshape(shape)
                    
                    img = Image.fromarray(img_array)
                    temp_img_path = f"temp_pdf_image_{os.path.basename(pdf_path)}_{idx}.png"
                    img.save(temp_img_path)
                    
                    result["images"].append({
                        "path": temp_img_path,
                        "page": img_data.get('page', 0),
                        "bbox": img_data.get('bbox', None)
                    })
                    temp_paths.append(temp_img_path)
            
            result["temp_paths"] = temp_paths
            
        
        if extract_tables and 'tables' in pdf_df.columns:
            tables_data = json.loads(pdf_df['tables'].iloc[0])
            for table in tables_data:
                if isinstance(table, dict) and 'data' in table:
                    result["tables"].append({
                        "page": table.get('page', 0),
                        "data": table.get('data'),
                        "caption": table.get('caption', '')
                    })
    
    except Exception as e:
        print(f"Error processing PDF {pdf_path}: {e}")
    
    return result

def process_image(image_path: str) -> Optional[str]:
    """Process image file and return path if valid"""
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return None
    
    try:
        
        Image.open(image_path)
        return image_path
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

def process_csv(csv_path: str, max_rows: int = 10) -> Optional[str]:
    """Process CSV file and return sample content"""
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return None
    
    try:
        data = pd.read_csv(csv_path)
        return data.head(max_rows).to_string()
    except Exception as e:
        print(f"Error processing CSV {csv_path}: {e}")
        return None

def extract_and_analyze(
    file_paths: List[str], 
    model: str = 'gemma3:4b',
    provider: str = 'ollama',
    preprocess: bool = False,
    extract_tables: bool = False,
    output_json: bool = False,
    output_file: str = None,
) -> Dict[str, Any]:
    """
    Extract content from files and analyze using an LLM
    
    Args:
        file_paths: List of paths to files (PDFs, images, CSVs)
        model: LLM model to use
        provider: LLM provider
        preprocess: Whether to do detailed preprocessing (True) or use attachment-based approach (False)
        extract_tables: Whether to extract tables from PDFs
        output_json: Whether to ask for structured JSON output
        output_file: Optional path to save results
        
    Returns:
        Dictionary containing analysis results
    """
    start_time = time.time()
    
    if not preprocess:
        
        print(f"Using simple attachment-based approach with {len(file_paths)} files")
        format_param = "json" if output_json else None
        
        response = get_llm_response(
            'Extract and analyze content from these files. Identify key concepts, data points, and provide a comprehensive analysis.',
            model=model,
            provider=provider,
            attachments=file_paths,
            format=format_param
        )
        
        result = {
            "analysis": response['response'],
            "processing_time": time.time() - start_time,
            "file_count": len(file_paths),
            "approach": "attachment-based"
        }
        
    else:
        
        print(f"Using detailed preprocessing approach with {len(file_paths)} files")
        pdf_results = []
        image_paths = []
        csv_contents = []
        temp_files = []
        
        
        for file_path in file_paths:
            _, ext = os.path.splitext(file_path)
            ext = ext.lower()
            
            if ext == '.pdf':
                print(f"Processing PDF: {file_path}")
                pdf_result = process_pdf(file_path, extract_tables=extract_tables)
                pdf_results.append({"path": file_path, "content": pdf_result})
                
                
                if "temp_paths" in pdf_result:
                    image_paths.extend(pdf_result["temp_paths"])
                    temp_files.extend(pdf_result["temp_paths"])
                    
            elif ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']:
                print(f"Processing image: {file_path}")
                img_path = process_image(file_path)
                if img_path:
                    image_paths.append(img_path)
                    
            elif ext == '.csv':
                print(f"Processing CSV: {file_path}")
                csv_content = process_csv(file_path)
                if csv_content:
                    csv_contents.append({"path": file_path, "content": csv_content})
        
        
        prompt = "Analyze the following content extracted from multiple documents:\n\n"
        
        
        for pdf_result in pdf_results:
            pdf_path = pdf_result["path"]
            pdf_content = pdf_result["content"]
            
            if pdf_content["text"]:
                prompt += f"PDF TEXT CONTENT ({os.path.basename(pdf_path)}):\n"
                
                for i, text_item in enumerate(pdf_content["text"][:5]):
                    prompt += f"- Page {text_item['page']}: {text_item['content'][:500]}...\n"
                prompt += "\n"
            
            
            if pdf_content["tables"]:
                prompt += f"PDF TABLES ({os.path.basename(pdf_path)}):\n"
                for i, table in enumerate(pdf_content["tables"][:3]):
                    prompt += f"- Table {i+1} (Page {table['page']}): {table['caption']}\n"
                    prompt += f"{str(table['data'])[:500]}...\n"
                prompt += "\n"
        
        
        for csv_item in csv_contents:
            prompt += f"CSV DATA ({os.path.basename(csv_item['path'])}):\n"
            prompt += f"{csv_item['content']}\n\n"
        
        
        prompt += "\nPlease provide a comprehensive analysis of the content above, identifying key concepts, patterns, and insights."
        
        if output_json:
            prompt += "\nFormat your response as a JSON object with the following structure: " + \
                      '{"key_concepts": [], "data_points": [], "analysis": "", "insights": []}'
        
        
        format_param = "json" if output_json else None
        response = get_llm_response(
            prompt=prompt,
            model=model,
            provider=provider,
            images=image_paths,
            format=format_param
        )
        
        result = {
            "analysis": response['response'],
            "processing_time": time.time() - start_time,
            "file_count": len(file_paths),
            "pdf_count": len(pdf_results),
            "image_count": len(image_paths),
            "csv_count": len(csv_contents),
            "approach": "detailed-preprocessing"
        }
        
        
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except Exception as e:
                    print(f"Error removing temp file {temp_file}: {e}")
    
    
    if output_file:
        try:
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"Results saved to {output_file}")
        except Exception as e:
            print(f"Error saving results to {output_file}: {e}")
    
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OCR Pipeline for extracting and analyzing document content")
    parser.add_argument('files', nargs='+', help='Paths to files (PDFs, images, CSVs)')
    parser.add_argument('--model', default='gemma3:4b', help='LLM model to use')
    parser.add_argument('--provider', default='ollama', help='LLM provider')
    parser.add_argument('--preprocess', action='store_true', help='Use detailed preprocessing (default: attachment-based)')
    parser.add_argument('--tables', action='store_true', help='Extract tables from PDFs')
    parser.add_argument('--json', action='store_true', help='Request JSON-formatted output')
    parser.add_argument('--output', help='Save results to file')
    
    args = parser.parse_args()
    
    result = extract_and_analyze(
        file_paths=args.files,
        model=args.model,
        provider=args.provider,
        preprocess=args.preprocess,
        extract_tables=args.tables,
        output_json=args.json,
        output_file=args.output
    )
    
    print("\nAnalysis Results:")
    print(result["analysis"])
    print(f"\nProcessing completed in {result['processing_time']:.2f} seconds")
    
    
    if not sys.argv[1:]:
        print("\nRunning example with default paths:")
        pdf_path = 'test_data/yuan2004.pdf'
        image_path = 'test_data/markov_chain.png'
        csv_path = 'test_data/sample_data.csv'
        
        result = extract_and_analyze(
            file_paths=[pdf_path, image_path, csv_path],
            model='gemma:4b',
            provider='ollama',
            preprocess=False
        )
        
        print("\nExample Analysis Results:")
        print(result["analysis"])
        print(f"\nExample processing completed in {result['processing_time']:.2f} seconds")
