import os
import sys
import fitz  # PyMuPDF
import pandas as pd
from tqdm import tqdm
import re
import logging
from pathlib import Path
import json
from datetime import datetime
import google.generativeai as genai
from dotenv import load_dotenv
import time
from tenacity import retry, stop_after_attempt, wait_exponential
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import torch
from torch import nn
import torch.nn.functional as F
from typing import List, Dict, Any, Tuple

# Import the standalone clustering module
from paper_clustering import cluster_papers, save_cluster_analysis

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

import os
import sys
import fitz  # PyMuPDF
import pandas as pd
from tqdm import tqdm
import re
import logging
from pathlib import Path
import json
from datetime import datetime
from dotenv import load_dotenv
import time
from tenacity import retry, stop_after_attempt, wait_exponential
import requests

# Import the standalone clustering module
from paper_clustering import cluster_papers, save_cluster_analysis

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configure OpenRouter API
API_KEY ="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"
API_URL = 'https://openrouter.ai/api/v1/chat/completions'
headers = {
    'Authorization': f'Bearer {API_KEY}',
    'Content-Type': 'application/json',
    'HTTP-Referer': 'https://your-domain.com',  # Required by OpenRouter
    'X-Title': 'Research Paper Analyzer'  # Custom header for identification
}

# Model configuration
MODEL_NAME = "deepseek/deepseek-chat:free"
GENERATION_CONFIG = {
    "temperature": 0.3,
    "top_p": 0.8,
    "max_tokens": 2048,
}

# Rate limiting configuration
RATE_LIMIT_DELAY = 1  # Delay between API calls in seconds
MAX_RETRIES = 4
INITIAL_RETRY_DELAY = 5  # Initial delay before first retry (in seconds)
MAX_RETRY_DELAY = 40  # Maximum delay between retries (in seconds)

@retry(
    stop=stop_after_attempt(MAX_RETRIES),
    wait=wait_exponential(multiplier=INITIAL_RETRY_DELAY, max=MAX_RETRY_DELAY)
)
def call_deepseek_api(prompt: str) -> str:
    """
    Call DeepSeek API through OpenRouter with optimized retry logic and rate limiting.
    """
    try:
        data = {
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": prompt}],
            **GENERATION_CONFIG
        }
        
        response = requests.post(API_URL, json=data, headers=headers)
        response.raise_for_status()
        
        result = response.json()
        time.sleep(RATE_LIMIT_DELAY)
        return result['choices'][0]['message']['content']
    
    except requests.exceptions.HTTPError as http_err:
        logger.error(f"HTTP error occurred: {http_err}")
        if response.status_code == 429:
            logger.warning("Rate limit exceeded. Retrying...")
            raise
        raise
    except Exception as e:
        logger.error(f"API call failed: {str(e)}")
        raise

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text from a PDF file using PyMuPDF.
    """
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text.strip()
    except Exception as e:
        logger.error(f"Error extracting text from {pdf_path}: {str(e)}")
        return ""

def extract_paper_info(text: str, filename: str) -> dict:
    """Extract structured information from paper text using DeepSeek API."""
    extracted_info = {
        "filename": filename,
        "extraction_status": "success",
        "Paper Title": "Not explicitly mentioned",
        "Authors": "Not explicitly mentioned",
        "Year of Publication": "Not explicitly mentioned",
        "Meta-Learning Method Used": "Not explicitly mentioned",
        "Meta-Features Chosen": "Not explicitly mentioned",
        "Algorithm Selection Method": "Not explicitly mentioned",
        "Algorithms Considered for Selection": "Not explicitly mentioned",
        "Evaluation Metrics": "Not explicitly mentioned",
        "Dataset(s) Used": "Not explicitly mentioned",
        "Performance of Meta-Learning Approach": "Not explicitly mentioned",
        "Key Findings/Contributions": "Not explicitly mentioned",
        "Meta-Feature Generation Process": "Not explicitly mentioned",
        "Limitations": "Not explicitly mentioned",
        "Simple Summary": "Not explicitly mentioned",
        "IEEE Citation": "Not explicitly mentioned"
    }
    
    try:
        # Extract basic information
        basic_info_prompt = f"""
        Extract ONLY the paper's title, authors, and publication year from this text.
        Respond with JSON using this exact structure:
        {{
            "Paper Title": "exact title or 'Not explicitly mentioned'",
            "Authors": "author names or 'Not explicitly mentioned'",
            "Year of Publication": "year or 'Not explicitly mentioned'"
        }}
        Text to analyze:
        {text[:5000]}
        """
        
        basic_info_response = call_deepseek_api(basic_info_prompt)
        if basic_info_response:
            clean_text = re.sub(r'^[^{]*', '', basic_info_response)
            clean_text = re.sub(r'[^}]*$', '', clean_text)
            
            try:
                basic_info = json.loads(clean_text)
                for key in ["Paper Title", "Authors", "Year of Publication"]:
                    extracted_info[key] = basic_info.get(key, "Not explicitly mentioned")
            except json.JSONDecodeError:
                logger.error(f"JSON decode error for basic info in {filename}")

        # Process text chunks
        max_chars = 15000
        text_chunks = [text[i:i+max_chars] for i in range(0, len(text), max_chars)]
        
        for chunk_index, chunk in enumerate(text_chunks):
            logger.info(f"Processing chunk {chunk_index+1}/{len(text_chunks)} for {filename}")
            
            detail_prompt = f"""
            Analyze this research paper chunk and extract information. Respond with JSON using:
            {{
                "Meta-Learning Method Used": "description",
                "Meta-Features Chosen": "description",
                "Algorithm Selection Method": "description",
                "Algorithms Considered": "list",
                "Evaluation Metrics": "list",
                "Datasets Used": "list",
                "Performance": "description",
                "Key Contributions": "description",
                "Meta-Feature Generation": "description",
                "Limitations": "description"
            }}
            Chunk content:
            {chunk}
            """
            
            try:
                chunk_response = call_deepseek_api(detail_prompt)
                if chunk_response:
                    clean_text = re.sub(r'^[^{]*', '', chunk_response)
                    clean_text = re.sub(r'[^}]*$', '', clean_text)
                    
                    try:
                        chunk_data = json.loads(clean_text)
                        for key in chunk_data:
                            if chunk_data[key] and chunk_data[key] != "Not explicitly mentioned":
                                extracted_info[key] = '; '.join(filter(None, [
                                    extracted_info[key],
                                    chunk_data[key]
                                ])).strip('; ')
                    except json.JSONDecodeError:
                        logger.error(f"JSON error in chunk {chunk_index+1}")

            except Exception as chunk_error:
                logger.error(f"Chunk processing error: {str(chunk_error)}")

        # Generate summary
        summary_prompt = f"""
        Create a concise 3-sentence summary of this research paper:
        {json.dumps(extracted_info, indent=2)}
        Focus on key contributions and methodology.
        """
        extracted_info["Simple Summary"] = call_deepseek_api(summary_prompt)

        # Generate citation
        extracted_info["IEEE Citation"] = generate_ieee_citation(
            extracted_info["Paper Title"],
            extracted_info["Authors"],
            extracted_info["Year of Publication"]
        )

        return extracted_info

    except Exception as e:
        logger.error(f"Processing failed for {filename}: {str(e)}")
        extracted_info["extraction_status"] = "error"
        extracted_info["error_message"] = str(e)
        return extracted_info

def generate_ieee_citation(title: str, authors: str, year: str) -> str:
    """Generate IEEE citation from extracted information."""
    if all([title, authors, year]):
        author_list = [a.strip() for a in authors.split(',')]
        author_text = ', '.join(author_list[:3]) + (' et al.' if len(author_list) > 3 else '')
        return f'{author_text}, "{title.strip()}", in Proc. Meta-Learning Conf., {year}.'
    return "Not explicitly mentioned"


def process_papers(input_folder: str, output_folder: str) -> None:
    """Process all PDF papers and generate analysis."""
    try:
        os.makedirs(output_folder, exist_ok=True)
    except PermissionError:
        logger.error(f"Permission denied when creating output folder: {output_folder}")
        return
    
    pdf_files = list(Path(input_folder).glob("*.pdf"))
    if not pdf_files:
        logger.error(f"No PDF files found in {input_folder}")
        return
    
    results = []
    
    # Process each PDF file with progress tracking
    for pdf_path in tqdm(pdf_files, desc="Processing papers"):
        logger.info(f"Starting to process {pdf_path.name}")
        
        try:
            text = extract_text_from_pdf(str(pdf_path))
            if not text:
                logger.warning(f"Could not extract text from {pdf_path.name}")
                continue
            
            extracted_info = extract_paper_info(text, pdf_path.name)
            results.append(extracted_info)
            
            # Save intermediate results after each paper
            try:
                df = pd.DataFrame(results)
                output_file = os.path.join(output_folder, "meta_learning_related_work.csv")
                df.to_csv(output_file, index=False)
                logger.info(f"Saved intermediate results after processing {pdf_path.name}")
            except Exception as save_error:
                logger.error(f"Error saving intermediate results: {str(save_error)}")
            
            time.sleep(RATE_LIMIT_DELAY)
            
        except Exception as e:
            logger.error(f"Error processing {pdf_path.name}: {str(e)}")
            continue
    
    if results:
        try:
            # Create DataFrame and perform clustering
            df = pd.DataFrame(results)
            
            # Perform clustering using the standalone module
            logger.info("Performing paper clustering...")
            df_with_clusters, cluster_analysis = cluster_papers(df)
            
            # Save results
            output_file = os.path.join(output_folder, "meta_learning_related_work.csv")
            df_with_clusters.to_csv(output_file, index=False)
            
            # Save cluster analysis
            analysis_file = os.path.join(output_folder, "cluster_analysis.json")
            save_cluster_analysis(cluster_analysis, analysis_file)
            
            logger.info(f"Results saved to {output_folder}")
            logger.info(f"Cluster analysis saved to {analysis_file}")
            logger.info("Cluster visualization saved as 'paper_clusters.png'")
            
        except Exception as e:
            logger.error(f"Error in final processing: {str(e)}")
    else:
        logger.error("No papers were successfully processed")

def main():
    if len(sys.argv) != 3:
        print("Usage: python generate_table.py <input_folder> <output_folder>")
        sys.exit(1)
    
    input_folder = sys.argv[1]
    output_folder = sys.argv[2]
    
    if not os.path.exists(input_folder):
        logger.error(f"Input folder {input_folder} does not exist")
        sys.exit(1)
    
    try:
        process_papers(input_folder, output_folder)
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 