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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configure Gemini API
GEMINI_API_KEY = "AIzaSyC_HrKoNXIEi-Eu7-xQjqRfsuzQjVyF0xI"
genai.configure(api_key=GEMINI_API_KEY)

# Model and generation configuration
GENERATION_CONFIG = {
    "temperature": 0.3,  # More focused and precise
    "top_p": 0.8,
    "top_k": 40,
    "max_output_tokens": 2048,
}

SAFETY_SETTINGS = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE",
    },
]

# Initialize the model
model = genai.GenerativeModel('gemini-1.5-pro')

# Rate limiting configuration
RATE_LIMIT_DELAY = 1  # Delay between API calls in seconds (reduced from 2)
MAX_RETRIES = 3  # Maximum number of retries for API calls
INITIAL_RETRY_DELAY = 5  # Initial delay before first retry (in seconds)
MAX_RETRY_DELAY = 20  # Maximum delay between retries (in seconds)

@retry(
    stop=stop_after_attempt(MAX_RETRIES),
    wait=wait_exponential(multiplier=INITIAL_RETRY_DELAY, max=MAX_RETRY_DELAY)
)
def call_gemini_api(prompt, generation_config=None, safety_settings=None):
    """
    Call Gemini API with optimized retry logic and rate limiting.
    - Uses 1 second delay between normal calls
    - Uses exponential backoff starting at 5 seconds for retries
    - Maximum retry delay capped at 20 seconds
    """
    try:
        response = model.generate_content(
            prompt,
            generation_config=generation_config or GENERATION_CONFIG,
            safety_settings=safety_settings or SAFETY_SETTINGS
        )
        time.sleep(RATE_LIMIT_DELAY)  # Short delay between successful calls
        return response
    except Exception as e:
        if "quota" in str(e).lower():
            logger.warning(f"API quota limit reached. Retrying with exponential backoff (max {MAX_RETRY_DELAY}s)...")
            raise  # Retry through decorator with exponential backoff
        raise

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text from a PDF file using PyMuPDF.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Extracted text as a string
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

def extract_paper_info_with_gemini(text: str, filename: str) -> dict:
    """Extract structured information from paper text using Gemini API."""
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
        # First, try to extract basic information (title, authors, year)
        basic_info_prompt = f"""
        Extract ONLY the paper's title, authors, and publication year from this text.
        
        RESPOND ONLY WITH THIS EXACT JSON STRUCTURE:
        {{
            "Paper Title": "exact title or 'Not explicitly mentioned'",
            "Authors": "author names or 'Not explicitly mentioned'",
            "Year of Publication": "year or 'Not explicitly mentioned'"
        }}

        Text to analyze:
        {text[:5000]}

        DO NOT include any other text or explanation. ONLY the JSON object is allowed.
        """
        
        try:
            basic_info_response = call_gemini_api(basic_info_prompt)
            if basic_info_response and hasattr(basic_info_response, 'text'):
                # Clean and parse the response
                clean_text = basic_info_response.text.strip()
                clean_text = re.sub(r'^[^{]*', '', clean_text)  # Remove any text before {
                clean_text = re.sub(r'[^}]*$', '', clean_text)  # Remove any text after }
                
                try:
                    basic_info = json.loads(clean_text)
                    for key in ["Paper Title", "Authors", "Year of Publication"]:
                        if key in basic_info and basic_info[key] != "Not explicitly mentioned":
                            extracted_info[key] = basic_info[key]
                except json.JSONDecodeError as je:
                    logger.error(f"JSON decode error for basic info in {filename}: {str(je)}")
                    logger.error(f"Problematic JSON string: {clean_text}")
        except Exception as e:
            logger.error(f"Error extracting basic info for {filename}: {str(e)}")
        
        # Split text into smaller chunks for detailed analysis
        max_chars = 15000
        text_chunks = [text[i:i + max_chars] for i in range(0, len(text), max_chars)]
        
        # Process each chunk for detailed information
        for chunk_index, chunk in enumerate(text_chunks):
            logger.info(f"Processing chunk {chunk_index + 1}/{len(text_chunks)} for {filename}")
            
            detail_prompt = f"""
            Analyze this portion of a research paper and extract specific information.
            
            RESPOND ONLY WITH THIS EXACT JSON STRUCTURE:
            {{
                "Meta-Learning Method Used": "description or 'Not explicitly mentioned'",
                "Meta-Features Chosen": "description or 'Not explicitly mentioned'",
                "Algorithm Selection Method": "description or 'Not explicitly mentioned'",
                "Algorithms Considered for Selection": "description or 'Not explicitly mentioned'",
                "Evaluation Metrics": "description or 'Not explicitly mentioned'",
                "Dataset(s) Used": "description or 'Not explicitly mentioned'",
                "Performance of Meta-Learning Approach": "description or 'Not explicitly mentioned'",
                "Key Findings/Contributions": "description or 'Not explicitly mentioned'",
                "Meta-Feature Generation Process": "description or 'Not explicitly mentioned'",
                "Limitations": "description or 'Not explicitly mentioned'"
            }}

            Text to analyze (chunk {chunk_index + 1} of {len(text_chunks)}):
            {chunk}

            IMPORTANT:
            1. DO NOT include any other text or explanation
            2. ONLY the JSON object is allowed in your response
            3. ALL property names MUST be exactly as shown above
            4. Use double quotes for ALL strings
            5. NO trailing commas
            6. If information is not found, use "Not explicitly mentioned"
            """
            
            try:
                response = call_gemini_api(detail_prompt)
                if response and hasattr(response, 'text'):
                    # Clean and parse the response
                    clean_text = response.text.strip()
                    clean_text = re.sub(r'^[^{]*', '', clean_text)  # Remove any text before {
                    clean_text = re.sub(r'[^}]*$', '', clean_text)  # Remove any text after }
                    
                    try:
                        result = json.loads(clean_text)
                        
                        # Update extracted_info with non-empty values
                        for key, value in result.items():
                            if key in extracted_info and value != "Not explicitly mentioned":
                                if extracted_info[key] == "Not explicitly mentioned":
                                    extracted_info[key] = value
                                else:
                                    # Combine new information
                                    current_info = set(extracted_info[key].split('; '))
                                    new_info = set(value.split('; '))
                                    combined_info = current_info.union(new_info)
                                    extracted_info[key] = '; '.join(sorted(filter(None, combined_info)))
                    
                    except json.JSONDecodeError as je:
                        logger.error(f"JSON decode error for chunk {chunk_index + 1} in {filename}: {str(je)}")
                        logger.error(f"Problematic JSON string: {clean_text}")
                        continue
            
            except Exception as chunk_error:
                logger.error(f"Error processing chunk {chunk_index + 1} for {filename}: {str(chunk_error)}")
                continue
        
        # Generate a summary if we have enough information
        if any(extracted_info[key] != "Not explicitly mentioned" for key in extracted_info.keys()):
            summary_prompt = f"""
            Create a brief summary of this research paper based on these extracted details.
            
            Paper information:
            {json.dumps(extracted_info, indent=2)}
            
            RESPOND WITH A SINGLE PARAGRAPH:
            1. Focus on the main contribution, method, and key findings
            2. Keep it concise (2-3 sentences)
            3. Use clear, academic language
            4. Do not include any JSON formatting
            """
            
            try:
                summary_response = call_gemini_api(summary_prompt)
                if summary_response and hasattr(summary_response, 'text'):
                    extracted_info["Simple Summary"] = summary_response.text.strip()
            except Exception as summary_error:
                logger.error(f"Error generating summary for {filename}: {str(summary_error)}")
        
        # Generate IEEE citation if needed
        if all(extracted_info[k] != "Not explicitly mentioned" for k in ["Paper Title", "Authors", "Year of Publication"]):
            extracted_info["IEEE Citation"] = generate_ieee_citation(
                extracted_info["Paper Title"],
                extracted_info["Authors"],
                extracted_info["Year of Publication"]
            )
        
        return extracted_info
    
    except Exception as e:
        logger.error(f"Error in Gemini API processing for {filename}: {str(e)}")
        extracted_info["extraction_status"] = "error"
        extracted_info["error_message"] = str(e)
        return extracted_info

def generate_ieee_citation(title: str, authors: str, year: str) -> str:
    """Generate IEEE citation from extracted information."""
    if title != "Not explicitly mentioned" and authors != "Not explicitly mentioned" and year != "Not explicitly mentioned":
        # Clean up authors (take first 3 if many)
        author_list = [a.strip() for a in authors.split(',')]
        if len(author_list) > 3:
            author_list = author_list[:3]
            author_text = ', '.join(author_list) + ', et al.'
        else:
            author_text = authors
        
        # Clean up title
        clean_title = title.strip('"').strip("'")
        
        return f"{author_text}, \"{clean_title},\" in Meta-Learning Research, {year}."
    
    return "Not explicitly mentioned"

def create_feature_vector(paper_info: dict) -> str:
    """Create a text representation for clustering."""
    features = [
        paper_info.get("Meta-Learning Method Used", ""),
        paper_info.get("Meta-Features Chosen", ""),
        paper_info.get("Algorithm Selection Method", ""),
        paper_info.get("Algorithms Considered for Selection", ""),
        paper_info.get("Dataset(s) Used", ""),
        paper_info.get("Key Findings/Contributions", ""),
    ]
    return " ".join([str(f) for f in features if f and f != "Not explicitly mentioned"])

def cluster_papers(papers_df: pd.DataFrame, n_clusters: int = 5) -> pd.DataFrame:
    """
    Cluster papers based on their content and methods.
    
    Args:
        papers_df: DataFrame containing paper information
        n_clusters: Number of clusters to create
        
    Returns:
        DataFrame with added clustering information
    """
    # Create feature vectors for clustering
    feature_texts = [create_feature_vector(row.to_dict()) for _, row in papers_df.iterrows()]
    
    # Convert text to TF-IDF vectors
    vectorizer = TfidfVectorizer(
        max_features=1000,
        stop_words='english',
        ngram_range=(1, 2)
    )
    
    try:
        # Transform texts to TF-IDF vectors
        tfidf_matrix = vectorizer.fit_transform(feature_texts)
        
        if tfidf_matrix.shape[0] < n_clusters:
            n_clusters = max(2, tfidf_matrix.shape[0] - 1)
        
        # Perform clustering
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=10
        )
        cluster_labels = kmeans.fit_predict(tfidf_matrix)
        
        # Add cluster labels to DataFrame
        papers_df['Cluster'] = cluster_labels
        
        # Generate cluster descriptions
        cluster_descriptions = {}
        for cluster_id in range(n_clusters):
            # Get papers in this cluster
            cluster_papers = papers_df[papers_df['Cluster'] == cluster_id]
            
            # Prepare cluster summary for Gemini
            cluster_summary = {
                "papers": cluster_papers[["Paper Title", "Meta-Learning Method Used", "Key Findings/Contributions"]].to_dict('records'),
                "size": len(cluster_papers)
            }
            
            # Generate cluster description using Gemini
            description_prompt = f"""
            Analyze these related papers and provide a short theme description.

            Papers in cluster:
            {json.dumps(cluster_summary, indent=2)}

            Provide a concise (1-2 sentences) description that captures the common theme or approach among these papers.
            Focus on the shared methodological or conceptual elements.
            """
            
            try:
                response = call_gemini_api(description_prompt)
                if response and hasattr(response, 'text'):
                    cluster_descriptions[cluster_id] = response.text.strip()
                else:
                    cluster_descriptions[cluster_id] = f"Cluster {cluster_id + 1}"
            except Exception as e:
                logger.error(f"Error generating description for cluster {cluster_id}: {str(e)}")
                cluster_descriptions[cluster_id] = f"Cluster {cluster_id + 1}"
        
        # Add cluster descriptions to DataFrame
        papers_df['Cluster Description'] = papers_df['Cluster'].map(cluster_descriptions)
        
        # Visualize clusters
        try:
            # Reduce dimensions for visualization
            pca = PCA(n_components=2)
            coords = pca.fit_transform(tfidf_matrix.toarray())
            
            # Create scatter plot
            plt.figure(figsize=(12, 8))
            scatter = plt.scatter(coords[:, 0], coords[:, 1], c=cluster_labels, cmap='viridis')
            plt.title('Paper Clusters Visualization')
            plt.xlabel('First Principal Component')
            plt.ylabel('Second Principal Component')
            
            # Add legend with cluster descriptions
            legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                        markerfacecolor=scatter.cmap(scatter.norm(i)), 
                                        label=f'Cluster {i + 1}: {desc[:50]}...' if len(desc) > 50 else desc,
                                        markersize=10)
                             for i, desc in cluster_descriptions.items()]
            plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
            
            # Save plot
            plt.tight_layout()
            plt.savefig('paper_clusters.png', bbox_inches='tight', dpi=300)
            plt.close()
            
        except Exception as e:
            logger.error(f"Error creating cluster visualization: {str(e)}")
        
        return papers_df
        
    except Exception as e:
        logger.error(f"Error in clustering: {str(e)}")
        papers_df['Cluster'] = 0
        papers_df['Cluster Description'] = "Clustering failed"
        return papers_df

def analyze_clusters(papers_df: pd.DataFrame) -> dict:
    """
    Analyze the clusters to identify patterns and trends.
    
    Args:
        papers_df: DataFrame with clustering information
        
    Returns:
        Dictionary containing cluster analysis
    """
    analysis = defaultdict(dict)
    
    try:
        for cluster_id in papers_df['Cluster'].unique():
            cluster_papers = papers_df[papers_df['Cluster'] == cluster_id]
            
            # Prepare cluster analysis
            cluster_info = {
                "size": len(cluster_papers),
                "papers": cluster_papers[["Paper Title", "Authors", "Year of Publication", 
                                        "Meta-Learning Method Used", "Key Findings/Contributions"]].to_dict('records'),
                "description": cluster_papers['Cluster Description'].iloc[0]
            }
            
            # Generate detailed analysis using Gemini
            analysis_prompt = f"""
            Analyze this cluster of related meta-learning papers and provide insights.

            Cluster Information:
            {json.dumps(cluster_info, indent=2)}

            Provide a detailed analysis that covers:
            1. Common methodological approaches
            2. Shared research goals or problems addressed
            3. Evolution of ideas within this group
            4. Key contributions and findings
            5. Potential future directions

            Format your response as a cohesive academic paragraph.
            """
            
            try:
                response = call_gemini_api(analysis_prompt)
                if response and hasattr(response, 'text'):
                    analysis[f"Cluster {cluster_id}"] = {
                        "description": cluster_info["description"],
                        "size": cluster_info["size"],
                        "detailed_analysis": response.text.strip()
                    }
                else:
                    analysis[f"Cluster {cluster_id}"] = {
                        "description": cluster_info["description"],
                        "size": cluster_info["size"],
                        "detailed_analysis": "Analysis generation failed"
                    }
            except Exception as e:
                logger.error(f"Error generating analysis for cluster {cluster_id}: {str(e)}")
                analysis[f"Cluster {cluster_id}"] = {
                    "description": cluster_info["description"],
                    "size": cluster_info["size"],
                    "detailed_analysis": f"Error in analysis: {str(e)}"
                }
    
    except Exception as e:
        logger.error(f"Error in cluster analysis: {str(e)}")
        analysis["error"] = str(e)
    
    return dict(analysis)

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
            
            extracted_info = extract_paper_info_with_gemini(text, pdf_path.name)
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
            
            # Perform clustering
            logger.info("Performing paper clustering...")
            df_with_clusters = cluster_papers(df)
            
            # Generate cluster analysis
            logger.info("Analyzing clusters...")
            cluster_analysis = analyze_clusters(df_with_clusters)
            
            # Save results
            output_file = os.path.join(output_folder, "meta_learning_related_work.csv")
            df_with_clusters.to_csv(output_file, index=False)
            
            # Save cluster analysis
            analysis_file = os.path.join(output_folder, "cluster_analysis.json")
            with open(analysis_file, 'w', encoding='utf-8') as f:
                json.dump(cluster_analysis, f, indent=2)
            
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