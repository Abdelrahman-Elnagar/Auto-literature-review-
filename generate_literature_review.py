import pandas as pd
import google.generativeai as genai
import logging
import json
import argparse
from pathlib import Path
import re
import os
from typing import List, Dict, Any
from collections import defaultdict
import time
from tenacity import retry, stop_after_attempt, wait_exponential
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configure paths
WORKSPACE_ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(WORKSPACE_ROOT, "output")
DEFAULT_INPUT_CSV = os.path.join(OUTPUT_DIR, "meta_learning_related_work_clustered.csv")
DEFAULT_OUTPUT_FILE = os.path.join(OUTPUT_DIR, "literature_review.md")

# Rate limiting configuration
RATE_LIMIT_DELAY = 3  # Delay between API calls in seconds
MAX_RETRIES = 4  # Maximum number of retries for API calls
RETRY_DELAY = 10  # Delay between retries in seconds

import os
import time
import logging
import openai
from tenacity import retry, stop_after_attempt, wait_exponential

# Constants
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds
RATE_LIMIT_DELAY = 1  # seconds

# Configure logging
logger = logging.getLogger(__name__)

# Configure DeepSeek API
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
if not DEEPSEEK_API_KEY:
    raise ValueError("DEEPSEEK_API_KEY not found in environment variables. Please check your .env file.")

openai.api_key = DEEPSEEK_API_KEY
openai.api_base = "https://api.deepseek.com/v1"  # DeepSeek's OpenAI-compatible endpoint

# Model and generation configuration
MODEL_NAME = "deepseek-chat"
GENERATION_CONFIG = {
    "temperature": 0.3,
    "top_p": 0.8,
    "max_tokens": 2048,
}

@retry(stop=stop_after_attempt(MAX_RETRIES), wait=wait_exponential(multiplier=RETRY_DELAY))
def call_gemini_api(prompt, generation_config=None):
    """
    Call DeepSeek API with retry logic and rate limiting.
    """
    try:
        messages = [{"role": "user", "content": prompt}]
        config = generation_config or GENERATION_CONFIG

        response = openai.ChatCompletion.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=config.get("temperature", 0.7),
            top_p=config.get("top_p", 1.0),
            max_tokens=config.get("max_tokens", 2048)
        )
        time.sleep(RATE_LIMIT_DELAY)
        return response["choices"][0]["message"]["content"]

    except Exception as e:
        if "quota" in str(e).lower():
            logger.warning("API quota limit reached. Waiting before retry...")
            time.sleep(RETRY_DELAY)
            raise  # Retry through decorator
        raise


def ensure_output_dir():
    """Ensure the output directory exists."""
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
    except Exception as e:
        logger.error(f"Error creating output directory: {str(e)}")
        raise

def load_papers_data(csv_path: str = DEFAULT_INPUT_CSV) -> pd.DataFrame:
    """Load and validate the CSV data."""
    try:
        # Convert to absolute path if needed
        abs_path = os.path.abspath(csv_path)
        
        if not os.path.exists(abs_path):
            raise FileNotFoundError(
                f"CSV file not found at: {abs_path}\n"
                f"Please ensure 'meta_learning_related_work_clustered.csv' exists in the output folder: {OUTPUT_DIR}"
            )
        
        logger.info(f"Reading CSV file from: {abs_path}")
        df = pd.read_csv(abs_path)
        
        required_columns = [
            "Paper Title", "Authors", "Year of Publication", 
            "Meta-Learning Method Used", "Meta-Features Chosen",
            "Algorithm Selection Method", "Algorithms Considered for Selection",
            "Evaluation Metrics", "Dataset(s) Used",
            "Performance of Meta-Learning Approach", "Key Findings/Contributions",
            "Meta-Feature Generation Process", "Limitations",
            "Simple Summary", "IEEE Citation", "Cluster", "Cluster Description"
        ]
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        if df.empty:
            raise ValueError("The CSV file is empty")
            
        return df
    except pd.errors.EmptyDataError:
        logger.error(f"The CSV file is empty: {abs_path}")
        raise
    except pd.errors.ParserError as e:
        logger.error(f"Error parsing CSV file: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error loading CSV file: {str(e)}")
        raise

def group_papers_by_clusters(df: pd.DataFrame) -> Dict[str, List[Dict]]:
    """Group papers by their clusters."""
    clusters = defaultdict(list)
    
    for _, paper in df.iterrows():
        cluster_id = paper["Cluster"]
        clusters[f"cluster_{cluster_id}"].append(paper.to_dict())
    
    return clusters

def generate_cluster_analysis(cluster_id: str, cluster_data: List[Dict]) -> str:
    """Generate analysis for a specific cluster using Gemini."""
    try:
        # Create a structured representation of the cluster data
        cluster_summary = json.dumps(cluster_data, indent=2)
        
        prompt = f"""
        You are an expert researcher in meta-learning. Write a comprehensive analysis of the papers in {cluster_id}.
        
        Focus on:
        1. Common themes and patterns in the research
        2. Key methodologies and approaches
        3. Major findings and their significance
        4. Current limitations and future research directions
        
        Use proper academic writing style and include IEEE-style citations [n].
        
        Papers to analyze:
        {cluster_summary}
        
        IMPORTANT INSTRUCTIONS:
        1. Write a cohesive paragraph that flows logically
        2. Maintain academic rigor and formal tone
        3. Focus on synthesis rather than summarization
        4. Ensure all claims are supported by the paper data
        5. Use clear transitions between ideas
        6. Keep citations in IEEE format [n]
        
        Begin your analysis directly without any preamble.
        """
        
        try:
            response = call_gemini_api(prompt)
            if response and hasattr(response, 'text'):
                # Clean up the response
                analysis = response.text.strip()
                if analysis:
                    return analysis
                else:
                    raise ValueError("Empty response from Gemini API")
            else:
                raise ValueError("Invalid response format from Gemini API")
        except Exception as api_error:
            logger.error(f"Gemini API error for cluster {cluster_id}: {str(api_error)}")
            return f"Error analyzing cluster {cluster_id}: API error - {str(api_error)}"
            
    except Exception as e:
        logger.error(f"Error generating cluster analysis: {str(e)}")
        return f"Error analyzing cluster {cluster_id}: {str(e)}"

def generate_literature_review(df: pd.DataFrame) -> Dict[str, str]:
    """Generate a structured literature review from the clustered paper data."""
    sections = {}
    
    try:
        # Group papers by clusters
        clusters = group_papers_by_clusters(df)
        
        # Generate introduction
        intro_prompt = f"""
        Write an introduction for a literature review on meta-learning research.
        
        Focus on:
        1. The importance of meta-learning in machine learning
        2. The scope of this review
        3. The structure of the review
        
        Use proper academic writing style.
        """
        
        try:
            response = call_gemini_api(intro_prompt)
            if response and hasattr(response, 'text'):
                sections["introduction"] = response.text.strip()
            else:
                sections["introduction"] = "Error generating introduction."
        except Exception as e:
            logger.error(f"Error generating introduction: {str(e)}")
            sections["introduction"] = "Error generating introduction."
        
        # Generate sections for each cluster
        for cluster_id, cluster_data in clusters.items():
            if cluster_data:
                section_title = f"Cluster {cluster_id.split('_')[1]}"
                sections[cluster_id] = generate_cluster_analysis(section_title, cluster_data)
        
        # Generate conclusion
        conclusion_prompt = f"""
        Write a conclusion for a literature review on meta-learning research.
        
        Focus on:
        1. Summary of key findings
        2. Current state of the field
        3. Future research directions
        
        Use proper academic writing style.
        """
        
        try:
            response = call_gemini_api(conclusion_prompt)
            if response and hasattr(response, 'text'):
                sections["conclusion"] = response.text.strip()
            else:
                sections["conclusion"] = "Error generating conclusion."
        except Exception as e:
            logger.error(f"Error generating conclusion: {str(e)}")
            sections["conclusion"] = "Error generating conclusion."
        
        return sections
    
    except Exception as e:
        logger.error(f"Error generating literature review: {str(e)}")
        return {"error": str(e)}

def format_ieee_references(df: pd.DataFrame) -> str:
    """Format IEEE references from the paper data."""
    try:
        # Filter out papers without proper citation information
        valid_citations = df[df["IEEE Citation"] != "Not explicitly mentioned"]
        
        if valid_citations.empty:
            return "No valid citations found."
        
        # Create numbered references
        references = []
        for i, (_, paper) in enumerate(valid_citations.iterrows(), 1):
            citation = paper["IEEE Citation"]
            if citation and citation != "Not explicitly mentioned":
                references.append(f"[{i}] {citation}")
        
        return "\n\n".join(references)
    
    except Exception as e:
        logger.error(f"Error formatting IEEE references: {str(e)}")
        return f"Error formatting references: {str(e)}"

def save_literature_review(sections: Dict[str, str], references: str, output_file: str) -> None:
    """Save the literature review to a markdown file."""
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            # Write title
            f.write("# Meta-Learning Literature Review\n\n")
            
            # Write introduction
            f.write("## Introduction\n\n")
            f.write(sections.get("introduction", "Introduction not available."))
            f.write("\n\n")
            
            # Write cluster sections
            for section_id, content in sections.items():
                if section_id not in ["introduction", "conclusion"]:
                    section_title = f"Cluster {section_id.split('_')[1]}"
                    f.write(f"## {section_title}\n\n")
                    f.write(content)
                    f.write("\n\n")
            
            # Write conclusion
            f.write("## Conclusion\n\n")
            f.write(sections.get("conclusion", "Conclusion not available."))
            f.write("\n\n")
            
            # Write references
            f.write("## References\n\n")
            f.write(references)
        
        logger.info(f"Literature review saved to {output_file}")
    
    except Exception as e:
        logger.error(f"Error saving literature review: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description='Generate a literature review from meta-learning papers data')
    parser.add_argument('--input', '-i', 
                      default=DEFAULT_INPUT_CSV,
                      help='Path to the input CSV file (default: output/meta_learning_related_work_clustered.csv)')
    parser.add_argument('--output', '-o',
                      default=DEFAULT_OUTPUT_FILE,
                      help='Path to save the literature review (default: output/literature_review.md)')
    args = parser.parse_args()
    
    try:
        # Ensure output directory exists
        ensure_output_dir()
        
        # Load paper data
        logger.info("Loading paper data...")
        df = load_papers_data(args.input)
        
        # Generate literature review
        logger.info("Generating literature review...")
        sections = generate_literature_review(df)
        
        # Format references
        logger.info("Formatting references...")
        references = format_ieee_references(df)
        
        # Save the review
        logger.info("Saving literature review...")
        save_literature_review(sections, references, args.output)
        
        logger.info(f"Literature review successfully generated and saved to {args.output}")
        
    except FileNotFoundError as e:
        logger.error(f"File not found error: {str(e)}")
        logger.error("Please ensure you've run run_clustering.py first to create the clustered CSV file")
        raise
    except PermissionError as e:
        logger.error(f"Permission error: {str(e)}")
        logger.error("Please check if you have the necessary permissions to read/write the files")
        raise
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main() 