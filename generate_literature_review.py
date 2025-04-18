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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configure paths
WORKSPACE_ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(WORKSPACE_ROOT, "output")
DEFAULT_INPUT_CSV = os.path.join(OUTPUT_DIR, "meta_learning_related_work.csv")
DEFAULT_OUTPUT_FILE = os.path.join(OUTPUT_DIR, "literature_review.md")

# Rate limiting configuration
RATE_LIMIT_DELAY = 2  # Delay between API calls in seconds
MAX_RETRIES = 3  # Maximum number of retries for API calls
RETRY_DELAY = 60  # Delay between retries in seconds

# Configure Gemini API
GEMINI_API_KEY = "AIzaSyC_HrKoNXIEi-Eu7-xQjqRfsuzQjVyF0xI"

@retry(stop=stop_after_attempt(MAX_RETRIES), wait=wait_exponential(multiplier=RETRY_DELAY))
def call_gemini_api(prompt, generation_config, safety_settings):
    """
    Call Gemini API with retry logic and rate limiting.
    """
    try:
        response = model.generate_content(
            prompt,
            generation_config=generation_config,
            safety_settings=safety_settings
        )
        time.sleep(RATE_LIMIT_DELAY)  # Rate limiting delay
        return response
    except Exception as e:
        if "quota" in str(e).lower():
            logger.warning("API quota limit reached. Waiting before retry...")
            time.sleep(RETRY_DELAY)  # Wait longer for quota issues
            raise  # Retry through decorator
        raise

try:
    genai.configure(api_key=GEMINI_API_KEY)
    
    # Initialize the model with specific configuration
    model = genai.GenerativeModel('gemini-1.5-pro')
    
    # Define generation config for better academic content
    GENERATION_CONFIG = {
        "temperature": 0.3,  # More focused and precise
        "top_p": 0.8,
        "top_k": 40,
        "max_output_tokens": 2048,  # Maximum output size
    }
    
    # Define safety settings for academic content
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
    
    logger.info("Successfully initialized Gemini API")
    
    # Test the model with configuration
    test_response = call_gemini_api(
        "Test connection",
        GENERATION_CONFIG,
        SAFETY_SETTINGS
    )
    if not test_response or not hasattr(test_response, 'text'):
        raise Exception("Failed to get valid response from Gemini API")
    logger.info("Successfully tested Gemini API connection")
    
except Exception as e:
    logger.error(f"Error configuring Gemini API: {str(e)}")
    logger.error("Please check your API key and internet connection")
    raise

def ensure_output_dir():
    """Ensure output directory exists."""
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        logger.info(f"Using output directory: {OUTPUT_DIR}")
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
                f"Please ensure 'meta_learning_related_work.csv' exists in the output folder: {OUTPUT_DIR}"
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
            "Simple Summary", "IEEE Citation"
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

def group_papers_by_themes(df: pd.DataFrame) -> Dict[str, List[Dict]]:
    """Group papers by different themes for analysis."""
    themes = {
        "meta_learning_methods": defaultdict(list),
        "algorithm_selection": defaultdict(list),
        "dataset_types": defaultdict(list),
        "publication_years": defaultdict(list)
    }
    
    for _, paper in df.iterrows():
        # Group by meta-learning methods
        if paper["Meta-Learning Method Used"] != "Not explicitly mentioned":
            themes["meta_learning_methods"][paper["Meta-Learning Method Used"]].append(paper.to_dict())
            
        # Group by algorithm selection methods
        if paper["Algorithm Selection Method"] != "Not explicitly mentioned":
            themes["algorithm_selection"][paper["Algorithm Selection Method"]].append(paper.to_dict())
            
        # Group by dataset types
        if paper["Dataset(s) Used"] != "Not explicitly mentioned":
            themes["dataset_types"][paper["Dataset(s) Used"]].append(paper.to_dict())
            
        # Group by publication years
        if paper["Year of Publication"] != "Not explicitly mentioned":
            themes["publication_years"][paper["Year of Publication"]].append(paper.to_dict())
    
    return themes

def generate_theme_analysis(theme_name: str, theme_data: Dict[str, List[Dict]]) -> str:
    """Generate analysis for a specific theme using Gemini."""
    try:
        # Create a structured representation of the theme data
        theme_summary = json.dumps(theme_data, indent=2)
        
        prompt = f"""
        You are an expert researcher in meta-learning. Write a comprehensive analysis of the papers grouped under the theme "{theme_name}".
        
        Focus on:
        1. Key patterns and trends in the research
        2. Comparison of different approaches
        3. Major findings and their significance
        4. Current limitations and future research directions
        
        Use proper academic writing style and include IEEE-style citations [n].
        
        Papers to analyze:
        {theme_summary}
        
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
            response = call_gemini_api(prompt, GENERATION_CONFIG, SAFETY_SETTINGS)
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
            logger.error(f"Gemini API error for theme {theme_name}: {str(api_error)}")
            return f"Error analyzing theme {theme_name}: API error - {str(api_error)}"
            
    except Exception as e:
        logger.error(f"Error generating theme analysis: {str(e)}")
        return f"Error analyzing theme {theme_name}: {str(e)}"

def generate_literature_review(df: pd.DataFrame) -> Dict[str, str]:
    """Generate a complete literature review with different sections."""
    try:
        # Group papers by themes
        themed_papers = group_papers_by_themes(df)
        
        # Convert DataFrame to a format suitable for the prompt
        papers_json = df.to_json(orient='records', indent=2)
        
        # Generate introduction with rate limiting
        intro_prompt = f"""
        Write an academic introduction for a literature review on meta-learning approaches in algorithm selection.
        
        Use this paper data to inform your writing:
        {papers_json}
        
        Your introduction MUST:
        1. Define the scope and objective clearly
        2. Provide context about meta-learning and algorithm selection
        3. Outline the structure of the review
        4. Highlight the significance of this research area
        
        IMPORTANT GUIDELINES:
        1. Use formal academic writing style
        2. Be concise but comprehensive
        3. Avoid unnecessary jargon
        4. Start directly with the content
        5. Focus on clarity and flow
        6. No need for abstract or keywords
        
        Begin your introduction directly without any preamble.
        """
        
        try:
            response = call_gemini_api(intro_prompt, GENERATION_CONFIG, SAFETY_SETTINGS)
            if response and hasattr(response, 'text'):
                introduction = response.text.strip()
                if not introduction:
                    introduction = "Error: Empty introduction generated"
            else:
                introduction = "Error: Invalid API response format"
            logger.info("Successfully generated introduction")
        except Exception as api_error:
            logger.error(f"Error generating introduction: {str(api_error)}")
            introduction = f"Error generating introduction: {str(api_error)}"
        
        # Generate themed sections with rate limiting
        sections = {
            "Introduction": introduction
        }
        
        # Process each theme with delays between calls
        theme_sections = [
            ("Evolution of Meta-Learning Methods", "Evolution of Meta-Learning", themed_papers["meta_learning_methods"]),
            ("Algorithm Selection Strategies", "Algorithm Selection", themed_papers["algorithm_selection"]),
            ("Datasets and Evaluation", "Datasets and Evaluation", themed_papers["dataset_types"]),
            ("Trends and Future Directions", "Trends", themed_papers["publication_years"])
        ]
        
        for section_title, theme_name, theme_data in theme_sections:
            logger.info(f"Generating section: {section_title}")
            sections[section_title] = generate_theme_analysis(theme_name, theme_data)
            time.sleep(RATE_LIMIT_DELAY)
        
        # Generate conclusion with rate limiting
        conclusion_prompt = f"""
        Write a conclusion for this meta-learning literature review.
        
        Previous sections:
        {json.dumps(sections, indent=2)}
        
        Your conclusion MUST:
        1. Summarize the key findings and patterns
        2. Identify important research gaps
        3. Suggest specific future research directions
        4. Emphasize the field's significance
        
        IMPORTANT GUIDELINES:
        1. Use formal academic style
        2. Be concise but comprehensive
        3. Make clear recommendations
        4. Connect back to the introduction
        5. End with a strong closing statement
        
        Begin your conclusion directly without any preamble.
        """
        
        try:
            response = call_gemini_api(conclusion_prompt, GENERATION_CONFIG, SAFETY_SETTINGS)
            if response and hasattr(response, 'text'):
                conclusion = response.text.strip()
                if not conclusion:
                    conclusion = "Error: Empty conclusion generated"
            else:
                conclusion = "Error: Invalid API response format"
            logger.info("Successfully generated conclusion")
        except Exception as api_error:
            logger.error(f"Error generating conclusion: {str(api_error)}")
            conclusion = f"Error generating conclusion: {str(api_error)}"
        
        sections["Conclusion"] = conclusion
        return sections
    
    except Exception as e:
        logger.error(f"Error generating literature review: {str(e)}")
        raise

def format_ieee_references(df: pd.DataFrame) -> str:
    """Format IEEE citations in the correct order."""
    try:
        # Extract and sort citations
        citations = df["IEEE Citation"].tolist()
        citations = [c for c in citations if c != "Not explicitly mentioned"]
        
        # Number the citations
        numbered_citations = [f"[{i+1}] {citation}" for i, citation in enumerate(citations)]
        
        return "\n\n".join(numbered_citations)
    except Exception as e:
        logger.error(f"Error formatting references: {str(e)}")
        return "Error generating references section"

def save_literature_review(sections: Dict[str, str], references: str, output_path: str = DEFAULT_OUTPUT_FILE):
    """Save the generated literature review to a file."""
    try:
        # Convert to absolute path if needed
        abs_path = os.path.abspath(output_path)
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        
        logger.info(f"Saving literature review to: {abs_path}")
        with open(abs_path, 'w', encoding='utf-8') as f:
            # Write each section
            for section_title, content in sections.items():
                f.write(f"# {section_title}\n\n")
                f.write(f"{content}\n\n")
            
            # Write references
            f.write("# References\n\n")
            f.write(references)
            
    except PermissionError:
        logger.error(f"Permission denied when writing to: {abs_path}")
        raise
    except Exception as e:
        logger.error(f"Error saving literature review: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description='Generate a literature review from meta-learning papers data')
    parser.add_argument('--input', '-i', 
                      default=DEFAULT_INPUT_CSV,
                      help='Path to the input CSV file (default: output/meta_learning_related_work.csv)')
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
        logger.error("Please ensure you've run generate_table.py first to create the CSV file")
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