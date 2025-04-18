import pandas as pd
import logging
import os
from pathlib import Path
from paper_clustering import cluster_papers, save_cluster_analysis

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
DEFAULT_OUTPUT_CSV = os.path.join(OUTPUT_DIR, "meta_learning_related_work_clustered.csv")
DEFAULT_ANALYSIS_FILE = os.path.join(OUTPUT_DIR, "cluster_analysis.json")

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

def main():
    try:
        # Ensure output directory exists
        ensure_output_dir()
        
        # Load paper data
        logger.info("Loading paper data...")
        df = load_papers_data()
        
        # Perform clustering
        logger.info("Performing paper clustering...")
        df_with_clusters, cluster_analysis = cluster_papers(df)
        
        # Save clustered data
        logger.info(f"Saving clustered data to {DEFAULT_OUTPUT_CSV}")
        df_with_clusters.to_csv(DEFAULT_OUTPUT_CSV, index=False)
        
        # Save cluster analysis
        logger.info(f"Saving cluster analysis to {DEFAULT_ANALYSIS_FILE}")
        save_cluster_analysis(cluster_analysis, DEFAULT_ANALYSIS_FILE)
        
        logger.info("Clustering process completed successfully")
        
    except Exception as e:
        logger.error(f"Error in clustering process: {str(e)}")
        raise

if __name__ == "__main__":
    main() 