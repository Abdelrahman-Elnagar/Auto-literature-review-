import os
import sys
import subprocess
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_command(command: str, description: str) -> None:
    """Run a command and log its output."""
    try:
        logger.info(f"Starting {description}...")
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            capture_output=True,
            text=True
        )
        logger.info(f"{description} completed successfully")
        if result.stdout:
            logger.info(f"Output:\n{result.stdout}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error in {description}: {str(e)}")
        if e.stdout:
            logger.error(f"Output:\n{e.stdout}")
        if e.stderr:
            logger.error(f"Error output:\n{e.stderr}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in {description}: {str(e)}")
        raise

def main():
    try:
        # Get the workspace root directory
        workspace_root = os.path.dirname(os.path.abspath(__file__))
        
        # Define paths
        input_dir = os.path.join(workspace_root, "input")
        output_dir = os.path.join(workspace_root, "output")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Run the table generation
        table_cmd = f"python generate_table.py {input_dir} {output_dir}"
        run_command(table_cmd, "table generation")
        
        # Run the clustering process
        clustering_cmd = "python run_clustering.py"
        run_command(clustering_cmd, "clustering process")
        
        # Run the literature review generation
        review_cmd = "python generate_literature_review.py"
        run_command(review_cmd, "literature review generation")
        
        logger.info("All processes completed successfully")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 