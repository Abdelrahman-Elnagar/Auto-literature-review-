# Research Paper Analysis Tool

This tool processes research papers in PDF format and extracts structured information using Google's Gemini AI. It's designed to analyze meta-learning research papers and generate a comprehensive CSV output with key information.

## Features

- PDF text extraction using PyMuPDF
- AI-powered analysis using Google's Gemini API
- Extracts key information including:
  - Paper title and authors
  - Publication year
  - Meta-learning methods
  - Meta-features
  - Algorithm selection methods
  - Evaluation metrics
  - Datasets used
  - Performance results
  - Key findings
  - Limitations
  - Simple summary
  - IEEE citation format
- Progress logging
- Error handling
- CSV output generation

## Installation

1. Clone this repository
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the script with input and output folder paths:

```bash
python generate_table.py <input_folder> <output_folder>
```

Example:
```bash
python generate_table.py ./papers ./output
```

## Output Format

The script generates a CSV file (`meta_learning_related_work.csv`) with the following columns:

- filename: Name of the PDF file
- extraction_status: Success or error status
- Paper Title: Title of the research paper
- Authors: List of authors
- Year of Publication: Publication year
- Meta-Learning Method Used: Description of the meta-learning approach
- Meta-Features Chosen: Features used for meta-learning
- Algorithm Selection Method: Method used for algorithm selection
- Algorithms Considered for Selection: List of algorithms evaluated
- Evaluation Metrics: Metrics used for evaluation
- Dataset(s) Used: Datasets used in the study
- Performance of Meta-Learning Approach: Performance results
- Key Findings/Contributions: Main contributions and findings
- Meta-Feature Generation Process: Process of generating meta-features
- Limitations: Study limitations and future work
- Simple Summary: Clear explanation of the paper in simple language
- IEEE Citation: Citation in IEEE format

## Error Handling

The script includes comprehensive error handling for:
- PDF text extraction errors
- File permission issues
- API errors
- Processing errors
- Invalid file formats

Errors are logged with descriptive messages to help diagnose issues.

## Dependencies

- PyMuPDF: PDF text extraction
- Google Generative AI: Gemini API integration
- pandas: CSV handling
- tqdm: Progress bars
- python-dotenv: Environment variable management

## Notes

- The script uses Google's Gemini API for advanced text analysis
- Requires an internet connection to access the Gemini API
- The API key is included in the script
- May incur costs based on Gemini API usage
- Results are based on AI-powered analysis of the paper content 