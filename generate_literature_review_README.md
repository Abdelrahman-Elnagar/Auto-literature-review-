# Meta-Learning Literature Review Generator

This tool automatically generates a structured literature review from a CSV file containing meta-learning paper information. It uses Google's Gemini AI to analyze and synthesize information from multiple papers into a cohesive academic review.

## Features

- Automated generation of a complete literature review
- Thematic analysis of papers
- IEEE-style citation management
- Structured sections following academic writing standards
- AI-powered synthesis and analysis

## Generated Review Structure

1. **Introduction**
   - Scope and objectives
   - Context and background
   - Review structure
   - Research significance

2. **Evolution of Meta-Learning Methods**
   - Historical development
   - Key approaches
   - Methodological trends

3. **Algorithm Selection Strategies**
   - Different approaches
   - Comparative analysis
   - Implementation considerations

4. **Datasets and Evaluation**
   - Common datasets
   - Evaluation metrics
   - Benchmark results

5. **Trends and Future Directions**
   - Current state of the field
   - Research gaps
   - Future opportunities

6. **Conclusion**
   - Summary of findings
   - Research implications
   - Future work suggestions

7. **References**
   - IEEE-formatted citations
   - Numbered in order of appearance

## Installation

1. Clone this repository
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the script with your CSV file:

```bash
python generate_literature_review.py input_csv_file output_markdown_file
```

Example:
```bash
python generate_literature_review.py meta_learning_related_work.csv literature_review.md
```

## Input CSV Format

The input CSV should contain the following columns:
- Paper Title
- Authors
- Year of Publication
- Meta-Learning Method Used
- Meta-Features Chosen
- Algorithm Selection Method
- Algorithms Considered for Selection
- Evaluation Metrics
- Dataset(s) Used
- Performance of Meta-Learning Approach
- Key Findings/Contributions
- Meta-Feature Generation Process
- Limitations
- Simple Summary
- IEEE Citation

## Output

The script generates a Markdown file containing:
1. A structured literature review
2. Properly formatted sections
3. In-text citations
4. IEEE-style reference list

## Dependencies

- pandas: Data processing
- google-generativeai: AI-powered analysis
- python-dotenv: Environment management

## Notes

- The generated review maintains academic writing standards
- Citations are automatically managed and numbered
- The review is organized thematically rather than chronologically
- Each section provides synthesis rather than just summarization
- The output is in Markdown format for easy conversion to other formats 