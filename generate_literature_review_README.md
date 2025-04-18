# Meta-Learning Literature Review Generator

This tool automatically generates a structured literature review from a CSV file containing meta-learning paper information. It uses Google's Gemini AI to analyze and synthesize information from multiple papers into a cohesive academic review.

## Features

- Automated generation of a complete literature review
- Advanced BERT-based semantic clustering of papers
- Thematic analysis using transformer embeddings
- IEEE-style citation management
- Structured sections following academic writing standards
- AI-powered synthesis and analysis
- Cluster visualization and analysis

## Paper Clustering

The system uses state-of-the-art transformer models for semantic clustering:

1. **BERT-based Embeddings**
   - Uses `sentence-transformers/all-MiniLM-L6-v2` for generating paper embeddings
   - Captures semantic relationships between papers
   - Supports batch processing for efficiency
   - Implements cosine similarity for cluster assignment

2. **Clustering Process**
   - Generates semantic embeddings for each paper
   - Uses K-means clustering on the embeddings
   - Identifies representative papers for each cluster
   - Creates t-SNE visualizations of clusters
   - Computes cluster centroids and similarities

3. **Cluster Analysis**
   - Identifies common themes and methodologies
   - Generates cluster descriptions using Gemini AI
   - Highlights representative papers in each cluster
   - Supports interactive visualization
   - Provides detailed cluster analysis in JSON format

## Installation

1. Clone this repository
2. Create a `.env` file in the root directory with your Gemini API key:
```
GEMINI_API_KEY=your_api_key_here
```
3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Dependencies

- pandas: Data processing
- transformers: BERT-based embeddings
- torch: Deep learning framework
- google-generativeai: AI-powered analysis
- python-dotenv: Environment management
- scikit-learn: Clustering and visualization
- matplotlib: Plotting and visualization
- numpy: Numerical computations

## Usage

Run the script with your CSV file:

```bash
python generate_literature_review.py --input input_csv_file --output output_markdown_file
```

Example:
```bash
python generate_literature_review.py --input meta_learning_related_work.csv --output literature_review.md
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
- Cluster (optional)
- Cluster Description (optional)

## Output

The script generates:
1. A structured literature review in Markdown format
2. Cluster analysis in JSON format
3. t-SNE visualization of paper clusters
4. IEEE-style reference list

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

## Notes

- The generated review maintains academic writing standards
- Citations are automatically managed and numbered
- The review is organized thematically using semantic clustering
- Each section provides synthesis rather than just summarization
- The output is in Markdown format for easy conversion to other formats
- Clustering helps identify related papers and research themes
- API key must be stored in the `.env` file 