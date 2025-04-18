import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BERTEncoder(nn.Module):
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def forward(self, texts: List[str]) -> torch.Tensor:
        # Tokenize sentences
        encoded_input = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
        
        # Compute token embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        
        # Perform pooling
        sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
        
        # Normalize embeddings
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        
        return sentence_embeddings

def create_feature_vector(paper_info: dict) -> str:
    """Create a text representation for clustering."""
    features = [
        paper_info.get("Paper Title", ""),
        paper_info.get("Meta-Learning Method Used", ""),
        paper_info.get("Meta-Features Chosen", ""),
        paper_info.get("Algorithm Selection Method", ""),
        paper_info.get("Algorithms Considered for Selection", ""),
        paper_info.get("Dataset(s) Used", ""),
        paper_info.get("Key Findings/Contributions", ""),
        paper_info.get("Simple Summary", "")
    ]
    return " ".join([str(f) for f in features if f and f != "Not explicitly mentioned"])

def compute_similarity_matrix(embeddings: torch.Tensor) -> torch.Tensor:
    """Compute cosine similarity matrix between embeddings."""
    return torch.mm(embeddings, embeddings.t())

def get_representative_papers(cluster_papers: pd.DataFrame, embeddings: torch.Tensor, 
                            cluster_indices: np.ndarray, n_papers: int = 3) -> pd.DataFrame:
    """Get the most representative papers for a cluster based on centroid similarity."""
    # Get embeddings for papers in this cluster
    cluster_embeddings = embeddings[cluster_indices]
    
    # Compute centroid
    centroid = cluster_embeddings.mean(dim=0)
    
    # Compute similarities to centroid
    similarities = F.cosine_similarity(cluster_embeddings, centroid.unsqueeze(0))
    
    # Get indices of most representative papers
    representative_indices = similarities.argsort(descending=True)[:n_papers]
    
    return cluster_papers.iloc[representative_indices.cpu().numpy()]

def generate_cluster_description(cluster_papers: pd.DataFrame, representative_papers: pd.DataFrame) -> str:
    """Generate a description of the cluster based on its papers."""
    # Get common themes from titles and methods
    titles = cluster_papers["Paper Title"].tolist()
    methods = cluster_papers["Meta-Learning Method Used"].tolist()
    
    # Create description
    description = (
        f"Cluster with {len(cluster_papers)} papers focusing on "
        f"{', '.join(set(methods) - {'Not explicitly mentioned'})}. "
        f"Key papers: {', '.join(representative_papers['Paper Title'].tolist())}"
    )
    
    return description

def cluster_papers(papers_df: pd.DataFrame, n_clusters: int = 5) -> Tuple[pd.DataFrame, Dict]:
    """
    Cluster papers based on their content using BERT embeddings.
    
    Args:
        papers_df: DataFrame containing paper information
        n_clusters: Number of clusters to create
        
    Returns:
        Tuple of (DataFrame with added clustering information, Dict with cluster analysis)
    """
    try:
        # Create feature vectors for clustering
        feature_texts = [create_feature_vector(row.to_dict()) for _, row in papers_df.iterrows()]
        
        # Initialize BERT encoder
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        encoder = BERTEncoder().to(device)
        encoder.eval()
        
        # Generate embeddings in batches
        batch_size = 32
        embeddings_list = []
        
        for i in range(0, len(feature_texts), batch_size):
            batch_texts = feature_texts[i:i + batch_size]
            with torch.no_grad():
                batch_embeddings = encoder(batch_texts)
            embeddings_list.append(batch_embeddings)
        
        # Concatenate all embeddings
        embeddings = torch.cat(embeddings_list, dim=0)
        
        # Move to CPU for clustering
        embeddings_np = embeddings.cpu().numpy()
        
        if embeddings_np.shape[0] < n_clusters:
            n_clusters = max(2, embeddings_np.shape[0] - 1)
        
        # Perform clustering
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=10
        )
        cluster_labels = kmeans.fit_predict(embeddings_np)
        
        # Add cluster labels to DataFrame
        papers_df['Cluster'] = cluster_labels
        
        # Generate cluster descriptions and analysis
        cluster_descriptions = {}
        cluster_analysis = {}
        
        for cluster_id in range(n_clusters):
            # Get papers in this cluster
            cluster_mask = papers_df['Cluster'] == cluster_id
            cluster_papers = papers_df[cluster_mask]
            cluster_indices = np.where(cluster_mask)[0]
            
            # Get representative papers
            representative_papers = get_representative_papers(cluster_papers, embeddings, cluster_indices)
            
            # Generate cluster description
            description = generate_cluster_description(cluster_papers, representative_papers)
            cluster_descriptions[cluster_id] = description
            
            # Generate cluster analysis
            analysis = {
                "description": description,
                "size": len(cluster_papers),
                "representative_papers": representative_papers["Paper Title"].tolist(),
                "common_methods": cluster_papers["Meta-Learning Method Used"].value_counts().head(3).to_dict(),
                "common_datasets": cluster_papers["Dataset(s) Used"].value_counts().head(3).to_dict(),
                "publication_years": cluster_papers["Year of Publication"].value_counts().sort_index().to_dict()
            }
            cluster_analysis[f"Cluster {cluster_id}"] = analysis
        
        # Add cluster descriptions to DataFrame
        papers_df['Cluster Description'] = papers_df['Cluster'].map(cluster_descriptions)
        
        # Visualize clusters using t-SNE
        try:
            # Reduce dimensions for visualization
            tsne = TSNE(n_components=2, random_state=42)
            coords = tsne.fit_transform(embeddings_np)
            
            # Create scatter plot
            plt.figure(figsize=(12, 8))
            scatter = plt.scatter(coords[:, 0], coords[:, 1], c=cluster_labels, cmap='viridis')
            plt.title('Paper Clusters Visualization (t-SNE)')
            plt.xlabel('First t-SNE Component')
            plt.ylabel('Second t-SNE Component')
            
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
            # Create a basic visualization without descriptions
            try:
                plt.figure(figsize=(10, 6))
                plt.scatter(coords[:, 0], coords[:, 1], c=cluster_labels, cmap='viridis')
                plt.title('Paper Clusters Visualization (t-SNE)')
                plt.xlabel('First t-SNE Component')
                plt.ylabel('Second t-SNE Component')
                plt.colorbar(label='Cluster')
                plt.savefig('paper_clusters.png', bbox_inches='tight', dpi=300)
                plt.close()
            except Exception as viz_error:
                logger.error(f"Error creating basic visualization: {str(viz_error)}")
        
        return papers_df, cluster_analysis
        
    except Exception as e:
        logger.error(f"Error in clustering: {str(e)}")
        papers_df['Cluster'] = 0
        papers_df['Cluster Description'] = "Clustering failed"
        return papers_df, {"error": str(e)}

def save_cluster_analysis(cluster_analysis: Dict, output_path: str) -> None:
    """Save cluster analysis to a JSON file."""
    try:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(cluster_analysis, f, indent=2)
            
        logger.info(f"Cluster analysis saved to {output_path}")
    except Exception as e:
        logger.error(f"Error saving cluster analysis: {str(e)}")

if __name__ == "__main__":
    # Example usage
    import json
    
    # Load your papers DataFrame
    papers_df = pd.read_csv("meta_learning_related_work.csv")
    
    # Perform clustering
    papers_df, cluster_analysis = cluster_papers(papers_df)
    
    # Save results
    papers_df.to_csv("meta_learning_related_work_clustered.csv", index=False)
    save_cluster_analysis(cluster_analysis, "cluster_analysis.json") 