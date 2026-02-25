import os
import sys
#Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from utils import load_data
from config_manager import load_config_with_args
from joblib import parallel_backend
import numpy as np

# Load configuration with CLI argument overrides
config = load_config_with_args(
    description="Check locality: find nearest neighbors of random sample points"
)

if __name__ == "__main__":\
    #pick N points at random, find the K nearest neighbors
    data = load_data.load_first_parquet(timing=True)
    sample_points = data.sample(n=config.data.n_samples_locality, random_state=config.training.random_seed)

    activations_layer = np.array(data[f"activation_layer_{config.model.layer_for_activation}"].tolist(), dtype=np.float32)
    sample_points_layer = sample_points[f"activation_layer_{config.model.layer_for_activation}"].tolist()
    from sklearn.neighbors import NearestNeighbors
    with parallel_backend(backend='loky', n_jobs=-1):
        nbrs = NearestNeighbors(n_neighbors=config.clustering.k_nearest_neighbors, algorithm='ball_tree', n_jobs=-1).fit(activations_layer)
        distances, indices = nbrs.kneighbors(sample_points_layer)
    
    #Make a pretty html that lists the sample points and their nearest neighbors in a table, with the distances
    #Make sure the prompts associated with each point are visible.

    def get_text_snippet(text, first_n=None, last_n=None):
        """Extract first and last n words from text."""
        if first_n is None:
            first_n = config.text.first_n_words
        if last_n is None:
            last_n = config.text.last_n_words
        words = str(text).split()
        if len(words) <= first_n + last_n:
            return ' '.join(words)
        first_words = ' '.join(words[:first_n])
        last_words = ' '.join(words[-last_n:])
        return f"{first_words} [...] {last_words}"
    
    html_output = "<html><head><title>Sample Points and Neighbors</title>"
    html_output += "<style>table {border-collapse: collapse; margin: 20px 0;} th, td {padding: 8px; text-align: left;} .text-snippet {max-width: 600px; font-size: 0.9em;}</style>"
    html_output += "</head><body>"
    for i, (idx, row) in enumerate(sample_points.iterrows()):
        html_output += f"<h2>Sample Point {i+1}</h2>"
        html_output += f"<p><strong>Prompt:</strong> {row['text_prefix']}</p>"
        html_output += "<table border='1'>"
        html_output += f"<tr><th>Neighbor Index</th><th>Distance</th><th>Text Snippet ({config.text.first_n_words}+{config.text.last_n_words} words)</th></tr>"
        for j in range(config.clustering.k_nearest_neighbors):
            neighbor_idx = indices[i][j]
            distance = distances[i][j]
            neighbor_row = data.iloc[neighbor_idx]
            text_snippet = get_text_snippet(neighbor_row['text_prefix'])
            html_output += f"<tr><td>{neighbor_idx}</td><td>{distance:.4f}</td><td class='text-snippet'>{text_snippet}</td></tr>"
        html_output += "</table>"
    html_output += "</body></html>"
    with open("sample_points_and_neighbors.html", "w") as f:
        f.write(html_output)

    