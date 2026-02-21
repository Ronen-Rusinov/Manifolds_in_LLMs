import os
import sys
#Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from utils import load_data
from joblib import parallel_backend
import numpy as np

if __name__ == "__main__":\
    #pick 10 points at random, find the 20 nearest neihbors
    data = load_data.load_first_parquet(timing=True)
    sample_points = data.sample(n=10, random_state=42)

    activations_layer_18 = np.array(data[f"activation_layer_18"].tolist(), dtype=np.float32)
    sample_points_layer_18 = sample_points[f"activation_layer_18"].tolist()
    from sklearn.neighbors import NearestNeighbors
    with parallel_backend(backend='loky',n_jobs=-1):
        nbrs = NearestNeighbors(n_neighbors=20, algorithm='ball_tree', n_jobs=-1).fit(activations_layer_18)
        distances, indices = nbrs.kneighbors(sample_points_layer_18)
    
    #Make a pretty html that lists the 10 sample points and their 20 nearest neighbors in a table, with the distances
    #Make sure the prompts associated with each point are visible.

    def get_text_snippet(text, first_n=20, last_n=20):
        """Extract first and last n words from text."""
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
        html_output += "<tr><th>Neighbor Index</th><th>Distance</th><th>Text Snippet</th></tr>"
        for j in range(20):
            neighbor_idx = indices[i][j]
            distance = distances[i][j]
            neighbor_row = data.iloc[neighbor_idx]
            text_snippet = get_text_snippet(neighbor_row['text_prefix'])
            html_output += f"<tr><td>{neighbor_idx}</td><td>{distance:.4f}</td><td class='text-snippet'>{text_snippet}</td></tr>"
        html_output += "</table>"
    html_output += "</body></html>"
    with open("sample_points_and_neighbors.html", "w") as f:
        f.write(html_output)

    