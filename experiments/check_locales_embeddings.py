#This file's primary purpose is to verify that the embeddings produced by Isomap still maintain 
#the same local behaviour as the original activations.
#Since the trustworthiness score for 20 nearest neighbors is 0.89,
#One would expect that ~89% of the 20 nearest neighbors in the original space are also among the 20 nearest neighbors in the Isomap embedding space.

#This file is heavily modeled after check_locales.py, but instead of comparing the nearest neighbors in the original activation space to the nearest neighbors in the original activation space, we will compare them to the nearest neighbors in the Isomap embedding space.
import os
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from utils import load_data
from joblib import parallel_backend
import joblib
import numpy as np
from sklearn.neighbors import NearestNeighbors


def get_text_snippet(text, first_n=20, last_n=20):
	"""Extract first and last n words from text."""
	words = str(text).split()
	if len(words) <= first_n + last_n:
		return ' '.join(words)
	first_words = ' '.join(words[:first_n])
	last_words = ' '.join(words[-last_n:])
	return f"{first_words} [...] {last_words}"


if __name__ == "__main__":
	sample_size = 10
	n_neighbors = 20
	random_state = 42

	data = load_data.load_first_parquet(timing=True)
	sample_points = data.sample(n=sample_size, random_state=random_state)

	activations_layer_18 = np.array(data["activation_layer_18"].tolist(), dtype=np.float32)
	sample_points_layer_18 = np.array(sample_points["activation_layer_18"].tolist(), dtype=np.float32)

	isomap_path = Path(__file__).parent.parent / "results" / "isomap_12D" / "isomap_n_neighbors_50_n_components_12.joblib"
	if not isomap_path.exists():
		raise FileNotFoundError(f"Isomap model not found at {isomap_path}. Train it first.")

	with parallel_backend(backend='loky', n_jobs=-1):
		isomap = joblib.load(isomap_path)
		embeddings = isomap.transform(activations_layer_18)
		sample_embeddings = isomap.transform(sample_points_layer_18)

		nbrs_orig = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree', n_jobs=-1).fit(activations_layer_18)
		orig_distances, orig_indices = nbrs_orig.kneighbors(sample_points_layer_18)

		nbrs_embed = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree', n_jobs=-1).fit(embeddings)
		embed_distances, embed_indices = nbrs_embed.kneighbors(sample_embeddings)

	html_output = "<html><head><title>Locality Check: Activations vs Isomap</title>"
	html_output += "<style>table {border-collapse: collapse; margin: 12px 0;} th, td {padding: 8px; text-align: left;} .text-snippet {max-width: 600px; font-size: 0.9em;} .badge {padding: 2px 6px; border-radius: 4px; background: #e6f3ff;}</style>"
	html_output += "</head><body>"
	html_output += "<h1>Locality Check: Activations vs Isomap</h1>"

	for i, (idx, row) in enumerate(sample_points.iterrows()):
		orig_set = set(orig_indices[i].tolist())
		embed_set = set(embed_indices[i].tolist())
		overlap = len(orig_set.intersection(embed_set))
		overlap_pct = (overlap / n_neighbors) * 100

		html_output += f"<h2>Sample Point {i + 1}</h2>"
		html_output += f"<p><strong>Prompt:</strong> {row['text_prefix']}</p>"
		html_output += f"<p><strong>Neighbor overlap:</strong> {overlap}/{n_neighbors} (<span class='badge'>{overlap_pct:.1f}%</span>)</p>"

		html_output += "<h3>Original Space Neighbors</h3>"
		html_output += "<table border='1'>"
		html_output += "<tr><th>Neighbor Index</th><th>Distance</th><th>Text Snippet</th></tr>"
		for j in range(n_neighbors):
			neighbor_idx = orig_indices[i][j]
			distance = orig_distances[i][j]
			neighbor_row = data.iloc[neighbor_idx]
			text_snippet = get_text_snippet(neighbor_row['text_prefix'])
			html_output += f"<tr><td>{neighbor_idx}</td><td>{distance:.4f}</td><td class='text-snippet'>{text_snippet}</td></tr>"
		html_output += "</table>"

		html_output += "<h3>Isomap Embedding Neighbors</h3>"
		html_output += "<table border='1'>"
		html_output += "<tr><th>Neighbor Index</th><th>Distance</th><th>In Original NN</th><th>Text Snippet</th></tr>"
		for j in range(n_neighbors):
			neighbor_idx = embed_indices[i][j]
			distance = embed_distances[i][j]
			neighbor_row = data.iloc[neighbor_idx]
			text_snippet = get_text_snippet(neighbor_row['text_prefix'])
			in_original = "Yes" if neighbor_idx in orig_set else "No"
			html_output += f"<tr><td>{neighbor_idx}</td><td>{distance:.4f}</td><td>{in_original}</td><td class='text-snippet'>{text_snippet}</td></tr>"
		html_output += "</table>"

	html_output += "</body></html>"

	output_path = Path(__file__).parent.parent / "sample_points_and_neighbors_embeddings.html"
	with open(output_path, "w") as f:
		f.write(html_output)