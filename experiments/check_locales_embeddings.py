import os
import sys
from pathlib import Path
from datetime import datetime

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from utils import load_data
from config_manager import load_config_with_args
from joblib import parallel_backend
import joblib
import numpy as np
from sklearn.neighbors import NearestNeighbors

# Load configuration with CLI argument overrides
config = load_config_with_args(
    description="Check locality preservation in Isomap embeddings"
)

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


if __name__ == "__main__":
	start_time = datetime.now()
	print(f"[{start_time.strftime('%Y-%m-%d %H:%M:%S')}] Starting execution...")
	
	sample_size = config.data.n_samples_locality
	n_neighbors = config.clustering.k_nearest_neighbors
	random_state = config.training.random_seed

	print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Loading data...")
	data = load_data.load_first_parquet(timing=True)
	sample_points = data.sample(n=sample_size, random_state=random_state)

	print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Extracting activations...")
	activations_layer = np.array(data[f"activation_layer_{config.model.layer_for_activation}"].tolist(), dtype=np.float32)
	sample_points_layer = np.array(sample_points[f"activation_layer_{config.model.layer_for_activation}"].tolist(), dtype=np.float32)

	isomap_path = Path(__file__).parent.parent / "results" / f"isomap_{config.dimensionality.n_components}D" / f"isomap_n_neighbors_{config.dimensionality.n_neighbors}_n_components_{config.dimensionality.n_components}.joblib"
	if not isomap_path.exists():
		raise FileNotFoundError(f"Isomap model not found at {isomap_path}. Train it first.")


	with parallel_backend(backend='loky', n_jobs=-1):
		print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Loading isomap model...")
		isomap = joblib.load(isomap_path)
		print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Transforming embeddings...")
		embeddings = isomap.transform(activations_layer)
		print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Finding nearest neighbors in isomap embedding space...")
		nbrs_embed = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree', n_jobs=-1).fit(embeddings)
		print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Finding nearest neighbors to sample points...")
		embed_distances, embed_indices = nbrs_embed.kneighbors(sample_points_layer)

	print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Generating HTML...")

	html_output = "<html><head><title>Locality Check: Activations vs Isomap</title>"
	html_output += "<style>table {border-collapse: collapse; margin: 12px 0;} th, td {padding: 8px; text-align: left;} .text-snippet {max-width: 600px; font-size: 0.9em;} .badge {padding: 2px 6px; border-radius: 4px; background: #e6f3ff;}</style>"
	html_output += "</head><body>"
	html_output += "<h1>Locality Check: Activations vs Isomap</h1>"

	for i, (idx, row) in enumerate(sample_points.iterrows()):
		html_output += f"<h2>Sample Point {i+1}</h2>"
		html_output += f"<p><strong>Prompt:</strong> {row['text_prefix']}</p>"
		
		html_output += "<h3>Isomap Embedding Neighbors</h3>"
		html_output += "<table border='1'>"
		html_output += f"<tr><th>Neighbor Index</th><th>Distance</th><th>Text Snippet ({config.text.first_n_words}+{config.text.last_n_words} words)</th></tr>"
		for j in range(n_neighbors):
			neighbor_idx = embed_indices[i][j]
			distance = embed_distances[i][j]
			neighbor_row = data.iloc[neighbor_idx]
			text_snippet = get_text_snippet(neighbor_row['text_prefix'])
			html_output += f"<tr><td>{neighbor_idx}</td><td>{distance:.4f}</td><td class='text-snippet'>{text_snippet}</td></tr>"
		html_output += "</table>"

	html_output += "</body></html>"

	output_path = Path(__file__).parent.parent / "sample_points_and_neighbors_embeddings.html"
	with open(output_path, "w") as f:
		f.write(html_output)
	
	end_time = datetime.now()
	elapsed = (end_time - start_time).total_seconds()
	print(f"[{end_time.strftime('%Y-%m-%d %H:%M:%S')}] Execution completed in {elapsed:.2f} seconds")