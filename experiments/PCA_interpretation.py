import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils import load_data
from src.config_manager import load_config, add_config_argument
import argparse
import numpy as np
from sklearn.decomposition import PCA

# Load configuration with CLI argument overrides
parser = argparse.ArgumentParser(description="Interpret PCA components of activation clusters around centroids")

# PCA parameters
parser.add_argument("--n_components", type=int, help="Number of PCA components")
parser.add_argument("--n_centroids", type=int, help="Number of centroids")
parser.add_argument("--pca_max_samples", type=int, help="Maximum activations sampled per PCA component")
parser.add_argument("--layer_for_activation", type=int, help="Layer index for activation extraction")

add_config_argument(parser)
args = parser.parse_args()
config = load_config(args.config)

# Override config with CLI arguments
if args.n_components is not None:
    config.dimensionality.n_components = args.n_components
if args.n_centroids is not None:
    config.clustering.n_centroids = args.n_centroids
if args.pca_max_samples is not None:
    config.numerical.pca_max_samples = args.pca_max_samples
if args.layer_for_activation is not None:
    config.model.layer_for_activation = args.layer_for_activation

if __name__ == "__main__":
    df = load_data.load_all_parquets(timing=True)
    #load centroids
    print(f"Loading centroids...")
    centroids_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results','minibatch_kmeans' ,'centroids.npy'))
    with open(centroids_path, "rb") as f:
        centroids = np.load(f)

    print(f"Centroids loaded from {centroids_path}.")
    print(f"Centroids shape: {centroids.shape}")
    #load nearest neighbors indices
    print(f"Loading nearest neighbors indices...")
    neighbors_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results','Balltree' ,'nearest_1000_neighbors_indices_1.npy'))
    with open(neighbors_path, "rb") as f:
        neighbors_indices = np.load(f)
    print(f"Nearest neighbors indices loaded from {neighbors_path}.")
    print(f"Nearest neighbors indices shape: {neighbors_indices.shape}")
    activations = np.array(df[f'activation_layer_{config.model.layer_for_activation}'].tolist(), dtype=np.float32)

    html_output = "<html><body>"
    indices = range(0, len(centroids), 10)

    for i in indices:
        print(f"Centroid {i}:")
        print(f"Indices of nearest neighbors: {neighbors_indices[i][:10]}...")
        cluster_activations = activations[neighbors_indices[i]]
        print(f"Shape of cluster activations: {cluster_activations.shape}")
        #compute PCA
        pca = PCA(n_components=config.dimensionality.n_components)
        reduced_activations = pca.fit_transform(cluster_activations)
        print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
        print(f"Total explained variance: {np.sum(pca.explained_variance_ratio_)}")

        html_output += f"<h2>Centroid {i}</h2>"

        #order the original activations by their projection on each principle component
        for component in range(config.dimensionality.n_components):
            component_projection = reduced_activations[:, component]
            sorted_indices = np.argsort(component_projection)
            #Argsort returns indices in the range of the array length.
            #We need to translate these indices back to the original indices of the activations, which are given by neighbors_indices[i].
            sorted_indices = neighbors_indices[i][sorted_indices]

            #Pull one out of every N activations, to produce a list of samples per component, ordered by their projection on that component.
            max_samples = config.numerical.pca_max_samples
            step = max(1, len(sorted_indices) // max_samples)
            selected_indices = sorted_indices[::step][:max_samples]

            #pull the first N words, the middle N words and the last N words associated with each of the indices
            #remember that the words of the prompt associated with an activation are saved in the "text_prefix" column of the dataframe.
            selected_words = []
            for idx in selected_indices:
                text_prefix = df.iloc[idx]['text_prefix']
                first_n_words = text_prefix.split()[:config.text.first_n_words]
                middle_n_words = text_prefix.split()[len(text_prefix.split())//2 - config.text.first_n_words//2:len(text_prefix.split())//2 + config.text.first_n_words//2]
                last_n_words = text_prefix.split()[-config.text.last_n_words:]
                selected_words.append((first_n_words, middle_n_words, last_n_words))

            html_output += f"<h3>Component {component}</h3>"
            html_output += "<table border='1'>"
            html_output += "<tr><th>First {0} Words</th><th>Middle {0} Words</th><th>Last {0} Words</th></tr>".format(config.text.first_n_words)
            for first, middle, last in selected_words:
                html_output += f"<tr><td>{' '.join(first)}</td><td>{' '.join(middle)}</td><td>{' '.join(last)}</td></tr>"
            html_output += "</table>"
    html_output += "</body></html>"

    #save html output
    output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results','PCA_interpretation' ,'pca_interpretation.html'))
    #make the path if it does not exist
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(html_output)
    print(f"HTML output saved to {output_path}")


