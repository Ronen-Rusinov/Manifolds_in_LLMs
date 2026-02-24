#This script seeks to see if the activations assigned to the neighborhood of each centroid (possibly overlapping)
#Can feasably be interpreted direction-wise.

#centroids. The script's prerequisits are the outputs of the script obtain_1000_nearest_to_centroids.py
#And subsequently relies on the output of minibatch_kmeans.py as well as
#on the output of produce_balltree.py

#Their outputs are stored in 
#/outputs/Balltree/nearest_1000_neighbors_indices_1.npy
#/outputs/minibatch_kmeans/centroids.npy
#and /outputs/Balltree/balltree_layer_18_all_parquets.pkl respectively.

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils import load_data
import numpy as np
from sklearn.decomposition import PCA

N_components = 12
cluster_count = 3

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
    activations = np.array(df['activation_layer_18'].tolist(), dtype=np.float32)

    html_output = "<html><body>"
    indices = range(0,len(centroids),10)

    for i in indices:
        print(f"Centroid {i}:")
        print(f"Indices of nearest neighbors: {neighbors_indices[i][:10]}...")
        cluster_activations = activations[neighbors_indices[i]]
        print(f"Shape of cluster activations: {cluster_activations.shape}")
        #compute PCA
        pca = PCA(n_components=N_components)
        reduced_activations = pca.fit_transform(cluster_activations)
        print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
        print(f"Total explained variance: {np.sum(pca.explained_variance_ratio_)}")

        html_output += f"<h2>Centroid {i}</h2>"

        #order the original activations by their projection on each principle component
        for component in range(N_components):
            component_projection = reduced_activations[:, component]
            sorted_indices = np.argsort(component_projection)
            #Argsort returns indices in the range of the array length.
            #We need to translate these indices back to the original indices of the activations, which are given by neighbors_indices[i].
            sorted_indices = neighbors_indices[i][sorted_indices]

            #Pull one out of every 100 activations, to produce a list of 100 activations per component, ordered by their projection on that component.
            step = max(1, len(sorted_indices) // 100)
            selected_indices = sorted_indices[::step][:100]

            #pull the first 20 words, the middle 20 words and the last 20 words associated with each of the indices
            #remember that the words of the prompt associated with an activation are saved in the "text_prefix" column of the dataframe.
            selected_words = []
            for idx in selected_indices:
                text_prefix = df.iloc[idx]['text_prefix']
                first_20_words = text_prefix.split()[:20]
                middle_20_words = text_prefix.split()[len(text_prefix.split())//2 - 10:len(text_prefix.split())//2 + 10]
                last_20_words = text_prefix.split()[-20:]
                selected_words.append((first_20_words, middle_20_words, last_20_words))

            html_output += f"<h3>Component {component}</h3>"
            html_output += "<table border='1'>"
            html_output += "<tr><th>First 20 Words</th><th>Middle 20 Words</th><th>Last 20 Words</th></tr>"
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


