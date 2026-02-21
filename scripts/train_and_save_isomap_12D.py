import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from sklearn.manifold import Isomap, trustworthiness
import numpy as np
import joblib
import datetime
from pathlib import Path

def train_and_save_isomap(dataframes,save_path, n_neighbors=50, n_components=12):

    train_data,val_data, test_data = dataframes
    #Use only 50% of train data. More than that is infeasable
    train_data = train_data.sample(frac=0.5, random_state=42)

    activations_layer_18 = np.array(train_data["activation_layer_18"].tolist())
    with joblib.parallel_backend(backend='loky', n_jobs=-1):
        isomap = Isomap(n_neighbors=n_neighbors, n_components=n_components, n_jobs=-1)
        embeddings = isomap.fit_transform(activations_layer_18)
        joblib.dump(isomap, save_path / f"isomap_n_neighbors_{n_neighbors}_n_components_{n_components}.joblib")
        #dump embeddings and original training data
        np.save(save_path / f"embeddings_n_neighbors_{n_neighbors}_n_components_{n_components}.npy", embeddings)
        train_data.to_parquet(save_path / f"train_data_n_neighbors_{n_neighbors}_n_components_{n_components}.parquet")
    
        #Evaluate on val data
        #use 20% of val data for evaluation, more than that is unnecessary
        val_data = val_data.sample(frac=0.2, random_state=42)
        activations_layer_18_val = np.array(val_data["activation_layer_18"].tolist())
        val_embeddings = isomap.transform(activations_layer_18_val)
        #check trustworthiness score for different values of k
        scores = {}
        for k in range(5,51,5):
            score = trustworthiness(activations_layer_18_val, val_embeddings, n_neighbors=k)
            scores[k] = score
            print(f"Trustworthiness score for k={k}: {score}")
    
    #save graph of scores
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,6))
    plt.plot(list(scores.keys()), list(scores.values()), marker='o')
    plt.title("Trustworthiness Score for Different Values of k")
    plt.xlabel("k (number of neighbors)")
    plt.ylabel("Trustworthiness Score")
    plt.xticks(list(scores.keys()))
    plt.grid()
    plt.savefig(save_path / f"trustworthiness_scores_n_neighbors_{n_neighbors}_n_components_{n_components}.png")
    plt.close()

if __name__ == "__main__":
    #print beggining timestamp
    
    print(f"Starting Isomap training at {datetime.datetime.now()}")
    
    from utils import load_data
    from pathlib import Path
    dataframes = load_data.load_train_test_val_first_parquet(train_size=0.7, val_size=0.2, timing=True)
    save_path = Path(__file__).parent.parent / "results" / "isomap_12D"
    save_path.mkdir(exist_ok=True)
    train_and_save_isomap(dataframes, save_path, n_neighbors=50, n_components=12)

    print(f"Finished Isomap training at {datetime.datetime.now()}")

