import numpy as np

def combine_embeddings(embeddings):
    # Average of embeddings = robust ID
    avg = np.mean(np.array(embeddings), axis=0)
    return avg.tolist()
