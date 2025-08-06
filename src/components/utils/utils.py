import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def standardise(data):
    mu=np.mean(data,axis=0)
    std=np.std(data,axis=0)
    data=(data-mu)/std

    return data




def compute_mfcc_similarity_metrics(original: np.ndarray, predicted: np.ndarray) -> dict:
    """
    Compute both Cosine Similarity and Mel Cepstral Distortion (MCD) between two MFCC sequences.

    Args:
        original: (num_frames, num_mfcc) array of ground-truth MFCCs
        predicted: (num_frames, num_mfcc) array of predicted MFCCs

    Returns:
        A dictionary with:
            - 'cosine_similarity': average per-frame cosine similarity (1.0 = perfect match)
            - 'mcd': Mel Cepstral Distortion in decibels (lower = better)
    """
    assert original.shape == predicted.shape, "Original and predicted must have the same shape"

    # Cosine similarity (frame-wise, diagonal only)
    cosine_sim_matrix = cosine_similarity(original, predicted)
    avg_cosine_similarity = np.mean(np.diag(cosine_sim_matrix))

    # MCD computation
    diff = original - predicted
    squared_diff = np.sum(diff**2, axis=1)
    mcd = (10.0 / np.log(10)) * np.sqrt(2) * np.mean(np.sqrt(squared_diff))

    return avg_cosine_similarity, mcd
