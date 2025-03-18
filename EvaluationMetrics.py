import sklearn.manifold as skm
import pandas as pd
import numpy as np


def compute_triplet_accuracy(original_distances: np.ndarray, embedding_distances: np.ndarray):

    # this is pretty slow due to the nested loop
    sensor_count = original_distances.shape[0]
    assert {*original_distances.shape, *embedding_distances.shape} == {sensor_count}, 'Distances shape is off.'

    triplet_count = 0
    correct_count = 0
    for idx in range(sensor_count):
        for idx2 in range(idx + 1, sensor_count):
            for idx3 in range(idx2+1, sensor_count):

                # get the distances from the original dataframe
                dist12 = original_distances[idx, idx2]
                dist13 = original_distances[idx, idx3]

                # get the embedding distances
                edist12 = embedding_distances[idx, idx2]
                edist13 = embedding_distances[idx, idx3]

                # check whether the relative ordering is the same
                correct_count += (dist12 < dist13) == (edist12 < edist13)
                triplet_count += 1

    return correct_count / triplet_count


def compute_trustworthiness(distance_df: pd.DataFrame, embeddings: np.ndarray, n_neighbors: int):
    return skm.trustworthiness(distance_df.to_numpy(), embeddings, n_neighbors=n_neighbors, metric='precomputed')