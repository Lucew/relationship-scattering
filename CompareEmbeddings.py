import time
import functools
import multiprocessing as mp
from glob import glob
import warnings

import pandas as pd
import numpy as np
import umap
import sklearn.manifold as skm
import threadpoolctl
import scipy.spatial.distance as spsd
from numpy.lib.function_base import extract
from tqdm import tqdm

import EvaluationMetrics as evm


def evaluate_comparison(input_tuple: tuple[str, str, dict, int]):

    # imap unordered from multiprocessing only support one input argument
    # so we need to unpack the tuple
    method_name, file_name, parameters, random_seed = input_tuple

    # read the distances into the memory and set the main diagonal to zero distance
    distances = pd.read_csv(file_name, index_col=0, header=0)
    assert distances.isna().sum().sum() == distances.shape[0], 'Something is off.'
    distances[distances.isna()] = 0

    # get the method and use the parameters
    method_dict = {"Isomap": functools.partial(skm.Isomap, metric="precomputed", **parameters),
                   "Met. MDS": functools.partial(skm.MDS, dissimilarity="precomputed", metric=True, **parameters),
                   "Non-Met. MDS": functools.partial(skm.MDS, dissimilarity="precomputed", metric=False, **parameters),
                   "TSNE": functools.partial(skm.TSNE, metric="precomputed", init="random", **parameters),
                   "UMAP": functools.partial(umap.UMAP, metric='precomputed', **parameters),}
    assert method_name in method_dict, f'Method {method_name} not defined.'

    # create the embedding using the methods
    method = method_dict[method_name]()

    # set the random seed so we have the comparisons
    np.random.seed(random_seed)

    # only use one thread
    with threadpoolctl.threadpool_limits(limits=1):

        # https://docs.python.org/3/library/warnings.html#temporarily-suppressing-warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            time_start = time.perf_counter()
            try:
                embedding = method.fit_transform(distances)
            except ValueError as e:
                print(method_name, file_name, distances.isnull().sum().sum(), distances.isna().sum().sum(), distances.isin([np.nan, np.inf, -np.inf]).sum().sum())
                print(e)
                return method_name, file_name, random_seed, -1, [(-1, -1.0), (-1, -1.0)], -1

        embedding_duration = time.perf_counter() - time_start

        # compute the pairwise Euclidean distance of the output
        pairwise_distances = spsd.squareform(spsd.pdist(embedding, metric='euclidean'))

        # compute the local metric (trustworthiness) for some neighborhood values
        trustworthiness = [(n_count, evm.compute_trustworthiness(distances, embedding, n_count)) for n_count in range(4, 20, 2)]

        # compute the global metric (triplet accuracy)
        triplet_accuracy = evm.compute_triplet_accuracy(distances.to_numpy(), pairwise_distances)

    return method_name, file_name, random_seed, embedding_duration, trustworthiness, triplet_accuracy


def main():

    # get the files from the data folder
    files = glob(f"data/*/*.csv")

    # define the seeds
    seeds = list(range(10, 60))

    # make a generator for the parameters
    data_generator = [(method_name, file, dict(), seed)
                      for seed in seeds
                      for file in files
                      for method_name in ["Isomap", "Met. MDS", "Non-Met. MDS", "TSNE", "UMAP"]]

    # create the jobs on the workers
    data = []
    with mp.Pool(mp.cpu_count()//2) as pool:
        for result in tqdm(pool.imap_unordered(evaluate_comparison, data_generator), "Computing embeddings", total=len(data_generator)):
            data.append(result)
    results = pd.DataFrame(data, columns=["Method", "File", "Random Seed", "Embedding Time", "Trustworthiness", "Triplet Accuracy"])
    results.to_parquet(f"Result_{time.time()}.parquet")



if __name__ == '__main__':
    main()
