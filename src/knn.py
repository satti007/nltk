"""K nearest neighbors search using faiss library."""

import faiss

res = faiss.StandardGpuResources()


def knn_faiss(database, queries, dim=300, k=5):
    """Get KNNs of all vectors in queries.

    Args:
        database - array of vectors from which neighbors are to be searched
        queries - array of vectors for which neighbors are to be searched
        dim - dimension of vectors
        k - k in KNN

    returns:
        dist - A matrix of shape (queries.shape[0], k)
             - distances of it's KNNs of each query

        idxs - A matrix of shape (queries.shape[0], k)
             - indicies of it's KNNs of each query
    """
    database = database.astype('float32')
    queries = queries.astype('float32')

    index = faiss.IndexFlatL2(dim)
    gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
    gpu_index.add(database)

    dist, idxs = gpu_index.search(queries, k)

    return dist, idxs
