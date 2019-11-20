"""Text Classification."""

import numpy as np
from knn import knn_faiss
from sklearn.metrics import accuracy_score, f1_score


def _classify(database, queries, dim, k, y_train):
    dist, idxs = knn_faiss(database, queries, dim=300, k=5)
    queries_pred = []

    for neighbors in idxs:
        classes = [y_train[n] for n in neighbors]
        queries_pred.append(max(set(classes), key=classes.count))

    return queries_pred


def get_results(x_train, x_test, y_train, y_test,
                idx2vec, emb_dim, k):
    database = [idx2vec[idx] for idx in x_train]
    queries = [idx2vec[idx] for idx in x_test]
    y_pred = _classify(np.vstack(database), np.vstack(queries),
                       emb_dim, k, y_train)

    accuracy = accuracy_score(y_test, y_pred)

    if np.unique(y_test).shape[0] == 2:
        avg = 'binary'
    else:
        avg = 'macro'

    f1_value = f1_score(y_test, y_pred, average=avg)

    return accuracy, f1_value
