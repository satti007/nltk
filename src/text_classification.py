"""Text Classification."""

import numpy as np
from os import path
from knn import knn_faiss
from net import get_model
from sklearn.metrics import accuracy_score
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint


def _classify(database, queries, dim, k, y_train):
    dist, idxs = knn_faiss(database, queries, dim=300, k=5)
    queries_pred = []

    for neighbors in idxs:
        classes = [y_train[n] for n in neighbors]
        queries_pred.append(max(set(classes), key=classes.count))

    return queries_pred


def _get_vectors(x_train, x_valid, x_test, idx2vec):
    train_vecs = [idx2vec[idx] for idx in x_train]
    valid_vecs = [idx2vec[idx] for idx in x_valid]
    test_vecs = [idx2vec[idx] for idx in x_test]

    return [train_vecs, valid_vecs, test_vecs]


def get_results(x_train, x_valid, x_test,
                y_train, y_valid, y_test,
                idx2vec, emb_dim, valid,
                lang, emb_src,
                model, k, nn_dims):
    """Get the clasiification results by the model(knn, net)."""
    vectors = _get_vectors(x_train, x_valid, x_test, idx2vec)

    if model == 'knn':
        database = vectors[0]
        if valid:
            queries = vectors[1]
        else:
            queries = vectors[3]

        y_pred = _classify(np.vstack(database), np.vstack(queries),
                           emb_dim, k, y_train)
    elif model == 'nn':
        X_train, X_valid, X_test = np.vstack(vectors[0]), np.vstack(vectors[1]),\
            np.vstack(vectors[2])

        num_classes = np.unique(y_test).shape[0]
        y_train = np.vstack(y_train)
        y_train = np.squeeze(np.eye(num_classes)[y_train.reshape(-1)])
        y_valid = np.vstack(y_valid)
        y_valid = np.squeeze(np.eye(num_classes)[y_valid.reshape(-1)])

        model = get_model(emb_dim, nn_dims, num_classes)
        # print(model.summary())

        model.fit(X_train, y_train,
                  validation_data=(X_valid, y_valid),
                  callbacks=[EarlyStopping(monitor='val_loss',
                             patience=5, mode='min', verbose=0),
                             ModelCheckpoint('../data/checkpoints/best_model_{}_{}.h5'.format(lang, emb_src),
                             monitor='val_accuracy', mode='max', verbose=0,
                             save_best_only=True)],
                  verbose=0,
                  epochs=100)

        if valid:
            y_pred = model.predict(X_valid)
        else:
            y_pred = model.predict(X_test)

        y_pred = np.argmax(y_pred, axis=1)
        y_valid = np.argmax(y_valid, axis=1)

    if valid:
        accuracy = accuracy_score(y_valid, y_pred)
    else:
        accuracy = accuracy_score(y_test, y_pred)

    # if np.unique(y_test).shape[0] == 2:
    #     avg = 'binary'
    # else:
    #     avg = 'macro'

    # f1_value = f1_score(y_test, y_pred, average=avg)
    f1_value = 0

    return accuracy, f1_value
