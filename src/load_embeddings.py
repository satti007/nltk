"""
Function to load word embeddings saved in a txt file.

The format of embeddings file should be as follows:
    dimension           * FIRST LINE(has the dimension of the embeddings)
    word1 x1 x2 ... xd  * SECOND LINE(word1 follwoed by space separated vector)
    word2 y1 y2 ... yd

"""

import io
import numpy as np


def read_txt_embeddings(emb_path, full_vocab=False,
                        max_vocab=200000, emb_dim=300):
    """Read the embeddings from the file.

    Args:
        emb_path - path to embeddings file
        full_vocab - boolean variable
                   - whether to load all the embeddings from the file
        max_vocab - max number of embeddings to load
        emb_dim - dimension of the embeddings

    returns:
        embeddings - an array of word vectors
                   - vector of word "W" can be accessed by:
                   - embeddings[word2id['W']]
        word2id - dictionary to store the index of word 'W' as:
                - word2id['W'] = id
        id2word - a dict from id to words
    """
    word2id = {}
    vectors = []

    with io.open(emb_path, 'r', newline='\n', errors='ignore') as f:
        for i, line in enumerate(f):
            if i == 0:
                split = line.split()
                assert len(split) == 2
                assert emb_dim == int(split[1])
                # print (split[0] + '\t' + split[1])
            else:
                # split the line into word and vector
                word, vect = line.rstrip().split(' ', 1)
                # form a vector from space separated string
                vect = np.fromstring(vect, sep=' ')
                # to avoid having null embeddings
                if np.linalg.norm(vect) == 0:
                    vect[0] = 0.01

                if word not in word2id:
                    if not vect.shape == (emb_dim,):
                        print(f'Invalid dimension {vect.shape[0]} for word {word} in line {i}')
                    else:
                        word2id[word] = len(word2id)
                        vectors.append(vect[None])

            if max_vocab > 0 and len(word2id) >= max_vocab and not full_vocab:
                break

    assert len(word2id) == len(vectors)
    print('Loaded %i word embeddings.' % len(vectors))

    id2word = {v: k for k, v in word2id.items()}
    embeddings = np.concatenate(vectors, 0)

    return embeddings, word2id, id2word
