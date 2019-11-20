"""File to read the datasets."""

import os
import numpy as np
from embeddings import read_txt_embeddings
from data_preprocess import sent_split, preprocess_sent
from sklearn.model_selection import train_test_split

seperator = '__________________________\n'
data_dir = '../data/classification_datasets/'


def read_data(lang):
    """Read the classification data.

    Args:
        lang - language code (2-letter ISO code)

    returns:
        datasets - dict containing texts and labels of each data_folder
                 - datasets[data_folder] has two dicts idx2words, idx2label
    """
    print(f'Reading {lang} language resources')
    datasets = {}
    data_folders = os.listdir(os.path.join(data_dir, lang))
    print('\t--Resources of classification data are: ', ', '.join(data_folders))

    data_folders.sort()
    for data_folder in data_folders:
        print(f'\t--Loading {data_folder} resource')
        idx2words = {}
        idx2label = {}
        data_files = os.listdir(os.path.join(data_dir, lang, data_folder))
        print('\t\t--Classes present are: ', ', '.join(data_files))

        idx = 0
        label = 0
        data_files.sort()
        for data_file in data_files:
            count = 0
            lines = open(os.path.join(data_dir, lang,
                                      data_folder, data_file), 'r').readlines()

            line_num = 0
            while line_num < len(lines):
                if lines[line_num] == seperator:
                    next_seperator = line_num + 1

                    while next_seperator < len(lines):
                        if lines[next_seperator] == seperator:
                            break
                        next_seperator += 1

                    text = lines[line_num + 1:next_seperator]
                    text = ''.join(text)
                    if len(text) > 0:
                        sentences = sent_split(text, lang)
                        words = []
                        for sent in sentences:
                            words.extend(preprocess_sent(sent, lang))
                        idx2words[idx] = words
                        idx2label[idx] = label
                        idx += 1
                        count += 1

                line_num = next_seperator

            label += 1
            print(f'\t\t--Num of data points in {data_file} are: {count}')

        datasets[data_folder] = {'words': idx2words, 'labels': idx2label}

    return datasets


def text2vec(args):
    """Get the represnetation of the text."""
    datasets = read_data(args.lang)

    print('\nLoading pre-trained word embeddings...')
    embeddings, word2id, id2word = read_txt_embeddings(args.emb_path,
                                                       args.full_vocab,
                                                       args.max_vocab,
                                                       args.emb_dim)
    OVV_words = {}
    for data_folder in datasets:
        idx2vec = {}
        idx2words = datasets[data_folder]['words']
        for idx in idx2words:
            words = idx2words[idx]
            vectors = []
            for w in words:
                try:
                    vectors.append(embeddings[word2id[w]])
                except KeyError:
                    if w in OVV_words:
                        OVV_words[w] += 1
                    else:
                        OVV_words[w] = 1

                    continue

            if len(vectors) > 0:
                idx2vec[idx] = np.mean(vectors, axis=0)
            else:
                idx2vec[idx] = np.zeros((args.emb_dim))

        datasets[data_folder]['vectors'] = idx2vec

    print('The number of OVV words are :{}\n'.format(len(OVV_words)))

    return datasets, OVV_words


def data_splitting(data, labels, fraction):
    """Data splitting."""
    X_train, X_valid, y_train, y_valid = train_test_split(data, labels,
                                                          test_size=fraction,
                                                          random_state=42)

    X_train, X_test, y_train, y_test = train_test_split(X_train,
                                                        y_train,
                                                        test_size=fraction /
                                                        (1 - fraction),
                                                        random_state=42)

    return X_train, X_valid, X_test, y_train, y_valid, y_test


def get_label_dist(labels):
    values_counts = np.unique(labels, return_counts=True)
    values, counts = values_counts[0], values_counts[1]
    val2c = {v: c for v, c in zip(values, counts)}

    return val2c

# def test():
#     """Testing read_data function."""
#     datasets = text2vec('te', ['../data/embeds/indicnlp.v1.te.vec', True, 100])
#     for data_folder in datasets:
#         print(data_folder)
#         idx2label = datasets[data_folder]['labels']
#         idx2vec = datasets[data_folder]['vectors']
#         labels = list(idx2label.values())
#         print(np.unique(labels, return_counts=True))
#         print(idx2vec[0].shape)

# test()
