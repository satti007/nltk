"""Getting and parsing the argumnets."""
import pprint
import argparse
from data_prep import text2vec, data_splitting, get_label_dist
from text_classification import get_results

parser = argparse.ArgumentParser()

parser.add_argument('--lang', type=str,
                    help='Language code (2-letter ISO code)')

parser.add_argument('--emb_path', type=str,
                    help='Path to the embeddings')

parser.add_argument('--full_vocab', type=str, default='True',
                    help='If all embeddings to be loaded')

parser.add_argument('--max_vocab', type=int, default=200000,
                    help='Max number of embeddings to load')

parser.add_argument('--emb_dim', type=int, default=300,
                    help='Dimension of the embeddings')

parser.add_argument('--k', type=int, default=5,
                    help='Number of neighbors to use for voting')

parser.add_argument('--valid', type=str, default=False,
                    help="""if true get results on validaion data,
                             useful for experimentation""")

args = parser.parse_args()
args.full_vocab = True if 'true' in args.full_vocab.lower() else False
args.valid = True if 'true' in args.valid.lower() else False
pprint.pprint(vars(args), width=10000)
print()

datasets, OVV_words = text2vec(args)

for data_folder in datasets:
    print('Classifying text {} resource'.format(data_folder))
    idx2vec, idx2label = datasets[data_folder]['vectors'], \
        datasets[data_folder]['labels']

    X_train, X_valid, X_test, y_train, y_valid, y_test = \
        data_splitting(list(idx2label.keys()), list(idx2label.values()), 0.10)

    print('\t--Label distribution in train: ', get_label_dist(y_train))
    print('\t--Label distribution in valid: ', get_label_dist(y_valid))
    print('\t--Label distribution in test: ', get_label_dist(y_test))

    if args.valid:
        accuracy, f1_value = get_results(X_train, X_valid, y_train, y_valid,
                                         idx2vec, args.emb_dim, args.k)
    else:
        accuracy, f1_value = get_results(X_train, X_test, y_train, y_test,
                                         idx2vec, args.emb_dim, args.k)

    print('\t--Metrics are accuracy: {:.4f}, f1_score: {:.4f}\n'.format(
        accuracy, f1_value))
