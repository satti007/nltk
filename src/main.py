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

parser.add_argument('--model', type=str, default='knn',
                    choices=['knn', 'nn'],
                    help='Model to used for classification')

parser.add_argument('--k', type=int, default=5,
                    help='Number of neighbors to use for voting')

parser.add_argument('--nn_dims', type=str, default='0',
                    help="""Dimensions of hidden layers of nn.
                    Format: hidden_layer1_dim, hidden_layer2_dim""")

parser.add_argument('--valid', type=str, default=False,
                    help="""if true get results on validaion data,
                             useful for experimentation""")

args = parser.parse_args()
args.full_vocab = True if 'true' in args.full_vocab.lower() else False
args.valid = True if 'true' in args.valid.lower() else False

if args.model == 'nn':
    args.nn_dims = args.nn_dims.split(',')
    args.nn_dims = [int(dim.strip()) for dim in args.nn_dims]

emb_src = args.emb_path.split('/')[-2]

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

    accuracy, f1_value = get_results(X_train, X_valid, X_test,
                                     y_train, y_valid, y_test,
                                     idx2vec, args.emb_dim, args.valid,
                                     args.lang, emb_src,
                                     args.model, args.k, args.nn_dims)

    print('\t--Metrics are accuracy: {:.4f}, f1_score: {:.4f}\n'.format(
        accuracy, f1_value))
