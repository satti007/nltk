# NLTK - Text Classification tool for Indian languages

usage: main.py [-h] [--lang LANG] [--emb_path EMB_PATH]
               [--full_vocab FULL_VOCAB] [--max_vocab MAX_VOCAB]
               [--emb_dim EMB_DIM] [--k K] [--valid VALID]

optional arguments:
  -h, --help            show this help message and exit
  --lang LANG           Language code (2-letter ISO code)
  --emb_path EMB_PATH   Path to the embeddings
  --full_vocab FULL_VOCAB
                        If all embeddings to be loaded
  --max_vocab MAX_VOCAB
                        Max number of embeddings to load
  --emb_dim EMB_DIM     Dimension of the embeddings
  --k K                 Number of neighbors to use for voting
  --valid VALID         if true get results on validaion data, useful for
                        experimentation
