# NLTK - Text Classification tool for Indian languages

**usage:** main.py [-h] [--lang LANG] [--emb_path EMB_PATH]<br/>
&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; [--full_vocab FULL_VOCAB] [--max_vocab MAX_VOCAB]<br/>
  &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp;             [--emb_dim EMB_DIM] [--k K] [--valid VALID]

## optional arguments: <br/>
  -h, --help               show this help message and exit<br/>
  --lang LANG              Language code (2-letter ISO code)<br/>
  --emb_path EMB_PATH      Path to the embeddings<br/>
  --full_vocab FULL_VOCAB  If all embeddings to be loaded<br/>
  --max_vocab MAX_VOCAB    Max number of embeddings to load<br/>
  --emb_dim EMB_DIM        Dimension of the embeddings<br/>
  --k K                    Number of neighbors to use for voting
  --valid VALID            if true get results on validaion data, useful for experimentation
