# Train word2vec model using gensim
The purpose of this repository is to train a word2vec model using gensim.
The information from [this blog](https://www.kaggle.com/pierremegret/gensim-word2vec-tutorial) was used.

### Prerequisites
Python 3.6 was used to create this project. It might work with older versions of Python.

### Python modules
A number of external modules need to be installed, which are listed in **requirements.txt**.
Depending on how you installed Python, you can probably install the requirements using one of following commands:
```bash
pip install -r requirements.txt
```

### Resources
Please run `bash install` to download the Dutch spaCy language models.
Please install models for other languages if required.

### How to use
1. edit `config/word2vec_settings.json` 
2. call `python lib/train_word2vec_model.py -h` for help on how to call the script

### Authors
* Marten Postma (m.c.postma@vu.nl)