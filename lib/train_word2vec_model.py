"""
Train a word2vec model using gensim

Usage:
  train_word2vec_model.py --path_config_json=<path_config_json> --verbose=<verbose>

Options:
    --path_config_json=<path_config_json> path to JSON with settings
    --verbose=<verbose> 0 nothing, 1 descriptive stats, 2 debugging information

Example:
    python train_word2vec_model.py --path_config_json="../config/word2vec_settings.json" --verbose="2"
"""
import logging  # Setting up the loggings to monitor gensim
logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt= '%H:%M:%S', level=logging.INFO)
from docopt import docopt
import json

import spacy

import word2vec_utils
import spacy_utils
import df_utils
import path_utils

# load arguments
arguments = docopt(__doc__)
print()
print('PROVIDED ARGUMENTS')
print(arguments)
print()

verbose = int(arguments['--verbose'])
settings = json.load(open(arguments['--path_config_json']))

# main components
nlp = spacy.load(settings['spacy']['modelname'],
                 disable=['ner', 'parser']) # disabling Named Entity Recognition for speed

# load data
df = df_utils.load_data(csv_path=settings['input']['path_to_csv'],
                        verbose=verbose)


# preprocessing
cleaned_txt = spacy_utils.clean_txt(df,
                                    nlp,
                                    settings,
                                    verbose=verbose)

# load txt df
text_df = df_utils.load_cleaned_df(cleaned_txt,
                                   verbose=verbose)

sentences = [row.split() for row in text_df['clean']]

model = word2vec_utils.initialize_word2vec_model(settings)

model = word2vec_utils.build_vocab(model, sentences, settings, verbose=verbose)

word2vec_utils.train_model(model, sentences, settings, verbose=verbose)

# save model
path_utils.save_model_and_settings(model, settings, verbose=verbose)


