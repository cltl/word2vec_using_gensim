from time import time
import multiprocessing

from gensim.models import Word2Vec

def initialize_word2vec_model(settings):
    w2v_model = Word2Vec(min_count=settings['word2vec']['min_count'],
                         window=settings['word2vec']['window'],
                         size=settings['word2vec']['size'],
                         sample=settings['word2vec']['sample'],
                         alpha=settings['word2vec']['alpha'],
                         min_alpha=settings['word2vec']['min_alpha'],
                         negative=settings['word2vec']['negative'],
                         workers=settings['word2vec']['workers'])

    return w2v_model

def build_vocab(model, sentences, settings, verbose=0):
    t = time()

    model.build_vocab(sentences, progress_per=settings['word2vec']['vocab_progress_per'])

    if verbose >= 2:
        print('Time to build vocab: {} mins'.format(round((time() - t) / 60, 2)))

    return model


def train_model(model,
                sentences,
                settings,
                init_sims=True,
                verbose=0):
    t = time()

    model.train(sentences,
                total_examples=model.corpus_count,
                epochs=settings['word2vec']['epochs'],
                report_delay=settings['word2vec']['report_delay'])

    if verbose >= 2:
        print('Time to train the model: {} mins'.format(round((time() - t) / 60, 2)))

    if init_sims:
        model.init_sims(replace=True)

    return model