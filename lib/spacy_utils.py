import re
from time import time  # To time our operations

def cleaning(doc,
             lemmatize=True,
             remove_stopwords=True,
             min_sent_length=3):
    # Lemmatizes and removes stopwords
    # doc needs to be a spacy Doc object
    if lemmatize:
        txt = []
        for token_obj in doc:
            token = token_obj.text
            if lemmatize:
                token = token_obj.lemma_

            if all([token_obj.is_stop,
                    remove_stopwords]):
                continue

            txt.append(token)


    # Word2Vec uses context words to learn the vector representation of a target word,
    # if a sentence is only one or two words long,
    # the benefit for the training is very small
    if len(txt) >= min_sent_length:
        return ' '.join(txt)

def clean_txt(df,
              nlp,
              settings,
              verbose=0):
    t = time()

    brief_cleaning = (re.sub("[^A-Za-z']+", ' ', str(row)).lower()
                      for row in df[settings['input']['column_to_process']])

    txt = [cleaning(doc,
                    lemmatize=settings['preprocessing']['lemmatize'],
                    remove_stopwords=settings['preprocessing']['remove_stopwords'],
                    min_sent_length=settings['preprocessing']['min_sent_length'])
           for doc in
           nlp.pipe(brief_cleaning,
                    batch_size=settings['spacy']['batch_size'],
                    n_threads=settings['spacy']['threads'])]

    if verbose >= 1:
        print('Time to clean up everything: {} mins'.format(round((time() - t) / 60, 2)))

    return txt



