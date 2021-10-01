#!/usr/bin/env python3

'''
Description:
    A basic script that uses multi-lingual sentence embeddings on the sentences
    and trains a simple SVM classifier with those.

-- TODO --
- use cached embeddings to speed up process -> by id
- add command line args instead of constants
- tune parameters svm???
- save models / output
'''

import pickle
import time
from pathlib import Path

import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import cross_val_predict
from sklearn.svm import LinearSVC

from utils import output_metrics

DATA_DIR = Path().cwd().parent / 'data'
RESULTS_DIR = Path().cwd() / 'results'
CACHE_DIR = Path().cwd() / 'cache'

# Model repo: https://huggingface.co/sentence-transformers/LaBSE
# Note that this script downloads the model (~2gb) the first time it is ran.
MODEL_NAME = 'sentence-transformers/LaBSE'
LANGUAGES = ['en', 'fr', 'it']
VERBOSE = True
USE_CACHE = True
CV_FOLDS = 5


def main():
    VERBOSE and print(f'Loading {MODEL_NAME} ...')
    transformer = SentenceTransformer(MODEL_NAME)

    for lang in LANGUAGES:
        VERBOSE and print(f'{lang.upper()} -> embedding ...')

        df = pd.read_csv(DATA_DIR / f'subtask_1_{lang}.csv')

        # TODO maybe add model-specific cache to avoid mistakes when swapping
        # out models -> for later
        cache_file = Path(CACHE_DIR / f'{lang}_embedding_cache.pickle')

        if USE_CACHE and cache_file.is_file():
            with open(cache_file, 'rb') as f:
                embeddings = pickle.load(f)
        else:
            embeddings = transformer.encode(df['sentence'].tolist())
            with open(cache_file, 'wb') as f:
                # TODO don't rely on ordering, use id of row instead
                # id2embedding = {
                #     id_: embeddings[i]
                #     for i, id_ in enumerate(df['id'].tolist())
                # }
                pickle.dump(embeddings, f)

        VERBOSE and print(f'{lang.upper()} -> cross validating ...')

        clf_for_lang = LinearSVC()
        y_pred = cross_val_predict(
            clf_for_lang, embeddings, df['labels'], cv=CV_FOLDS
        )

        result_label = f'{lang} {CV_FOLDS}-fold CV'
        report = output_metrics(df['labels'], y_pred, result_label)

        result_filename = f'results_{result_label}_{time.time()}.txt'
        with open(RESULTS_DIR / result_filename, 'w') as f:
            f.write(report)
        print(report)


if __name__ == '__main__':
    main()
