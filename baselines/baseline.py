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
from sklearn.svm import LinearSVC, SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from utils import output_metrics
import argparse

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


def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_file", default='reviews.txt', type=str,
                        help="Input file to learn from (default reviews.txt)")
    parser.add_argument("-mf", "--most_frequent", action="store_true",
                        help="Use the most frequent class baseline")
    parser.add_argument("-s", "--SVC", action="store_true",
                        help="Use the SVC with linear kernel as baseline")
    parser.add_argument("-e", "--embeddings", action="store_true",
                        help="Use the multi-lingual sentence embeddings baseline")

    args = parser.parse_args()
    return args


def identity(x):
    """Dummy function that just returns the input"""
    return x


def main():
    args = create_arg_parser()

    ## Baseline 1: Multi-lingual embeddings:
    if args.embeddings:
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

    ## Baseline 2. Using SVC without embeddings
    elif args.SVC:

        # We use a dummy function as tokenizer and preprocessor,
        # since the texts are already preprocessed and tokenised.
        vec = TfidfVectorizer(preprocessor=identity, tokenizer=identity)

        # Combine the vectorizer with the SVC classifier
        clf = SVC(random_state=0)
        classifier = Pipeline([('vec', vec), ('cls', clf)])

        for lang in LANGUAGES:

            # Read the language file
            df = pd.read_csv(DATA_DIR / f'subtask_1_{lang}.csv')
            VERBOSE and print(f'{lang.upper()} -> cross validating ...')

            # Perform K-fold cross validation
            y_pred = cross_val_predict(
                classifier, df['sentence'], df['labels'], cv=CV_FOLDS
            )

            # Perform the evaluation, write the scores to a file and print the scores as well
            result_label = f'{lang} {CV_FOLDS}-fold CV'
            report = output_metrics(df['labels'], y_pred, result_label)
            result_filename = f'SVC_results_{result_label}_{time.time()}.txt'
            with open(RESULTS_DIR / result_filename, 'w') as f:
                f.write(report)
            print(report)


    ## Baseline 3. Using the most frequent class
    elif args.most_frequent:
        # setup classifier, no need to vectorize
        clf = DummyClassifier(strategy="most_frequent", random_state=0)
   
        for lang in LANGUAGES:     
            df = pd.read_csv(DATA_DIR / f'subtask_1_{lang}.csv')
            # perform K-fold cross validation
            y_pred = cross_val_predict(clf, df['sentence'], df['labels'])
            
            # Perform the evaluation, write the scores to a file and print the scores as well
            result_label = f'{lang} {CV_FOLDS}-fold CV'
            report = output_metrics(df['labels'], y_pred, result_label)
            result_filename = f'MF_results_{result_label}_{time.time()}.txt'
            with open(RESULTS_DIR / result_filename, 'w') as f:
                f.write(report)
            print(report)
        
        


if __name__ == '__main__':
    main()
