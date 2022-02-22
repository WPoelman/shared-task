#!/usr/bin/env python3

"""
Description:
    A script that shows several baselines for the classification subtask for
    the PreTENS shared task. It has the following baselines:
        - A basic most frequent baseline
        - A TF-IDF with SVC classifier
        - A SVC classifier using multi-lingual sentence embeddings
    By default it runs the selected baseline(s) for all languages.

"""

import argparse
import pickle
import time
from pathlib import Path

import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import *
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

DATA_DIR = Path().cwd().parent / "data"
RESULTS_DIR = Path().cwd() / "results"
CACHE_DIR = Path().cwd() / "cache"


def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--most_frequent",
        action="store_true",
        help="Use the most frequent class baseline",
    )
    parser.add_argument(
        "-s", "--svc", action="store_true", help="Use the TF-IDF and SVC as baseline"
    )
    parser.add_argument(
        "-e",
        "--embeddings",
        action="store_true",
        help="Use the multi-lingual sentence embedding baseline",
    )
    parser.add_argument(
        "-l",
        "--languages",
        nargs="+",
        help="Language datasets to use for the baselines",
        default=["en", "fr", "it"],
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=True,
        help="Show progression output",
    )
    parser.add_argument(
        "-c",
        "--cache",
        default=False,
        action="store_true",
        help="Use cached sentence embeddings with -e",
    )
    parser.add_argument(
        "-cv", "--cross_validation", default=5, help="Cross validation folds"
    )
    # Default model repo:
    #   https://huggingface.co/sentence-transformers/paraphrase-xlm-r-multilingual-v1
    # Also tried with:
    #   https://huggingface.co/sentence-transformers/LaBSE
    # Note that this script downloads the selected model (~2gb) the first time
    # it is ran.
    parser.add_argument(
        "-mo",
        "--model",
        default="paraphrase-xlm-r-multilingual-v1",
        help="Sentence embedding model to use with -e",
    )

    args = parser.parse_args()
    return args


def rnd(x, digits=5):
    """Helper to make rounding consistent"""
    return round(x, ndigits=digits)


def output_metrics(y_true, y_pred, dataset_label, digits=5):
    """Output the usual performance metrics and a classification report"""

    ac = accuracy_score(y_true, y_pred)
    pr = precision_score(y_true, y_pred, average="macro", zero_division=0)
    re = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    msg = f"""
    --- {dataset_label} ---
    {classification_report(y_true, y_pred, digits=digits, zero_division=0)}

    Accuracy:   {rnd(ac, digits)}
    Precision:  {rnd(pr, digits)}
    Recall:     {rnd(re, digits)}
    F-score:    {rnd(f1, digits)}
    """
    return msg


def evaluate_model(args, model, lang, X, y, model_name):
    """Evaluate a given model"""
    args.verbose and print(f"Evaluating {model_name} for {lang.upper()} ...")

    y_pred = cross_val_predict(model, X, y, cv=args.cross_validation)
    result_label = f"{lang} {args.cross_validation}-fold CV"
    report = output_metrics(y, y_pred, result_label)

    args.verbose and print(report)

    result_filename = f"{model_name}_results_{lang}_{time.time()}.txt"
    with open(RESULTS_DIR / result_filename, "w") as f:
        f.write(report)


def embedding_model(args, lang, X, y):
    """Create sentence embeddings and evaluate a simple linear model"""
    # huggingface namingstyle is 'org/model', which is an illegal path
    model_id = args.model.replace("/", "-")
    cache_file = Path(CACHE_DIR / f"{lang}_{model_id}_cache.pickle")

    args.verbose and print(f"Creating sentence embeddings ...")
    if args.cache and cache_file.is_file():
        with open(cache_file, "rb") as f:
            embeddings = pickle.load(f)
    else:
        args.verbose and print(f"Loading {args.model} ...")
        transformer = SentenceTransformer(args.model)
        embeddings = transformer.encode(X)
        with open(cache_file, "wb") as f:
            pickle.dump(embeddings, f)

    evaluate_model(args, SVC(), lang, embeddings, y, f"{model_id}_svc")


def svc_model(args, lang, X, y):
    """Create tf-idf embeddings and evaluate a simple linear model"""
    vectorizer = TfidfVectorizer()
    clf = SVC(random_state=0)
    model = Pipeline([("vec", vectorizer), ("cls", clf)])
    evaluate_model(args, model, lang, X, y, "tfidf_svc")


def most_freq_model(args, lang, X, y):
    """Create a most frequent classifier"""
    model = DummyClassifier(strategy="most_frequent", random_state=0)
    evaluate_model(args, model, lang, X, y, "most_frequent")


def main():
    args = create_arg_parser()

    for lang in args.languages:
        df = pd.read_csv(DATA_DIR / f"subtask_1_{lang}.csv")
        X = df["sentence"].tolist()
        y = df["labels"].tolist()

        if args.embeddings:
            embedding_model(args, lang, X, y)
        if args.svc:
            svc_model(args, lang, X, y)
        if args.most_frequent:
            most_freq_model(args, lang, X, y)


if __name__ == "__main__":
    main()
