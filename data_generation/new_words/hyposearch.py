#!/usr/bin/python3

'''
Manually search for hyponyms to add to the wordlists of all categories. The first argument is the word
you want all hyponyms of.
'''

from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
import warnings
import sys


def main():
    warnings.filterwarnings("ignore")
    lemma = WordNetLemmatizer()

    term = sys.argv[1]

    w1_hypolist = list(
        set(
            [
                w
                for s in wn.synsets(lemma.lemmatize(term))[0].closure(
                    lambda s: s.hyponyms()
                )
                for w in s.lemma_names()
            ]
        )
    )
    print(w1_hypolist)


if __name__ == "__main__":
    main()
