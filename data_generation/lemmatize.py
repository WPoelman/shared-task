#!/usr/bin/python3

'''
Code to lemmatize the nouns and verbs of a dataset.
'''

import random
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
import warnings
import csv
import spacy


def lemmatize_all(sentence):
    wnl = WordNetLemmatizer()
    for word, tag in pos_tag(word_tokenize(sentence)):
        if tag.startswith("NN"):
            yield wnl.lemmatize(word, pos="n")
        elif tag.startswith("VB"):
            yield wnl.lemmatize(word, pos="v")
        else:
            yield word


def main():
    # nlp = spacy.load('it_core_news_md')
    # with open('It-Subtask1-test_lemma.tsv', 'w', encoding='UTF8') as f:
    # 	with open('It-Subtask1-test.tsv', 'r', encoding='UTF8') as o:
    # 		writer = csv.writer(f)
    # 		warnings.filterwarnings("ignore")
    # 		lemma = WordNetLemmatizer()

    # 		reader = csv.reader(o, delimiter='\t')
    # 		next(reader)
    # 		for line in reader:
    # 			doc = nlp(line[1])
    # 			print(line[0]," ".join([token.lemma_ for token in doc]))
    # 			writer.writerow([line[0]," ".join([token.lemma_ for token in doc])])

    with open(
        "3_newhypownt_newwordswnt_pegasus_shuffled_lemma.csv", "w", encoding="UTF8"
    ) as f:
        with open(
            "3_newhypownt_newwordswnt_pegasus_shuffled.csv", "r", encoding="UTF8"
        ) as o:
            writer = csv.writer(f)
            warnings.filterwarnings("ignore")
            lemma = WordNetLemmatizer()

            reader = csv.reader(o, delimiter=",")
            for line in reader:
                print(line[0], " ".join(lemmatize_all(line[1])).lower(), line[2])
                writer.writerow(
                    [line[0], " ".join(lemmatize_all(line[1])).lower(), line[2]]
                )


if __name__ == "__main__":
    main()
