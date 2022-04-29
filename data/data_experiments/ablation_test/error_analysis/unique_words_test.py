#!/usr/bin/python3

'''
Extract all unique nouns out of the task test data.
'''


import csv
import spacy
from pattern.en import pluralize, singularize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import PorterStemmer


def main():
    nlp = spacy.load("en_core_web_sm")
    ps = PorterStemmer()

    words = []
    with open("evaluation_results.csv", newline="") as csvfile:
        spamreader = csv.reader(csvfile, delimiter=",")
        for line in spamreader:
            k = nlp(line[2])
            subjects = [tok for tok in k if (tok.pos_ == "NOUN")]
            for subject in subjects:
                words.append(str(subject))
    words = list(set(words))
    with open("uniquewords_test.txt", "w") as outfile:
        outfile.write("\n".join(words))


if __name__ == "__main__":
    main()
