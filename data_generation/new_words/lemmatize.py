#!/usr/bin/python3

import random
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
import warnings
import csv


def lemmatize_all(sentence):
    wnl = WordNetLemmatizer()
    for word, tag in pos_tag(word_tokenize(sentence)):
        if tag.startswith("NN"):
            yield wnl.lemmatize(word, pos='n')
        elif tag.startswith('VB'):
            yield wnl.lemmatize(word, pos='v')
        else:
            yield word

def main():
	with open('1_newhypownt_newwordswnt_inverse_pegasus_pegasus_lemma.csv', 'w', encoding='UTF8') as f:
		with open('1_newhypownt_newwordswnt_inverse_pegasus_pegasus.csv', 'r', encoding='UTF8') as o:
			writer = csv.writer(f)
			warnings.filterwarnings("ignore")
			lemma = WordNetLemmatizer()

			reader = csv.reader(o, delimiter=',')
			for line in reader:
				print(line[0],' '.join(lemmatize_all(line[1])).lower(),line[2])
				writer.writerow([line[0],' '.join(lemmatize_all(line[1])).lower(),line[2]])


	

if __name__ == "__main__":
	main()