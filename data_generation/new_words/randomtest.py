from random_word import RandomWords


import random
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
import warnings



def main():
	warnings.filterwarnings("ignore")
	lemma = WordNetLemmatizer()


		


	w1_hypolist = list(set([w for s in wn.synsets(lemma.lemmatize('scientist'))[0].closure(lambda s:s.hyponyms()) for w in s.lemma_names()]))

	print(w1_hypolist)
	# x hypernym of y: 0
	# y hypernym of x: 1
	# no relation: 2






	#print(list(set([w for s in wn.synsets(lemma.lemmatize('drinks'))[0].closure(lambda s:s.hyponyms()) for w in s.lemma_names()])))
if __name__ == "__main__":
	main()
