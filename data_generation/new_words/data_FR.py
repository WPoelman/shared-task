#!/usr/bin/python3

import random
import spacy
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer

class Sentence:
	def __init__(self,verb,word1,word2):
		self.verb = verb
		self.word1 = word1
		self.word2 = word2
	def output(self):
		templatelist = [("J' {verb} {word1} , sauf {word2} .",0), 
						("J' {verb} {word1} , et plus particulièrement {word2} .",0),
						("J' {verb} {word1} , mais pas {word2} .",0),
						("J' {verb} {word1} , un type intéressant de {word2} .",1),
						("J' {verb} {word1} plus que {word2} .",2),
						("Je n' aime pas {word1} , je préfère {word2} .",2),
						("J' {verb} {word1} , et aussi {word2} .",2)
						]
		sent_label = random.choice(templatelist)
		return sent_label[0].format(verb=self.verb,word1=self.word1,word2=self.word2), sent_label[1]


def main():
	words = ['les arbres','les chênes','les Kawasakis', 'les motos', 'les terriers', 'les caniches', 'les chiens', 'les chats']
	verbs = ['aime','utilise', 'ai rencontré', 'e fais confiance']

	nlp = spacy.load('fr_core_news_lg')

	for i in range(10):
		verb = random.choice(verbs)
		word1 = random.choice(words)
		word2 = random.choice(words)
		
		while word1 == word2:
			word2 = random.choice(words)

		generated_sent = Sentence(verb, word1, word2).output()

		try:
			lemma1 = [tok.lemma_ for tok in nlp(word1.split()[1])][0]
			lemma2 = [tok.lemma_ for tok in nlp(word2.split()[1])][0]
			w1_hypolist = list(set([w for s in wn.synsets(lemma1, lang='fra')[0].closure(lambda s:s.hyponyms()) for w in s.lemma_names()]))
			w2_hypolist = list(set([w for s in wn.synsets(lemma2, lang='fra')[0].closure(lambda s:s.hyponyms()) for w in s.lemma_names()]))

			# x hypernym of y: 0
			# y hypernym of x: 1
			# no relation: 2

			if lemma.lemmatize(word2) in w1_hypolist:
				x = 0
			elif lemma.lemmatize(word1) in w2_hypolist:
				x = 1
			else:
				x = 2
		except:
			x = 2

		if generated_sent[1] == x:
			print(generated_sent[0], 1)
		else:
			print(generated_sent[0], 0)

if __name__ == "__main__":
	main()
