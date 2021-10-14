#!/usr/bin/python3

import random
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer

class Sentence:
	def __init__(self,verb,word1,word2):
		self.verb = verb
		self.word1 = word1
		self.word2 = word2
	def output(self):
		templatelist = [("I {verb} {word1} , except {word2} .",0), 
						("I {verb} {word1} , and more specifically {word2} .",0), 
						("I {verb} {word1} , but not {word2} .",0),
						("I {verb} {word1} , an interesting type of {word2} .",1),
						("I {verb} {word1} more than {word2} .",2),
						("I do not like {word1} , I prefer {word2} .",2),
						("I {verb} {word1} , and {word2} too .",2)
						]
		sent_label = random.choice(templatelist)
		return sent_label[0].format(verb=self.verb,word1=self.word1,word2=self.word2), sent_label[1]


def main():
	words = ['trees','oaks','Kawasakis', 'motorcycles', 'terriers', 'poodles', 'dogs', 'cats']
	verbs = ['like','use', 'met', 'trust']

	lemma = WordNetLemmatizer()

	for i in range(10):
		verb = random.choice(verbs)
		word1 = random.choice(words)
		word2 = random.choice(words)
		generated_sent = Sentence(verb, word1, word2).output()

		try:
			w1_hypolist = list(set([w for s in wn.synsets(lemma.lemmatize(word1))[0].closure(lambda s:s.hyponyms()) for w in s.lemma_names()]))
			w2_hypolist = list(set([w for s in wn.synsets(lemma.lemmatize(word2))[0].closure(lambda s:s.hyponyms()) for w in s.lemma_names()]))

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