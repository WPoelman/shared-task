#!/usr/bin/python3

import random
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
import warnings


class Sentence:
	def __init__(self,verb,word1,word2):
		self.verb = verb
		self.word1 = word1
		self.word2 = word2
	def output(self):
		templatelist = [("I {verb} {word1} , except {word2} .",0), 
						("I {verb} {word1} , and more specifically {word2} .",0), 
						("I {verb} {word1} , but not {word2} .",0 and 2),
						("I {verb} {word1} , an interesting type of {word2} .",1),
						("I {verb} {word1} more than {word2} .",2),
						("I do not {verb} {word1} , I prefer {word2} .",2),
						("I {verb} {word1} , and {word2} too .",2)
						]
		sent_label = random.choice(templatelist)
		if sent_label[1] == 1:
			lemma = WordNetLemmatizer()
			return sent_label[0].format(verb=self.verb,word1=self.word1,word2=lemma.lemmatize(self.word2)), sent_label[1]
		else:
			return sent_label[0].format(verb=self.verb,word1=self.word1,word2=self.word2), sent_label[1]

def read(domains):
	domain_dict = {}
	for domain in domains:
		wordlist = []
		with open("domains/{0}.txt".format(domain)) as file:
			for line in file: 
				line = line.strip() 
				wordlist.append(line) 
		domain_dict[domain] = wordlist
	return domain_dict

def main():
	warnings.filterwarnings("ignore")
	lemma = WordNetLemmatizer()

	domain_verbs = {'words':['trust','like'],
			   'people':['met','like'],
			   'materials':['use','like'],
			   'food':['eat','like'],
			   'music':['listen to','like'],
			   'animals':['like'],
			   'wearables':['wear','like'],
			   'movies':['watch','like'],
			   'books':['read','like'],
			   'entertainment':['like'],
			   'transport':['like'],
			   'drinks':['drink','like'],
			   'furniture':['like'],
			   'plants':['like']
			   }
	domain_dict = read(list(domain_verbs.keys()))


	# number of sentences per label per relation per domain
	# final number of sentences: n_sent * 2 * 3 * 14
	n_sent = 9

	sent_count = 0

	for domain in list(domain_verbs.keys()):
		for relation in [0,1,2]:
			count0 = 0
			count1 = 0
			while not (count0 == n_sent and count1 == n_sent):
				verb = random.choice(domain_verbs[domain])
				word1 = random.choice(domain_dict[domain])
				word2 = random.choice(domain_dict[domain])
				while word1 == word2:
					word2 = random.choice(domain_dict[domain])

				generated_sent = Sentence(verb, word1, word2).output()

				try:
					# make 2 part word Wordnet searchable
					word1 = word1.split()
					word1 = '_'.join(word1)
					word2 = word2.split()
					word2 = '_'.join(word2)			

					w1_hypolist = list(set([w for s in wn.synsets(lemma.lemmatize(word1))[0].closure(lambda s:s.hyponyms()) for w in s.lemma_names()]))
					w2_hypolist = list(set([w for s in wn.synsets(lemma.lemmatize(word2))[0].closure(lambda s:s.hyponyms()) for w in s.lemma_names()]))

					# x hypernym of y: 0
					# y hypernym of x: 1
					# no relation: 2

					if word2 in w1_hypolist or lemma.lemmatize(word2) in w1_hypolist:
						x = 0
					elif word1 in w2_hypolist or lemma.lemmatize(word1) in w2_hypolist:
						x = 1
					else:
						x = 2
				except:
					x = 2

				

				if x == relation:
					if generated_sent[1] == x:
						if count1 != n_sent:
							sent_count += 1
							print("word-{0}".format(sent_count),generated_sent[0], 1)
							count1 += 1

					else:
						if count0 != n_sent:
							sent_count += 1
							print("word-{0}".format(sent_count),generated_sent[0], 0)
							count0 += 1

if __name__ == "__main__":
	main()