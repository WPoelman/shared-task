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
						("I {verb} {word1} , but not {word2} .",0 or 2),
						("I {verb} {word1} , an interesting type of {word2} .",1),
						("I {verb} {word1} more than {word2} .",2),
						("I do not like {word1} , I prefer {word2} .",2),
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

	domains = ['words','people','materials','food','music','animals','wearables','movies','books','entertainment','transport','drinks','furniture','plants','senses']
	domain_dict = read(domains)
	verb_domain_dict = {'use':domain_dict['materials'],
						'met':domain_dict['people'],
						'eat':domain_dict['food'],
						'listen to':domain_dict['music'],
						'wear':domain_dict['wearables'],
						'watch':domain_dict['movies'],
						'read':domain_dict['books'],
						'drink':domain_dict['drinks'],
						'trust':domain_dict['words']
						}
	verbs = list(verb_domain_dict.keys()) + ['like']


	n_sent = 60 # number of sentences per label
	count0 = 0
	count1 = 0

	while not (count0 == n_sent and count1 == n_sent):
		verb = random.choice(verbs)
		if verb == 'like':
			domain = random.choice(domains)
			word1 = random.choice(domain_dict[domain])
			word2 = random.choice(domain_dict[domain])
			while word1 == word2:
				word2 = random.choice(domain_dict[domain])

		else:
			word1 = random.choice(verb_domain_dict[verb])
			word2 = random.choice(verb_domain_dict[verb])

			while word1 == word2:
				word2 = random.choice(verb_domain_dict[verb])

		generated_sent = Sentence(verb, word1, word2).output()

		try:
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
			x = 3

		
		if generated_sent[1] == x:
			if count1 != n_sent:
				print(generated_sent[0], 1)
				count1 += 1
		else:
			if count0 != n_sent:
				print(generated_sent[0], 0)
				count0 += 1

	#print(list(set([w for s in wn.synsets(lemma.lemmatize('drinks'))[0].closure(lambda s:s.hyponyms()) for w in s.lemma_names()])))
if __name__ == "__main__":
	main()