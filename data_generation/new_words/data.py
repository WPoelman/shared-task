#!/usr/bin/python3

import random
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
import warnings
import csv

class Sentence:
	def __init__(self,pron,verb,poss,word1,word2):
		self.pron = pron
		self.verb = verb
		self.poss = poss
		self.word1 = word1
		self.word2 = word2
	def output(self):
		templatelist = [("{pron} {verb} {word1} , except {word2} .",[0]), 
						("{pron} {verb} {word1} , and more specifically {word2} .",[0]), 
						("{pron} {verb} {word1} , but not {word2} .",[0,2]),
						("{pron} {verb} {word1} , an interesting type of {word2} .",[1]),
						("{pron} {verb} {word1} more than {word2} .",[2]),
						("{pron} do not {verb} {word1} , {pron} prefer {word2} .",[2]),
						("{pron} {verb} {word1} , and {word2} too .",[2]), # new templates
						("{pron} {verb} {word1} , and particularly {word2} .",[0]),
						("{pron} {verb} {word1} , and especially {word2} .",[0]),
						("{pron} {verb} {word1} , and in particular {word2} .",[0]),
						("if {word1} did not exist , then {word2} would not exist either .",[0]),
						("{pron} {verb} {word1} , and to be more specific {word2} .",[0]),
						("{pron} {verb} {word1} , and narrowing it down , {word2} .",[0]),
						("{pron} {verb} {word1} , less generally {word2} .",[0]),
						("{pron} {verb} {word1} , but most of all {word2} .",[0]),
						("{pron} {verb} {word1} in general , but {word2} are my favourite .",[0]),
						("{pron} {verb} {word1} in general , but {word2} are my favorite .",[0]),
						("{pron} {verb} {word1} , which is a kind of {word2} .",[1]),
						("{pron} {verb} {word1} , which is a type of {word2} .",[1]),
						("{pron} {verb} {word1} , an example of {word2} .",[1]),
						("{pron} {verb} {word1} , which is an example of {word2} .",[1]),
						("{word1} is a subtype of {word2} .",[1]),
						("{word1} is a subclass of {word2} .",[1]),
						("{word1} is {poss} favourite type of {word2} .",[1]),
						("{word1} is {poss} favorite type of {word2} .",[1]),
						("{pron} {verb} {word1} and , more generally , {word2} .",[1]),
						("{pron} {verb} {word1} and , broadly speaking , {word2} .",[1]),
						("{pron} {verb} {word1} and , to generalize , {word2} .",[1]),
						("{pron} {verb} {word1} and , to generalise , {word2} .",[1]),
						("{pron} {verb} {word1} and , in general , {word2} .",[1]),
						("{pron} prefer {word1} over {word2} .",[2]),
						("{pron} would pick {word1} over {word2} .",[2]),
						("{pron} would choose {word1} over {word2} .",[2]),
						("{pron} {verb} {word1} , but above all , {word2} .",[2]),
						("{pron} {verb} {word1} , but more so {word2} .",[2]),
						("{pron} {verb} {word1} , but even more {word2} .",[2]),
						("{pron} {verb} {word1} , but even more so {word2} .",[2]),
						("{pron} {verb} {word1} , but {word2} are {poss} favourite .",[2]),
						("{pron} {verb} {word1} , but {word2} are {poss} favorite .",[2]),
						("{pron} {verb} {word1} , but {poss} liking for {word2} is greater .",[2]),
						("{pron} {verb} {word1} , but {poss} appreciation of {word2} is greater .",[2]),
						("{pron} {verb} both {word1} and {word2} .",[2]),
						("{pron} {verb} {word1} as well as {word2} .",[2]),
						("{pron} {verb} not only {word1} but also {word2} .",[2]),
						("apart from {word1}, {pron} {verb} {word2} .",[2]),
						("more than {word1}, {pron} {verb} {word2} .",[2]),
						("in addition to {word1}, {pron} {verb} {word2} .",[2]),
						("{pron} {verb} {word1} to a greater extent than {word2} .",[2]),
						("{pron} {verb} {word1} and {word2} .",[2]),
						("{pron} {verb} {word1} and additionally {word2} .",[2]),
						("{pron} {verb} {word1} and moreover {word2} .",[2]),
						("{pron} {verb} {word1} , and moreover , {pron} like {word2} .",[2]),
						("{pron} {verb} {word1} , however {pron} do not {verb} {word2} .",[0,2]),
						("{pron} {verb} {word1} , however not {word2} .",[0,2]),
						("{pron} {verb} {word1} , but on the contrary , {pron} do not {verb} {word2} .",[0,2]),
						("{pron} {verb} {word1} , but {pron} do not {verb} {word2} .",[0,2]),
						("{pron} {verb} {word1} while {pron} do not {verb} {word2} .",[0,2]),
						("{pron} {verb} {word1} , although {pron} do not {verb} {word2} .",[0,2]),
						("{pron} {verb} {word1} , yet {pron} do not {verb} {word2} .",[0,2]),
						("{pron} {verb} {word1} , though {pron} do not {verb} {word2} .",[0,2]),
						("{pron} do not {verb} {word1} , but {pron} {verb} {word2} .",[0,1]),
						("{pron} do not {verb} {word1} , but on the contrary , {pron} {verb} {word2} .",[0,1]),
						("although {pron} {verb} {word1} , {pron} do not {verb} {word2} .",[0,1]),
						("while {pron} {verb} {word1} , {pron} do not {verb} {word2} .",[0,1]),
						("though {pron} {verb} {word1} , {pron} do not {verb} {word2} .",[0,1]),
						]
		sent_label = random.choice(templatelist)
		return sent_label[0].format(pron=self.pron,verb=self.verb,poss=self.poss,word1=self.word1,word2=self.word2), sent_label[1]

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

def lemmatize_all(sentence):
    wnl = WordNetLemmatizer()
    for word, tag in pos_tag(word_tokenize(sentence)):
        if tag.startswith("NN"):
            yield wnl.lemmatize(word, pos='n')
        elif tag.startswith('VB'):
            yield wnl.lemmatize(word, pos='v')
        elif tag.startswith('JJ'):
            yield wnl.lemmatize(word, pos='a')
        else:
            yield word

def main():
	with open('newdata.csv', 'w', encoding='UTF8') as f:
		writer = csv.writer(f)
		warnings.filterwarnings("ignore")
		lemma = WordNetLemmatizer()

		domain_verbs = {'misc':['trust','like','love','enjoy','feel'],
				   'people':['met','like','love'],
				   'materials':['use','like','love'],
				   'food':['eat','like','love','enjoy'],
				   'music':['listen to','like','love','enjoy'],
				   'animals':['like','love'],
				   'wearables':['wear','like','love'],
				   'movies':['watch','like','love','enjoy'],
				   'books':['read','like','love','enjoy'],
				   'transport':['like','love','enjoy'],
				   'drinks':['drink','like','love','enjoy'],
				   'furniture':['like','love'],
				   'plants':['like','love'],
				   'games':['play','like','love','enjoy'],
				   }
		domain_dict = read(list(domain_verbs.keys()))
		pron_dict = {'I':'my',
					 'he':'his',
					 'she':'her',
					 'they':'their'
					}


		# number of sentences per label per relation per domain
		# final number of sentences: n_sent * 2 * 3 * 14
		n_sent = 250

		sent_count = 0

		for domain in list(domain_verbs.keys()):
			for relation in [0,1,2]:
				count0 = 0
				count1 = 0
				while not (count0 == n_sent and count1 == n_sent):
					pron = random.choice(list(pron_dict.keys()))
					verb = random.choice(domain_verbs[domain])
					poss = pron_dict[pron]
					word1 = random.choice(domain_dict[domain])
					word2 = random.choice(domain_dict[domain])
					while word1 == word2:
						word2 = random.choice(domain_dict[domain])

					generated_sent = Sentence(pron, verb, poss, word1, word2).output()
					sent = generated_sent[0][0].upper() + generated_sent[0][1:]

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
						if x in generated_sent[1]:
							if count1 != n_sent:
								sent_count += 1
								writer.writerow(["new-{0}".format(sent_count),sent, 1])
								print("new-{0}".format(sent_count),sent, 1)
								count1 += 1

						else:
							if count0 != n_sent:
								sent_count += 1
								writer.writerow(["new-{0}".format(sent_count),sent, 0])
								print("new-{0}".format(sent_count),sent, 0)
								count0 += 1

if __name__ == "__main__":
	main()