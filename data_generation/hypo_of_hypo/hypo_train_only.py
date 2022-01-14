# !/usr/bin/env python3

'''
Description:
	A script that takes existing hyponyms and searches Wordnet for hyponyms of these hyponyms.
	To be able to run, excecute: pip install pattern3
'''

from nltk.corpus import wordnet as wn
from pathlib import Path
import pandas as pd
import re
import csv
import argparse
from pattern.en import pluralize, singularize
from random import choice

DATA_DIR = Path().cwd() / 'data'


def create_arg_parser():
	parser = argparse.ArgumentParser()
	parser.add_argument('-cf', '--csv-file', type=str, default='en_new_train_split.csv',
	                    help='The csv file that we extract the templates from')
	args = parser.parse_args()
	return args


def extract_hypxnyms(sentences, labels, hypx_dict, regex_templates, target_label, order):
	"""Searches the sentences for a regex match with the correct label, then
	extracts the hyponym and hypernym depending on the given order"""
	for sent, label in zip(sentences, labels):
		for regex in regex_templates:
			match = re.search(regex, sent)
			if match and label == target_label:
				if order == "first":
					hyponym = sent.split(" ")[2]
					hypernym = sent.split(" ")[-2]
				else:
					hyponym = sent.split(" ")[-2]
					hypernym = sent.split(" ")[2]
				hypx_dict[hyponym] = hypernym

	return hypx_dict


def generate_hyponyms(current_hyponyms, n=5):
	"""Takes a list of current hyponyms, searches WordNet for hyponyms
	these hyponyms and returns the first N hyponyms in a dict"""
	generated_hyponyms = dict()
	for hyponym in current_hyponyms:
		syn = wn.synsets(hyponym)
		if syn:
			new_hyponyms = syn[0].hyponyms()
			for new_hyp in new_hyponyms[:4]:
				new_hyp = new_hyp.lemmas()[0].name().replace("_", " ")
				generated_hyponyms[new_hyp] = hyponym

	return generated_hyponyms


def get_plural(noun):
	"""Gets the plural form of a noun"""
	exceptions = {'built-in bed': 'built-in beds', 'creeps': 'creeps', 'prosciutto': 'prosciutto',
	              'calculus': 'calculus',
	              'beluga caviar': 'beluga caviar', 'bellbottom trousers': 'bellbottom trousers',
	              'churidars': 'churidars'}
	if 'ness' in noun:
		plural = noun
	elif noun in exceptions:
		plural = exceptions[noun]
	else:
		plural = pluralize(noun)

	return plural


def get_singular(noun):
	"""Gets the singular form of a noun"""
	if 'ness' in noun:
		singular = noun
	else:
		singular = singularize(noun)

	return singular


def generate_sentences(regex_templates, hypo_hyper_dict):
	"""Searches the sentences for a regex match with the correct label, then
	extracts the hyponym and hypernym depending on the given order"""
	sen_label_tuplist = []  # list with (sentence, label) tuples
	all_hypernyms = list(hypo_hyper_dict.values())
	all_hyponyms = list(hypo_hyper_dict.keys())

	for template, properties in regex_templates.items():
		relation = properties[0]
		hyper_pos = properties[1]
		hypo_pos = properties[2]
		number = properties[3]
		tokens = template.split(" ")

		if relation == "No superset":
			# These sentences are valid if Y is not a superset of X
			# 'I like animals but not pigs' vs 'I like pigs but not animals' vs 'I like pigs but not cats'
			for hypo, hyper in hypo_hyper_dict.items():
				# Make valid sentence with first the hypernym, then hyponym
				tokens[hyper_pos] = hyper
				tokens[hypo_pos] = get_plural(hypo)
				sen_label_tuplist.append((" ".join(tokens), 1))

				# Make valid sentence with two hyponyms
				second_hypo = choice(all_hyponyms)
				while second_hypo == hypo:
					second_hypo = choice(all_hyponyms)
				tokens[hyper_pos] = get_plural(second_hypo)
				tokens[hypo_pos] = get_plural(hypo)
				sen_label_tuplist.append((" ".join(tokens), 1))

				# Make invalid sentence, by using hyponym first then hypernym
				tokens[hyper_pos] = get_plural(hypo)
				tokens[hypo_pos] = hyper
				sen_label_tuplist.append((" ".join(tokens), 0))

		elif relation:
			# These sentences are valid if there is the right relation between the two nouns
			for hypo, hyper in hypo_hyper_dict.items():
				# Make valid sentence
				tokens[hyper_pos] = get_singular(hyper) if number == 'sg' else hyper
				tokens[hypo_pos] = get_plural(hypo)
				sen_label_tuplist.append((" ".join(tokens), 1))

				# Make invalid sentence, by using a wrong hypernym
				wrong_hyper = choice(all_hypernyms)
				while wrong_hyper == hyper:
					wrong_hyper = choice(all_hypernyms)
				tokens[hyper_pos] = get_singular(wrong_hyper) if number == 'sg' else wrong_hyper
				tokens[hypo_pos] = get_plural(hypo)
				sen_label_tuplist.append((" ".join(tokens), 0))

		else:
			# The sentences are valid if there is no relation between the two nouns
			for hypo, hyper in hypo_hyper_dict.items():
				# Make valid sentence, using two hyponyms
				second_hypo = choice(all_hyponyms)
				while second_hypo == hypo:
					second_hypo = choice(all_hyponyms)
				tokens[hyper_pos] = second_hypo if number == 'sg' else get_plural(second_hypo)
				tokens[hypo_pos] = get_plural(hypo)
				sen_label_tuplist.append((" ".join(tokens), 1))

				# Make invalid sentences, by using a hypo and hypernym in both positions
				tokens[hyper_pos] = get_singular(hyper) if number == 'sg' else hyper
				tokens[hypo_pos] = get_plural(hypo)
				sen_label_tuplist.append((" ".join(tokens), 0))

				tokens[hypo_pos] = get_singular(hyper) if number == 'sg' else hyper
				tokens[hyper_pos] = get_plural(hypo)
				sen_label_tuplist.append((" ".join(tokens), 0))

	return sen_label_tuplist



def main():
	args = create_arg_parser()

	# Load the ids, sentences and labels from the csv file
	df = pd.read_csv(DATA_DIR / args.csv_file)
	ids, sentences, labels = df['id'].tolist(), df['sentence'].tolist(), df['labels'].tolist()

	# Make a dictionary and define the templates lists, depending on the label that a template should have
	# whether the hyponym comes first or last in the sentence for the extraction part
	hypo_hyper = dict()
	temp_1_last = [r'I like \w+ , except \w+ .', r'He likes \w+ , except \w+ .', ]
	temp_1_first = [r'I like \w+ , an interesting type of \w+ .', r'He likes \w+ , an interesting type of \w+ .']

	# Fill the hyponym-hypernym dictionary:
	hypo_hyper = extract_hypxnyms(sentences, labels, hypo_hyper, temp_1_first, 1, 'first')
	hypo_hyper = extract_hypxnyms(sentences, labels, hypo_hyper, temp_1_last, 1, 'last')

	# If we manually find weird exceptions, we take them out here. e.g. 'crabs' will not refer
	# to the seafood and 'bracelets' will not refer to the jewelry.
	exceptions = ['crabs', 'bracelets', 'firs', 'tables', 'rock']
	for exception in exceptions:
		del hypo_hyper[exception]

	# Find new hyponyms for our hyponyms, using WordNet
	new_hyponyms = generate_hyponyms(list(hypo_hyper.keys()), n=5)

	# Define the templates for the generation of new sentences
	# We also store in a list whether there is a relation between the two nouns,
	# The hypernym position, the hyponym position and whether the hypernym is sg or pl
	templates = {r'I like \w+ , except \w+ .': [True, 2, -2, 'pl'],
	             r'He likes \w+ , except \w+ .': [True, 2, -2, 'pl'],
	             r'I do not like \w+ , I prefer \w+ .': [False, 4, -2, 'pl'],
	             r'He does not like \w+ , he prefers \w+ .': [False, 4, -2, 'pl'],
	             r'I like \w+ , more than \w+ .': [False, 2, -2, 'pl'],
	             r'He likes \w+ , more than \w+ .': [False, 2, -2, 'pl'],
	             r'I like \w+ , an interesting type of \w+ .': [True, -2, 2, 'sg'],
	             r'He likes \w+ , an interesting type of \w+ .': [True, -2, 2, 'sg']}

	# Generate new valid and invalid sentences
	all_sentences = generate_sentences(templates, new_hyponyms)  # List of (sentence, label) tuples

	# Write to new file
	with open('subtask_1_en_new_hyponyms_only_train.csv', 'w') as csvfile:
		writer = csv.writer(csvfile, lineterminator='\n')
		writer.writerow(["ID", 'Sentence', 'Labels'])
		c = 0
		for sent, label in all_sentences:
			id = "H" + str(c)
			writer.writerow([id, sent, label])
			c += 1


if __name__ == '__main__':
	main()
