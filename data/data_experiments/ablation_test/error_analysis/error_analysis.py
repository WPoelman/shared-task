#!/usr/bin/python3

'''
Outputs a dataset with the incorrect predictions for all ablation tests and adds the columns
of new and old templates and words, and the category of the sentence.
'''

import csv
import spacy
from nltk.stem.wordnet import WordNetLemmatizer

def read(domains):
    domain_dict = {}
    for domain in domains:
        wordlist = []
        with open("test categories/{0}.txt".format(domain)) as file:
            for line in file:
                line = line.strip()
                wordlist.append(line)
        domain_dict[domain] = wordlist
    return domain_dict

def main():
	no_pegasus = []
	no_pegasus.append('No pegasus')
	no_newwords = []
	no_newwords.append('No new words')
	no_hypo = []
	no_hypo.append('No hypo of hypo')

	with open('evaluation_results.csv', 'r') as file:
	    reader = csv.reader(file)
	    for row in reader:
	        if row[3] != row[4] and row[3] == row[5] and row[3] == row[6]:
	        	no_pegasus.append((row[1],row[2],row[3])) 

	        if row[3] != row[5] and row[3] == row[4] and row[3] == row[6]:
	        	no_newwords.append((row[1],row[2],row[3]))

	        if row[3] != row[6] and row[3] == row[4] and row[3] == row[5]:
	        	no_hypo.append((row[1],row[2],row[3]))	

	old_temp_cat = ['andtoo','butnot','prefer']
	old_temp_sent = ['more than',' except ','interesting type', 'specifically']
	new_temp_cat = ['drather','generally','unlike']
	new_temp_sent = ['less than', 'as much', 'exception','other types','particular']

	nlp = spacy.load("en_core_web_sm")
	stoplist = ['exception','type','types','text']
	unique_words = []
	lemma = WordNetLemmatizer()
	with open('unique_words.txt','r') as f:
		for line in f:
			unique_words.append(lemma.lemmatize(line.strip()))

	domains = ['misc',
	"people",
	"materials",
	'food',
	'entertainment',
	'animals',
	'clothing',
	'transportation',
	'drinks',
	'furniture',
	'plants',
	'weather',
	'places']
	domain_dict = read(domains)


	with open('error_analysis.csv', 'w') as f:	
		writer = csv.writer(f)
		writer.writerow(['ablation','construction','sentence','gold_label','template','words','category'])
		for abl in [no_pegasus,no_newwords,no_hypo]:
			for line in abl:
				k = nlp(line[1])
				subjects = [tok.lemma_ for tok in k if (tok.pos_ == "NOUN")]
				ll = [x for x in subjects if x not in stoplist]
				try:
					for key,value in domain_dict.items():
						if ll[0] in value:
							if line[0] in old_temp_cat or any(item in line[1] for item in old_temp_sent):
								if any(x not in unique_words for x in ll):
									writer.writerow([abl[0],line[0],line[1],line[2],'old','new',key])
								else:
									writer.writerow([abl[0],line[0],line[1],line[2],'old','old',key])
							if line[0] in new_temp_cat or any(item in line[1] for item in new_temp_sent):
								if any(x not in unique_words for x in ll):
									writer.writerow([abl[0],line[0],line[1],line[2],'new','new',key])
								else:
									writer.writerow([abl[0],line[0],line[1],line[2],'new','old',key])
				except:
					if line[0] in old_temp_cat or any(item in line[1] for item in old_temp_sent):
						if any(x not in unique_words for x in ll):
							writer.writerow([abl[0],line[0],line[1],line[2],'old','new','entertainment'])
						else:
							writer.writerow([abl[0],line[0],line[1],line[2],'old','old','entertainment'])
					if line[0] in new_temp_cat or any(item in line[1] for item in new_temp_sent):
						if any(x not in unique_words for x in ll):
							writer.writerow([abl[0],line[0],line[1],line[2],'new','new','entertainment'])
						else:
							writer.writerow([abl[0],line[0],line[1],line[2],'new','old','entertainment'])		
	
if __name__ == '__main__':
	main()