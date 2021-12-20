 #!/usr/bin/env python3

'''
Description:
    A script that takes the inverse of several templates to create more training data.
'''

from pathlib import Path
import pandas as pd
import re
import csv
import argparse

DATA_DIR = Path().cwd() / 'data'


def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--languages', nargs='+',
                        help='Language datasets to use for the inverting of the labels',
                        default=['en', 'fr', 'it'])
    args = parser.parse_args()
    return args


def new_label(label, instruction_string):
    """Generates the new label, depending on the instructional string'"""
    if instruction_string == "same":
        return label
    elif instruction_string == "one_to_zero" and label == 1:
        return 0
    elif instruction_string == "zero_to_one" and label == 0:
        return 1

    return 'NA'


def create_swapped_data(list_of_pattern_lists, df):
    """Generates new sentences with the keywords swapped
    and the corresponding new labels"""
    ids, sentences, labels = df['id'].tolist(), df['sentence'].tolist(), df['labels'].tolist()
    nids, nsentences, nlabels = [], [], []

    for pattern_list in list_of_pattern_lists:
        for key in list(pattern_list.keys()):
            sRegex = re.compile(key)
            # Search the sentences for the patterns that are specified in each list
            for id, sent, label in zip(ids, sentences, labels):
                regex_match = sRegex.search(sent)
                if regex_match:
                    # Get the new (swapped) sentence and the new label
                    new_sentence = re.sub(key, pattern_list[key], sent)
                    nlabel = new_label(label, pattern_list['instruction'])
                    # Add the new sentence and label if the combination does not already exist
                    if nlabel != 'NA' and new_sentence not in sentences:
                        nids.append(str(id) + "i")
                        nsentences.append(new_sentence)
                        nlabels.append(nlabel)

    return nids, nsentences, nlabels


def main():
    args = create_arg_parser()

    # ENGLISH regex pattern lists
    # todo: currently only the same_label sentences produce new templates;
    #  the one_to_zero and zero_to_one sentences are examples of how these could work but the inverted
    #  sentences and labels for these templates are currently already in the dataset

    #
    one_to_zero_en = {'instruction': 'one_to_zero',
                      r'((\w+ \w+)|(\w+ \w+ not \w+)) ((\w+)|(his \w+)) , and more specifically ((\w+)|(his \w+)) \.': r'\1 \7 , and more specifically \4 .',
                      r'((\w+ \w+)|(\w+ \w+ not \w+)) ((\w+)|(his \w+)) , except ((\w+)|(his \w+)) \.': r'\1 \7 , except \4 .'}
    same_label_en = {'instruction': 'same',
                     r'((\w+ \w+)|(\w+ \w+ not \w+)) ((\w+)|(his \w+)) more than ((\w+)|(his \w+)) \.': r'\1 \7 more than \4 .',
                     r'((\w+ \w+)|(\w+ \w+ not \w+)) ((\w+)|(his \w+)) , I prefer ((\w+)|(his \w+)) \.': r'\1 \7 , I prefer \4 .',
                     r'((\w+ \w+)|(\w+ \w+ not \w+)) ((\w+)|(his \w+)) , he prefers ((\w+)|(his \w+)) \.': r'\1 \7 , he prefers \4 .',
                     r'((\w+ \w+)|(\w+ \w+ not \w+)) ((\w+)|(his \w+)) , and ((\w+)|(his \w+)) too \.': r'\1 \7 , and \4 too .',}
    zero_to_one_en = {'instruction': 'zero_to_one',
                      r'((\w+ \w+)|(\w+ \w+ not \w+)) ((\w+)|(his \w+)) , but not ((\w+)|(his \w+)) \.': r'\1 \7 , but not \4 .',}

    # FRENCH regex pattern lists
    same_label_fr = {'instruction': 'same',
                     r"(((J'|Il|Je) (adore|apprécie|utilise|aime|ai rencontré|n' aime pas))|(Je peux comprendre)) (.*) plus que (.*) \.":
                         r"\1 \7 plus que \6 .",
                     r"(Il fait confiance|Il ne fait pas confiance) (à|aux) (.*) plus que (à|aux) (.*) \.":
                         r"\1 \4 \5 plus que \2 \3 .",
                     r"(((J'|Il|Je) (adore|apprécie|utilise|aime|ai rencontré|n' aime pas))|(Je peux comprendre)) (.*) , je préfère (.*) \.":
                         r"\1 \7 , je préfère \6 .",
                     r"(((J'|Il|Je) (adore|apprécie|utilise|aime|ai rencontré|n' aime pas))|(Je peux comprendre)) (.*) , il préfère (.*) \.":
                         r"\1 \7 , il préfère \6 .",
                     r"(((J'|Il|Je) (adore|apprécie|utilise|aime|ai rencontré|n' aime pas))|(Je peux comprendre)) (.*) , et aussi (.*) \.":
                         r"\1 \7 , et aussi \6 ."}

    # ITALIAN regex pattern lists
    same_label_it = {'instruction': 'same',
                     r"(Amo|Uso|Apprezzo|Posso capire) l(.) (.*) più dell(.) (.*) \.": r"\1 l\4 \5 più dell\2 \3 .",
                     r"(Amo|Uso|Apprezzo|Posso capire) l(.) (.*) più dei (.*) \.": r"\1 i \4 più dell\2 \3 .",
                     r"(Amo|Uso|Apprezzo|Posso capire) l(.) (.*) più del (.*) \.": r"\1 il \4 più dell\2 \3 .",
                     r"(Amo|Uso|Apprezzo|Posso capire) l(.) (.*) più degli (.*) \.": r"\1 gli \4 più dell\2 \3 .",
                     r"(Amo|Uso|Apprezzo|Posso capire) gli (.*) più dei (.*) \.": r"\1 i \3 più degli \2 .",
                     r"(Amo|Uso|Apprezzo|Posso capire) gli (.*) più dell(.) (.*) \.": r"\1 l\3 \4 più degli \2 .",
                     r"(Amo|Uso|Apprezzo|Posso capire) gli (.*) più del (.*) \.": r"\1 il \3 più degli \2 .",
                     r"(Amo|Uso|Apprezzo|Posso capire) gli (.*) più degli (.*) \.": r"\1 gli \3 più degli \2 .",
                     r"(Amo|Uso|Apprezzo|Posso capire) il (.*) più dei (.*) \.": r"\1 i \3 più del \2 .",
                     r"(Amo|Uso|Apprezzo|Posso capire) il (.*) più dell(.) (.*) \.": r"\1 l\3 \4 più del \2 .",
                     r"(Amo|Uso|Apprezzo|Posso capire) il (.*) più del (.*) \.": r"\1 il \3 più del \2 .",
                     r"(Amo|Uso|Apprezzo|Posso capire) il (.*) più degli (.*) \.": r"\1 gli \3 più del \2 .",
                     r"(Amo|Uso|Apprezzo|Posso capire) i (.*) più dei (.*) \.": r"\1 i \3 più dei \2 .",
                     r"(Amo|Uso|Apprezzo|Posso capire) i (.*) più dell(.) (.*) \.": r"\1 l\3 \4 più dei \2 .",
                     r"(Amo|Uso|Apprezzo|Posso capire) i (.*) più del (.*) \.": r"\1 il \3 più dei \2 .",
                     r"(Amo|Uso|Apprezzo|Posso capire) i (.*) più degli (.*) \.": r"\1 gli \3 più dei \2 .",
                     r"Si fida (.*) più che (.*) \.": r"Si fida \2 più che \1 .",
                     r"Non ama (.*) , preferisce (.*) \.": r"Non ama \2 , preferisce \1 .",
                     r"Non amo (.*) , preferisco (.*) \.": r"Non amo \2 , preferisco \1 .",
                     r"(Amo|Uso|Apprezzo|Posso capire|Ho incontrato) (.*) , ed anche (.*) \.": r"\1 \3 , ed anche \2 ."}

    pattern_lists = {'en': [same_label_en, one_to_zero_en, zero_to_one_en],
                     'fr': [same_label_fr],
                     'it': [same_label_it]}

    # For each language, create the new templates and write to a new tsv file
    for lang in args.languages:
        filename = 'subtask_1_' + lang + '.csv'
        df = pd.read_csv(DATA_DIR / filename)
        nids, nsentences, nlabels = create_swapped_data(pattern_lists[lang], df)

        # Create list containing lists of the new templates:
        new_templates_list = [[id, sent, label] for id, sent, label in zip(nids, nsentences, nlabels)]

        # Write new templates to tsv file:
        with open('subtask_1_' + lang + '_inverse.tsv', 'w') as tsvfile:
            writer = csv.writer(tsvfile, delimiter='\t', lineterminator='\n')
            writer.writerow(["ID", 'Sentence', 'Labels'])
            for template_list in new_templates_list:
                writer.writerow(template_list)






if __name__ == '__main__':
    main()
