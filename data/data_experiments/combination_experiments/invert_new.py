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
    parser.add_argument('-m', '--manual_templates', default=False, action='store_true',
                        help='Include manually written templates in the inverting process')
    parser.add_argument('-f', '--csv_file', default="en_new_train_split.csv",
                        help='CSV file containing the data that needs to be inverted')
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
        keys_list = list(pattern_list.keys())
        keys_list.remove("instruction")
        for key in keys_list:
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
    if args.manual_templates:
        one_to_zero_en = {'instruction': 'one_to_zero',
                          r'((\w+ \w+)|(\w+ \w+ not \w+)) ((\w+)|(his \w+)) , and more specifically ((\w+)|(his \w+)) \.': r'\1 \7 , and more specifically \4 .',
                          r'((\w+ \w+)|(\w+ \w+ not \w+)) ((\w+)|(his \w+)) , except ((\w+)|(his \w+)) \.': r'\1 \7 , except \4 .',
                          r"I (\w+) (\w+) , and particularly (\w+) \.": r'I \1 \3 , and particularly \2 .',
                          r"I (\w+) (\w+) , and especially (\w+) \.": r"I \1 \3 , and especially \2 .",
                          r"I (\w+) (\w+) , and in particular (\w+) \.": r"I \1 \3 , and in particular \2 .",
                          r"If (\w+) did not exist , then (\w+) would not exist either \.": r"If \2 did not exist , then \1 would not exist either .",
                          r"I (\w+) (\w+) , and to be more specific (\w+) \.": r"I \1 \3 , and to be more specific \2 .",
                          r"I (\w+) (\w+) , and narrowing it down , (\w+) \.": r"I \1 \3 , and narrowing it down , \2 .",
                          r"I (\w+) (\w+) , less generally (\w+) \.": r"I \1 \3 , less generally \2 .",
                          r"I (\w+) (\w+) , but most of all (\w+) \.": r"I \1 \3 , but most of all \2 .",
                          r"I (\w+) (\w+) in general , but (\w+) are my favourite \.": r"I \1 \3 in general , but \2 are my favourite .",
                          r"I (\w+) (\w+) in general , but (\w+) are my favorite \.": r"I \1 \3 in general , but \2 are my favorite .",
                          r"I (\w+) (\w+) , which are a kind of (\w+) \.": r"I \1 \3 , which are a kind of \2 .",
                          r"I (\w+) (\w+) , which are a type of (\w+) \.": r"I \1 \3 , which are a type of \2 .",
                          r"I (\w+) (\w+) , an example of (\w+) \.": r"I \1 \3 , an example of \2 .",
                          r"I (\w+) (\w+) , which are an example of (\w+) \.": r"I \1 \3 , which are an example of \2 .",
                          r"(\w+) are a subtype of (\w+) \.": r"\2 are a subtype of \1 .",
                          r"(\w+) are a subclass of (\w+) \.": r"\2 are a subclass of \1 .",
                          r"(\w+) are my favourite type of (\w+) \.": r"\2 are my favourite type of \1 .",
                          r"(\w+) are my favorite type of (\w+) \.": r"\2 are my favorite type of \1 .",
                          r"I (\w+) (\w+) and , more generally , (\w+) \.": r"I \1 \3 and , more generally , \2 .",
                          r"I (\w+) (\w+) and , broadly speaking , (\w+) \.": r"I \1 \3 and , broadly speaking , \2 .",
                          r"I (\w+) (\w+) and , to generalize , (\w+) \.": r"I \1 \3 and , to generalize , \2 .",
                          r"I (\w+) (\w+) and , to generalise , (\w+) \.": r"I \1 \3 and , to generalise , \2 .",
                          r"I (\w+) (\w+) and , in general , (\w+) \.": r"I \1 \3 and , in general , \2 .",
        }
        same_label_en = {'instruction': 'same',
                         r'((\w+ \w+)|(\w+ \w+ not \w+)) ((\w+)|(his \w+)) more than ((\w+)|(his \w+)) \.': r'\1 \7 more than \4 .',
                         r'((\w+ \w+)|(\w+ \w+ not \w+)) ((\w+)|(his \w+)) , I prefer ((\w+)|(his \w+)) \.': r'\1 \7 , I prefer \4 .',
                         r'((\w+ \w+)|(\w+ \w+ not \w+)) ((\w+)|(his \w+)) , he prefers ((\w+)|(his \w+)) \.': r'\1 \7 , he prefers \4 .',
                         r'((\w+ \w+)|(\w+ \w+ not \w+)) ((\w+)|(his \w+)) , and ((\w+)|(his \w+)) too \.': r'\1 \7 , and \4 too .',
                         r"I prefer (\w+) over (\w+) .": r"I prefer \2 over \1 .",
                         r"I would pick (\w+) over (\w+) \.": r"I would pick \2 over \1 .",
                         r"I would choose (\w+) over (\w+) \.": r"I would choose \2 over \1 .",
                         r"I (\w+) (\w+) , but above all , (\w+) \.": r"I \1 \3 , but above all , \2 .",
                         r"I (\w+) (\w+) , but more so (\w+) \.": r"I \1 \3 , but more so \2 .",
                         r"I (\w+) (\w+) , but even more (\w+) \.": r"I \1 \3 , but even more \2 .",
                         r"I (\w+) (\w+) , but even more so (\w+) \.": r"I \1 \3 , but even more so \2 .",
                         r"I (\w+) (\w+) , but (\w+) are my favourite \.": r"I \1 \3 , but \2 are my favourite .",
                         r"I (\w+) (\w+) , but (\w+) are my favorite \.": r"I \1 \3 , but \2 are my favorite .",
                         r"I (\w+) (\w+) , but my liking for (\w+) is greater \.": r"I \1 \3 , but my liking for \2 is greater .",
                         r"I (\w+) (\w+) , but my appreciation of (\w+) is greater \.": r"I \1 \3 , but my appreciation of \2 is greater .",
                         r"I (\w+) both (\w+) and (\w+) \.": r"I \1 both \3 and \2 .",
                         r"I (\w+) (\w+) as well as (\w+) \.": r"I \1 \3 as well as \2 .",
                         r"I (\w+) not only (\w+) but also (\w+) \.": r"I \1 not only \3 but also \2 .",
                         r"Apart from (\w+), I (\w+) (\w+) \.": r"Apart from \3, I \2 \1 .",
                         r"More than (\w+), I (\w+) (\w+) \.": r"More than \3, I \2 \1 .",
                         r"In addition to (\w+), I (\w+) (\w+) \.": r"In addition to \3, I \2 \1 .",
                         r"I (\w+) (\w+) to a greater extent than (\w+) \.": r"I \1 \3 to a greater extent than \2 .",
                         r"I (\w+) (\w+) and (\w+) \.": r"I \1 \3 and \2 .",
                         r"I (\w+) (\w+) and aditionally (\w+) \.": r"I \1 \3 and aditionally \2 .",
                         r"I (\w+) (\w+) and moreover (\w+) \.": r"I \1 \3 and moreover \2 .",
                         r"I like (\w+) , and moreover , I like (\w+) \.": r"I like \2 , and moreover , I like \1 .",
                         }
        zero_to_one_en = {'instruction': 'zero_to_one',
                          r'((\w+ \w+)|(\w+ \w+ not \w+)) ((\w+)|(his \w+)) , but not ((\w+)|(his \w+)) \.': r'\1 \7 , but not \4 .',
                          r"I (\w+) (\w+) , however I do not (\w+) (\w+) \.": r"I \1 \4 , however I do not \3 \2 .",
                          r"I (\w+) (\w+) , however not (\w+) \.": r"I \1 \3 , however not \2 .",
                          r"I (\w+) (\w+) , but on the contrary , I do not (\w+) (\w+) \.": r"I \1 \4 , but on the contrary , I do not \3 \2 .",
                          r"I (\w+) (\w+) , but I do not (\w+) (\w+) \.": r"I \1 \4 , but I do not \3 \2 .",
                          r"I (\w+) (\w+) while I do not (\w+) (\w+) \.": r"I \1 \4 while I do not \3 \2 .",
                          r"I (\w+) (\w+) , although I do not (\w+) (\w+) \.": r"I \1 \4 , although I do not \3 \2 .",
                          r"I (\w+) (\w+) , yet I do not (\w+) (\w+) \.": r"I \1 \4 , yet I do not \3 \2 .",
                          r"I (\w+) (\w+) , though I do not (\w+) (\w+) \.": r"I \1 \4 , though I do not \3 \2 .",
                          r"I do not (\w+) (\w+) , but I (\w+) (\w+) \.": r"I do not \1 \4 , but I \3 \2 .",
                          r"I do not (\w+) (\w+) , but on the contrary , I (\w+) (\w+) \.": r"I do not \1 \4 , but on the contrary , I \3 \2 .",
                          r"Although I (\w+) (\w+) , I do not (\w+) (\w+) \.": r"Although I \1 \4 , I do not \3 \2 .",
                          r"While I (\w+) (\w+) , I do not (\w+) (\w+) \.": r"While I \1 \4 , I do not \3 \2 .",
                          r"Though I (\w+) (\w+) , I do not (\w+) (\w+) \.": r"Though I \1 \4 , I do not \3 \2 ."}
    else:
        one_to_zero_en = {'instruction': 'one_to_zero',
                          r'((\w+ \w+)|(\w+ \w+ not \w+)) ((\w+)|(his \w+)) , and more specifically ((\w+)|(his \w+)) \.': r'\1 \7 , and more specifically \4 .',
                          r'((\w+ \w+)|(\w+ \w+ not \w+)) ((\w+)|(his \w+)) , except ((\w+)|(his \w+)) \.': r'\1 \7 , except \4 .'}
        same_label_en = {'instruction': 'same',
                         r'((\w+ \w+)|(\w+ \w+ not \w+)) ((\w+)|(his \w+)) more than ((\w+)|(his \w+)) \.': r'\1 \7 more than \4 .',
                         r'((\w+ \w+)|(\w+ \w+ not \w+)) ((\w+)|(his \w+)) , I prefer ((\w+)|(his \w+)) \.': r'\1 \7 , I prefer \4 .',
                         r'((\w+ \w+)|(\w+ \w+ not \w+)) ((\w+)|(his \w+)) , he prefers ((\w+)|(his \w+)) \.': r'\1 \7 , he prefers \4 .',
                         r'((\w+ \w+)|(\w+ \w+ not \w+)) ((\w+)|(his \w+)) , and ((\w+)|(his \w+)) too \.': r'\1 \7 , and \4 too .',}
        zero_to_one_en = {'instruction': 'zero_to_one',
                          r'((\w+ \w+)|(\w+ \w+ not \w+)) ((\w+)|(his \w+)) , but not ((\w+)|(his \w+)) \.': r'\1 \7 , but not \4 .', }


    pattern_lists = [same_label_en, one_to_zero_en, zero_to_one_en]

    # For each language, create the new templates and write to a new tsv file
    df = pd.read_csv(args.csv_file)
    nids, nsentences, nlabels = create_swapped_data(pattern_lists, df)

    # Create list containing lists of the new templates:
    new_templates_list = [[id, sent, label] for id, sent, label in zip(nids, nsentences, nlabels)]

    # Write new templates to tsv file:
    filename = 'inverse_with_manual.csv' if args.manual_templates else 'inverse.csv'
    with open(filename, 'w') as csvfile:
        writer = csv.writer(csvfile, lineterminator='\n')
        writer.writerow(["id", 'sentence', 'labels'])
        for template_list in new_templates_list:
            writer.writerow(template_list)






if __name__ == '__main__':
    main()
