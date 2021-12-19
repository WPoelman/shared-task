#!/usr/bin/env python

"""
Filename:   create_en_data_split.py
Date:       12-12-2021
Description:
    Program to create a train and a test split, depending on specific templates and
    noun categories, without template overlap.
"""

from nltk import pos_tag


def main():

    # Open file as list to allow for multiple iteration
    f = open("subtask_1_en.csv").read().splitlines()

    # Get person and material nouns
    spec_nouns = get_special_nouns(f)

    # Generate splits
    test_lines = create_test_set(f, spec_nouns)
    write_to_file(test_lines, "en_new_test_split.csv")

    train_lines = create_train_set(f, test_lines)
    write_to_file(train_lines, "en_new_train_split.csv")


def create_test_set(file, nouns):
    """ Return lines that contain special nouns and / or selected templates """
    noun_lines = [line for line in file for n in nouns if n in line]
    template_lines = [line for line in file if "more specifically" in line or " too " in line or "but not" in line]

    return noun_lines + template_lines


def create_train_set(f, test_lines):
    """ Return lines that do not occur in test set """
    return [line for line in f if line not in test_lines]


def get_special_nouns(file):
    """ Return set of nouns that occur in sentences that use the verb 'met' or 'use' """
    nouns = set()
    filter_words = ["type", "schools", "offices", "restaurants", "factories"]
    for line in file:
        if " use " in line or " met " in line:
            # Every 'use' sentence starts with 'I' and ends in a period:
            line = line[line.index("I"):line.index(".")+1]

            pos = pos_tag(line.split(" "))
            for w, p in pos:
                if p.startswith("NN") and w not in filter_words:  # manual correction: remove other noun categories
                    nouns.add(w)

    nouns.add("professor")  # manual correction: people are not mass nouns

    return nouns


def write_to_file(lines, outfile):
    """ Write lines to file """
    with open(outfile, "w") as o:
        o.write("id,sentence,labels" + "\n")
        for line in set(lines):
            if "id,sentence,labels" not in line:
                o.write(line + "\n")


if __name__ == "__main__":
    main()
