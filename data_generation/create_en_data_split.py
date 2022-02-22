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

    # Open files as list to allow for multiple iteration
    with open("subtask_1_en.csv") as f1, open("subtask_1_fr.csv") as f2, open(
        "subtask_1_it.csv"
    ) as f3:
        f_en = f1.read().splitlines()  # English
        f_fr = f2.read().splitlines()  # French
        f_it = f3.read().splitlines()  # Italian

    # Get person and material nouns
    spec_nouns = get_special_nouns(f_en)

    # Generate splits for English
    en_test_lines = create_test_set(f_en, spec_nouns)
    write_to_file(en_test_lines, "en_new_test_split.csv")
    write_to_file(create_train_set(f_en, en_test_lines), "en_new_train_split.csv")

    test_ids = [line.split(",")[0] for line in en_test_lines]

    # Generate splits for French
    fr_test_lines, fr_train_lines = make_splits_from_ids(f_fr, test_ids)
    write_to_file(fr_test_lines, "fr_new_test_split.csv")
    write_to_file(fr_train_lines, "fr_new_train_split.csv")

    # Generate splits for Italian
    it_test_lines, it_train_lines = make_splits_from_ids(f_it, test_ids)
    write_to_file(it_test_lines, "it_new_test_split.csv")
    write_to_file(it_train_lines, "it_new_train_split.csv")


def create_test_set(file, nouns):
    """Return lines that contain special nouns and / or selected templates"""
    noun_lines = [line for line in file for n in nouns if n in line]
    template_lines = [
        line
        for line in file
        if "more specifically" in line or " too " in line or "but not" in line
    ]

    return noun_lines + template_lines


def create_train_set(f, test_lines):
    """Return lines that do not occur in test set"""
    return [line for line in f if line not in test_lines]


def get_special_nouns(file):
    """Return set of nouns that occur in sentences that use the verb 'met' or 'use'"""
    nouns = set()
    filter_words = ["type", "schools", "offices", "restaurants", "factories"]
    for line in file:
        if " use " in line or " met " in line:
            # Every 'use' sentence starts with 'I' and ends in a period:
            line = line[line.index("I") : line.index(".") + 1]

            pos = pos_tag(line.split(" "))
            for w, p in pos:
                if (
                    p.startswith("NN") and w not in filter_words
                ):  # manual correction: remove other noun categories
                    nouns.add(w)

    nouns.add("professor")  # manual correction: people are not mass nouns

    return nouns


def write_to_file(lines, outfile):
    """Write lines to file"""
    with open(outfile, "w") as o:
        o.write("id,sentence,labels" + "\n")
        for line in set(lines):
            if "id,sentence,labels" not in line:
                o.write(line + "\n")


def make_splits_from_ids(f, ids):
    """Generate splits for given the ids of the lines that should
    only occur in the test set (to be used for French and Italian here)"""
    test_lines = [line for line in f if line.split(",")[0] in ids]
    train_lines = create_train_set(f, test_lines)

    return test_lines, train_lines


if __name__ == "__main__":
    main()
