# !/usr/bin/env python3

"""
Description:
    This scripts translates English data (more specifically, our hyponym dataset) into French and Italian.
"""

import argparse
import pandas as pd
from deep_translator import GoogleTranslator, MicrosoftTranslator, PonsTranslator
import csv


def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-l",
        "--target_language",
        type=str,
        default="fr",
        help="The target language(s) to translate into: French (fr) or Italian (it)",
    )
    parser.add_argument(
        "-s",
        "--source_text",
        type=str,
        default="data_experiments/train_base.csv",
        help="Location of the source text, i.e. the file to be translated",
    )
    parser.add_argument(
        "-t",
        "--translator",
        type=str,
        default="google",
        help="Translation system: either google, microsoft or pons (for now)",
    )
    args = parser.parse_args()
    return args


def translate(source_lang, target_lang, sentence, translator):
    """Translate input from one language to the other"""
    if translator == "google":
        return GoogleTranslator(source_lang, target_lang).translate(sentence)
    elif translator == "microsoft":
        return MicrosoftTranslator(source_lang, target_lang).translate(sentence)
    elif translator == "pons":
        return PonsTranslator(source_lang, target_lang).translate(sentence)
    else:
        raise ValueError("Translator was not correctly specified")


def main():

    args = create_arg_parser()

    # Get target language
    if args.target_language in ["fr", "it"]:
        t_lang = args.target_language
    else:
        raise ValueError(
            "Please specify one of the following target languages: 'fr' or 'it'."
        )

    # Get translation system
    if args.translator in ["google", "microsoft", "pons"]:
        translator = args.translator
    else:
        raise ValueError(
            "Please specify one of the following translators: 'google', 'microsoft', 'pons'."
        )

    # Get input and output files
    df = pd.read_csv(args.source_text)
    outfile = args.source_text.split(".")[0] + "_" + args.target_language + ".csv"

    # Translate and write sentences to new csv
    with open(outfile, "w") as translated_file:
        writer = csv.writer(translated_file, lineterminator="\n")
        writer.writerow(["id", "sentence", "labels"])
        for sent_id, sent, label in zip(df["id"], df["sentence"], df["labels"]):
            translated_sent = translate("en", t_lang, sent, "google")
            writer.writerow([sent_id, translated_sent, label])


if __name__ == "__main__":
    main()
