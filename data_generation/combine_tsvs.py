import sys
from pathlib import Path

import pandas as pd


def usage():
    print(
        """
Usage: 
    combine_tsvs.py <path to train_subtask-1> <output dir for csvs>

Description:
    Script that combines the separate `folds` from the shared task dataset repo
    into single csv files per language.
    """
    )


def main():
    if len(sys.argv) < 3:
        usage()
        exit(1)
    base_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])

    for lang in ["en", "fr", "it"]:
        files_for_lang = Path(base_dir / lang).glob("*.tsv")

        lang_df = pd.concat([pd.read_csv(f, sep="\t") for f in files_for_lang])
        lang_df.columns = lang_df.columns.str.lower()
        lang_df.to_csv(Path(output_dir / f"subtask_1_{lang}.csv"), index=False)


if __name__ == "__main__":
    main()
