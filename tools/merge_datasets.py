import sys
from pathlib import Path

import pandas as pd


def main():
    # Should be the path to the 'train_subtask-1' folder
    base_folder = Path(sys.argv[1])

    for lang in ['en', 'fr', 'it']:
        files_for_lang = Path(base_folder / lang).glob('*.tsv')

        lang_df = pd.concat([pd.read_csv(f, sep='\t') for f in files_for_lang])
        lang_df.columns= lang_df.columns.str.lower()
        lang_df.to_csv(f'subtask_1_{lang}.csv', index=False)


if __name__ == '__main__':
    main()
