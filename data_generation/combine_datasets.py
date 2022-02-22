import sys

import pandas as pd


def usage():
    print(
        """
Usage: 
    combine_datasets.py <output_csv_filepath> <input_csv_1> ... <input_csv_n>

Description:
    Combines datasets from several csv's into a single csv and removes 
    duplicate sentences.
    """
    )


def main():
    if len(sys.argv) < 3:
        usage()
        exit(1)

    output_file_path = sys.argv[1]
    input_paths = sys.argv[2:]

    full_df = pd.concat([pd.read_csv(path) for path in input_paths])
    full_df.drop_duplicates(["sentence"], inplace=True)

    full_df.to_csv(output_file_path, index=False)

    print(f"Created combined dataset with {len(full_df)} rows")


if __name__ == "__main__":
    main()
