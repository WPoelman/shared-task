import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, BertForSequenceClassification, Trainer


def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_path", required=True,
                        help="Input sentences to predict.")
    parser.add_argument("-o", "--output_path", required=True,
                        help="Output to write predictions to.")
    parser.add_argument("-m", "--model_path", required=True,
                        help="Model to use.")
    parser.add_argument("-n", "--original_model", required=True,
                        help="Model tokenizer to use.")
    return parser.parse_args()


def main():
    args = create_arg_parser()

    dataset = load_dataset(
        'csv',
        data_files={'test': args.input_path},
        delimiter='\t'
    )

    # torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch_device = 'cuda'
    tokenizer = AutoTokenizer.from_pretrained(args.original_model)

    tokenized_dataset = dataset.map(
        lambda examples: tokenizer(
            examples["Sentence"],
            padding="max_length",
            truncation=True
        ),
        batched=True
    )

    # Columns need to have the same names as in the training phase and in the
    # training face, the models expect 'text' and 'label'. Something to keep
    # in mind.
    tokenized_dataset = tokenized_dataset.rename_column("Sentence", "text")

    df = pd.read_csv(args.input_path, sep='\t')

    model = BertForSequenceClassification.from_pretrained(
        args.model_path, num_labels=2
    ).to(torch_device)
    test_trainer = Trainer(model)
    raw_pred, _, _ = test_trainer.predict(tokenized_dataset['test'])
    y_pred = np.argmax(raw_pred, axis=1)

    df['Labels'] = y_pred
    df.to_csv(args.output_path, index=False, sep='\t')


if __name__ == '__main__':
    main()
