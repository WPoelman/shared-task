import argparse

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, BertForSequenceClassification, Trainer

def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_path", required=True,
                        help="Input sentences to predict (csv for now).")
    parser.add_argument('-m', '--model_path', required=True,
                        help="Model to use.")
    parser.add_argument("-o", "--output_path", required=True,
                        help="Output to write predictions to (csv for now).")
    # TODO: check if tokenizer used at training time needs to be included here?
    parser.add_argument("-t", "--tokenizer", default="bert-base-cased",
                        help="Tokenizer to use.")
    return parser.parse_args()


def main():
    args = create_arg_parser()

    dataset = load_dataset('csv', data_files={'test': args.input_path})

    torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = BertForSequenceClassification.from_pretrained(
        args.model_path, num_labels=2
    ).to(torch_device)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    tokenized_dataset = dataset.map(
        lambda examples: tokenizer(
            examples["sentence"],
            padding="max_length",
            truncation=True
        ),
        batched=True
    )

    # Columns need to have the same names as in the training phase and in the
    # training face, the models expect 'text' and 'label'. Something to keep
    # in mind.
    tokenized_dataset = tokenized_dataset.rename_column("labels", "label")
    tokenized_dataset = tokenized_dataset.rename_column("sentence", "text")

    test_trainer = Trainer(model)
    raw_pred, _, _ = test_trainer.predict(tokenized_dataset['test'])
    y_pred = np.argmax(raw_pred, axis=1)

    # Dit is flink lelijk, alles 2x inladen, even uitzoeken waarom de datasets
    # to_csv() zo raar deed. Aan de andere kant hebben we overal pandas
    # gebruikt, zo erg is het ook weer niet.
    df = pd.read_csv(args.input_path)
    df['prediction'] = y_pred
    df.to_csv(args.output_path, index=False)


if __name__ == '__main__':
    main()
