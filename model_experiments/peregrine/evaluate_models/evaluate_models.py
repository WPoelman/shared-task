import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, BertForSequenceClassification, Trainer


def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_path", required=True,
                        help="Input sentences to predict (csv for now).")
    parser.add_argument('-s', '--starting_path', required=True,
                        help="Search for output models.")
    parser.add_argument("-o", "--output_path", required=True,
                        help="Output to write predictions to (csv for now).")
    parser.add_argument("-t", "--tokenizer", default="bert-base-cased",
                        help="Tokenizer to use.")
    return parser.parse_args()


def main():
    args = create_arg_parser()

    dataset = load_dataset('csv', data_files={'test': args.input_path})

    # torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch_device = 'cuda'
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

    df = pd.read_csv(args.input_path)

    all_models = list(Path(args.starting_path).rglob('checkpoint-*'))
    final_models = list(Path(args.starting_path).rglob('final-model'))

    all_models.extend(final_models)

    for model_path in all_models:
        parent_dir_name = model_path.parent.stem
        current_model = model_path.stem

        # Als we nieuwe modellen willen testen, dan kunnen we het eval bestand
        # meegeven en dan gewoon columns inserten. Dit zou done modellen
        # dan moeten overslaan.
        # evaluation_id = f'prediction-{parent_dir_name}-{current_model}'
        # if evaluation_id in df.columns:
        #     continue

        print(f'Testing: {parent_dir_name} - {current_model}')

        model = BertForSequenceClassification.from_pretrained(
            model_path, num_labels=2
        ).to(torch_device)
        test_trainer = Trainer(model)
        raw_pred, _, _ = test_trainer.predict(tokenized_dataset['test'])
        y_pred = np.argmax(raw_pred, axis=1)

        df[f'prediction-{parent_dir_name}-{current_model}'] = y_pred
    df.to_csv(args.output_path, index=False)


if __name__ == '__main__':
    main()
