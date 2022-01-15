from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from transformers import (AutoTokenizer, BertForSequenceClassification,
                          EarlyStoppingCallback, Trainer, TrainingArguments)


def get_args():
    parser = ArgumentParser()
    parser.add_argument('-m', '--model', default="bert-base-cased",
                        help='Huggingface model id to use.')
    parser.add_argument('-d', '--data_directory', default="./data",
                        help='Directory with all csv files to test.')
    parser.add_argument('-t', '--test_set', help='Test dataset csv path.')
    parser.add_argument('-o', '--output_dir',
                        help='Directory to write model outputs to.')
    return parser.parse_args()


def compute_metrics(p):
    pred, labels = p
    pred = np.argmax(pred, axis=1)

    return {
        "accuracy": accuracy_score(y_true=labels, y_pred=pred),
        "precision": precision_score(y_true=labels, y_pred=pred),
        "recall": recall_score(y_true=labels, y_pred=pred),
        "f1":  f1_score(y_true=labels, y_pred=pred),
    }


def main():
    if not torch.cuda.is_available():
        print('No gpu available!')
        exit(1)

    args = get_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    def tokenize_function(examples):
        return tokenizer(
            examples["sentence"],
            padding="max_length",
            truncation=True
        )

    for data_file in Path(args.data_directory).glob('*.csv'):
        output_dir = Path(args.output_dir) / \
            str(data_file.stem).replace('.csv', '')

        dataset = load_dataset(
            'csv',
            data_files={'train': str(data_file), 'test': args.test_set}
        )

        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        tokenized_dataset = tokenized_dataset.rename_column("labels", "label")
        tokenized_dataset = tokenized_dataset.rename_column("sentence", "text")

        model = BertForSequenceClassification.from_pretrained(
            args.model,
            num_labels=2
        )

        training_args = TrainingArguments(
            output_dir=output_dir,
            evaluation_strategy="steps",
            eval_steps=1000,
            save_steps=5000,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=4,
            seed=0,
            load_best_model_at_end=True,
            disable_tqdm=True,
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset['train'],
            eval_dataset=tokenized_dataset['test'],
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        )
        trainer.train()
        trainer.save_model(f'{output_dir}/final-model')


if __name__ == '__main__':
    main()
