"""
Description:
    A script that creates (paraphrases) new sentences based on the original 
    input data. Note that this is only intended to work for English!
"""

import argparse

import pandas as pd
import spacy
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from pathlib import Path

print('Loading pegasus model')
# Model url: https://huggingface.co/tuner007/pegasus_paraphrase
MODEL_NAME = "tuner007/pegasus_paraphrase"
TORCH_DEVICE = "cuda"
TOKENIZER = PegasusTokenizer.from_pretrained(MODEL_NAME)
MODEL = PegasusForConditionalGeneration.from_pretrained(
    MODEL_NAME).to(TORCH_DEVICE)

print('Loading spacy model')
# maybe add spacy model to args
NLP = spacy.load("en_core_web_sm", disable=["ner"])


def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_directory', default="./data",
                        help='Directory with all csv files to test.')
    parser.add_argument('-o', '--output_dir',
                        help='Directory to write model outputs to.')
    return parser.parse_args()


def get_response(input_texts):

    # Random crashes? Call your local obsure huggingface forum helper from a
    # year ago: https://discuss.huggingface.co/t/out-of-index-error-when-using-pre-trained-pegasus-model/5196/2
    batch = TOKENIZER(
        input_texts, 
        truncation=True, 
        padding='longest', 
        max_length=200, 
        return_tensors="pt"
    ).to(TORCH_DEVICE)

    translated = MODEL.generate(
        **batch, 
        #max_length=200, 
        max_length=60,
        num_beams=10, 
        num_return_sequences=10, 
        temperature=1.5
    )

    tgt_text = TOKENIZER.batch_decode(translated, skip_special_tokens=True)

    return tgt_text


def validate_results(original_sent, new_sents):
    # TODO find a neat way to dedupe the sentences while keeping the original
    # casing (if it is even a problem that there are dupes?)
    #
    # # First dedupe the results
    # original_lowered = original_sent.lower()
    # # use set to remove dupes in result
    # new_lowered = {s.lower() for s in new_sents}
    # # remove dupes of the original
    # if original_lowered in new_lowered:
    #     new_lowered.remove(original_lowered)

    original_sent_doc = NLP(original_sent)
    original_sent_pos = {t.pos_ for t in original_sent_doc}

    filtered_new = []
    for sent in new_sents:
        new_sent_doc = NLP(sent)
        new_sent_pos = {t.pos_ for t in new_sent_doc}
        # Not sure if this is always correct, need to check
        if new_sent_pos.issubset(original_sent_pos) or original_sent_pos.issubset(
            new_sent_pos
        ):
            filtered_new.append(sent)

    # filter on:
    #   - POS and/or DEP tags (at least the ones in original)
    #   - order of words if they are the same (nouns, except "I" and such)

    # print(filtered_new)

    return filtered_new


def main():
    args = create_arg_parser()

    all_files = list(Path(args.data_directory).glob('*.csv'))

    for i, data_file in enumerate(all_files):
        filename = str(data_file.stem).replace('.csv', '')
        output_file = f'{filename}_pegasus.csv'

        df = pd.read_csv(data_file)
        print(
            f"""
    Working on file {filename} ({i+1} / {len(all_files)})
    Creating new sentences for {len(df)} sentences
        """
        )

        records = []
        for _, row in df.iterrows():
            original_sent_id = row.id
            records.append(
                {"id": row.id, "sentence": row.sentence, "labels": row.labels}
            )

            # NOTE: maybe try to convert all sentences in one go, not sure if that
            # will speed it up, but it might be handy if we are sure we use the
            # total len measures, need to test!
            try:
                results = get_response([row.sentence])
            except Exception:
                continue

            results = validate_results(row.sentence, results)

            records.extend(
                [
                    {
                        "id": f"{original_sent_id}-pegaus-{i}",
                        "sentence": result,
                        "labels": row.labels,
                    }
                    for i, result in enumerate(results)
                ]
            )
        pd.DataFrame().from_records(records).to_csv(
            Path(args.output_dir) / output_file,
            index=False
        )


if __name__ == "__main__":
    main()
