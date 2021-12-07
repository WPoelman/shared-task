# shared-task
PreTENS shared task.

* Shared task website: [https://sites.google.com/view/semeval2022-pretens/home-page](https://sites.google.com/view/semeval2022-pretens/home-page)
* Original dataset repo: [https://github.com/shammur/SemEval2022Task3](https://github.com/shammur/SemEval2022Task3)


# Usage
1. (Optionally) create a virtual environment for this project (tested on python 3.8+)
2. `pip install -r requirements.txt`
3. Download an English spacy model for the data generation: `python -m spacy download en_core_web_sm`
4. `python3 baseline.py -mse` (for all languages, for all baselines)

## Optional arguments
```
  -m, --most_frequent   Use the most frequent class baselines

  -s, --svc             Use the TF-IDF and SVC as baseline

  -e, --embeddings      Use the multi-lingual sentence embedding baseline

  -l LANGUAGES [LANGUAGES ...], --languages LANGUAGES [LANGUAGES ...]	Language datasets to use for the baselines

  -v, --verbose         Show progression output

  -c, --cache           Use cached sentence embeddings with -e

  -cv CROSS_VALIDATION, --cross_validation CROSS_VALIDATION		Cross validation folds
  
  -mo MODEL, --model MODEL	Sentence embedding model to use with -e
```

## Data generation
### Pegasus English paraphrase
```
usage: generate_pegasus_data.py [-h] [-i INPUT_PATH] [-o OUTPUT_PATH] [-m MAX_NEW] [-tl]

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT_PATH, --input_path INPUT_PATH
                        English CSV with templates to expand.
  -o OUTPUT_PATH, --output_path OUTPUT_PATH
                        Output path for CSV with results.
  -m MAX_NEW, --max_new MAX_NEW
                        Max new sentences to generate per sentence
  -tl, --total_len      Use the overall min and max sentence lengths of the entire corpus, instead of individual sentences
```

## TODO
- [ ] test set maken (nieuwe templates, nieuwe woordsoorten/onderwerpen)
- [ ] unieke woorden uit de data halen en in woordenlijsten zetten
- [ ] wordnet 'zij-relaties' eruit halen (eventueel Babelnet -> has kind)
- [ ] code voor inverten erop zetten
- [ ] eerst focussen op Engels, daarna de nieuwe dataset daarvoor vertalen naar IT en FR
- [x] PEGASUS script erop zetten
- [ ] data generen samenvoegen in een script
- [ ] als we de nieuwe datasets hebben -> uitproberen wat het effect is van de datageneratietechnieken (los en gecombineerd)
- [ ] als we verbeteringen zien, Bert model verder trainen op de task ipv embeddings in SVM (peregine uitproberen)
