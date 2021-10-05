# shared-task
PreTENS shared task.

* Shared task website: [https://sites.google.com/view/semeval2022-pretens/home-page](https://sites.google.com/view/semeval2022-pretens/home-page)
* Original dataset repo: [https://github.com/shammur/SemEval2022Task3](https://github.com/shammur/SemEval2022Task3)


# Usage
1. `pip install -r requirements.txt`
2. `python3 baseline.py`

# Optional arguments

  -m, --most_frequent   Use the most frequent class baselines

  -s, --svc             Use the TF-IDF and SVC as baseline

  -e, --embeddings      Use the multi-lingual sentence embedding baseline

  -l LANGUAGES [LANGUAGES ...], --languages LANGUAGES [LANGUAGES ...]	Language datasets to use for the baselines

  -v, --verbose         Show progression output

  -c, --cache           Use cached sentence embeddings with -e

  -cv CROSS_VALIDATION, --cross_validation CROSS_VALIDATION		Cross validation folds
  
  -mo MODEL, --model MODEL	Sentence embedding model to use with -e