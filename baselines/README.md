# Baselines
These are the baseline scripts we used to try out various methods.
These were required for an assignment and are here mostly for completeness sake.
To use the newly generated dataset, please use `model_experiments/run_baselines.py`

## Arguments baselines
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