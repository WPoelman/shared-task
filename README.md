# shared-task
PreTENS shared task.

* Shared task website: [https://sites.google.com/view/semeval2022-pretens/home-page](https://sites.google.com/view/semeval2022-pretens/home-page)
* Original dataset repo: [https://github.com/shammur/SemEval2022Task3](https://github.com/shammur/SemEval2022Task3)


# Usage
1. `pip install -r requirements.txt`
2. `python3 baseline.py -mse` (for all languages, for all baselines)

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

## TODO
- [ ] test set maken (nieuwe templates, nieuwe woordsoorten/onderwerpen)
- [ ] unieke woorden uit de data halen en in woordenlijsten zetten
- [ ] wordnet 'zij-relaties' eruit halen (eventueel Babelnet -> has kind)
- [ ] code voor inverten erop zetten
- [ ] eerst focussen op Engels, daarna de nieuwe dataset daarvoor vertalen naar IT en FR
- [ ] PEGASUS script erop zetten
- [ ] data generen samenvoegen in een script
- [ ] als we de nieuwe datasets hebben -> uitproberen wat het effect is van de datageneratietechnieken (los en gecombineerd)
- [ ] als we verbeteringen zien, Bert model verder trainen op de task ipv embeddings in SVM (peregine uitproberen)
